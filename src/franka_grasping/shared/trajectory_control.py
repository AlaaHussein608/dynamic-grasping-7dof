"""
trajectory_control.py
=====================
Trajectory generation and torque-level control laws shared across the
scripts:

  * quintic joint trajectories (closed-form 10s³−15s⁴+6s⁵ blend) sampled
    with forward finite differences,
  * TOPP-RA time-optimal retiming of a straight joint-space segment,
  * RNEA feedforward torque precomputation,
  * adaptive sliding-mode control law,
  * feedback-linearization (computed-torque) control law,
  * gravity-compensated PD hold torque,
  * end-effector tip-wrench → joint torque (Pinocchio frame Jacobian).

All dynamics use the fast ``arm_mass_bias`` helper (crba + rnea, armature
always added) except the sliding-mode law, which needs the Coriolis matrix
explicitly and therefore keeps ``computeAllTerms``.
"""

import numpy as np
import pinocchio as pin

from .franka_common import DQ_MAX, TAU_MAX, arm_mass_bias


# ─────────────────────────────────────────────────────────────────────────────
#  Quintic trajectories
# ─────────────────────────────────────────────────────────────────────────────

def quintic_trajectory(theta_start, theta_end, Tf, N_sim):
    """Closed-form quintic joint path (positions only), shape (N, dof).

    Uses the minimum-jerk time scaling s(t) = 10(t/Tf)³ − 15(t/Tf)⁴ + 6(t/Tf)⁵
    with theta(t) = s·theta_end + (1−s)·theta_start (zero boundary velocity
    and acceleration)."""
    N = int(N_sim)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((N, len(theta_start)))
    for i in range(N):
        t = timegap * i
        s = 10 * (t / Tf)**3 - 15 * (t / Tf)**4 + 6 * (t / Tf)**5
        traj[i] = s * theta_end + (1 - s) * theta_start
    return traj


def sample_quintic_trajectory(q_start, q_end, Tf, n_steps, nq=7):
    """
    Quintic joint trajectory over Tf seconds with n_steps samples.

    Positions from :func:`quintic_trajectory`; velocities / accelerations
    from forward finite differences with dt = Tf / (n_steps - 1) and a zero
    first sample.

    Returns (pos, vel, acc) arrays of shape (n_steps, nq).
    """
    pos = quintic_trajectory(q_start, q_end, Tf, n_steps)
    vel = np.zeros((n_steps, nq))
    acc = np.zeros((n_steps, nq))
    dt  = Tf / (n_steps - 1.0)
    for i in range(n_steps - 1):
        vel[i + 1] = (pos[i + 1] - pos[i]) / dt
        acc[i + 1] = (vel[i + 1] - vel[i]) / dt
    return pos, vel, acc


# ─────────────────────────────────────────────────────────────────────────────
#  TOPP-RA retiming
# ─────────────────────────────────────────────────────────────────────────────

def toppra_segment(q_start, q_end, pin_model, pin_data):
    """
    Time-optimal retiming of the straight joint-space segment q_start → q_end
    under the Panda velocity and torque limits.
    """
    import toppra as ta
    import toppra.constraint as constraint

    def _inv_dyn(q, qd, qdd):
        return pin.rnea(pin_model, pin_data, q, qd, qdd)

    path      = ta.SplineInterpolator([0, 1], [q_start, q_end])
    pc_vel    = constraint.JointVelocityConstraint(DQ_MAX)
    tau_lim   = np.column_stack([-TAU_MAX, TAU_MAX])
    fs_coef   = np.zeros(len(TAU_MAX))
    pc_torque = constraint.JointTorqueConstraint(_inv_dyn, tau_lim, fs_coef)
    instance  = ta.algorithm.TOPPRA([pc_vel, pc_torque], path)
    return instance.compute_trajectory()


def sample_toppra_trajectory(toppra_traj, Tf, dt, min_steps=1):
    """Sample a TOPP-RA trajectory object at rate dt over [0, min(Tf, duration)].

    Returns (pos, vel, acc, n_steps)."""
    n_steps = max(int(Tf / dt), min_steps)
    t_pts   = np.linspace(0, min(Tf, toppra_traj.duration), n_steps)
    pos = np.array([toppra_traj(t)    for t in t_pts])
    vel = np.array([toppra_traj(t, 1) for t in t_pts])
    acc = np.array([toppra_traj(t, 2) for t in t_pts])
    return pos, vel, acc, n_steps


# ─────────────────────────────────────────────────────────────────────────────
#  Feedforward torque
# ─────────────────────────────────────────────────────────────────────────────

def precompute_feedforward(pin_model, pin_data, pos, vel, acc, nv=7):
    """tau_ff[i] = M*ddq_d + C*dq_d + g  via RNEA (computed offline)."""
    n_steps = pos.shape[0]
    tau_ff  = np.zeros((n_steps, nv))
    for i in range(n_steps):
        tau_ff[i] = pin.rnea(pin_model, pin_data, pos[i], vel[i], acc[i])
    return tau_ff


# ─────────────────────────────────────────────────────────────────────────────
#  Control laws
# ─────────────────────────────────────────────────────────────────────────────

def sliding_mode_control(q, q_dot, q_d, q_d_dot, q_d_ddot,
                         pin_model, pin_data, armature_diag,
                         Lambda, phi, K_min, K_max, alpha_adapt):
    """
    Adaptive sliding-mode (Slotine-Li) control for the 7-DOF Panda.

    tau = H q_r_ddot + C q_r_dot + g - K tanh(s/phi), with K adapting as
    K = clip(K_min + alpha*|s|, K_min, K_max). The Coriolis MATRIX is needed
    (it multiplies the reference velocity, not the actual one), so this law
    uses computeAllTerms; H has the armature added to its diagonal.

    Returns (tau, s).
    """
    n = 7

    # Errors and sliding surface
    q_tilde     = q     - q_d
    q_tilde_dot = q_dot - q_d_dot
    s           = q_tilde_dot + Lambda @ q_tilde

    # Reference signals
    q_r_dot  = q_d_dot  - Lambda @ q_tilde
    q_r_ddot = q_d_ddot - Lambda @ q_tilde_dot

    # Model estimates via Pinocchio (Coriolis matrix required)
    pin.computeAllTerms(pin_model, pin_data, q, q_dot)
    H_hat = pin_data.M + np.diag(armature_diag)
    C_hat = pin_data.C
    g_hat = pin.rnea(pin_model, pin_data, q, np.zeros(n), np.zeros(n))

    # Nominal control (computed torque)
    tau_hat = H_hat @ q_r_ddot + C_hat @ q_r_dot + g_hat

    # Adaptive K: gain grows with |s|, clamped to [K_min, K_max]
    K_new = np.empty(n)
    for i in range(n):
        K_new[i] = np.clip(K_min[i] + alpha_adapt[i] * abs(s[i]),
                           K_min[i], K_max[i])

    # Smooth switching term
    sgn_s  = np.array([np.tanh(s[i] / phi[i]) for i in range(n)])
    tau_sw = K_new * sgn_s

    return tau_hat - tau_sw, s


def feedback_linearization_control(q, q_dot, q_d, q_d_dot, q_d_ddot,
                                   pin_model, pin_data, armature_diag,
                                   kp, kd, tau_max=TAU_MAX):
    """
    Feedback-linearization (computed-torque) law:

        tau = M(q) (ddq_d + kp e + kd de) + C(q, dq) dq + g(q)

    with M = crba + diag(armature) and C dq + g computed together as
    rnea(q, dq, 0) (fast), clipped to tau_max.
    """
    error  = q_d     - q
    derror = q_d_dot - q_dot

    M, h = arm_mass_bias(pin_model, pin_data, q, q_dot, armature_diag)
    tau = M @ (q_d_ddot + kp * error + kd * derror) + h
    return np.clip(tau, -tau_max, tau_max)


def hold_pd_torque(pin_model, pin_data, q, q_dot, q_ref, kp, kd,
                   armature_diag, tau_max=None):
    """
    Gravity-compensated PD torque holding the arm at q_ref:

        tau = M (kp (q_ref - q) - kd dq) + C dq + g

    M = crba + diag(armature); C dq + g = rnea(q, dq, 0).
    ``tau_max`` (if given) clips the result.
    """
    M, h = arm_mass_bias(pin_model, pin_data, q, q_dot, armature_diag)
    tau = M @ (kp * (q_ref - q) - kd * q_dot) + h
    if tau_max is not None:
        tau = np.clip(tau, -tau_max, tau_max)
    return tau


# ─────────────────────────────────────────────────────────────────────────────
#  End-effector tip wrench → joint torque (Pinocchio frame Jacobian)
# ─────────────────────────────────────────────────────────────────────────────

def ee_tip_torque(pin_model, pin_data, q, ftip, frame_id):
    """
    Joint torques produced by an end-effector wrench ``ftip`` applied at
    ``frame_id`` (the gripper frame): tau = J(q)^T ftip.

    Replaces modern-robotics ``JacobianBody``; the frame Jacobian is taken in
    the LOCAL frame. ``ftip`` is in Pinocchio Force convention:
    [f_x, f_y, f_z, tau_x, tau_y, tau_z] (linear first, angular last).
    """
    J = pin.computeFrameJacobian(pin_model, pin_data, q, frame_id,
                                 pin.ReferenceFrame.LOCAL)
    return J.T @ ftip
