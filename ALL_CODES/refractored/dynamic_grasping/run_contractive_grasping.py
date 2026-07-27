"""
run_contractive_grasping.py — contractive MPC for dynamic ball grasping (v1).

Contraction-factor selection (error-based variant) lives in
contraction_gains.py; IK, grasp geometry and workspace checks come from the
shared modules.

Requires the contraction-constrained ACADOS solver
(run build_contractive_solver.py → franka_point_stab_contractive.json).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import pinocchio as pin

from shared.franka_common import (SPHERE_SCENE_XML, Q_HOME, TAU_MAX,
                                  NQ, NV, NU, check_workspace, single_grasp_pose,
                                  ee_position, ee_pose, IK_pinocchio,
                                  compute_F_thresh, arm_mass_bias,
                                  ik_model, ik_data, ik_ee_id, script_dir)
from shared.trajectory_control import sample_quintic_trajectory
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state)
from contraction_gains import compute_alpha_error_based, intercept_is_feasible

SCRIPT_DIR = script_dir(__file__)

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

EPSILON_PREGRASP   = 0.15   # [rad] convergence tolerance in pregrasp phase
EPSILON_GRASP      = 0.05   # [rad] convergence tolerance in grasp / hold phase
PREGRASP_PROXIMITY = 0.10   # [m]   EE-to-target distance to latch into grasp

GRIPPER_CLOSE_TIME  = 0.20   # [s] gripper lead time
INTERCEPT_XY_THRESH = 0.035  # [m] XY error threshold to enter wait/grasp routine
INTERCEPT_Z_THRESH  = 0.012  # [m] Z  error threshold to enter wait/grasp routine
WAIT_TIMEOUT        = 8.0    # [s] safety cap on the wait loop

GRIP_SAFETY        = 1.8     # [-] multiply the held grip torque for lift margin

DZ_LIFT      = 0.15
FINAL_CONFIG = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
TF_ZLIFT     = 2.0    # [s] vertical pull-up duration
TF_SWING     = 3.5    # [s] swing-to-retract duration

MPC_DT  = 0.01        # MPC discretisation step [s]
N       = 10          # prediction horizon (steps)
T_CATCH = N * MPC_DT  # lookahead horizon [s]


# ══════════════════════════════════════════════════════════════════════════════
#  Kinematics helpers (shared IK Pinocchio model)
# ══════════════════════════════════════════════════════════════════════════════

def get_ee_position(q):
    return ee_position(ik_model, ik_data, ik_ee_id, q)


def get_ee_pose(q):
    return ee_pose(ik_model, ik_data, ik_ee_id, q)


# ══════════════════════════════════════════════════════════════════════════════
#  MuJoCo model
# ══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)

data.qpos[:7]  = Q_HOME.copy()
data.qpos[-7:] = [0.5, -0.8, 0.03, 1., 0., 0., 0.]
data.qvel[-6:] = [0., 0.4, 0., 0., 0., 0.]
data.qvel[:7]  = np.zeros(NV)
mujoco.mj_forward(model, data)

STEPS_PER_INTERVAL = int(MPC_DT / model.opt.timestep)
armature = model.dof_armature[:7]

sphere_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
sphere_mass = model.body_mass[sphere_id]
sphere_geom_id = next(i for i in range(model.ngeom)
                      if model.geom_bodyid[i] == sphere_id)
sphere_radius = model.geom_size[sphere_geom_id, 0]

left_pad_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
finger_geoms = {left_pad_id, right_pad_id}
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

# Pre-grasp hover offset equals the ball radius so the EE just clears the top.
PREGRASP_OFFSET = sphere_radius


# ══════════════════════════════════════════════════════════════════════════════
#  Grasp geometry
# ══════════════════════════════════════════════════════════════════════════════

def compute_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius=None):
    """4×4 EE target: X along ball XY velocity, Z pointing down."""
    if ball_radius is None:
        ball_radius = sphere_radius
    return single_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius,
                             pregrasp_offset=PREGRASP_OFFSET)


def solve_timed_intercept(p_ball, v_ball, q_curr, lead_time=GRIPPER_CLOSE_TIME):
    """Compute a feasible intercept point T_CATCH + lead_time seconds ahead.
    Returns (p_intercept, q_intercept) on success, None otherwise."""
    T_int    = T_CATCH + lead_time
    p_int    = p_ball + v_ball * T_int
    p_int[2] = sphere_radius

    if not check_workspace(p_int):
        return None

    T_pose     = compute_grasp_pose(p_int, v_ball, use_pregrasp=False)
    q_cand, ok = IK_pinocchio(T_pose, q_curr)
    if not ok:
        return None

    error_norm = np.linalg.norm(q_curr - q_cand)
    if not intercept_is_feasible(error_norm, EPSILON_GRASP, N, MPC_DT):
        return None

    return p_int.copy(), q_cand.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamics helpers
# ══════════════════════════════════════════════════════════════════════════════

def F_thresh_now():
    return compute_F_thresh(model, sphere_geom_id, left_pad_id, sphere_mass,
                            rule="mean")


def arm_torque_hold(theta_d):
    """Gravity-compensated PD torque that holds the arm at theta_d (fast)."""
    q  = data.qpos[:7].copy()
    dq = data.qvel[:7].copy()
    M, h = arm_mass_bias(ik_model, ik_data, q, dq, armature)
    return M @ (500.0 * (theta_d - q) - 200.0 * dq) + h


def contact_force_ok(F_thresh):
    """True when a finger pad reports a normal force above 10× F_thresh."""
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if ((g1 == sphere_geom_id and g2 in finger_geoms) or
                (g2 == sphere_geom_id and g1 in finger_geoms)):
            mujoco.mj_contactForce(model, data, i, force)
            if abs(force[0]) > 10.0 * F_thresh:
                return True
    return False


def ee_support_torque(q, dq, ddq_des):
    """Joint torques supporting the grasped ball's weight (feedforward)."""
    pin.forwardKinematics(ik_model, ik_data, q, dq, ddq_des)
    pin.updateFramePlacements(ik_model, ik_data)
    wrench = np.array([0.0, 0.0, sphere_mass * 9.81, 0.0, 0.0, 0.0])
    J = pin.computeFrameJacobian(ik_model, ik_data, q, ik_ee_id,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J.T @ wrench


# ══════════════════════════════════════════════════════════════════════════════
#  Gripper routines (mj_contactForce contact check, phase-integrated)
# ══════════════════════════════════════════════════════════════════════════════

def close_gripper(TT, theta_d):
    """Ramp fingers closed over 0.9*TT, then squeeze until contact is firm."""
    T_ramp   = 0.9 * TT
    F_thresh = F_thresh_now()
    dt       = model.opt.timestep
    n_steps  = max(int(T_ramp / dt), 1)

    q_finger_open = data.qpos[7]
    target_closed = sphere_radius
    kp, kd = 500.0, 50.0

    for step in range(n_steps):
        t_ratio        = step / n_steps
        gripper_target = q_finger_open - t_ratio * (q_finger_open - target_closed)
        data.ctrl[7]   = kp * (gripper_target - data.qpos[7]) + kd * (-data.qvel[7])
        data.ctrl[:7]  = arm_torque_hold(theta_d)
        mujoco.mj_step(model, data)
        viewer.sync()

    alpha = 1.0
    while True:
        data.ctrl[7]  = -alpha * F_thresh
        data.ctrl[:7] = arm_torque_hold(theta_d)
        mujoco.mj_step(model, data)
        viewer.sync()
        if contact_force_ok(F_thresh):
            return alpha * F_thresh
        alpha += 0.4


def wait_and_grasp(theta_hold):
    """Hold theta_hold, fingers open, until the ball is one gripper-close time
    away, then close. Returns the grip torque."""
    dt             = model.opt.timestep
    max_wait_steps = int(WAIT_TIMEOUT / dt)

    for _ in range(max_wait_steps):
        p_ball_now = data.xpos[sphere_id].copy()
        v_ball_now = data.qvel[-6:-3].copy()
        speed_now  = np.linalg.norm(v_ball_now[:2])
        ee_xy      = data.site_xpos[ee_site_id][:2]
        dist_2d    = np.linalg.norm(p_ball_now[:2] - ee_xy)

        data.ctrl[:7] = arm_torque_hold(theta_hold)
        data.ctrl[7]  = 0.0

        if speed_now > 1e-6 and dist_2d <= speed_now * GRIPPER_CLOSE_TIME:
            break

        mujoco.mj_step(model, data)
        viewer.sync()

    speed_now = np.linalg.norm(data.qvel[-6:-3][:2])
    dist_2d   = np.linalg.norm(
        data.xpos[sphere_id][:2] - data.site_xpos[ee_site_id][:2])
    T_close = (dist_2d / speed_now) if speed_now > 1e-6 else GRIPPER_CLOSE_TIME
    return close_gripper(T_close, theta_hold)


def run_trajectory(Tf, theta_start, theta_end, grip_torque,
                   open_grip=False, use_ftip=False, kp=800.0, kd=40.0, ki=300.0):
    """Execute a quintic PID trajectory over Tf seconds."""
    dt = model.opt.timestep
    n  = max(int(Tf / dt), 2)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_start, theta_end, Tf, n)
    integral = np.zeros(NQ)
    t_start  = data.time

    while True:
        elapsed = data.time - t_start
        if elapsed >= Tf:
            break

        i          = min(int((elapsed / Tf) * (n - 1)), n - 1)
        thetalist  = data.qpos[:7].copy()
        dthetalist = data.qvel[:7].copy()

        error     = thetamatd[i]  - thetalist
        derror    = dthetamatd[i] - dthetalist
        integral += error * dt

        M, h = arm_mass_bias(ik_model, ik_data, thetalist, dthetalist, armature)
        tau  = M @ (ddthetamatd[i] + kp * error + kd * derror + ki * integral) + h

        if use_ftip:
            tau += ee_support_torque(thetalist, dthetalist, ddthetamatd[i])

        data.ctrl[:7] = np.clip(tau, -TAU_MAX, TAU_MAX)
        data.ctrl[7]  = (200.0 * (0.04 - data.qpos[7])
                         if open_grip else -grip_torque)

        mujoco.mj_step(model, data)
        viewer.sync()


# ══════════════════════════════════════════════════════════════════════════════
#  ACADOS solver setup
# ══════════════════════════════════════════════════════════════════════════════

print("Loading ACADOS solver …")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab_contractive.json')
print("Solver loaded.")

_Q_cost = 200.0 * np.diag([300.0] * NQ + [30.0] * NV)
_R_cost = np.diag([0.5] * NU)
apply_cost_weights(solver, N, _Q_cost, _R_cost)

_q_init = data.qpos[:7].copy()
_x_init = np.concatenate([_q_init, data.qvel[:7]])
init_warm_start(solver, N, _x_init, nu=NU)


# ══════════════════════════════════════════════════════════════════════════════
#  Main MPC loop
# ══════════════════════════════════════════════════════════════════════════════

viewer = mujoco.viewer.launch_passive(model, data)

phase            = 'pregrasp'   # 'pregrasp' | 'grasp' | 'hold'
q_target_prev    = _q_init.copy()
q_hold           = None
grip_torque      = None
intercept_locked = None   # (p_intercept, q_intercept) once locked
print("Starting dynamic grasping MPC …")

while viewer.is_running():

    q_curr  = data.qpos[:7].copy()
    dq_curr = data.qvel[:7].copy()
    x_curr  = np.concatenate([q_curr, dq_curr])

    p_ball = data.xpos[sphere_id].copy()
    v_ball = data.qvel[-6:-3].copy()
    p_pred = p_ball + v_ball * T_CATCH

    ee_pos = get_ee_position(q_curr)

    # ── Phase logic ───────────────────────────────────────────────────────────
    if phase == 'pregrasp':
        T_pre = compute_grasp_pose(p_pred, v_ball, use_pregrasp=True)
        T_pre[0, 3] += v_ball[0] * 0.2
        T_pre[1, 3] += v_ball[1] * 0.2

        if check_workspace(T_pre[:3, 3]):
            q_target, ok = IK_pinocchio(T_pre, q_curr)
            if not ok:
                q_target = q_target_prev.copy()
        else:
            q_target = q_target_prev.copy()

        if np.linalg.norm(ee_pos - T_pre[:3, 3]) < PREGRASP_PROXIMITY:
            intercept_locked = None
            phase = 'grasp'

    elif phase == 'grasp':
        if intercept_locked is None:
            result = solve_timed_intercept(p_ball, v_ball, q_curr)
            if result is not None:
                intercept_locked = result

        if intercept_locked is not None:
            p_intercept, q_target = intercept_locked
        else:
            pp_pred    = p_ball + v_ball * (T_CATCH + GRIPPER_CLOSE_TIME)
            T_hover    = compute_grasp_pose(pp_pred, v_ball, use_pregrasp=True)
            q_cand, ok = IK_pinocchio(T_hover, q_curr)
            if ok:
                q_target    = q_cand
                p_intercept = T_hover[:3, 3]
            else:
                q_target    = q_target_prev.copy()
                p_intercept = ee_pos.copy()

        xy_err = np.linalg.norm(ee_pos[:2] - p_intercept[:2])
        z_err  = abs(ee_pos[2] - sphere_radius)

        if (intercept_locked is not None
                and xy_err < INTERCEPT_XY_THRESH
                and z_err  < INTERCEPT_Z_THRESH):

            _, q_intercept = intercept_locked
            raw_grip    = wait_and_grasp(q_intercept)
            grip_torque = GRIP_SAFETY * raw_grip

            T_zlift        = get_ee_pose(data.qpos[:7])
            T_zlift[2, 3] += DZ_LIFT
            q_zlift, ok_z  = IK_pinocchio(T_zlift, data.qpos[:7])
            if ok_z:
                run_trajectory(TF_ZLIFT, data.qpos[:7], q_zlift,
                               grip_torque=grip_torque, use_ftip=True)

            run_trajectory(TF_SWING, data.qpos[:7], FINAL_CONFIG,
                           grip_torque=grip_torque, use_ftip=True)

            q_hold           = FINAL_CONFIG.copy()
            phase            = 'hold'
            intercept_locked = None

            q_curr   = data.qpos[:7].copy()
            dq_curr  = data.qvel[:7].copy()
            x_curr   = np.concatenate([q_curr, dq_curr])
            q_target = q_hold

    else:  # phase == 'hold'
        q_target = q_hold

    q_target_prev = q_target.copy()

    # ── Gripper command ───────────────────────────────────────────────────────
    if phase == 'hold' and grip_torque is not None:
        data.ctrl[7] = -grip_torque
    else:
        data.ctrl[7] = 200.0 * (0.04 - data.qpos[7])   # fingers open

    # ── Contraction factor ────────────────────────────────────────────────────
    epsilon    = EPSILON_PREGRASP if phase == 'pregrasp' else EPSILON_GRASP
    error_norm = float(np.linalg.norm(q_curr - q_target))
    alpha      = compute_alpha_error_based(error_norm, epsilon, N, MPC_DT)

    # ── Solver references ─────────────────────────────────────────────────────
    xs_curr = np.concatenate([q_target, np.zeros(NV)])
    p_vec   = np.concatenate([xs_curr, [alpha]])
    yref    = np.concatenate([xs_curr, np.zeros(NU)])
    for k in range(N):
        solver.set(k, 'p',    p_vec)
        solver.set(k, 'yref', yref)
    solver.set(N, 'yref', xs_curr)

    pin_initial_state(solver, x_curr)
    solver.solve()

    u = solver.get(0, 'u').copy()
    data.ctrl[:7] = np.clip(u, -TAU_MAX, TAU_MAX)

    for _ in range(STEPS_PER_INTERVAL):
        mujoco.mj_step(model, data)
        viewer.sync()

    shift_warm_start(solver, N, zero_last_u=True, nu=NU)
