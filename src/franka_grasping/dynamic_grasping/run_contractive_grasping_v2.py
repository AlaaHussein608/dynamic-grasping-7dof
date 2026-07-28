"""
run_contractive_grasping_v2.py — contractive MPC for dynamic ball grasping (v2).

v2 improvements over run_contractive_grasping.py: state-aware contraction
factor, grid-searched timed intercept with a hover parking configuration,
lock-drift validation, QP-failure fallback (last clean torque + warm-start
reset), and a wait routine that detects a receding / stopped ball.

Contraction-factor selection (state-aware variant) and the bang-bang
travel-time bound live in contraction_gains.py; IK, grasp geometry and
workspace checks come from the shared modules.

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
                               init_warm_start, pin_initial_state)
from contraction_gains import (compute_alpha_state_aware, min_travel_time,
                               ALPHA_HI)

SCRIPT_DIR = script_dir(__file__)

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

EPSILON_PREGRASP   = 0.15
EPSILON_GRASP      = 0.05
PREGRASP_PROXIMITY = 0.10

GRIPPER_CLOSE_TIME  = 0.20
INTERCEPT_XY_THRESH = 0.035
INTERCEPT_Z_THRESH  = 0.012
WAIT_TIMEOUT        = 8.0
RECEDE_STEPS        = 100

GRIP_SAFETY        = 1.8

DZ_LIFT      = 0.15
FINAL_CONFIG = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
TF_ZLIFT     = 2.0
TF_SWING     = 3.5

MPC_DT  = 0.01
N       = 10
T_CATCH = N * MPC_DT

# ── Intercept planning ────────────────────────────────────────────────────────
T_INT_MIN      = T_CATCH + GRIPPER_CLOSE_TIME
T_INT_MAX      = 3.0
T_INT_STEP     = 0.05
TIME_MARGIN    = 1.25
PLANNER_PERIOD = 5
LOCK_DRIFT_TOL = 0.05
T_DESCEND      = 0.8
HOVER_MARGIN   = 0.03


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
data.qvel[-6:] = [0., 0.8, 0., 0., 0., 0.]
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

# Hover offset: ball radius + margin, so the parked gripper clears a passing ball.
PREGRASP_OFFSET = sphere_radius + HOVER_MARGIN


# ══════════════════════════════════════════════════════════════════════════════
#  Grasp geometry
# ══════════════════════════════════════════════════════════════════════════════

def compute_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius=None):
    """4×4 EE target: X along ball XY velocity, Z pointing down."""
    if ball_radius is None:
        ball_radius = sphere_radius
    return single_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius,
                             pregrasp_offset=PREGRASP_OFFSET)


def solve_timed_intercept(p_ball, v_ball, q_curr, q_seed):
    """
    Search the grid T_INT_MIN … T_INT_MAX for the earliest feasible intercept
    (in workspace, IK-solvable, reachable at least GRIPPER_CLOSE_TIME early).
    Returns (T_int, p_int, q_int, q_hover) or None.
    """
    was_inside = False
    for T_int in np.arange(T_INT_MIN, T_INT_MAX + 1e-9, T_INT_STEP):
        p_int    = p_ball + v_ball * T_int
        p_int[2] = sphere_radius

        if not check_workspace(p_int):
            if was_inside:
                break               # ball has left the workspace — stop
            continue
        was_inside = True

        T_pose     = compute_grasp_pose(p_int, v_ball, use_pregrasp=False)
        q_cand, ok = IK_pinocchio(T_pose, q_seed)
        if not ok:
            continue

        if min_travel_time(q_curr, q_cand, TIME_MARGIN) <= T_int - GRIPPER_CLOSE_TIME:
            T_hover       = compute_grasp_pose(p_int, v_ball, use_pregrasp=True)
            q_hover, ok_h = IK_pinocchio(T_hover, q_cand)
            if not ok_h:
                q_hover = q_cand
            return float(T_int), p_int.copy(), q_cand.copy(), q_hover.copy()

    return None


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
    pin.forwardKinematics(ik_model, ik_data, q, dq, ddq_des)
    pin.updateFramePlacements(ik_model, ik_data)
    wrench = np.array([0.0, 0.0, sphere_mass * 9.81, 0.0, 0.0, 0.0])
    J = pin.computeFrameJacobian(ik_model, ik_data, q, ik_ee_id,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J.T @ wrench


# ══════════════════════════════════════════════════════════════════════════════
#  Gripper routines
# ══════════════════════════════════════════════════════════════════════════════

def close_gripper(TT, theta_d):
    """Ramp fingers closed over 0.75*TT, then squeeze until contact is firm."""
    T_ramp   = 0.75 * TT
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
    """
    Hold theta_hold (fingers actively open) until the ball is one gripper-close
    time away, then close. A ball stopped within the intercept tolerance is
    closed on directly. Returns the grip torque, or None (receding / stopped
    out of reach / timed out) so the caller can replan.
    """
    dt             = model.opt.timestep
    max_wait_steps = int(WAIT_TIMEOUT / dt)
    prev_dist      = np.inf
    receding       = 0
    triggered      = False

    for _ in range(max_wait_steps):
        p_ball_now = data.xpos[sphere_id].copy()
        v_ball_now = data.qvel[-6:-3].copy()
        speed_now  = np.linalg.norm(v_ball_now[:2])
        ee_xy      = data.site_xpos[ee_site_id][:2]
        dist_2d    = np.linalg.norm(p_ball_now[:2] - ee_xy)

        data.ctrl[:7] = arm_torque_hold(theta_hold)
        data.ctrl[7]  = 200.0 * (0.04 - data.qpos[7])

        if speed_now > 1e-6 and dist_2d <= speed_now * GRIPPER_CLOSE_TIME:
            triggered = True
            break

        if speed_now < 1e-3:
            if dist_2d < INTERCEPT_XY_THRESH:
                triggered = True                        # stopped in the grasp
                break
            return None                                 # stopped out of reach

        receding = receding + 1 if dist_2d > prev_dist + 1e-6 else 0
        if receding > RECEDE_STEPS:
            return None                                 # missed — replan
        prev_dist = dist_2d

        mujoco.mj_step(model, data)
        viewer.sync()

    if not triggered:
        return None

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


def reset_warm_start(x):
    init_warm_start(solver, N, x, nu=NU)


_q_init = data.qpos[:7].copy()
_x_init = np.concatenate([_q_init, data.qvel[:7]])
reset_warm_start(_x_init)


# ══════════════════════════════════════════════════════════════════════════════
#  Main MPC loop
# ══════════════════════════════════════════════════════════════════════════════

viewer = mujoco.viewer.launch_passive(model, data)

phase         = 'pregrasp'          # 'pregrasp' | 'grasp' | 'hold'
q_target_prev = _q_init.copy()
q_hold        = None
grip_torque   = None
# Consistent lock: (t_lock, T_int, p_int, q_int, q_hover).
intercept_locked = None

qp_fail    = 0
u_prev     = np.zeros(NU)
iter_count = 0

print("Starting dynamic grasping MPC …")

while viewer.is_running():

    q_curr  = data.qpos[:7].copy()
    dq_curr = data.qvel[:7].copy()
    x_curr  = np.concatenate([q_curr, dq_curr])

    p_ball = data.xpos[sphere_id].copy()
    v_ball = data.qvel[-6:-3].copy()

    ee_pos = get_ee_position(q_curr)

    # ── Intercept planning (validate lock, replan when unlocked) ─────────────
    if phase in ('pregrasp', 'grasp'):
        if intercept_locked is not None:
            t_lock, T_int, p_int, _, _ = intercept_locked
            t_rem   = t_lock + T_int - data.time
            p_check = p_ball + v_ball * max(t_rem, 0.0)
            if t_rem < -0.5 or np.linalg.norm(p_check[:2] - p_int[:2]) > LOCK_DRIFT_TOL:
                intercept_locked = None
                phase = 'pregrasp'

        if intercept_locked is None and iter_count % PLANNER_PERIOD == 0:
            result = solve_timed_intercept(p_ball, v_ball, q_curr, q_target_prev)
            if result is not None:
                intercept_locked = (data.time, *result)

    # ── Phase logic ───────────────────────────────────────────────────────────
    if phase == 'pregrasp':
        if intercept_locked is not None:
            _, _, p_int, _, q_hover = intercept_locked
            p_hover = p_int + np.array([0.0, 0.0, PREGRASP_OFFSET])
            q_target = q_hover

            if np.linalg.norm(ee_pos - p_hover) < PREGRASP_PROXIMITY:
                phase = 'grasp'
        else:
            p_pred = p_ball + v_ball * T_CATCH
            T_pre  = compute_grasp_pose(p_pred, v_ball, use_pregrasp=True)
            T_pre[0, 3] += v_ball[0] * 0.2
            T_pre[1, 3] += v_ball[1] * 0.2

            if check_workspace(T_pre[:3, 3]):
                q_cand, ok = IK_pinocchio(T_pre, q_target_prev)
                q_target = q_cand if ok else q_target_prev.copy()
            else:
                q_target = q_target_prev.copy()

    elif phase == 'grasp':
        if intercept_locked is None:
            q_target = q_target_prev.copy()
            phase = 'pregrasp'
        else:
            t_lock, T_int, p_int, q_int, q_hover = intercept_locked
            t_rem = t_lock + T_int - data.time

            if t_rem > T_DESCEND:
                q_target = q_hover
            else:
                q_target = q_int

                xy_err = np.linalg.norm(ee_pos[:2] - p_int[:2])
                z_err  = abs(ee_pos[2] - sphere_radius)

                if xy_err < INTERCEPT_XY_THRESH and z_err < INTERCEPT_Z_THRESH:
                    raw_grip = wait_and_grasp(q_int)

                    if raw_grip is not None:
                        grip_torque = GRIP_SAFETY * raw_grip

                        T_zlift        = get_ee_pose(data.qpos[:7])
                        T_zlift[2, 3] += DZ_LIFT
                        q_zlift, ok_z  = IK_pinocchio(T_zlift, data.qpos[:7])
                        if ok_z:
                            run_trajectory(TF_ZLIFT, data.qpos[:7], q_zlift,
                                           grip_torque=grip_torque, use_ftip=True)

                        run_trajectory(TF_SWING, data.qpos[:7], FINAL_CONFIG,
                                       grip_torque=grip_torque, use_ftip=True)

                        q_hold = FINAL_CONFIG.copy()
                        phase  = 'hold'
                    else:
                        phase = 'pregrasp'   # missed the ball — replan

                    intercept_locked = None
                    q_curr  = data.qpos[:7].copy()
                    dq_curr = data.qvel[:7].copy()
                    x_curr  = np.concatenate([q_curr, dq_curr])
                    q_target = q_hold if phase == 'hold' else q_curr.copy()
                    reset_warm_start(x_curr)

    else:  # phase == 'hold'
        q_target = q_hold

    q_target_prev = q_target.copy()

    # ── Gripper command ───────────────────────────────────────────────────────
    if phase == 'hold' and grip_torque is not None:
        data.ctrl[7] = -grip_torque
    else:
        data.ctrl[7] = 200.0 * (0.04 - data.qpos[7])   # fingers open

    # ── Contraction factor (relaxed geometrically after QP failures) ─────────
    epsilon = EPSILON_PREGRASP if phase == 'pregrasp' else EPSILON_GRASP
    alpha   = compute_alpha_state_aware(q_target - q_curr, dq_curr, epsilon,
                                        N, MPC_DT)
    if qp_fail:
        alpha = min(1.0 - (1.0 - alpha) * 0.5 ** min(qp_fail, 6), ALPHA_HI)

    # ── Solver references ─────────────────────────────────────────────────────
    xs_curr = np.concatenate([q_target, np.zeros(NV)])
    p_vec   = np.concatenate([xs_curr, [alpha]])
    yref    = np.concatenate([xs_curr, np.zeros(NU)])
    for k in range(N):
        solver.set(k, 'p',    p_vec)
        solver.set(k, 'yref', yref)
    solver.set(N, 'yref', xs_curr)

    pin_initial_state(solver, x_curr)

    status = solver.solve()
    if status == 0:
        u_prev  = solver.get(0, 'u').copy()
        qp_fail = max(qp_fail - 1, 0)
    else:
        qp_fail += 1
        reset_warm_start(x_curr)    # stale internal trajectory — start clean

    data.ctrl[:7] = np.clip(u_prev, -TAU_MAX, TAU_MAX)

    for _ in range(STEPS_PER_INTERVAL):
        mujoco.mj_step(model, data)
        viewer.sync()

    if status == 0:
        for k in range(N - 1):
            solver.set(k, 'x', solver.get(k + 1, 'x'))
            solver.set(k, 'u', solver.get(k + 1, 'u'))
        solver.set(N - 1, 'u', np.zeros(NU))

    iter_count += 1
