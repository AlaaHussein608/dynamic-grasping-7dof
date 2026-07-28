"""
benchmark_grasping_methods.py
─────────────────────────────
Compares three dynamic-grasping methods over N_TRIALS random ball launches.

Methods
-------
  TOPPRA      : intercept_planner_toppra.find_intercept  (brute-force scan)
  CROFT       : intercept_planner_croft.find_intercept   (bracketed root-finding)
  CONTRACTIVE : online reactive MPC — tracks the ball continuously and
                commits to an intercept as soon as the arm can reach it in time

All three use the same ACADOS solver (franka_point_stab_planning.json,
h=0.005, N=8 — build with build_solver.py --planning). The contractive method
has no offline planning phase and uses a quintic-time feasibility check.

Metrics (mean over successful trials unless noted): success, catch_time,
search_time, tau_rms, ee_path_length, joint_jerk_rms, grasp_distance.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import pinocchio as pin

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_HOME,
                                  TAU_MAX, NQ, NV, NU,
                                  PANDA_MIN_REACH, PANDA_BODY_RADIUS,
                                  check_workspace, single_grasp_pose,
                                  IK_pinocchio, quintic_min_time,
                                  compute_F_thresh, script_dir)
from shared.trajectory_control import (toppra_segment as _toppra_segment,
                                       hold_pd_torque)
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state)
import intercept_planner_croft
import intercept_planner_toppra

SCRIPT_DIR = script_dir(__file__)

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

N_TRIALS      = 20
MAX_SIM_TIME  = 30.0
MAX_WALL_TIME = 30.0
HEADLESS      = True
SEED          = 42

RNG = np.random.default_rng(SEED)

X_RANGE  = (-2.0,  2.0)
Y_RANGE  = (-2.0,  2.0)
VX_RANGE = ( 0.0,  0.5)
VY_RANGE = ( 0.0,  0.5)
Z_INIT   = 0.03

ARM_Q0 = Q_HOME.copy()

P_H = 0.005      # MPC timestep [s]
P_N = 8          # MPC horizon

GRIP_THRESH_MULT = 10.0

# ── Contractive MPC specific params ──────────────────────────────────────────
C_PREGRASP_LOOKAHEAD = 0.5
C_GRIPPER_CLOSE_TIME = 0.20
C_INTERCEPT_XY       = 0.035
C_INTERCEPT_Z        = 0.012
C_WAIT_TIMEOUT       = 8.0
C_INTERCEPT_SCAN_DT  = 0.05
C_INTERCEPT_T_MAX    = 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  MuJoCo model
# ══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)

sphere_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "sphere")
sphere_mass    = model.body_mass[sphere_id]
sphere_geom_id = next(i for i in range(model.ngeom)
                      if model.geom_bodyid[i] == sphere_id)
sphere_radius  = model.geom_size[sphere_geom_id, 0]

left_pad_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
finger_geoms = {left_pad_id, right_pad_id}
ee_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  "gripper")
finger1_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")

P_STEPS  = int(P_H / model.opt.timestep)
armature = model.dof_armature[:7]


def reset_sim(p0, v0):
    mujoco.mj_resetData(model, data)
    data.qpos[:7]  = ARM_Q0
    data.qpos[-7:] = [p0[0], p0[1], p0[2], 1., 0., 0., 0.]
    data.qvel[-6:] = [v0[0], v0[1], 0., 0., 0., 0.]
    data.qvel[:7]  = np.zeros(NQ)
    mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════════════════════════════════════
#  Trial generation  (workspace-filtered)
# ══════════════════════════════════════════════════════════════════════════════

def _ball_enters_workspace(p0, v0, t_max=15.0, dt=0.1):
    for t in np.arange(0.0, t_max, dt):
        p   = p0 + v0 * t
        d3  = np.linalg.norm(p)
        dxy = np.linalg.norm(p[:2])
        if (PANDA_MIN_REACH <= d3 <= 0.5
                and dxy >= PANDA_BODY_RADIUS
                and p[2] >= 0.0):
            return True
    return False


def sample_trials(n):
    trials, attempts = [], 0
    while len(trials) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError(
                f"Only generated {len(trials)}/{n} workspace-reachable trials "
                f"after {attempts} samples — widen VX/VY range.")
        x  = float(RNG.uniform(*X_RANGE))
        y  = float(RNG.uniform(*Y_RANGE))
        vx = float(RNG.uniform(*VX_RANGE))
        vy = float(RNG.uniform(*VY_RANGE))
        p0 = np.array([x, y, Z_INIT])
        v0 = np.array([vx, vy, 0.0])
        if _ball_enters_workspace(p0, v0):
            trials.append({"p0": p0, "v0": v0})
    print(f"Generated {n} workspace-reachable trials "
          f"({attempts} total samples, {attempts - n} rejected).")
    return trials


# ══════════════════════════════════════════════════════════════════════════════
#  Pinocchio model
# ══════════════════════════════════════════════════════════════════════════════

pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()
ee_id     = pin_model.getFrameId("gripper")


def get_ee_pos():
    pin.forwardKinematics(pin_model, pin_data, data.qpos[:7].copy())
    pin.updateFramePlacement(pin_model, pin_data, ee_id)
    return pin_data.oMf[ee_id].translation.copy()


def get_ee_pos_q(q):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacement(pin_model, pin_data, ee_id)
    return pin_data.oMf[ee_id].translation.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  ACADOS solver  (single solver used by all three methods)
# ══════════════════════════════════════════════════════════════════════════════

print("Loading solver (h=0.005, N=8) …")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab_planning.json')
print("Solver loaded.\n")

_P_Q = 200.0 * np.diag([300.] * NQ + [30.] * NV)
_P_R = np.diag([1.0] * NU)
apply_cost_weights(solver, P_N, _P_Q, _P_R)


def _warmstart():
    x0 = np.concatenate([data.qpos[:7], data.qvel[:7]])
    init_warm_start(solver, P_N, x0, nu=NU)


def _shift():
    shift_warm_start(solver, P_N)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def F_thresh_now():
    return compute_F_thresh(model, sphere_geom_id, left_pad_id, sphere_mass,
                            rule="mean")


def contact_force_ok(F_thresh):
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if ((g1 == sphere_geom_id and g2 in finger_geoms) or
                (g2 == sphere_geom_id and g1 in finger_geoms)):
            mujoco.mj_contactForce(model, data, i, force)
            if abs(force[0]) > GRIP_THRESH_MULT * F_thresh:
                return True
    return False


def hold_steady(kp=8000, kd=40):
    q_hold = data.qpos[:7].copy()
    q, dq  = data.qpos[:7], data.qvel[:7]
    tau = hold_pd_torque(pin_model, pin_data, q, dq, q_hold, kp, kd, armature)
    data.ctrl[:7] = np.clip(tau, -TAU_MAX, TAU_MAX)


def toppra_segment(q_start, q_end):
    # TOPP-RA fails on near-zero paths (arm already at target) — skip them.
    if np.max(np.abs(q_end - q_start)) < 1e-4:
        return None
    return _toppra_segment(q_start, q_end, pin_model, pin_data)


def ik_panda(T_target, q0):
    return IK_pinocchio(T_target, q0)


def grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius=None):
    if ball_radius is None:
        ball_radius = sphere_radius
    return single_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius)


def quintic_time(q_from, q_to):
    return quintic_min_time(q_from, q_to)


def arm_hold_torque(theta_d):
    q  = data.qpos[:7].copy()
    dq = data.qvel[:7].copy()
    return hold_pd_torque(pin_model, pin_data, q, dq, theta_d, 500., 200., armature)


# ══════════════════════════════════════════════════════════════════════════════
#  Contractive MPC — gripper close / wait helpers
# ══════════════════════════════════════════════════════════════════════════════

def close_gripper_hold(TT, theta_d, rec, viewer):
    F_thresh = F_thresh_now()
    dt       = model.opt.timestep
    n_ramp   = max(int(0.9 * TT / dt), 1)
    q_open   = data.qpos[7]
    kp, kd   = 500., 50.

    for step in range(n_ramp):
        t_r          = step / n_ramp
        grip_tgt     = q_open - t_r * (q_open - sphere_radius)
        data.ctrl[7] = kp * (grip_tgt - data.qpos[7]) + kd * (-data.qvel[7])
        data.ctrl[:7] = arm_hold_torque(theta_d)
        rec.step(); mujoco.mj_step(model, data)
        if not HEADLESS: viewer.sync()

    alpha = 1.0
    t_sq  = data.time
    while data.time - t_sq < 3.0:          # bounded: max 3 s of squeezing
        data.ctrl[7]  = -alpha * F_thresh
        data.ctrl[:7] = arm_hold_torque(theta_d)
        rec.step(); mujoco.mj_step(model, data)
        if not HEADLESS: viewer.sync()
        if contact_force_ok(F_thresh):
            return alpha * F_thresh
        alpha = min(alpha + 0.4, 50.0)     # cap squeeze force
    return None


def wait_then_grasp(theta_hold, rec, viewer):
    dt        = model.opt.timestep
    max_steps = int(C_WAIT_TIMEOUT / dt)

    for _ in range(max_steps):
        p_b   = data.xpos[sphere_id].copy()
        v_b   = data.qvel[-6:-3].copy()
        speed = np.linalg.norm(v_b[:2])
        dist  = np.linalg.norm(p_b[:2] - data.site_xpos[ee_site_id][:2])
        data.ctrl[:7] = arm_hold_torque(theta_hold)
        data.ctrl[7]  = 0.
        if speed > 1e-6 and dist <= speed * C_GRIPPER_CLOSE_TIME:
            break
        rec.step(); mujoco.mj_step(model, data)
        if not HEADLESS: viewer.sync()

    speed = np.linalg.norm(data.qvel[-6:-3][:2])
    dist  = np.linalg.norm(
        data.xpos[sphere_id][:2] - data.site_xpos[ee_site_id][:2])
    T_close = (dist / speed) if speed > 1e-6 else C_GRIPPER_CLOSE_TIME
    grip = close_gripper_hold(T_close, theta_hold, rec, viewer)
    return grip is not None


# ══════════════════════════════════════════════════════════════════════════════
#  Intercept search for contractive MPC
# ══════════════════════════════════════════════════════════════════════════════

def find_intercept_online(p_ball, v_ball, q_curr):
    t_candidates = np.arange(C_INTERCEPT_SCAN_DT,
                             C_INTERCEPT_T_MAX + C_INTERCEPT_SCAN_DT,
                             C_INTERCEPT_SCAN_DT)

    for t in t_candidates:
        p_int    = p_ball + v_ball * t
        p_int[2] = sphere_radius

        if not check_workspace(p_int):
            continue

        T_pose     = grasp_pose(p_int, v_ball, use_pregrasp=False)
        q_cand, ok = ik_panda(T_pose, q_curr)
        if not ok:
            continue

        t_arm = quintic_time(q_curr, q_cand)
        if t_arm <= t - C_GRIPPER_CLOSE_TIME:
            return p_int.copy(), q_cand.copy()

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Metric recorder
# ══════════════════════════════════════════════════════════════════════════════

class Recorder:
    def __init__(self):
        self.tau_history = []
        self.ee_history  = []
        self.dt          = model.opt.timestep

    def step(self):
        self.tau_history.append(data.ctrl[:7].copy())
        self.ee_history.append(get_ee_pos())

    def compute(self):
        tau = np.array(self.tau_history)
        ee  = np.array(self.ee_history)
        dt  = self.dt

        tau_rms  = float(np.sqrt(np.mean(tau ** 2)))
        path_len = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))

        if len(ee) >= 4:
            vel      = np.diff(ee, axis=0) / dt
            acc      = np.diff(vel, axis=0) / dt
            jerk     = np.diff(acc, axis=0) / dt
            jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
        else:
            jerk_rms = float('nan')

        return tau_rms, path_len, jerk_rms


# ══════════════════════════════════════════════════════════════════════════════
#  Trial runners
# ══════════════════════════════════════════════════════════════════════════════

FAILURE = dict(success=0, catch_time=np.nan, search_time=np.nan,
               tau_rms=np.nan, ee_path_length=np.nan,
               joint_jerk_rms=np.nan, grasp_distance=np.nan)


def _timed_out(t_wall_start):
    return (time.perf_counter() - t_wall_start) > MAX_WALL_TIME


def _mpc_run_to_target(q_target, Tf, traj, rec, t_wall_start,
                       open_grip=False, viewer=None):
    """Execute a TOPPRA trajectory via MPC for exactly Tf seconds."""
    n      = max(int(Tf / P_H), 2)
    t_samp = np.linspace(0, min(Tf, traj.duration), n)
    pos    = np.array([traj(t)    for t in t_samp])
    vel    = np.array([traj(t, 1) for t in t_samp])
    acc    = np.array([traj(t, 2) for t in t_samp])
    tau_ff = np.array([pin.rnea(pin_model, pin_data, pos[i], vel[i], acc[i])
                       for i in range(n)])

    _warmstart()
    t0 = data.time

    while data.time - t0 < Tf:
        if _timed_out(t_wall_start):
            return False
        idx    = min(int((data.time - t0) / P_H), n - 1)
        x_curr = np.concatenate([data.qpos[:7], data.qvel[:7]])

        for k in range(P_N):
            j = min(idx + k, n - 1)
            solver.set(k, 'yref', np.concatenate([pos[j], vel[j], tau_ff[j]]))
        solver.set(P_N, 'yref', np.concatenate([pos[min(idx + P_N, n - 1)],
                                                 vel[min(idx + P_N, n - 1)]]))
        pin_initial_state(solver, x_curr)
        solver.solve()

        u = solver.get(0, 'u').copy()
        data.ctrl[:7] = np.clip(u, -TAU_MAX, TAU_MAX)

        for _ in range(P_STEPS):
            if open_grip:
                data.ctrl[7] = 200.0 * (0.04 - data.qpos[finger1_id])
            rec.step()
            mujoco.mj_step(model, data)
            if not HEADLESS and viewer is not None:
                viewer.sync()
        _shift()

    return True


def _mpc_track_target(q_target, rec, t_wall_start, viewer=None):
    """Single MPC step tracking a joint-space target (point stab)."""
    if _timed_out(t_wall_start):
        return False

    x_curr  = np.concatenate([data.qpos[:7], data.qvel[:7]])
    xs_ref  = np.concatenate([q_target, np.zeros(NV)])
    yref    = np.concatenate([xs_ref, np.zeros(NU)])

    for k in range(P_N):
        solver.set(k, 'yref', yref)
    solver.set(P_N, 'yref', xs_ref)

    pin_initial_state(solver, x_curr)
    solver.solve()

    u = solver.get(0, 'u').copy()
    data.ctrl[:7] = np.clip(u, -TAU_MAX, TAU_MAX)

    for _ in range(P_STEPS):
        rec.step()
        mujoco.mj_step(model, data)
        if not HEADLESS and viewer is not None:
            viewer.sync()
    _shift()
    return True


# ── Planning methods (TOPPRA + CROFT) ─────────────────────────────────────────

def _run_planning_method(find_intercept_fn, p0, v0, viewer):
    reset_sim(p0, v0)
    rec          = Recorder()
    t_wall_start = time.perf_counter()

    # Live get_state: paces the sim by wall-clock, holding the arm steady, and
    # returns the current ball + arm state to the planner.
    last_wall = [time.perf_counter()]
    acc_time  = [0.0]

    def get_state():
        now = time.perf_counter()
        acc_time[0] += now - last_wall[0]
        last_wall[0] = now
        n = int(acc_time[0] / model.opt.timestep)
        acc_time[0] -= n * model.opt.timestep
        for _ in range(n):
            hold_steady()
            mujoco.mj_step(model, data)
        if not HEADLESS and viewer is not None and n:
            viewer.sync()
        return (data.qpos[-7:-4].copy(),
                data.qvel[-6:-3].copy(),
                data.qpos[:7].copy())

    # ── Search ────────────────────────────────────────────────────────────────
    t_search_start = time.perf_counter()
    result = find_intercept_fn(p0, v0, ARM_Q0, verbose=False, get_state=get_state)
    search_time = time.perf_counter() - t_search_start

    if result is None or _timed_out(t_wall_start):
        return {**FAILURE, "search_time": search_time}

    sim_start = data.time

    # ── Phase 1: home → pregrasp ──────────────────────────────────────────────
    if not _mpc_run_to_target(result['q_pregrasp'], result['T1'],
                               result['traj1'], rec, t_wall_start,
                               open_grip=True, viewer=viewer):
        return {**FAILURE, "search_time": search_time}

    # ── Phase 2: pregrasp → grasp ─────────────────────────────────────────────
    traj2 = toppra_segment(data.qpos[:7].copy(), result['q_grasp'])
    if traj2 is None:
        return {**FAILURE, "search_time": search_time}
    if not _mpc_run_to_target(result['q_grasp'], traj2.duration,
                               traj2, rec, t_wall_start, viewer=viewer):
        return {**FAILURE, "search_time": search_time}

    # ── Phase 3: correction time (hold at grasp config) ───────────────────────
    corr = result.get('correction_time', 0.0)
    if corr > 1e-3 and not _timed_out(t_wall_start):
        traj3 = toppra_segment(data.qpos[:7].copy(), result['q_grasp'])
        if traj3 is not None:
            _mpc_run_to_target(result['q_grasp'], corr, traj3, rec,
                               t_wall_start, viewer=viewer)

    # ── Phase 4: wait for ball ────────────────────────────────────────────────
    F_thresh = F_thresh_now()
    dist     = 0.0
    t_wait   = data.time
    while data.time - t_wait < MAX_SIM_TIME and not _timed_out(t_wall_start):
        p_b   = data.xpos[sphere_id]
        v_now = np.linalg.norm(data.qvel[-6:-4])
        dist  = np.linalg.norm(p_b[:2] - data.site_xpos[ee_site_id][:2])
        hold_steady()
        rec.step()
        mujoco.mj_step(model, data)
        if not HEADLESS and viewer is not None: viewer.sync()
        if v_now > 1e-6 and dist <= v_now * result['GRIPPER_CLOSE_TIME']:
            break

    if _timed_out(t_wall_start):
        return {**FAILURE, "search_time": search_time}

    # ── Phase 5: close gripper ────────────────────────────────────────────────
    theta_d      = data.qpos[:7].copy()
    q_open       = data.qpos[7]
    closing_dist = q_open - sphere_radius
    speed_now    = np.linalg.norm(data.qvel[-6:-4])
    T_close      = (dist / speed_now) if speed_now > 1e-6 else 0.5
    n_close      = max(int(T_close / model.opt.timestep), 1)

    for step in range(n_close):
        t_ratio      = step / n_close
        grip_tgt     = q_open - t_ratio * closing_dist
        data.ctrl[7] = 500 * (grip_tgt - data.qpos[7]) - 50 * data.qvel[7]
        data.ctrl[:7] = arm_hold_torque(theta_d)
        rec.step(); mujoco.mj_step(model, data)
        if not HEADLESS and viewer is not None: viewer.sync()

    alpha   = 1.0
    success = False
    t_sq    = data.time
    while data.time - t_sq < 3.0 and not _timed_out(t_wall_start):
        data.ctrl[7]  = -alpha * F_thresh
        data.ctrl[:7] = arm_hold_torque(theta_d)
        rec.step(); mujoco.mj_step(model, data)
        if not HEADLESS and viewer is not None: viewer.sync()
        if contact_force_ok(F_thresh):
            success = True
            break
        alpha += 0.2

    catch_time = data.time - sim_start
    grasp_dist = np.linalg.norm(data.xpos[sphere_id] - data.site_xpos[ee_site_id])
    tau_rms, path_len, jerk_rms = rec.compute()

    return dict(
        success        = int(success),
        catch_time     = catch_time if success else np.nan,
        search_time    = search_time,
        tau_rms        = tau_rms,
        ee_path_length = path_len,
        joint_jerk_rms = jerk_rms,
        grasp_distance = grasp_dist if success else np.nan,
    )


def run_toppra(p0, v0, viewer):
    return _run_planning_method(intercept_planner_toppra.find_intercept,
                                p0, v0, viewer)


def run_croft(p0, v0, viewer):
    return _run_planning_method(intercept_planner_croft.find_intercept,
                                p0, v0, viewer)


# ── Contractive (online reactive) MPC ─────────────────────────────────────────

def run_contractive(p0, v0, viewer):
    reset_sim(p0, v0)
    rec          = Recorder()
    t_wall_start = time.perf_counter()
    search_time  = 0.0   # no offline planning
    sim_start    = data.time

    _warmstart()

    phase            = 'pregrasp'
    q_target         = data.qpos[:7].copy()
    intercept_locked = None
    success          = False

    search_every = 4
    step_count   = 0

    while (data.time - sim_start < MAX_SIM_TIME
           and not _timed_out(t_wall_start)):

        q_curr = data.qpos[:7].copy()
        p_ball = data.xpos[sphere_id].copy()
        v_ball = data.qvel[-6:-3].copy()
        ee_pos = get_ee_pos_q(q_curr)
        step_count += 1

        if phase == 'pregrasp':
            p_hover = p_ball + v_ball * C_PREGRASP_LOOKAHEAD
            T_hover = grasp_pose(p_hover, v_ball, use_pregrasp=True)
            if check_workspace(T_hover[:3, 3]):
                q_hover, ok = ik_panda(T_hover, q_curr)
                if ok:
                    q_target = q_hover

            if step_count % search_every == 0:
                result = find_intercept_online(p_ball, v_ball, q_curr)
                if result is not None:
                    intercept_locked = result
                    q_target         = result[1]
                    phase            = 'grasp'

        elif phase == 'grasp':
            p_intercept, q_intercept = intercept_locked

            if step_count % search_every == 0:
                result_new = find_intercept_online(p_ball, v_ball, q_curr)
                if result_new is not None:
                    intercept_locked         = result_new
                    p_intercept, q_intercept = result_new

            q_target = q_intercept

            xy_err = np.linalg.norm(ee_pos[:2] - p_intercept[:2])
            z_err  = abs(ee_pos[2] - sphere_radius)

            if xy_err < C_INTERCEPT_XY and z_err < C_INTERCEPT_Z:
                success = wait_then_grasp(q_intercept, rec, viewer)
                break

        data.ctrl[7] = 200.0 * (0.04 - data.qpos[finger1_id])   # fingers open
        if not _mpc_track_target(q_target, rec, t_wall_start, viewer):
            break   # wall-clock timeout

    catch_time = data.time - sim_start
    grasp_dist = np.linalg.norm(data.xpos[sphere_id] - data.site_xpos[ee_site_id])
    tau_rms, path_len, jerk_rms = rec.compute()

    return dict(
        success        = int(success),
        catch_time     = catch_time if success else np.nan,
        search_time    = search_time,
        tau_rms        = tau_rms,
        ee_path_length = path_len,
        joint_jerk_rms = jerk_rms,
        grasp_distance = grasp_dist if success else np.nan,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregation and printing
# ══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = ["success", "catch_time", "search_time",
               "tau_rms", "ee_path_length", "joint_jerk_rms", "grasp_distance"]


def aggregate(results):
    agg = {}
    for key in METRIC_KEYS:
        vals = np.array([r[key] for r in results], dtype=float)
        agg[key] = float(np.mean(vals) if key == "success" else np.nanmean(vals))
    return agg


def print_results(name, agg, n_trials):
    n_ok = int(round(agg["success"] * n_trials))
    print(f"\n{'─'*56}")
    print(f"  {name}")
    print(f"{'─'*56}")
    print(f"  Success rate     : {agg['success']*100:.1f}%  ({n_ok}/{n_trials})")
    print(f"  Catch time       : {agg['catch_time']:.3f} s")
    print(f"  Search time      : {agg['search_time']:.4f} s")
    print(f"  Torque RMS       : {agg['tau_rms']:.2f} Nm")
    print(f"  EE path length   : {agg['ee_path_length']:.3f} m")
    print(f"  Jerk RMS         : {agg['joint_jerk_rms']:.3f} m/s³")
    print(f"  Grasp distance   : {agg['grasp_distance']*1000:.2f} mm")


def print_comparison(results_dict, n_trials):
    methods = list(results_dict.keys())
    col     = 17
    width   = 22 + col * len(methods)
    print(f"\n{'═'*width}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*width}")
    print(f"{'Metric':<22}" + "".join(f"{m:>{col}}" for m in methods))
    print("─" * width)

    rows = [
        ("Success rate (%)",  "success",        lambda v: f"{v*100:.1f}"),
        ("Catch time (s)",    "catch_time",      lambda v: f"{v:.3f}"),
        ("Search time (s)",   "search_time",     lambda v: f"{v:.4f}"),
        ("Torque RMS (Nm)",   "tau_rms",         lambda v: f"{v:.2f}"),
        ("EE path (m)",       "ee_path_length",  lambda v: f"{v:.3f}"),
        ("Jerk RMS (m/s³)",   "joint_jerk_rms",  lambda v: f"{v:.3f}"),
        ("Grasp dist (mm)",   "grasp_distance",  lambda v: f"{v*1000:.2f}"),
    ]

    for label, key, fmt in rows:
        row = f"{label:<22}"
        for m in methods:
            row += f"{fmt(results_dict[m][key]):>{col}}"
        print(row)

    print(f"{'═'*width}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trials = sample_trials(N_TRIALS)
    viewer = None if HEADLESS else mujoco.viewer.launch_passive(model, data)

    METHODS = {
        "TOPPRA (brute-force)": run_toppra,
        "CROFT":                run_croft,
        "Contractive MPC":      run_contractive,
    }

    all_results = {}

    for method_name, run_fn in METHODS.items():
        print(f"\n{'═'*56}")
        print(f"  Running: {method_name}  ({N_TRIALS} trials)")
        print(f"{'═'*56}")
        trial_results = []
        for i, trial in enumerate(trials):
            print(f"  Trial {i+1:>2}/{N_TRIALS}  "
                  f"p0=({trial['p0'][0]:+.2f},{trial['p0'][1]:+.2f})  "
                  f"v0=({trial['v0'][0]:.2f},{trial['v0'][1]:.2f}) … ",
                  end="", flush=True)
            r = run_fn(trial['p0'], trial['v0'], viewer)
            trial_results.append(r)
            if r["success"]:
                print(f"✓  t={r['catch_time']:.2f}s  "
                      f"d={r['grasp_distance']*1000:.1f}mm")
            else:
                print("✗")

        agg = aggregate(trial_results)
        all_results[method_name] = agg
        print_results(method_name, agg, N_TRIALS)

    print_comparison(all_results, N_TRIALS)
