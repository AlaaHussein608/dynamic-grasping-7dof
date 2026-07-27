"""
run_grasping_nmpc_toppra.py
===========================
Live dynamic-grasping demo: TOPP-RA intercept planning (brute-force scan)
executed with the ACADOS NMPC tracking controller, then wait / gripper-close
/ lift phases.

Requires the pre-compiled grasping solver (run build_solver.py).
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
                                  TAU_MAX, compute_F_thresh, script_dir)
from shared.trajectory_control import (precompute_feedforward, hold_pd_torque,
                                       sample_toppra_trajectory,
                                       quintic_trajectory, ee_tip_torque)
from shared.gripper import open_gripper, close_gripper
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state, set_trajectory_references)
import intercept_planner_toppra
from intercept_planner_toppra import toppra_segment

SCRIPT_DIR = script_dir(__file__)

# ── Load model ────────────────────────────────────────────────────────────────

model   = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data    = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

# ── Initial state ─────────────────────────────────────────────────────────────

data.qvel[-6:] = [-0., 0.7, 0, 0, 0, 0]
data.qpos[-7:] = [0.5, -0.8, 0.03, 1, 0, 0, 0]
data.qpos[:7]  = Q_HOME.copy()
mujoco.mj_forward(model, data)

p0     = data.qpos[-7:-4].copy()
v_ball = data.qvel[-6:-3].copy()
q0     = data.qpos[:7].copy()

# ── Object and gripper properties ─────────────────────────────────────────────

sphere_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
sphere_mass = model.body_mass[sphere_id]

sphere_geom_id = next(i for i in range(model.ngeom)
                      if model.geom_bodyid[i] == sphere_id)
sphere_radius = model.geom_size[sphere_geom_id, 0]

left_pad_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
finger_geoms = {left_pad_id, right_pad_id}

ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  "gripper")
finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")

# ── Pinocchio model ───────────────────────────────────────────────────────────

pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()
ee_id     = pin_model.getFrameId("gripper")

# ── NMPC parameters (must match the build script) ─────────────────────────────

h = 0.005
N = 8
nq = 7
nv = 7
nu = nv

steps_per_interval = int(h / model.opt.timestep)   # inner sim steps per MPC step

# ── Load pre-compiled ACADOS solver ───────────────────────────────────────────

print("Loading ACADOS solver …")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab.json')
print("Solver loaded.")

Q = 200.0 * np.diag([300.0] * 7 + [30.0] * 7)
R = np.diag([1.000] * 7)
apply_cost_weights(solver, N, Q, R)

# ── hold_steady ───────────────────────────────────────────────────────────────

def hold_steady(q_ref, use_ftip=False, kp=8000, kd=120):
    thetalist  = data.qpos[:7]
    dthetalist = data.qvel[:7]
    tau = hold_pd_torque(pin_model, pin_data, thetalist, dthetalist, q_ref,
                         kp, kd, model.dof_armature[:7])
    if use_ftip:
        # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
        Ftip = np.array([0, 0, -9.81 * sphere_mass, 0, 0, 0])
        tau += ee_tip_torque(pin_model, pin_data, thetalist, Ftip, ee_id)
    data.ctrl[0:7] = np.clip(tau, -TAU_MAX, TAU_MAX)


def hold_gripper_open(fpos_desired=0.04):
    data.ctrl[7] = 200.0 * (fpos_desired - data.qpos[finger1_id])

# ── NMPC helpers ──────────────────────────────────────────────────────────────

def _generate_traj_mpc_rate(Tf, theta_start, theta_end):
    """Quintic trajectory sampled at MPC rate h (not sim rate)."""
    N_traj    = max(int(Tf / h), 2)
    t_samples = np.linspace(0, Tf, N_traj)

    dt_sim  = model.opt.timestep
    N_sim   = int(Tf / dt_sim)
    pos_sim = quintic_trajectory(theta_start, theta_end, Tf, N_sim)

    thetamatd   = np.zeros((N_traj, 7))
    dthetamatd  = np.zeros((N_traj, 7))
    ddthetamatd = np.zeros((N_traj, 7))
    for i, t in enumerate(t_samples):
        sim_idx      = min(int(t / dt_sim), N_sim - 1)
        thetamatd[i] = pos_sim[sim_idx]
    for i in range(1, N_traj):
        dthetamatd[i]  = (thetamatd[i]  - thetamatd[i-1])  / h
        ddthetamatd[i] = (dthetamatd[i] - dthetamatd[i-1]) / h

    return thetamatd, dthetamatd, ddthetamatd, N_traj


def _build_ref(Tf, theta_start, theta_end, toppra_traj):
    if toppra_traj is not None:
        pos, vel, acc, N_traj = sample_toppra_trajectory(
            toppra_traj, Tf, h, min_steps=2)
    else:
        pos, vel, acc, N_traj = _generate_traj_mpc_rate(Tf, theta_start, theta_end)
    tau_ff = precompute_feedforward(pin_model, pin_data, pos, vel, acc, nv=nv)
    return pos, vel, acc, tau_ff, N_traj


def run_trajectory(Tf, theta_start, theta_end,
                   toppra_traj=None, open_grip=False, use_ftip=False):
    """Execute a joint-space trajectory using NMPC (ACADOS SQP-RTI)."""
    # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
    Ftip = np.array([0., 0., -9.81 * sphere_mass, 0., 0., 0.])

    pos, vel, acc, tau_ff, N_traj = _build_ref(
        Tf, theta_start, theta_end, toppra_traj)

    x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
    init_warm_start(solver, N, x_curr, nu=nu)

    t_start = data.time
    while True:
        elapsed = data.time - t_start
        if elapsed >= Tf:
            break

        traj_idx = min(int(elapsed / h), N_traj - 1)
        set_trajectory_references(solver, N, traj_idx, pos, vel, tau_ff)
        pin_initial_state(solver, x_curr)
        solver.solve()

        u = solver.get(0, 'u').copy()
        if use_ftip:
            u += ee_tip_torque(pin_model, pin_data, data.qpos[:7].copy(), Ftip, ee_id)
        data.ctrl[:7] = np.clip(u, -TAU_MAX, TAU_MAX)

        for _ in range(steps_per_interval):
            if open_grip:
                hold_gripper_open()
            mujoco.mj_step(model, data)
            viewer.sync()

        x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
        shift_warm_start(solver, N)

# ── Launch viewer ─────────────────────────────────────────────────────────────

viewer = mujoco.viewer.launch_passive(model, data)

# ── Phase 1: find_intercept (live sim via get_state callback) ─────────────────

q_ref_search = data.qpos[:7].copy()
last_wall    = [time.perf_counter()]
acc_time     = [0.0]

def get_state():
    now = time.perf_counter()
    acc_time[0] += now - last_wall[0]
    last_wall[0] = now

    n = int(acc_time[0] / model.opt.timestep)
    acc_time[0] -= n * model.opt.timestep

    for _ in range(n):
        hold_steady(q_ref_search)
        mujoco.mj_step(model, data)
    if n:
        viewer.sync()

    return (data.qpos[-7:-4].copy(),
            data.qvel[-6:-3].copy(),
            data.qpos[:7].copy())


print("Searching for intercept (simulation running) …")

result = intercept_planner_toppra.find_intercept(
    p0, v_ball, q0, verbose=False, get_state=get_state)

if result is None:
    quit()

print(f"\nIntercept found!")
print(f"  T1 (home → pregrasp) : {result['T1']:.3f} s")
print(f"  T2 (pregrasp → grasp): {result['T2']:.3f} s")
print(f"  p_intercept          : {result['p_intercept']}")
print(f"  margin               : {result['margin']}")

# ── Phase 2: home → pre-grasp (NMPC + TOPPRA reference) ──────────────────────
print(f"Moving to pre-grasp in {result['T1']:.3f} s …")
run_trajectory(result['T1'], data.qpos[:7].copy(), result['q_pregrasp'],
               toppra_traj=result['traj1'], open_grip=True)

# ── Phase 3: pre-grasp → grasp (NMPC + TOPPRA reference) ─────────────────────
print(f"Moving to grasp in {result['T2']:.3f} s …")
traj2 = toppra_segment(data.qpos[:7].copy(), result['q_grasp'])
run_trajectory(traj2.duration, data.qpos[:7].copy(), result['q_grasp'],
               toppra_traj=traj2, open_grip=True)

correction_time = result['correction_time']
run_trajectory(correction_time, data.qpos[:7].copy(), result['q_grasp'],
               open_grip=True)

# ── Phase 4: wait until ball is close enough ──────────────────────────────────

q_ref_wait = data.qpos[:7].copy()
ball_ok    = True

while True:
    d_vec = data.xpos[sphere_id][:2] - data.site_xpos[site_id][:2]
    v_now = data.qvel[-6:-3][:2]
    dist  = np.linalg.norm(d_vec)
    speed = np.linalg.norm(v_now)

    if dist <= speed * result['GRIPPER_CLOSE_TIME']:
        break
    if d_vec @ v_now > 0.0 or speed < 1e-3:
        print("Ball receding or stopped — aborting wait.")
        ball_ok = False
        break

    hold_steady(q_ref_wait)
    hold_gripper_open()
    mujoco.mj_step(model, data)
    viewer.sync()

# ── Phase 5: close gripper (shared primitive) ────────────────────────────────

F_thresh  = compute_F_thresh(model, sphere_geom_id, left_pad_id, sphere_mass, rule="max")
speed_now = max(np.linalg.norm(data.qvel[-6:-3][:2]), 1e-3)
dist_now  = np.linalg.norm(data.xpos[sphere_id][:2] - data.site_xpos[site_id][:2])
T_close   = 0.75 * dist_now / speed_now if ball_ok else 0.2

f = close_gripper(model, data, viewer, pin_model, pin_data,
                  sphere_geom_id, finger_geoms, F_thresh,
                  ramp_time=T_close, sphere_radius=sphere_radius,
                  contact_mult=20.0, alpha_step=1.0)
print(f"Grip achieved: {f:.3f} N")

print(np.linalg.norm(data.xpos[sphere_id][:2] - data.site_xpos[site_id][:2]))

# ── Phase 6: lift ─────────────────────────────────────────────────────────────

run_trajectory(0.8, data.qpos[:7].copy(), result['q_pregrasp'], use_ftip=True)
run_trajectory(3.0, data.qpos[:7].copy(),
               np.array([0, 0.2, 0, -1.57079, 0, 2.0, -0.7853]), use_ftip=True)

q_ref_final = data.qpos[:7].copy()
while viewer.is_running():
    hold_steady(q_ref_final, use_ftip=True)
    mujoco.mj_step(model, data)
    viewer.sync()
