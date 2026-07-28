"""
run_grasping_pid_quintic.py
===========================
Live dynamic-grasping demo: quintic intercept planning (closed-form
minimum-time law) executed with a PID computed-torque tracking controller,
then wait / gripper-close / lift phases.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import pinocchio as pin
from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, TAU_MAX,
                                  compute_F_thresh, arm_mass_bias)
from shared.trajectory_control import sample_quintic_trajectory, ee_tip_torque
from shared.gripper import close_gripper
from intercept_planner_quintic import find_intercept

# ── Load model ────────────────────────────────────────────────────────────────

model   = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data    = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

# ── Initial state ─────────────────────────────────────────────────────────────

data.qvel[-6:] = [0., 0.4, 0, 0, 0, 0]
data.qpos[-7:] = [0.5, -0.8, 0.03, 1, 0, 0, 0]
data.qpos[:7]  = np.zeros(7)
mujoco.mj_forward(model, data)

p0     = data.qpos[-7:-4].copy()
v_ball = data.qvel[-6:-3].copy()
q0     = data.qpos[:7].copy()

# ─── Object and gripper properties ───────────────────────────────────────────

sphere_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
sphere_mass = model.body_mass[sphere_id]

sphere_geom_id = next(i for i in range(model.ngeom)
                      if model.geom_bodyid[i] == sphere_id)
sphere_radius = model.geom_size[sphere_geom_id, 0]

left_pad_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
finger_geoms = {left_pad_id, right_pad_id}

ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")

# ── Pinocchio model ───────────────────────────────────────────────────────────

pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()
ee_id     = pin_model.getFrameId("gripper")

armature = model.dof_armature[:7]

# ── Helpers ───────────────────────────────────────────────────────────────────

_hold_integral = np.zeros(7)

def hold_steady(use_ftip=False, kp=800, kd=40, ki=300, max_integral=5):
    """Single PID step holding the arm at its current position (fast dynamics).
    Safe to call from inside find_intercept (live-sim mode); does NOT step."""
    global _hold_integral
    dt = model.opt.timestep

    q_hold     = data.qpos[:7].copy()
    thetalist  = data.qpos[:7]
    dthetalist = data.qvel[:7]

    error  = q_hold - thetalist
    _hold_integral += error * dt
    _hold_integral  = np.clip(_hold_integral, -max_integral, max_integral)

    M, h = arm_mass_bias(pin_model, pin_data, thetalist, dthetalist, armature)
    tau = M @ (kp * error + kd * (-dthetalist) + ki * _hold_integral) + h

    if use_ftip:
        # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
        Ftip = np.array([0, 0, -9.81 * sphere_mass, 0, 0, 0])
        tau += ee_tip_torque(pin_model, pin_data, thetalist, Ftip, ee_id)

    data.ctrl[0:7] = np.clip(tau, -TAU_MAX, TAU_MAX)


def run_trajectory(Tf, theta_start, theta_end, open_grip=False, use_ftip=False,
                   kp=800, kd=40, ki=300):
    """Execute a quintic trajectory from theta_start to theta_end in Tf seconds."""
    dt = model.opt.timestep
    N  = int(Tf / dt)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_start, theta_end, Tf, N)
    integral     = np.zeros(7)
    t_start      = data.time
    fpos_desired = 0.04
    # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
    Ftip         = np.array([0, 0, -9.81 * sphere_mass, 0, 0, 0])

    while True:
        elapsed = data.time - t_start
        if elapsed >= Tf:
            break

        i = min(int((elapsed / Tf) * (N - 1)), N - 1)

        thetalist  = data.qpos[:7]
        dthetalist = data.qvel[:7]

        error     = thetamatd[i]  - thetalist
        derror    = dthetamatd[i] - dthetalist
        integral += error * dt

        M, h = arm_mass_bias(pin_model, pin_data, thetalist, dthetalist, armature)
        tau  = M @ (ddthetamatd[i] + kp * error + kd * derror + ki * integral) + h
        tau  = np.clip(tau, -TAU_MAX, TAU_MAX)
        data.ctrl[0:7] = tau

        if use_ftip:
            tau += ee_tip_torque(pin_model, pin_data, thetalist, Ftip, ee_id)

        if open_grip:
            data.ctrl[7] = 200.0 * (fpos_desired - data.qpos[finger1_id])

        mujoco.mj_step(model, data)
        viewer.sync()

# ── Launch viewer ─────────────────────────────────────────────────────────────

viewer = mujoco.viewer.launch_passive(model, data)

# ── Phase 1: find_intercept with live simulation ──────────────────────────────

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
    if n:
        viewer.sync()

    return (data.qpos[-7:-4].copy(),
            data.qvel[-6:-3].copy(),
            data.qpos[:7].copy())


print("Searching for intercept (simulation running) …")

result = find_intercept(p0, v_ball, q0, verbose=True, get_state=get_state)

if result is None:
    quit()

print(f"\nIntercept found!")
print(f"  T1 (home → pregrasp) : {result['T1']:.3f} s")
print(f"  T2 (pregrasp → grasp): {result['T2']:.3f} s")
print(f"  p_intercept          : {result['p_intercept']}")
print(f"  q_pregrasp           : {np.round(result['q_pregrasp'], 3)}")
print(f"  q_grasp              : {np.round(result['q_grasp'],    3)}")

# ── Phase 2: home → pre-grasp ─────────────────────────────────────────────────

print(f"Moving to pre-grasp in {result['T1']:.3f} s …")
run_trajectory(result['T1'], data.qpos[:7].copy(), result['q_pregrasp'],
               open_grip=True)

# ── Phase 3: pre-grasp → grasp ────────────────────────────────────────────────

print(f"Moving to grasp in {result['T2']:.3f} s …")
run_trajectory(result['T2'], data.qpos[:7].copy(), result['q_grasp'])

# ── Phase 4: wait until ball is close enough to close gripper ────────────────

speed = np.linalg.norm(v_ball)
dist  = np.linalg.norm(data.xpos[sphere_id][:2] - data.site_xpos[site_id][:2])
while dist > speed * result['GRIPPER_CLOSE_TIME']:
    dist = np.linalg.norm(data.xpos[sphere_id][:2] - data.site_xpos[site_id][:2])
    hold_steady()
    mujoco.mj_step(model, data)
    viewer.sync()

# ── Phase 5: close gripper (shared primitive) ────────────────────────────────

F_thresh = compute_F_thresh(model, sphere_geom_id, left_pad_id, sphere_mass, rule="mean")
f = close_gripper(model, data, viewer, pin_model, pin_data,
                  sphere_geom_id, finger_geoms, F_thresh,
                  ramp_time=0.7 * dist / speed, sphere_radius=sphere_radius,
                  contact_mult=7.0, alpha_step=0.2)
print(f"Grip achieved: {f:.3f} N")

# ── Phase 6: lift to home-ish pose ───────────────────────────────────────────

run_trajectory(0.8, data.qpos[:7].copy(), result['q_pregrasp'], use_ftip=True)
run_trajectory(3, data.qpos[:7].copy(),
               np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853]), use_ftip=True)

while viewer.is_running():
    hold_steady(use_ftip=True)
    mujoco.mj_step(model, data)
    viewer.sync()
