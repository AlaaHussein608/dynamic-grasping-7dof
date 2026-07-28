"""
track_smc_quintic.py
====================
Live MuJoCo demo: adaptive sliding-mode control tracking a quintic joint
trajectory to Q_TARGET_DEMO.

The SMC law lives in shared.trajectory_control.sliding_mode_control; this
script only builds the reference and runs the simulation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_TARGET_DEMO,
                                  DQ_MAX, TAU_MAX, load_pinocchio_model)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       sliding_mode_control, ee_tip_torque)

pin_model, pin_data = load_pinocchio_model(PANDA_URDF)
ee_id = pin_model.getFrameId("gripper")

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]

armature     = model.dof_armature[:7]
q_above_cube = Q_TARGET_DEMO.copy()

# Sliding control parameters
Lambda      = np.diag([40, 40, 40, 40, 40, 40, 40])
phi         = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
K_min       = np.array([ 4.0,  4.0,  4.0,  4.0,  0.6,  0.6,  0.6])
K_max       = np.array([60.0, 60.0, 60.0, 60.0,  8.0,  8.0,  8.0])
alpha_adapt = np.array([30.0, 30.0, 20.0, 15.0,  3.0,  3.0,  2.0])


def generate_traj(theta_s, theta_end):
    """Quintic reference at the simulation timestep; duration from the
    quintic minimum-time law under DQ_MAX."""
    dt = model.opt.timestep
    T  = float(np.max(15 * np.abs(theta_end - theta_s) / (8 * DQ_MAX)))
    N  = int(T / dt)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_s, theta_end, T, N)
    return thetamatd, dthetamatd, ddthetamatd, N, T


def move(theta_end, use_ftip=False):
    theta_start = data.qpos[:7].copy()
    t_start = data.time
    thetamatd, dthetamatd, ddthetamatd, N, Tf = generate_traj(theta_start, theta_end)
    # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
    Ftip = np.array([0, 0, -9.81 * 0.1, 0, 0, 0])

    while True:
        elapsed = data.time - t_start
        t = min(elapsed, Tf)
        if t >= Tf:
            break

        i = min(int((t / Tf) * (N - 1)), N - 1)
        theta_d, dtheta_d, ddtheta_d = thetamatd[i], dthetamatd[i], ddthetamatd[i]

        q = data.qpos[:7].copy()
        v = data.qvel[:7].copy()

        tau, s = sliding_mode_control(q, v, theta_d, dtheta_d, ddtheta_d,
                                      pin_model, pin_data, armature,
                                      Lambda, phi, K_min, K_max, alpha_adapt)
        if use_ftip:
            tau += ee_tip_torque(pin_model, pin_data, q, Ftip, ee_id)

        data.ctrl[:7] = np.clip(tau, -TAU_MAX, TAU_MAX)
        mujoco.mj_step(model, data)
        viewer.sync()


with mujoco.viewer.launch_passive(model, data) as viewer:
    move(q_above_cube, use_ftip=False)