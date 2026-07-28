"""
track_fl_quintic.py
===================
Live MuJoCo demo: feedback-linearization (computed-torque) control tracking
a quintic joint trajectory to Q_TARGET_DEMO.

The control law lives in
shared.trajectory_control.feedback_linearization_control; this script only
builds the reference and runs the simulation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_TARGET_DEMO,
                                  DQ_MAX, load_pinocchio_model)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       feedback_linearization_control,
                                       ee_tip_torque)

pin_model, pin_data = load_pinocchio_model(PANDA_URDF)
ee_id = pin_model.getFrameId("gripper")

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]

armature     = model.dof_armature[:7]
q_above_cube = Q_TARGET_DEMO.copy()


def generate_traj(theta_s, theta_end):
    """Quintic reference at the simulation timestep; duration from the
    quintic minimum-time law under DQ_MAX."""
    dt = model.opt.timestep
    T  = float(np.max(15 * np.abs(theta_end - theta_s) / (8 * DQ_MAX)))
    N  = int(T / dt)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_s, theta_end, T, N)
    return thetamatd, dthetamatd, ddthetamatd, N, T


def move(theta_end, kp=400, kd=30, use_ftip=False):
    theta_start = data.qpos[:7].copy()
    t = 0
    t_start = data.time
    thetamatd, dthetamatd, ddthetamatd, N, Tf = generate_traj(theta_start, theta_end)
    # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
    Ftip = np.array([0, 0, -9.81 * 0.1, 0, 0, 0])

    while t < Tf:
        elapsed = data.time - t_start
        t = min(elapsed, Tf)
        i = min(int((t / Tf) * (N - 1)), N - 1)

        q  = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()

        tau = feedback_linearization_control(q, dq, thetamatd[i], dthetamatd[i],
                                             ddthetamatd[i], pin_model, pin_data,
                                             armature, kp, kd)
        if use_ftip:
            tau += ee_tip_torque(pin_model, pin_data, q, Ftip, ee_id)

        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        viewer.sync()



with mujoco.viewer.launch_passive(model, data) as viewer:
    move(q_above_cube, kp=400, kd=30, use_ftip=False)
