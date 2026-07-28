"""
track_fl_toppra.py
==================
Live MuJoCo demo: feedback-linearization (computed-torque) control tracking
a TOPP-RA time-optimal joint trajectory to Q_TARGET_DEMO.

The control law and TOPP-RA retiming live in shared.trajectory_control; this
script only builds the reference and runs the simulation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_TARGET_DEMO,
                                  load_pinocchio_model)
from shared.trajectory_control import (toppra_segment as _toppra_segment,
                                       feedback_linearization_control,
                                       ee_tip_torque)

pin_model, pin_data = load_pinocchio_model(PANDA_URDF)
ee_id = pin_model.getFrameId("gripper")

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]

armature     = model.dof_armature[:7]
q_above_cube = Q_TARGET_DEMO.copy()


def toppra_segment(q_start, q_end):
    return _toppra_segment(q_start, q_end, pin_model, pin_data)


def move(theta_end, kp=400, kd=300, use_ftip=False):
    theta_start = data.qpos[:7].copy()
    t = 0
    t_start = data.time
    toppra_traj = toppra_segment(theta_start, theta_end)
    Tf = toppra_traj.duration
    # Ftip in Pinocchio convention: [f_x, f_y, f_z, τ_x, τ_y, τ_z]
    Ftip = np.array([0, 0, -9.81 * 0.1, 0, 0, 0])

    while t < Tf:
        elapsed = data.time - t_start
        t = min(elapsed, Tf)

        theta_d   = toppra_traj(t)
        dtheta_d  = toppra_traj(t, 1)
        ddtheta_d = toppra_traj(t, 2)

        q  = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()

        tau = feedback_linearization_control(q, dq, theta_d, dtheta_d, ddtheta_d,
                                             pin_model, pin_data, armature, kp, kd)
        if use_ftip:
            tau += ee_tip_torque(pin_model, pin_data, q, Ftip, ee_id)

        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        viewer.sync()


with mujoco.viewer.launch_passive(model, data) as viewer:
    move(q_above_cube, kp=400, kd=300, use_ftip=False)
