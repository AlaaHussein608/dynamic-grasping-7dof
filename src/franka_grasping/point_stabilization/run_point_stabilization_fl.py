"""
run_point_stabilization_fl.py
=============================
Feedback-linearization (computed-torque) point stabilization: regulate the arm
to the constant setpoint q_d = Q_TARGET_DEMO (dq_d = 0, ddq_d = 0) and hold it
there, live in the MuJoCo viewer.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import (PANDA_URDF, CUBE_SCENE_XML, Q_HOME,
                                  Q_TARGET_DEMO, load_pinocchio_model)
from shared.trajectory_control import feedback_linearization_control

# ── Gains ────────────────────────────────────────────────────────────────────
KP = 16.0
KD = 8.0

# ── Models ───────────────────────────────────────────────────────────────────
pin_model, pin_data = load_pinocchio_model(PANDA_URDF)
model = mujoco.MjModel.from_xml_path(CUBE_SCENE_XML)
data = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]

data.qpos[:7] = Q_HOME.copy()
data.qvel[:7] = np.zeros(7)
mujoco.mj_forward(model, data)

armature = model.dof_armature[:7]
q_target = Q_TARGET_DEMO.copy()
zeros = np.zeros(7)

# ── Control loop: regulate to the fixed setpoint and hold ────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = feedback_linearization_control(q, dq, q_target, zeros, zeros,
                                             pin_model, pin_data, armature, KP, KD)
        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        viewer.sync()
