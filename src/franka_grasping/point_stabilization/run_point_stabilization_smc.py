"""
run_point_stabilization_smc.py
==============================
Adaptive sliding-mode point stabilization: drive the arm from Q_HOME to
Q_TARGET_DEMO by regulating to the constant target
(q_d = Q_TARGET_DEMO, dq_d = 0, ddq_d = 0), live in the MuJoCo viewer.

Same task as run_point_stabilization_mpc.py, controller-only (no solver).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import (PANDA_URDF, CUBE_SCENE_XML, Q_HOME,
                                  Q_TARGET_DEMO, TAU_MAX, load_pinocchio_model)
from shared.trajectory_control import sliding_mode_control

# ── Sliding-mode parameters ──────────────────────────────────────────────────
Lambda      = np.diag([40, 40, 40, 40, 40, 40, 40])
phi         = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
K_min       = np.array([ 4.0,  4.0,  4.0,  4.0,  0.6,  0.6,  0.6])
K_max       = np.array([60.0, 60.0, 60.0, 60.0,  8.0,  8.0,  8.0])
alpha_adapt = np.array([30.0, 30.0, 20.0, 15.0,  3.0,  3.0,  2.0])

# ── Models ───────────────────────────────────────────────────────────────────
pin_model, pin_data = load_pinocchio_model(PANDA_URDF)

model = mujoco.MjModel.from_xml_path(CUBE_SCENE_XML)
data  = mujoco.MjData(model)
data.qpos[:7] = Q_HOME.copy()
data.qvel[:7] = np.zeros(7)
mujoco.mj_forward(model, data)
model.opt.gravity[:] = [0, 0, -9.81]

armature = model.dof_armature[:7]
q_target = Q_TARGET_DEMO.copy()
zeros    = np.zeros(7)

viewer = mujoco.viewer.launch_passive(model, data)

while np.linalg.norm(data.qpos[:7] - q_target) > 0.01:
    q  = data.qpos[:7].copy()
    dq = data.qvel[:7].copy()

    tau, s = sliding_mode_control(q, dq, q_target, zeros, zeros,
                                  pin_model, pin_data, armature,
                                  Lambda, phi, K_min, K_max, alpha_adapt)
    data.ctrl[:7] = np.clip(tau, -TAU_MAX, TAU_MAX)
    mujoco.mj_step(model, data)
    viewer.sync()


print("Reached target.")
print("Final joint positions:", data.qpos[:7])
