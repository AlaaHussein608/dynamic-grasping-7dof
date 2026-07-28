"""
run_point_stabilization_mpc.py
==============================
Load the pre-compiled solver and drive the arm from Q_HOME to Q_TARGET_DEMO
with point-stabilization MPC, live in the MuJoCo viewer.

Run build_point_stabilization_solver.py first (→ franka_point_stab.json).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from shared.franka_common import CUBE_SCENE_XML, Q_HOME, Q_TARGET_DEMO, script_dir
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state, set_point_reference)

SCRIPT_DIR = script_dir(__file__)

# ── Parameters (must match the build script) ─────────────────────────────────
h  = 0.01
N  = 10
nq = 7
nv = 7
nx = nq + nv
nu = nv

# ── Initial and target configuration ─────────────────────────────────────────
q_init = Q_HOME.copy()
xs     = np.concatenate([Q_TARGET_DEMO, np.zeros(nv)])

# ── Load pre-compiled solver ─────────────────────────────────────────────────
print("Loading pre-compiled solver...")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab.json')
print("Solver loaded.")

# ── Cost matrices  ← tune here without rebuilding ────────────────────────────
Q = 200.0 * np.diag([300] * 7 + [30] * 7)
R = np.diag([0.5] * 7)
apply_cost_weights(solver, N, Q, R)

# ── References (fixed target) ────────────────────────────────────────────────
set_point_reference(solver, N, xs, nu=nu)

# ── MuJoCo setup ─────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(CUBE_SCENE_XML)
data  = mujoco.MjData(model)
data.qpos[:7] = q_init
data.qvel[:7] = np.zeros(nv)
mujoco.mj_forward(model, data)
viewer = mujoco.viewer.launch_passive(model, data)
print(f"MuJoCo timestep: {model.opt.timestep}  |  Steps per MPC interval: {int(h / model.opt.timestep)}")
model.opt.timestep = 0.005

# ── Warm-start: initialise all stages at q_init ──────────────────────────────
x_init_state = np.concatenate([q_init, np.zeros(nv)])
init_warm_start(solver, N, x_init_state, nu=nu)


def shift(t0):
    """Apply first control, simulate one MPC step, return next state."""
    tau           = solver.get(0, 'u')
    data.ctrl[:7] = tau
    for _ in range(int(h / model.opt.timestep)):
        mujoco.mj_step(model, data)
        viewer.sync()
    q_next  = data.qpos[:7].copy()
    dq_next = data.qvel[:7].copy()
    return t0 + h, np.concatenate([q_next, dq_next])


# ── MPC loop ─────────────────────────────────────────────────────────────────
x_curr  = x_init_state.copy()
t0      = 0.0
mpciter = 0

while np.linalg.norm(x_curr[:nq] - xs[:nq]) > 0.01:
    pin_initial_state(solver, x_curr)

    status = solver.solve()
    if status not in [0, 2]:
        print(f"WARNING: solver status {status} at iter {mpciter}")

    t0, x_curr = shift(t0)
    shift_warm_start(solver, N)

    mpciter += 1

print("Final joint positions:", data.qpos[:7])
