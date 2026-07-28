"""
build_trajectory_tracking_solver.py
===================================
Run ONCE to generate and compile the ACADOS solver used by the
trajectory-tracking scripts (h=0.005, N=8).

Refactored from: MPC trajectory/ACADOS_build_solver.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import PANDA_URDF, CUBE_SCENE_XML, script_dir
from shared.acados_mpc import build_franka_ocp_solver

SCRIPT_DIR = script_dir(__file__)

# ── MPC parameters ────────────────────────────────────────────────────────────
H = 0.005
N = 8

# ── Cost:  stage (x - xs)^T Q (x - xs) + u^T R u ─────────────────────────────
Q = 200 * np.diag([300] * 7 + [30] * 7)
R = np.diag([1.] * 7)

if __name__ == "__main__":
    build_franka_ocp_solver(
        model_name='franka_point_stab',
        h=H, N=N, Q=Q, R=R,
        urdf_path=PANDA_URDF,
        mjcf_path=CUBE_SCENE_XML,
        json_file=SCRIPT_DIR / 'franka_point_stab.json',
        export_dir=SCRIPT_DIR / 'franka_point_stab_generated',
    )
