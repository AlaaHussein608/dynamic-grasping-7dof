"""
build_point_stabilization_solver.py
===================================
Run ONCE to generate and compile the ACADOS solver used by the
point-stabilization scripts (h=0.01, N=10).

Refactored from: MPC_point_stablization/ACADOS_build_solver.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import PANDA_URDF, CUBE_SCENE_XML, script_dir
from shared.acados_mpc import build_franka_ocp_solver

SCRIPT_DIR = script_dir(__file__)

# ── MPC parameters ────────────────────────────────────────────────────────────
H = 0.01
N = 10

# ── Cost:  stage (x - xs)^T Q (x - xs) + u^T R u ─────────────────────────────
Q = 200.0 * np.diag([300] * 7 + [30] * 7)
R = np.diag([0.5] * 7)

if __name__ == "__main__":
    build_franka_ocp_solver(
        model_name='franka_point_stab',
        h=H, N=N, Q=Q, R=R,
        urdf_path=PANDA_URDF,
        mjcf_path=CUBE_SCENE_XML,
        json_file=SCRIPT_DIR / 'franka_point_stab.json',
        export_dir=SCRIPT_DIR / 'franka_point_stab_generated',
    )
