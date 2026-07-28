"""
build_grasping_solver.py
========================
Run ONCE to generate and compile the ACADOS solver used by the dynamic
grasping scripts (h=0.005, N=8).

The grasping benchmark expects the same solver under the name
franka_point_stab_planning.json — pass --planning (or copy the JSON) to
produce that file as well.

Refactored from: dynamic grasping/intercept planning/ACADOS_build_solver.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import PANDA_URDF, SPHERE_SCENE_XML, script_dir
from shared.acados_mpc import build_franka_ocp_solver

SCRIPT_DIR = script_dir(__file__)

# ── MPC parameters ────────────────────────────────────────────────────────────
H = 0.005
N = 8

# ── Cost:  stage (x - xs)^T Q (x - xs) + u^T R u ─────────────────────────────
Q = 200.0 * np.diag([300.0] * 7 + [30.0] * 7)
R = np.diag([1.000] * 7)

if __name__ == "__main__":
    json_name = ('franka_point_stab_planning.json'
                 if '--planning' in sys.argv else 'franka_point_stab.json')
    build_franka_ocp_solver(
        model_name='franka_point_stab',
        h=H, N=N, Q=Q, R=R,
        urdf_path=PANDA_URDF,
        mjcf_path=SPHERE_SCENE_XML,
        json_file=SCRIPT_DIR / json_name,
        export_dir=SCRIPT_DIR / 'franka_point_stab_generated',
    )
