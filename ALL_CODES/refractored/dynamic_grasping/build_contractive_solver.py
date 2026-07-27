"""
build_contractive_solver.py
===========================
Run ONCE to generate and compile the contraction-constrained ACADOS solver
(h=0.01, N=10) used by the contractive dynamic-grasping scripts.

The contraction stage constraint
    ‖e(k+1)‖² − α² ‖e(k)‖² ≤ 0,   e(k) = x(k)[:nq] − x_ref[:nq]
is added by the shared builder via contraction=True; the parameters
p = [x_ref (nx), alpha (1)] are set online with solver.set(k, 'p', ...).

NOTE: the run scripts historically load 'franka_point_stab.json' — rename or
copy the produced JSON accordingly (see README).

Refactored from: dynamic grasping/Contractive MPC/contractive_build_solver.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import PANDA_URDF, SPHERE_SCENE_XML, script_dir
from shared.acados_mpc import build_franka_ocp_solver

SCRIPT_DIR = script_dir(__file__)

# ── MPC parameters ────────────────────────────────────────────────────────────
H = 0.01
N = 10

# ── Cost ──────────────────────────────────────────────────────────────────────
Q = 100 * np.diag([200] * 7 + [30] * 7)
R = np.diag([2.] * 7)

if __name__ == "__main__":
    build_franka_ocp_solver(
        model_name='franka_point_stab',
        h=H, N=N, Q=Q, R=R,
        urdf_path=PANDA_URDF,
        mjcf_path=SPHERE_SCENE_XML,
        json_file=SCRIPT_DIR / 'franka_point_stab_contractive.json',
        export_dir=SCRIPT_DIR / 'franka_point_stab_contractive_generated',
        contraction=True,
    )
