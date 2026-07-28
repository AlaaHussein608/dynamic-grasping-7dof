"""
intercept_planner_quintic.py
============================
Intercept-point selection for dynamic grasping — quintic variant: motion time
comes from the closed-form quintic minimum-time law instead of TOPP-RA, so the
two planners are drop-in comparable.

The forward scan, grasp-pose IK and timing estimators live in
:mod:`dg_common`; this file just binds the scan to the quintic estimator.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dg_common import forward_scan_intercept, estimate_quintic


def find_intercept(p0, v_ball, q_current,
                   ball_radius=0.03, base_pos=np.zeros(3),
                   t_min=0.0, t_max=40.0, verbose=True,
                   get_state=None, correction_time=0.06):
    """Forward-scan intercept search with closed-form quintic segment timing.

    See :func:`dg_common.forward_scan_intercept` for the arguments and the
    returned result dict.
    """
    return forward_scan_intercept(
        p0, v_ball, q_current, estimate_quintic,
        ball_radius=ball_radius, base_pos=base_pos,
        t_min=t_min, t_max=t_max, correction_time=correction_time,
        get_state=get_state, verbose=verbose)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p0     = np.array([0.5, -0.8, 0.03])
    v_ball = np.array([-0.1,  0.2,  0.0])
    q0     = np.zeros(7)

    result = find_intercept(p0, v_ball, q0, verbose=True)

    if result:
        print("\n─── Intercept found ───")
        print(f"  T1      : {result['T1']:.3f} s")
        print(f"  T2      : {result['T2']:.3f} s")
        print(f"  t_sim   : {result['t_sim']:.3f} s")
        print(f"  t_solve : {result['t_solve']:.3f} s")
        print(f"  margin  : {result['margin']:.3f} s")
    else:
        print("No feasible intercept found.")
