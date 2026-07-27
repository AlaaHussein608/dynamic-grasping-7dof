"""
intercept_planner_quintic.py
============================
Intercept-point selection for dynamic grasping of a rolling ball with a
Franka Emika Panda arm.

Quintic-trajectory variant: motion time comes from the closed-form quintic
minimum-time law instead of TOPP-RA.  All other machinery — analytic
reachable window, frantik IK, bang-bang prefilter, get_state callback,
measured-solve-time margin — mirrors the TOPP-RA planner so the two are
drop-in comparable (and is imported from the shared modules).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import (GRIPPER_CLOSE_TIME,
                                  IK_frantik, check_workspace,
                                  reachable_window, compute_grasp_pose,
                                  lower_bound_time, quintic_min_time)


# ── Quintic minimum-time estimate ─────────────────────────────────────────────

def estimate_real_time_quintic(q_start, q_pg, q_g):
    """
    Quintic durations for both segments.
    Returns T1, T2, gripper_close — no trajectory object: a quintic is
    regenerated at execution time from (q_start, q_end, T).
    """
    T1 = quintic_min_time(q_start, q_pg)
    T2 = quintic_min_time(q_pg,    q_g)
    return T1, T2, GRIPPER_CLOSE_TIME


# ── Main intercept search ─────────────────────────────────────────────────────

def find_intercept(p0, v_ball, q_current,
                   ball_radius=0.03, base_pos=np.zeros(3),
                   t_min=0.0, t_max=40.0, verbose=True,
                   get_state=None,
                   correction_time=0.06):
    """
    get_state : optional callback () -> (p_ball_now, v_ball_now, q_now).
                In live mode the caller owns sim stepping, arm holding and
                viewer syncing inside this callback. When None, offline
                straight-line prediction from (p0, v_ball) is used.
    """
    t  = t_min
    dt = 0.1

    t_search_start = time.perf_counter()

    while t < t_max:

        if get_state is not None:
            p_now, v_now, q_now = get_state()
        else:
            t_elapsed = time.perf_counter() - t_search_start
            p_now = p0 + v_ball * t_elapsed
            v_now = v_ball
            q_now = q_current

        # ── Analytic reachable window: skip or terminate early ───────────────
        window = reachable_window(p_now, v_now, base_pos, t, t_max, ball_radius)
        if window is None:
            if verbose: print("Ball never (re)enters the workspace — aborting.")
            return None

        t_enter, t_exit = window
        if t < t_enter:
            if verbose: print(f"t={t:.2f}  skip ahead to window entry t={t_enter:.2f}")
            t = t_enter
        elif t > t_exit:
            if verbose: print("Ball has permanently left the workspace — aborting.")
            return None

        p_ball = p_now + v_now * t

        # ── Workspace check (exact, incl. inner exclusions + pregrasp) ───────
        T_pregrasp, T_grasp = compute_grasp_pose(p_ball, v_now, ball_radius)

        if not (check_workspace(T_pregrasp[:3, 3], base_pos) and
                check_workspace(T_grasp[:3, 3],    base_pos)):
            if verbose: print(f"t={t:.2f}  REJECT workspace")
            t += dt; continue

        # ── IK ────────────────────────────────────────────────────────────────
        q_pg, ok = IK_frantik(T_pregrasp, q_now)
        if not ok:
            if verbose: print(f"t={t:.2f}  REJECT IK pregrasp")
            t += dt; continue

        q_g, ok = IK_frantik(T_grasp, q_pg)
        if not ok:
            if verbose: print(f"t={t:.2f}  REJECT IK grasp")
            t += dt; continue

        # ── Stage 1: bang-bang lower-bound rejection (provably safe) ─────────
        T1, T2, gripper_close = lower_bound_time(q_now, q_pg, q_g)
        t_lb = T1 + T2 + gripper_close + correction_time

        if t_lb > t:
            if verbose: print(f"t={t:.2f}  REJECT lower bound: {t_lb:.3f} > {t:.3f}")
            t += dt; continue

        # ── Stage 2: quintic timing on the accepted candidate ────────────────
        t_solve_start = time.perf_counter()
        T1_q, T2_q, gripper_close = estimate_real_time_quintic(q_now, q_pg, q_g)
        t_solve = time.perf_counter() - t_solve_start

        t_sim  = T1_q + T2_q + gripper_close + correction_time
        margin = t - t_sim

        # Require slack even after paying the measured solve time
        if margin <= t_solve:
            if verbose:
                print(f"t={t:.2f}  REJECT quintic: exec {t_sim:.3f} s "
                      f"+ solve {t_solve:.3f} s > budget {t:.3f} s")
            t += dt; continue

        margin -= t_solve

        # ── Solution found ────────────────────────────────────────────────────
        if verbose:
            t_elapsed = time.perf_counter() - t_search_start
            print(f"Solution accepted | search_time={t_elapsed:.3f} s | "
                  f"lb={t_lb:.3f} s | quintic={t_sim:.3f} s | "
                  f"solve={t_solve:.3f} s | margin={margin:.3f} s")

        return {
            "T1"                 : T1_q,
            "T2"                 : T2_q,
            "margin"             : margin,      # net of measured solve time
            "t_solve"            : t_solve,     # measured quintic timing wall time
            "GRIPPER_CLOSE_TIME" : gripper_close,
            "p_intercept"        : p_ball.copy(),
            "T_pregrasp"         : T_pregrasp,
            "T_grasp"            : T_grasp,
            "q_pregrasp"         : q_pg,
            "q_grasp"            : q_g,
            "t_sim"              : t_sim,
            "correction_time"    : correction_time,
        }

    return None


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
