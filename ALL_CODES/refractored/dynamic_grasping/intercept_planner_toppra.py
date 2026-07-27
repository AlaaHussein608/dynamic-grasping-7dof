"""
intercept_planner_toppra.py
===========================
Intercept-point selection for dynamic grasping of a rolling ball with a
Franka Emika Panda arm: brute-force forward scan over candidate times with
a bang-bang lower-bound prefilter, TOPP-RA giving the true time-optimal
motion time on accepted candidates.

Shared machinery (frantik IK, workspace check, reachable window, grasp-pose
construction, bang-bang lower bound, TOPP-RA retiming) lives in the shared
modules; this file contains only the search algorithm.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import (GRIPPER_CLOSE_TIME, Q_MIN, Q_MAX,
                                  IK_frantik, check_workspace,
                                  reachable_window, compute_grasp_pose,
                                  lower_bound_time, ik_model, ik_data)
from shared.trajectory_control import toppra_segment as _toppra_segment


def toppra_segment(q_start, q_end):
    """TOPP-RA retiming under the Panda limits (shared IK Pinocchio model)."""
    return _toppra_segment(q_start, q_end, ik_model, ik_data)


# ── TOPP-RA: true time-optimal estimate (called once on accepted candidate) ───

def estimate_real_time_toppra(q_start, q_pg, q_g):
    """
    True time-optimal durations via TOPP-RA.
    Returns T1, T2, gripper_close, and the two trajectory objects
    so the caller can use them directly for execution.
    """
    traj1 = toppra_segment(q_start, q_pg)
    traj2 = toppra_segment(q_pg,    q_g)
    return traj1.duration, traj2.duration, GRIPPER_CLOSE_TIME, traj1, traj2


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

        # ── Stage 2: TOPP-RA on the accepted candidate (solve time measured) ─
        if verbose: print(f"t={t:.2f}  lower bound ok ({t_lb:.3f} s) — running TOPP-RA …")

        t_solve_start = time.perf_counter()
        T1_opt, T2_opt, gripper_close, traj1, traj2 = estimate_real_time_toppra(
            q_now, q_pg, q_g
        )
        t_solve = time.perf_counter() - t_solve_start

        t_sim_opt = T1_opt + T2_opt + gripper_close + correction_time
        margin    = t - t_sim_opt

        # Require slack even after paying the measured solve time
        if margin <= t_solve:
            if verbose:
                print(f"t={t:.2f}  REJECT TOPP-RA: exec {t_sim_opt:.3f} s "
                      f"+ solve {t_solve:.3f} s > budget {t:.3f} s")
            t += dt; continue

        margin -= t_solve

        # ── Solution found ────────────────────────────────────────────────────
        if verbose:
            t_elapsed = time.perf_counter() - t_search_start
            print(f"Solution accepted | search_time={t_elapsed:.3f} s | "
                  f"lb={t_lb:.3f} s | toppra={t_sim_opt:.3f} s | "
                  f"solve={t_solve:.3f} s | margin={margin:.3f} s")

        return {
            "T1"                 : T1_opt,
            "T2"                 : T2_opt,
            "margin"             : margin,      # net of measured solve time
            "t_solve"            : t_solve,     # measured TOPP-RA wall time
            "GRIPPER_CLOSE_TIME" : gripper_close,
            "p_intercept"        : p_ball.copy(),
            "T_pregrasp"         : T_pregrasp,
            "T_grasp"            : T_grasp,
            "q_pregrasp"         : q_pg,
            "q_grasp"            : q_g,
            "t_sim"              : t_sim_opt,
            "traj1"              : traj1,   # TOPP-RA trajectory: home → pregrasp
            "traj2"              : traj2,   # TOPP-RA trajectory: pregrasp → grasp
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
