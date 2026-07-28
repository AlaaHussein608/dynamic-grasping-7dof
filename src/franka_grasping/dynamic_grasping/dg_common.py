"""
dg_common.py
============
Dynamic-grasping helpers — a thin, grasping-specific layer over the ``shared``
modules so the intercept planners and run scripts stop re-implementing the
same machinery.

Provides:
  * :func:`ee_pos`          — end-effector FK bound to the shared IK model,
  * :func:`toppra_segment`  — TOPP-RA retiming bound to the shared IK model,
  * :func:`solve_grasp_ik`  — grasp-pose construction + workspace check + the
                              two frantik IK solves every planner needs,
  * :func:`estimate_toppra` / :func:`estimate_quintic` — segment-timing
    estimators,
  * :func:`forward_scan_intercept` — the brute-force forward-scan intercept
    search shared by the TOPP-RA and quintic planners (parameterised by the
    timing estimator).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import (GRIPPER_CLOSE_TIME, IK_frantik,
                                  check_workspace, reachable_window,
                                  compute_grasp_pose, lower_bound_time,
                                  quintic_min_time, ee_position,
                                  ik_model, ik_data, ik_ee_id)
from shared.trajectory_control import toppra_segment as _toppra_segment


# ─────────────────────────────────────────────────────────────────────────────
#  End-effector FK and TOPP-RA, bound to the shared IK Pinocchio model
# ─────────────────────────────────────────────────────────────────────────────

def ee_pos(q):
    """End-effector position for configuration ``q`` (shared IK model)."""
    return ee_position(ik_model, ik_data, ik_ee_id, q)


def toppra_segment(q_start, q_end):
    """TOPP-RA time-optimal retiming q_start → q_end (shared IK model)."""
    return _toppra_segment(q_start, q_end, ik_model, ik_data)


# ─────────────────────────────────────────────────────────────────────────────
#  Grasp-pose IK evaluation (shared by every intercept planner)
# ─────────────────────────────────────────────────────────────────────────────

def solve_grasp_ik(p_ball, v_ball, q_seed, ball_radius=0.03,
                   base_pos=np.zeros(3)):
    """
    Grasp-pose inverse kinematics for a predicted ball position.

    Builds the pregrasp/grasp poses, checks both lie inside the reachable
    workspace, and solves frantik IK for each (grasp seeded from the pregrasp
    solution). Returns ``(q_pg, q_g, T_pregrasp, T_grasp)`` or ``None`` on any
    workspace/IK failure.
    """
    T_pregrasp, T_grasp = compute_grasp_pose(p_ball, v_ball, ball_radius)

    if not (check_workspace(T_pregrasp[:3, 3], base_pos) and
            check_workspace(T_grasp[:3, 3], base_pos)):
        return None

    q_pg, ok = IK_frantik(T_pregrasp, q_seed)
    if not ok:
        return None

    q_g, ok = IK_frantik(T_grasp, q_pg)
    if not ok:
        return None

    return q_pg, q_g, T_pregrasp, T_grasp


# ─────────────────────────────────────────────────────────────────────────────
#  Segment-timing estimators
# ─────────────────────────────────────────────────────────────────────────────

def estimate_toppra(q_start, q_pg, q_g):
    """TOPP-RA time-optimal durations + the two trajectory objects.

    Returns ``(T1, T2, gripper_close, (traj1, traj2))``.
    """
    traj1 = toppra_segment(q_start, q_pg)
    traj2 = toppra_segment(q_pg,    q_g)
    return traj1.duration, traj2.duration, GRIPPER_CLOSE_TIME, (traj1, traj2)


def estimate_quintic(q_start, q_pg, q_g):
    """Closed-form quintic minimum-time durations (no trajectory object).

    Returns ``(T1, T2, gripper_close, None)``.
    """
    T1 = quintic_min_time(q_start, q_pg)
    T2 = quintic_min_time(q_pg,    q_g)
    return T1, T2, GRIPPER_CLOSE_TIME, None


# ─────────────────────────────────────────────────────────────────────────────
#  Brute-force forward-scan intercept search (TOPP-RA and quintic planners)
# ─────────────────────────────────────────────────────────────────────────────

def forward_scan_intercept(p0, v_ball, q_current, estimate,
                           ball_radius=0.03, base_pos=np.zeros(3),
                           t_min=0.0, t_max=40.0, dt=0.1,
                           correction_time=0.06,
                           get_state=None, verbose=True):
    """
    Forward scan over candidate intercept times, shared by the TOPP-RA and
    quintic planners.

    ``estimate(q_start, q_pg, q_g)`` returns ``(T1, T2, gripper_close, extra)``
    where ``extra`` is the pair of TOPP-RA trajectories (TOPP-RA planner) or
    ``None`` (quintic planner). ``get_state`` is an optional callback
    ``() -> (p_ball, v_ball, q)`` giving the live sim state; when ``None`` an
    offline straight-line ball model is used.

    Acceptance is two-stage: a bang-bang lower bound prefilters candidates,
    then the (measured) estimate must still leave positive margin after paying
    its own solve time. Returns the intercept result dict, or ``None``.
    """
    t = t_min
    t_search_start = time.perf_counter()

    while t < t_max:

        if get_state is not None:
            p_now, v_now, q_now = get_state()
        else:
            t_elapsed = time.perf_counter() - t_search_start
            p_now = p0 + v_ball * t_elapsed
            v_now = v_ball
            q_now = q_current

        # ── Analytic reachable window: skip ahead or terminate early ─────────
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

        # ── Grasp-pose workspace + IK (shared helper) ────────────────────────
        sol = solve_grasp_ik(p_ball, v_now, q_now, ball_radius, base_pos)
        if sol is None:
            if verbose: print(f"t={t:.2f}  REJECT workspace/IK")
            t += dt; continue
        q_pg, q_g, T_pregrasp, T_grasp = sol

        # ── Stage 1: bang-bang lower-bound rejection (provably safe) ─────────
        T1, T2, gripper_close = lower_bound_time(q_now, q_pg, q_g)
        t_lb = T1 + T2 + gripper_close + correction_time
        if t_lb > t:
            if verbose: print(f"t={t:.2f}  REJECT lower bound: {t_lb:.3f} > {t:.3f}")
            t += dt; continue

        # ── Stage 2: full timing estimate (solve time measured) ──────────────
        t_solve_start = time.perf_counter()
        T1_e, T2_e, gripper_close, extra = estimate(q_now, q_pg, q_g)
        t_solve = time.perf_counter() - t_solve_start

        t_sim  = T1_e + T2_e + gripper_close + correction_time
        margin = t - t_sim
        if margin <= t_solve:                 # require slack after paying solve time
            if verbose:
                print(f"t={t:.2f}  REJECT estimate: exec {t_sim:.3f} s "
                      f"+ solve {t_solve:.3f} s > budget {t:.3f} s")
            t += dt; continue
        margin -= t_solve

        # ── Solution found ───────────────────────────────────────────────────
        if verbose:
            t_elapsed = time.perf_counter() - t_search_start
            print(f"Solution accepted | search_time={t_elapsed:.3f} s | "
                  f"lb={t_lb:.3f} s | est={t_sim:.3f} s | "
                  f"solve={t_solve:.3f} s | margin={margin:.3f} s")

        result = {
            "T1"                 : T1_e,
            "T2"                 : T2_e,
            "margin"             : margin,      # net of measured solve time
            "t_solve"            : t_solve,     # measured estimate wall time
            "GRIPPER_CLOSE_TIME" : gripper_close,
            "p_intercept"        : p_ball.copy(),
            "T_pregrasp"         : T_pregrasp,
            "T_grasp"            : T_grasp,
            "q_pregrasp"         : q_pg,
            "q_grasp"            : q_g,
            "t_sim"              : t_sim,
            "correction_time"    : correction_time,
        }
        if extra is not None:                   # TOPP-RA planner: attach trajectories
            result["traj1"], result["traj2"] = extra
        return result

    return None
