"""
intercept_planner_croft.py — Rendezvous-point selection (Croft, Fenton,
Benhabib 1998, IEEE Trans. SMC-B 28(2):192-204).

Implements: Eq. (7) interception-time objective, Eq. (8) y_hat = min{y_j},
Eq. (10)/(14) temporal convergence, Sec. III polynomial r(t) surrogate
intersected with h(t) = t, Fig. 10 counter rule (stop after 2 consecutive
worsenings of y_hat).

Only the current evaluation's own delay (state read -> TOPP-RA done) eats
into t_cand's budget; it is MEASURED and charged: r_adj = r_bar + dt_eval.

Sections V-VI (uncertainty tolerance region) omitted: sim state is exact.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import GRIPPER_CLOSE_TIME, reachable_window
from dg_common import ee_pos, toppra_segment, solve_grasp_ik

T_P_SEG  = 0.0179                 # one toppra_segment call (measure_tp.py)
T_P_EVAL = 2.0 * T_P_SEG          # one candidate evaluation (two segments)


def interception_time(r_j, h_j, V_r, V_t):
    """Eq. (7): y = h if robot early; catching-line if late."""
    if r_j <= h_j:
        return h_j
    if V_r <= V_t:
        return np.inf
    return r_j + (r_j - h_j) / (V_r / V_t - 1.0)


def _intersect_h(ts, rs, t_lo, t_hi):
    """Fit r(t) (deg <= 2) and return earliest root of r(t) - t in window."""
    p = np.polyfit(ts, rs, min(len(ts) - 1, 2))
    p[-2] -= 1.0
    real = sorted(r.real for r in np.roots(p)
                  if abs(r.imag) < 1e-9 and t_lo <= r.real <= t_hi)
    return real[0] if real else None


def find_intercept(p0, v_ball, q_current,
                   ball_radius=0.03, base_pos=np.zeros(3),
                   t_min=0.0, t_max=40.0,
                   get_state=None,
                   t_p_eval=T_P_EVAL,
                   correction_time=0.06,
                   max_iters=12,
                   verbose=True):
    """
    get_state : callback () -> (p_ball_now, v_ball_now, q_now); caller owns
                sim stepping inside it. None = offline straight-line mode.
    t_p_eval  : predicted cost of the next evaluation, for Eq. (14).
    """
    t_search_start = time.perf_counter()

    j_count = 0

    def read_state():
        if get_state is not None:
            return get_state()
        elapsed = time.perf_counter() - t_search_start
        return p0 + v_ball * elapsed, v_ball.copy(), q_current.copy()

    def full_eval(t_cand):
        """IK + TOPP-RA at candidate time t_cand (relative to 'now')."""
        nonlocal j_count

        t0 = time.perf_counter()                  # t_cand's clock starts here
        p_now, v_now, q_now = read_state()
        p_ball = p_now + v_now * t_cand

        sol = solve_grasp_ik(p_ball, v_now, q_now, ball_radius, base_pos)
        if sol is None:
            return None
        q_pg, q_g, T_pregrasp, T_grasp = sol

        traj1 = toppra_segment(q_now, q_pg)
        traj2 = toppra_segment(q_pg,  q_g)
        r_bar = (traj1.duration + traj2.duration +
                 GRIPPER_CLOSE_TIME + correction_time)

        j_count += 1
        dt_eval = time.perf_counter() - t0        # measured planning delay
        r_adj   = r_bar + dt_eval                 # Eq. (12), per-eval form

        V_t = np.linalg.norm(v_now[:2])
        V_r = np.linalg.norm(ee_pos(q_now) - T_grasp[:3, 3]) / r_bar
        y   = interception_time(r_adj, t_cand, V_r, V_t)   # Eq. (7)

        return dict(t=t_cand, r_bar=r_bar, r_adj=r_adj, y=y,
                    p_ball=p_ball.copy(),
                    T_pregrasp=T_pregrasp, T_grasp=T_grasp,
                    q_pg=q_pg, q_g=q_g, traj1=traj1, traj2=traj2)

    def eval_with_fallback(t_cand, t_anchor, tries=3):
        for _ in range(tries):
            e = full_eval(t_cand)
            if e is not None:
                return e
            t_cand = 0.5 * (t_cand + t_anchor)
        return None

    # ── two initial states spanning the reachable window ─────────────────────
    p_now, v_now, _ = read_state()
    window = reachable_window(p_now, v_now, base_pos, t_min, t_max, ball_radius)
    if window is None:
        return None
    t_enter, t_exit = window
    span = t_exit - t_enter
    if span < 1e-3:
        return None

    mid = t_enter + 0.5 * span
    e1 = eval_with_fallback(t_enter + 0.05 * span, mid)
    e2 = eval_with_fallback(t_exit  - 0.05 * span, mid)
    evals = [e for e in (e1, e2) if e is not None]
    if not evals:
        return None

    best  = min(evals, key=lambda e: e["y"])
    y_hat = best["y"]
    counter = 0

    # ── iterate: fit surrogate, intersect with h(t)=t, evaluate ──────────────
    for _ in range(max_iters):
        if len(evals) < 2:
            break

        t_new = _intersect_h([e["t"] for e in evals],
                             [e["r_adj"] for e in evals], t_enter, t_exit)
        if t_new is None or any(abs(t_new - e["t"]) < 1e-3 for e in evals):
            t_new = 0.5 * (best["t"] +
                           (t_enter if best["t"] > mid else t_exit))

        if t_p_eval > y_hat - t_new:              # Eq. (14), next-eval cost
            if verbose:
                print(f"Temporal convergence: t_p_eval={t_p_eval:.4f} > "
                      f"Δt={y_hat - t_new:.4f}")
            break

        e_new = eval_with_fallback(t_new, best["t"])
        if e_new is None:
            break
        evals.append(e_new)

        if e_new["y"] < y_hat:
            y_hat, best, counter = e_new["y"], e_new, 0
        else:
            counter += 1
            if counter >= 2:                      # Fig. 10
                break

    # ── feasibility of the selected rendezvous-point ──────────────────────────
    t_star, t_sim = best["t"], best["r_bar"]
    margin = t_star - t_sim
    if margin < 0:
        if verbose:
            print(f"Best candidate infeasible: r={t_sim:.3f} > t={t_star:.3f}")
        return None

    if verbose:
        print(f"t*={t_star:.3f}  y_hat={y_hat:.3f}  r={t_sim:.3f}  "
              f"margin={margin:.3f}  evals={j_count}")

    return {
        "T1"                : best["traj1"].duration,
        "T2"                : best["traj2"].duration,
        "GRIPPER_CLOSE_TIME": GRIPPER_CLOSE_TIME,
        "margin"            : margin,
        "t_sim"             : t_sim,
        "t_intercept"       : t_star,
        "y_hat"             : y_hat,
        "n_evals"           : j_count,
        "correction_time"   : correction_time,
        "p_intercept"       : best["p_ball"],
        "T_pregrasp"        : best["T_pregrasp"],
        "T_grasp"           : best["T_grasp"],
        "q_pregrasp"        : best["q_pg"],
        "q_grasp"           : best["q_g"],
        "traj1"             : best["traj1"],
        "traj2"             : best["traj2"],
    }


if __name__ == "__main__":
    p0     = np.array([0.5, -0.8, 0.03])
    v_ball = np.array([-0.1,  0.2,  0.0])
    q0     = np.zeros(7)

    result = find_intercept(p0, v_ball, q0, verbose=True)

    if result:
        print(f"\nt*={result['t_intercept']:.3f}  T1={result['T1']:.3f}  "
              f"T2={result['T2']:.3f}  margin={result['margin']:.3f}  "
              f"evals={result['n_evals']}")
    else:
        print("No feasible rendezvous-point found.")
