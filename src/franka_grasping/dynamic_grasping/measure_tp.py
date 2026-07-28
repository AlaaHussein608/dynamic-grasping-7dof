"""
measure_tp.py — calibrate t_p (paper Sec. IV, Eq. 12).

t_p is the wall-clock time of ONE rendezvous-point evaluation. The paper
assumed a known constant (t_p = 0.1 s in their simulation); here it is
measured on your machine.

Two quantities are timed:
  A) one toppra_segment call            (a single PTP time-parameterization)
  B) one full CROFT candidate evaluation (2x IK_frantik + 2x toppra_segment)

Use B as T_P in find_intercept — that is what one candidate actually costs.
Median is reported as the recommended value (robust to first-call and GC
spikes).

Run:  python measure_tp.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import IK_frantik, Q_MIN, Q_MAX, ik_model, ik_data
from intercept_planner_toppra import toppra_segment

N_SAMPLES = 30
_EE_FRAME = ik_model.getFrameId("gripper")


def fk_pose(q):
    """4x4 pose of the gripper frame at configuration q."""
    pin.forwardKinematics(ik_model, ik_data, q)
    pin.updateFramePlacements(ik_model, ik_data)
    M = ik_data.oMf[_EE_FRAME]
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3]  = M.translation
    return T


def stats(name, samples):
    s = np.asarray(samples)
    print(f"{name:38s} n={len(s):3d}  "
          f"mean={s.mean():.4f} s  median={np.median(s):.4f} s  "
          f"std={s.std():.4f} s  max={s.max():.4f} s")
    return float(np.median(s))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # warm-up: first calls pay import/JIT/allocation costs
    for _ in range(3):
        qa, qb = rng.uniform(Q_MIN, Q_MAX, (2, 7))
        toppra_segment(qa, qb)

    # ── A: single TOPP-RA segment ────────────────────────────────────────────
    times_toppra = []
    for _ in range(N_SAMPLES):
        qa, qb = rng.uniform(Q_MIN, Q_MAX, (2, 7))
        t0 = time.perf_counter()
        toppra_segment(qa, qb)
        times_toppra.append(time.perf_counter() - t0)

    # ── B: full candidate evaluation (2x IK + 2x TOPP-RA) ────────────────────
    # Target poses come from FK of random configurations, so they are
    # guaranteed reachable and IK exercises its realistic cost.
    times_eval = []
    attempts = 0
    while len(times_eval) < N_SAMPLES and attempts < 5 * N_SAMPLES:
        attempts += 1
        q_now = rng.uniform(Q_MIN, Q_MAX)
        T_pg  = fk_pose(rng.uniform(Q_MIN, Q_MAX))
        T_g   = fk_pose(rng.uniform(Q_MIN, Q_MAX))

        t0 = time.perf_counter()
        q_pg, ok = IK_frantik(T_pg, q_now)
        if not ok:
            continue
        q_g, ok = IK_frantik(T_g, q_pg)
        if not ok:
            continue
        toppra_segment(q_now, q_pg)
        toppra_segment(q_pg,  q_g)
        times_eval.append(time.perf_counter() - t0)

    print()
    stats("A: toppra_segment (one PTP)", times_toppra)
    t_p = stats("B: full evaluation (2 IK + 2 TOPP-RA)", times_eval)

    print(f"\nRecommended constant for CROFT.find_intercept:\n"
          f"    T_P = {t_p:.4f}   # seconds, measured on this machine")
