"""
contraction_gains.py
====================
Contraction-factor (alpha) selection for the contractive MPC — the core
algorithm extracted from the two contractive dynamic-grasping scripts.

Two variants exist:

  * error-based  (v1, run_contractive_grasping.py):
    alpha lower bound from the velocity-limited one-step displacement only.

  * state-aware  (v2, run_contractive_grasping_v2.py):
    alpha lower bound from the guaranteed one-step displacement toward the
    target given the current joint velocity and torque authority; also
    provides the bang-bang travel-time bound used by the v2 planner.

Both share the geometric decay requirement
    alpha_req = (epsilon / ||e||)^(1/N).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import DQ_MAX, TAU_MAX

# Maximum inertia diagonal — bounds the guaranteed per-joint torque authority.
M_MAX = np.array([3.69124433, 3.70289864, 1.50436853, 1.0414124,
                  0.05894302, 0.0554556,  0.00668415])

ALPHA_LO = 0.05
ALPHA_HI = 0.999

# v2 constants: guaranteed accel authority and accel/decel time allowance
A_MIN   = float(np.min(TAU_MAX / M_MAX))   # [rad/s²]
DQ_MIN  = float(np.min(DQ_MAX))            # [rad/s]
T_ACCEL = 2.0 * DQ_MIN / A_MIN             # [s]


def alpha_required(error_norm, epsilon, n_horizon):
    """
    Alpha needed so that ||e|| decays geometrically to epsilon in n_horizon
    steps:  alpha_req = (epsilon / ||e||)^(1/N).
    Returns 0.0 when already inside the basin (||e|| <= epsilon).
    """
    if error_norm <= epsilon:
        return 0.0
    return (epsilon / error_norm) ** (1.0 / n_horizon)


# ─────────────────────────────────────────────────────────────────────────────
#  v1 — error-based variant
# ─────────────────────────────────────────────────────────────────────────────

def alpha_min_error_based(error_norm, dt):
    """
    Lower bound on alpha: the contracted reference must not require a
    single-step displacement exceeding the reachable set.

        alpha_min = 1 - min_j(dq_max_j) * dt / ||e||
    """
    if error_norm < 1e-6:
        return 0.99
    return max(1.0 - float(np.min(DQ_MAX)) * dt / error_norm, 0.05)


def compute_alpha_error_based(error_norm, epsilon, n_horizon, dt):
    """Canonical contraction factor (v1): larger of the geometrically-required
    alpha and the one-step-reachability lower bound, clipped to (0.05, 0.999)."""
    alpha_req = alpha_required(error_norm, epsilon, n_horizon)
    alpha_lb  = alpha_min_error_based(error_norm, dt)
    return float(np.clip(max(alpha_req, alpha_lb), 0.05, 0.999))


def intercept_is_feasible(error_norm, epsilon, n_horizon, dt):
    """
    True when the required contraction is achievable given the one-step
    reachability constraint. Infeasible only when alpha_required < alpha_min
    AND the error is already outside the basin (alpha_required != 0.0).
    """
    alpha_req = alpha_required(error_norm, epsilon, n_horizon)
    if alpha_req == 0.0:
        return True   # already inside basin — trivially reachable
    return alpha_req >= alpha_min_error_based(error_norm, dt)


# ─────────────────────────────────────────────────────────────────────────────
#  v2 — state-aware variant
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha_state_aware(e, dq, epsilon, n_horizon, dt):
    """
    State-aware contraction factor (v2).

    The one-step displacement toward the target guaranteed by the dynamics is
        d1 = <dq, e_hat> * dt + 0.5 * a_min * dt^2,   a_min = min_j(tau_max/M_max),
    capped at min_j(dq_max_j) * dt. The loosest admissible contraction is
        alpha_lb = 1 - d1 / ||e||,
    saturating at ALPHA_HI when d1 <= 0. The returned alpha is
    max(alpha_req, alpha_lb) clipped to [ALPHA_LO, ALPHA_HI].
    """
    error_norm = float(np.linalg.norm(e))
    if error_norm < 1e-6:
        return ALPHA_LO

    alpha_req = alpha_required(error_norm, epsilon, n_horizon)

    v_toward = float(dq @ e) / error_norm
    d1 = v_toward * dt + 0.5 * A_MIN * dt ** 2
    d1 = min(d1, DQ_MIN * dt)

    alpha_lb = ALPHA_HI if d1 <= 0.0 else 1.0 - d1 / error_norm
    return float(np.clip(max(alpha_req, alpha_lb), ALPHA_LO, ALPHA_HI))


def min_travel_time(q_from, q_to, time_margin=1.25):
    """
    Bang-bang-style lower bound on joint-space travel time: slowest-joint
    velocity-limit time with a safety margin, plus a fixed accel/decel
    allowance.
    """
    t_vel = float(np.max(np.abs(q_to - q_from) / DQ_MAX))
    return time_margin * t_vel + T_ACCEL
