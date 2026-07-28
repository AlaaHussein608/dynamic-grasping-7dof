"""
benchmark_metrics.py
====================
Per-episode metric extraction, seed aggregation and report printing shared
by the FL / SMC / MPC controller benchmarks (and the Q/R sweep).

Extracted verbatim from the identical blocks that were duplicated in
benchmark_fl.py, benchmark_smc.py and both benchmark_mpc.py copies.

Episode results are dicts with keys:
    q_desired, q_actual, tau, time, wall_times   (numpy arrays)
"""

import numpy as np

from shared.franka_common import Q_MIN, Q_MAX, TAU_MAX


# ─────────────────────────────────────────────────────────────────────────────
#  Per-episode metrics
# ─────────────────────────────────────────────────────────────────────────────

def gather_metrics(result, q_target):
    q_des = result["q_desired"]
    q_act = result["q_actual"]
    tau   = result["tau"]
    t_vec = result["time"]
    walls = result["wall_times"]

    # 1. RMSE
    rmse_val = float(np.sqrt(np.mean((q_des - q_act) ** 2)))

    # 2. Effort
    tau_rms = np.sqrt(np.mean(tau ** 2, axis=0))           # (7,)
    energy  = float(np.sum(np.trapz(tau ** 2, t_vec, axis=0)))

    # 3. Smoothness — mean norm of 2nd diff of torque
    if tau.shape[0] >= 3:
        jerk = float(np.mean(np.linalg.norm(np.diff(np.diff(tau, axis=0), axis=0), axis=1)))
    else:
        jerk = float("nan")

    # 4. Computational cost
    wall_mean = float(np.mean(walls) * 1000)
    wall_max  = float(np.max(walls)  * 1000)

    # 5. Constraint violations
    q_viol   = float(np.mean(np.any((q_act < Q_MIN - 1e-4) | (q_act > Q_MAX + 1e-4), axis=1)))
    tau_viol = float(np.mean(np.any(np.abs(tau) > TAU_MAX + 1e-4, axis=1)))

    # 8. Settling time & overshoot
    err_norm = np.linalg.norm(q_act - q_target, axis=1)
    peak_err = err_norm[0] if err_norm[0] > 1e-8 else (np.max(err_norm) + 1e-8)
    thresh   = 0.02 * peak_err
    settled  = err_norm < thresh
    crossings = np.where(np.diff(settled.astype(int)) > 0)[0]
    settling_time = float(t_vec[crossings[-1]]) if len(crossings) else float(t_vec[-1])
    approach  = np.linalg.norm(q_target - q_act[0])
    overshoot = max(0.0, (float(np.max(err_norm)) - approach) / approach) if approach > 1e-8 else 0.0

    return {
        "rmse":          rmse_val,
        "tau_rms":       tau_rms,
        "energy":        energy,
        "jerk":          jerk,
        "wall_mean_ms":  wall_mean,
        "wall_max_ms":   wall_max,
        "q_viol":        q_viol,
        "tau_viol":      tau_viol,
        "settling_time": settling_time,
        "overshoot":     overshoot,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate: mean over all seeds
# ─────────────────────────────────────────────────────────────────────────────

def mean_metrics(metrics_list):
    """Average all scalar fields; for tau_rms average element-wise."""
    keys = [k for k in metrics_list[0] if k != "tau_rms"]
    out  = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    out["tau_rms"] = np.mean(np.stack([m["tau_rms"] for m in metrics_list]), axis=0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 68
SEP2 = "-" * 68


def print_results(controller_label, traj_label, n_seeds,
                  m_normal, m_dist, m_noise,
                  disturbance_std, disturbance_steps, sensor_noise_std,
                  cost_label="controller law only"):
    print(f"\n{SEP}")
    print(f"  {controller_label:<4} |  Trajectory: {traj_label.upper()}  |  mean over {n_seeds} seeds")
    print(SEP)

    print(f"\n[1] Tracking RMSE — Normal")
    print(f"    RMSE  : {m_normal['rmse']:.6f}  rad")

    print(f"\n[2] Control Effort")
    rms = m_normal['tau_rms']
    print(f"    tau RMS  [N·m]  : " + "  ".join(f"J{i+1}={rms[i]:.2f}" for i in range(7)))
    print(f"    Total energy    : {m_normal['energy']:.2f}  N²m²s")

    print(f"\n[3] Smoothness")
    print(f"    Mean jerk norm  : {m_normal['jerk']:.4f}  (N·m/s²  proxy)")

    print(f"\n[4] Computational Cost  ({cost_label})")
    print(f"    Mean step time  : {m_normal['wall_mean_ms']:.4f}  ms")
    print(f"    Max  step time  : {m_normal['wall_max_ms']:.4f}  ms")

    print(f"\n[5] Constraint Handling")
    print(f"    Joint pos. violation rate : {m_normal['q_viol']*100:.2f} %")
    print(f"    Torque    violation rate  : {m_normal['tau_viol']*100:.2f} %")

    print(f"\n[6] Robustness — Disturbance Injection"
          f"  (std={disturbance_std} N·m, {disturbance_steps} steps)")
    print(f"    RMSE under disturbance : {m_dist['rmse']:.6f}  rad")

    print(f"\n[7] Robustness — Sensor Noise  (std={sensor_noise_std} rad)")
    print(f"    RMSE under sensor noise: {m_noise['rmse']:.6f}  rad")

    print(f"\n[8] Stability & Convergence")
    print(f"    Settling time  : {m_normal['settling_time']:.4f}  s")
    print(f"    Overshoot      : {m_normal['overshoot']*100:.2f} %")
    print(f"\n{SEP2}")
