"""
sweep_n_qr.py
=============
Joint sweep of horizon length N and cost matrices (Q, R) for the
trajectory-tracking MPC on the Franka Panda.

Structure
---------
  Outer loop : N ∈ N_VALUES         — each N builds (or loads) a solver
  Inner loop : (q_scale, q_pos, r)  — Q/R grid, reuses the same solver

Scoring (lower = better):
    score = W_TRACK * mean_rmse_normal + W_DIST * mean_rmse_disturbance

After the full sweep, the globally best (N, Q, R) triple is used for
a final detailed benchmark and the complete metric report is printed.

Refactored from: sweep_horizon_length.py  +  sweep_qr_weights.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import pinocchio as pin

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_HOME,
                                  DQ_MAX, TAU_MAX, NQ, NV, NX, NU,
                                  sample_targets, script_dir)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       toppra_segment as _toppra_segment,
                                       precompute_feedforward)
from shared.acados_mpc import (build_franka_ocp_solver, load_solver as _load,
                               apply_cost_weights, init_warm_start,
                               shift_warm_start, pin_initial_state,
                               set_trajectory_references)
from benchmark_metrics import gather_metrics, mean_metrics

SCRIPT_DIR = script_dir(__file__)

# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP RANGES
# ─────────────────────────────────────────────────────────────────────────────

N_VALUES       = list(range(2, 21, 3))       # [2, 5, 8, 11, 14, 17, 20]

Q_SCALE_VALUES = np.linspace(50,   200,  4)  # overall Q multiplier
Q_POS_VALUES   = np.linspace(100,  300,  4)  # position diagonal element
Q_VEL_RATIO    = 0.10                        # vel diag = Q_VEL_RATIO * q_pos
R_VALUES       = np.linspace(0.5,  5.0,  4)  # R diagonal element

# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION BUDGET
# ─────────────────────────────────────────────────────────────────────────────

N_EVAL_SEEDS  = 5    # seeds used during grid search (fast)
N_FINAL_SEEDS = 5    # seeds used for the final benchmark
MASTER_SEED   = 42

W_TRACK = 0.5        # weight for normal RMSE in score
W_DIST  = 0.5        # weight for disturbance RMSE in score

# ─────────────────────────────────────────────────────────────────────────────
#  MPC / DISTURBANCE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

MPC_H             = 0.005
DISTURBANCE_STD   = 2.0
DISTURBANCE_STEPS = 80
SENSOR_NOISE_STD  = 0.002
TRAJ_TYPE         = "toppra"
HEADLESS          = True

# ─────────────────────────────────────────────────────────────────────────────
#  MuJoCo + Pinocchio models  (shared across all experiments)
# ─────────────────────────────────────────────────────────────────────────────

pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()

mj_model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
mj_data  = mujoco.MjData(mj_model)
mj_data.qpos[:7] = Q_HOME.copy()
mujoco.mj_forward(mj_model, mj_data)
mj_model.opt.gravity[:] = [0, 0, -9.81]

# ─────────────────────────────────────────────────────────────────────────────
#  Target pools
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TARGETS  = sample_targets(N_EVAL_SEEDS,  MASTER_SEED)
FINAL_TARGETS = sample_targets(N_FINAL_SEEDS, MASTER_SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  Solver build / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solver_json(N_val):
    return SCRIPT_DIR / f"franka_N{N_val}.json"


def get_solver(N_val):
    """Build solver for N_val if not cached, then load and return it."""
    json_file = _solver_json(N_val)
    if not os.path.exists(json_file):
        print(f"  [build] Compiling solver for N={N_val} ...")
        build_franka_ocp_solver(
            model_name=f"franka_N{N_val}",
            h=MPC_H, N=N_val,
            Q=np.diag([200.0] * NQ + [20.0] * NV),   # placeholder — overwritten per candidate
            R=np.diag([2.0] * NU),
            urdf_path=PANDA_URDF,
            mjcf_path=SPHERE_SCENE_XML,
            json_file=json_file,
            export_dir=SCRIPT_DIR / f"franka_N{N_val}_generated",
            tolerances=(1e-4, 1e-4),
        )
        print(f"  [build] Done → {json_file}")
    return _load(json_file)

# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory generators
# ─────────────────────────────────────────────────────────────────────────────

def toppra_segment(q_start, q_end):
    return _toppra_segment(q_start, q_end, pin_model, pin_data)


def generate_quintic(q_start, q_end, h):
    Tf     = float(np.max(15 * np.abs(q_end - q_start) / (8 * DQ_MAX)))
    Tf     = max(Tf, 0.05)
    N_traj = max(int(Tf / h), 2)
    pos, vel, acc = sample_quintic_trajectory(q_start, q_end, Tf, N_traj, nq=NQ)
    return pos, vel, acc, N_traj, Tf

# ─────────────────────────────────────────────────────────────────────────────
#  Sim reset
# ─────────────────────────────────────────────────────────────────────────────

def reset():
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:7] = Q_HOME.copy()
    mj_data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(mj_model, mj_data)

# ─────────────────────────────────────────────────────────────────────────────
#  Single MPC episode
# ─────────────────────────────────────────────────────────────────────────────

def run_mpc_episode(solver, N_val, h, traj_type, q_target,
                    viewer=None, disturbance_tau=None, sensor_noise_std=0.0):
    reset()
    rng = np.random.default_rng(0)

    q_start = mj_data.qpos[:7].copy()

    if traj_type == "toppra":
        traj        = toppra_segment(q_start, q_target)
        Tf          = traj.duration
        N_traj      = max(int(Tf / h), 1)
        t_pts       = np.linspace(0, Tf, N_traj)
        thetamatd   = np.array([traj(t)    for t in t_pts])
        dthetamatd  = np.array([traj(t, 1) for t in t_pts])
        ddthetamatd = np.array([traj(t, 2) for t in t_pts])
    else:
        thetamatd, dthetamatd, ddthetamatd, N_traj, Tf = \
            generate_quintic(q_start, q_target, h)

    tau_ff = precompute_feedforward(pin_model, pin_data,
                                    thetamatd, dthetamatd, ddthetamatd, nv=NV)

    x_curr = np.concatenate([mj_data.qpos[:7].copy(), mj_data.qvel[:7].copy()])
    init_warm_start(solver, N_val, x_curr, nu=NU)

    steps_per_interval = max(int(h / mj_model.opt.timestep), 1)

    q_desired_hist, q_actual_hist = [], []
    tau_hist, time_hist, wall_hist = [], [], []

    for traj_idx in range(N_traj):
        set_trajectory_references(solver, N_val, traj_idx,
                                  thetamatd, dthetamatd, tau_ff)
        pin_initial_state(solver, x_curr)

        t0     = time.perf_counter()
        solver.solve()
        wall_hist.append(time.perf_counter() - t0)

        tau_raw = solver.get(0, "u").copy()
        if disturbance_tau is not None and traj_idx < len(disturbance_tau):
            tau_raw = tau_raw + disturbance_tau[traj_idx]
        tau_clipped = np.clip(tau_raw, -TAU_MAX, TAU_MAX)

        q_desired_hist.append(thetamatd[traj_idx].copy())
        q_actual_hist.append(mj_data.qpos[:7].copy())
        tau_hist.append(tau_clipped.copy())
        time_hist.append(traj_idx * h)

        mj_data.ctrl[:7] = tau_clipped
        for _ in range(steps_per_interval):
            mujoco.mj_step(mj_model, mj_data)
            if viewer is not None:
                viewer.sync()

        x_curr = np.concatenate([mj_data.qpos[:7].copy(), mj_data.qvel[:7].copy()])
        shift_warm_start(solver, N_val)

    return {
        "q_desired":  np.array(q_desired_hist),
        "q_actual":   np.array(q_actual_hist),
        "tau":        np.array(tau_hist),
        "time":       np.array(time_hist),
        "wall_times": np.array(wall_hist),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate one (Q, R) candidate on a fixed solver
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidate(solver, N_val, h, Q_mat, R_mat,
                       targets, disturbance, viewer):
    apply_cost_weights(solver, N_val, Q_mat, R_mat)
    rmse_n_list, rmse_d_list = [], []
    for q_target in targets:
        r = run_mpc_episode(solver, N_val, h, TRAJ_TYPE, q_target, viewer)
        rmse_n_list.append(gather_metrics(r, q_target)["rmse"])
        r = run_mpc_episode(solver, N_val, h, TRAJ_TYPE, q_target, viewer,
                            disturbance_tau=disturbance)
        rmse_d_list.append(gather_metrics(r, q_target)["rmse"])
    mean_n = float(np.mean(rmse_n_list))
    mean_d = float(np.mean(rmse_d_list))
    return mean_n, mean_d, W_TRACK * mean_n + W_DIST * mean_d

# ─────────────────────────────────────────────────────────────────────────────
#  Full benchmark with best (N, Q, R)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_benchmark(solver, N_val, h, Q_mat, R_mat,
                       targets, disturbance, viewer):
    apply_cost_weights(solver, N_val, Q_mat, R_mat)
    m_normal, m_dist, m_noise = [], [], []
    for q_target in targets:
        r = run_mpc_episode(solver, N_val, h, TRAJ_TYPE, q_target, viewer)
        m_normal.append(gather_metrics(r, q_target))
        r = run_mpc_episode(solver, N_val, h, TRAJ_TYPE, q_target, viewer,
                            disturbance_tau=disturbance)
        m_dist.append(gather_metrics(r, q_target))
        r = run_mpc_episode(solver, N_val, h, TRAJ_TYPE, q_target, viewer,
                            sensor_noise_std=SENSOR_NOISE_STD)
        m_noise.append(gather_metrics(r, q_target))
    return mean_metrics(m_normal), mean_metrics(m_dist), mean_metrics(m_noise)

# ─────────────────────────────────────────────────────────────────────────────
#  Printers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 76
SEP2 = "-" * 76


def print_sweep_summary(all_results):
    """all_results: list of dicts with keys N, q_scale, q_pos, r_val, score, ..."""
    sorted_r = sorted(all_results, key=lambda x: x["score"])
    print(f"\n{SEP}")
    print("  JOINT SWEEP RESULTS  (N × Q/R) — sorted best → worst")
    print(SEP)
    print(f"  {'Rank':>4}  {'N':>4}  {'q_scale':>8}  {'q_pos':>8}  {'r_val':>6}"
          f"  {'RMSE_normal':>12}  {'RMSE_dist':>10}  {'Score':>10}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}"
          f"  {'-'*12}  {'-'*10}  {'-'*10}")
    for rank, res in enumerate(sorted_r[:20], 1):      # top-20 rows
        marker = "  ◀ BEST" if rank == 1 else ""
        print(f"  {rank:>4}  {res['N']:>4}  {res['q_scale']:>8.1f}"
              f"  {res['q_pos']:>8.1f}  {res['r_val']:>6.2f}"
              f"  {res['rmse_normal']:>12.6f}  {res['rmse_dist']:>10.6f}"
              f"  {res['score']:>10.6f}{marker}")
    print(SEP2)


def print_full_metrics(m_normal, m_dist, m_noise, n_seeds, N_val, Q_mat, R_mat):
    print(f"\n{SEP}")
    print(f"  FINAL BENCHMARK  |  N={N_val}  |  Traj: {TRAJ_TYPE.upper()}"
          f"  |  {n_seeds} seeds")
    print(f"    Q diag (pos) : {Q_mat[0,0]:.1f}   (vel) {Q_mat[NQ,NQ]:.1f}")
    print(f"    R diag       : {R_mat[0,0]:.3f}")
    print(SEP)

    print(f"\n[1] Tracking RMSE — Normal")
    print(f"    RMSE : {m_normal['rmse']:.6f} rad")
    print(f"\n[2] Control Effort")
    rms = m_normal['tau_rms']
    print(f"    tau RMS [N·m] : " + "  ".join(f"J{i+1}={rms[i]:.2f}" for i in range(7)))
    print(f"    Total energy  : {m_normal['energy']:.2f} N²m²s")
    print(f"\n[3] Smoothness")
    print(f"    Mean jerk norm : {m_normal['jerk']:.4f}")
    print(f"\n[4] Solver Cost")
    print(f"    Mean step time : {m_normal['wall_mean_ms']:.4f} ms")
    print(f"    Max  step time : {m_normal['wall_max_ms']:.4f} ms")
    print(f"\n[5] Constraint Handling")
    print(f"    Joint pos. violation : {m_normal['q_viol']*100:.2f} %")
    print(f"    Torque violation     : {m_normal['tau_viol']*100:.2f} %")
    print(f"\n[6] Robustness — Disturbance  (std={DISTURBANCE_STD} N·m, {DISTURBANCE_STEPS} steps)")
    print(f"    RMSE under disturbance : {m_dist['rmse']:.6f} rad")
    print(f"\n[7] Robustness — Sensor Noise  (std={SENSOR_NOISE_STD} rad)")
    print(f"    RMSE under noise       : {m_noise['rmse']:.6f} rad")
    print(f"\n[8] Stability")
    print(f"    Settling time : {m_normal['settling_time']:.4f} s")
    print(f"    Overshoot     : {m_normal['overshoot']*100:.2f} %")
    print(f"\n{SEP2}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dist_rng    = np.random.default_rng(MASTER_SEED + 1)
    disturbance = dist_rng.normal(0, DISTURBANCE_STD, (DISTURBANCE_STEPS, 7))

    viewer = None
    if not HEADLESS:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Build Q/R candidate list once — same for every N
    candidates = [(float(qs), float(qp), float(r))
                  for qs in Q_SCALE_VALUES
                  for qp in Q_POS_VALUES
                  for r  in R_VALUES]
    n_cand = len(candidates)

    total_episodes = len(N_VALUES) * n_cand * N_EVAL_SEEDS * 2
    print(f"\n{SEP}")
    print(f"  JOINT SWEEP: N × Q/R")
    print(f"  N values     : {N_VALUES}")
    print(f"  Q/R candidates : {n_cand}")
    print(f"  Eval seeds   : {N_EVAL_SEEDS}")
    print(f"  Total episodes: {total_episodes}")
    print(SEP)

    all_results = []     # one entry per (N, q_scale, q_pos, r_val)
    best_global = None   # track globally best entry

    try:
        for N_val in N_VALUES:
            print(f"\n{'─'*76}")
            print(f"  N = {N_val}  —  loading / building solver ...")
            print(f"{'─'*76}")

            solver = get_solver(N_val)

            for cand_idx, (q_scale, q_pos, r_val) in enumerate(candidates):
                q_vel = Q_VEL_RATIO * q_pos
                Q_mat = q_scale * np.diag([q_pos] * NQ + [q_vel] * NV)
                R_mat = np.diag([r_val] * NU)

                print(f"  N={N_val} | cand {cand_idx+1:>3}/{n_cand}"
                      f"  q_scale={q_scale:.0f}  q_pos={q_pos:.0f}"
                      f"  r={r_val:.2f} ...", end="\r")

                rmse_n, rmse_d, score = evaluate_candidate(
                    solver, N_val, MPC_H, Q_mat, R_mat,
                    EVAL_TARGETS, disturbance, viewer)

                entry = {
                    "N":          N_val,
                    "q_scale":    q_scale,
                    "q_pos":      q_pos,
                    "r_val":      r_val,
                    "rmse_normal": rmse_n,
                    "rmse_dist":   rmse_d,
                    "score":       score,
                    "Q_mat":       Q_mat.copy(),
                    "R_mat":       R_mat.copy(),
                    "solver":      solver,     # keep reference for final benchmark
                }
                all_results.append(entry)

                if best_global is None or score < best_global["score"]:
                    best_global = entry

            print(f"  N={N_val} done — {n_cand} candidates evaluated.          ")

        # ── print combined table ──────────────────────────────────────────────
        print_sweep_summary(all_results)

        b = best_global
        print(f"\n  Globally best combination:")
        print(f"    N       = {b['N']}")
        print(f"    q_scale = {b['q_scale']:.1f}")
        print(f"    q_pos   = {b['q_pos']:.1f}")
        print(f"    r_val   = {b['r_val']:.3f}")
        print(f"    Score   = {b['score']:.6f}")
        print(f"\n  Q_mat = {b['q_scale']:.1f} * diag([{b['q_pos']:.1f}]*NQ"
              f" + [{b['q_pos']*Q_VEL_RATIO:.1f}]*NV)")
        print(f"  R_mat = diag([{b['r_val']:.3f}]*NU)")

        # ── full benchmark with globally best (N, Q, R) ───────────────────────
        print(f"\n{'─'*76}")
        print(f"  Running full {N_FINAL_SEEDS}-seed benchmark with best (N, Q, R) ...")
        print(f"{'─'*76}")

        final_dist = np.random.default_rng(MASTER_SEED + 1).normal(
            0, DISTURBANCE_STD, (DISTURBANCE_STEPS, 7))

        m_normal, m_dist, m_noise = run_full_benchmark(
            b["solver"], b["N"], MPC_H,
            b["Q_mat"], b["R_mat"],
            FINAL_TARGETS, final_dist, viewer)

        print_full_metrics(m_normal, m_dist, m_noise,
                           N_FINAL_SEEDS, b["N"], b["Q_mat"], b["R_mat"])

    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()