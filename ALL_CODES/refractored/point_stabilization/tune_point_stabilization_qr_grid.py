"""
tune_point_stabilization_qr_grid.py
===================================
Merged build + run + Q/R tuning for the Franka Panda point-stabilization MPC.

Flow
----
1.  Build 9 ACADOS solvers for every (N, h) pair:
        N  ∈ {5, 10, 20}
        h  ∈ {0.01, 0.05, 0.10}   [seconds]
    Each solver is saved to  franka_N<N>_h<h>.json

2.  For each (N, h) solver, sweep a grid of (Q, R) cost matrices
    WITHOUT rebuilding — just call solver.cost_set() at runtime.

3.  Each candidate is scored on N_EVAL_SEEDS random targets by:
        score = w_err * mean_final_error + w_time * mean_time_to_reach

4.  Best (Q, R) per (N, h) is reported in a final summary table.

Refactored from: MPC_point_stablization/tune_mpc.py.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.franka_common import (PANDA_URDF, CUBE_SCENE_XML, Q_HOME,
                                  NQ, NV, NX, NU, TAU_MAX,
                                  sample_targets, script_dir)
from shared.acados_mpc import (build_franka_ocp_solver, load_solver,
                               apply_cost_weights, init_warm_start,
                               shift_warm_start, pin_initial_state,
                               set_point_reference)

SCRIPT_DIR = script_dir(__file__)

# ─────────────────────────────────────────────────────────────────────────────
#  SOLVER GRID  (N, h) pairs — 9 total
# ─────────────────────────────────────────────────────────────────────────────

N_VALUES = [5, 10, 20]
H_VALUES = [0.01, 0.05, 0.10]

# ─────────────────────────────────────────────────────────────────────────────
#  Q / R SEARCH GRID
# ─────────────────────────────────────────────────────────────────────────────

Q_SCALE_VALUES = np.linspace(50,   200, 4)   # overall multiplier
Q_POS_VALUES   = np.linspace(100,  300, 4)   # position diagonal
Q_VEL_RATIO    = 0.10                        # vel = Q_VEL_RATIO * q_pos
R_VALUES       = np.linspace(0.5,  5.0, 4)   # R diagonal

# ─────────────────────────────────────────────────────────────────────────────
#  SCORING WEIGHTS
#  score = W_ERR * mean_final_error + W_TIME * mean_time_to_reach
# ─────────────────────────────────────────────────────────────────────────────

W_ERR  = 0.6
W_TIME = 0.4

# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

N_EVAL_SEEDS  = 5
MASTER_SEED   = 42
CONV_THRESH   = 0.04    # ||q - q_target|| < this → converged
MAX_ITER      = 500     # safety cap per episode

Q_START = Q_HOME.copy()

# ─────────────────────────────────────────────────────────────────────────────
#  MUJOCO MODEL  (single instance, reset between episodes)
# ─────────────────────────────────────────────────────────────────────────────

mj_model = mujoco.MjModel.from_xml_path(CUBE_SCENE_XML)
mj_data  = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.005

# ─────────────────────────────────────────────────────────────────────────────
#  TARGET CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TARGETS = sample_targets(N_EVAL_SEEDS, MASTER_SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — BUILD ONE SOLVER  for a given (N, h)
# ─────────────────────────────────────────────────────────────────────────────

def solver_json_name(N, h):
    h_tag = f"{h:.2f}".replace(".", "p")
    return str(SCRIPT_DIR / f"franka_N{N}_h{h_tag}.json")


def build_solver(N, h):
    """Generate and compile one ACADOS OCP solver; only N and h are
    parametric.  Cost is an identity placeholder overwritten at runtime."""
    json_file = solver_json_name(N, h)
    if os.path.exists(json_file):
        print(f"    [build] {json_file} already exists — skipping rebuild.")
        return json_file

    print(f"    [build] Compiling  N={N}  h={h} → {json_file} ...")
    build_franka_ocp_solver(
        model_name=f'franka_N{N}_h{int(h*100):03d}',
        h=h, N=N,
        Q=np.eye(NX), R=np.eye(NU),   # placeholder — overwritten at runtime
        urdf_path=PANDA_URDF,
        mjcf_path=CUBE_SCENE_XML,
        json_file=json_file,
        export_dir=SCRIPT_DIR / f'franka_N{N}_h{int(h*100):03d}_generated',
    )
    print(f"    [build] Done → {json_file}")
    return json_file

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — RESET MUJOCO
# ─────────────────────────────────────────────────────────────────────────────

def reset_sim():
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:NQ] = Q_START.copy()
    mj_data.qvel[:NV] = np.zeros(NV)
    mujoco.mj_forward(mj_model, mj_data)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — RUN ONE MPC EPISODE
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(solver, N, h, q_target):
    """
    Drive the arm from Q_START toward q_target using the loaded solver.
    Returns (final_error, time_to_reach).
    time_to_reach = elapsed sim time when ||q - q_target|| < CONV_THRESH,
                    or MAX_ITER * h if it never converged.
    """
    reset_sim()

    steps_per_interval = max(int(h / mj_model.opt.timestep), 1)

    # full state target
    xs = np.concatenate([q_target, np.zeros(NV)])
    set_point_reference(solver, N, xs, nu=NU)

    # warm-start
    x0 = np.concatenate([Q_START, np.zeros(NV)])
    init_warm_start(solver, N, x0, nu=NU)

    x_curr    = x0.copy()
    t_elapsed = 0.0
    t_reached = None

    for it in range(MAX_ITER):
        pin_initial_state(solver, x_curr)

        status = solver.solve()

        # apply first control, simulate one MPC interval
        tau = solver.get(0, "u")
        mj_data.ctrl[:NQ] = np.clip(tau, -TAU_MAX, TAU_MAX)
        for _ in range(steps_per_interval):
            mujoco.mj_step(mj_model, mj_data)
        t_elapsed += h

        x_curr = np.concatenate([mj_data.qpos[:NQ].copy(),
                                 mj_data.qvel[:NV].copy()])

        shift_warm_start(solver, N)

        err = np.linalg.norm(x_curr[:NQ] - q_target)
        if err < CONV_THRESH and t_reached is None:
            t_reached = t_elapsed

        # stop early once converged
        if t_reached is not None:
            break

    final_error   = float(np.linalg.norm(x_curr[:NQ] - q_target))
    time_to_reach = t_reached if t_reached is not None else float(MAX_ITER * h)
    return final_error, time_to_reach

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — EVALUATE ONE (Q, R) CANDIDATE  over all eval targets
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidate(solver, N, h, Q_mat, R_mat):
    apply_cost_weights(solver, N, Q_mat, R_mat)

    errors = []
    times  = []
    for q_target in EVAL_TARGETS:
        err, t = run_episode(solver, N, h, q_target)
        errors.append(err)
        times.append(t)

    mean_err  = float(np.mean(errors))
    mean_time = float(np.mean(times))
    score     = W_ERR * mean_err + W_TIME * mean_time
    return mean_err, mean_time, score

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — GRID SEARCH over (Q, R) for one solver
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_QR(solver, N, h):
    """Return the best result dict and the full ranked list."""
    candidates = [
        (float(qs), float(qp), float(rv))
        for qs in Q_SCALE_VALUES
        for qp in Q_POS_VALUES
        for rv in R_VALUES
    ]
    n_cand  = len(candidates)
    results = []

    for idx, (q_scale, q_pos, r_val) in enumerate(candidates):
        q_vel = Q_VEL_RATIO * q_pos
        Q_mat = q_scale * np.diag([q_pos] * NQ + [q_vel] * NV)
        R_mat = np.diag([r_val] * NU)

        print(f"    cand {idx+1:>3}/{n_cand}"
              f"  qs={q_scale:.0f} qp={q_pos:.0f} r={r_val:.2f} ...",
              end="\r")

        mean_err, mean_time, score = evaluate_candidate(
            solver, N, h, Q_mat, R_mat)

        results.append({
            "q_scale":   q_scale,
            "q_pos":     q_pos,
            "r_val":     r_val,
            "mean_err":  mean_err,
            "mean_time": mean_time,
            "score":     score,
            "Q_mat":     Q_mat.copy(),
            "R_mat":     R_mat.copy(),
        })

    print(f"    {n_cand} candidates done.                          ")
    results_sorted = sorted(results, key=lambda x: x["score"])
    return results_sorted[0], results_sorted   # best, all

# ─────────────────────────────────────────────────────────────────────────────
#  PRINTERS
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 76
SEP2 = "-" * 76

def print_grid_table(N, h, ranked):
    print(f"\n{SEP}")
    print(f"  Grid results  N={N}  h={h}")
    print(SEP)
    print(f"  {'Rank':>4}  {'q_scale':>8}  {'q_pos':>7}  {'r_val':>6}"
          f"  {'FinalErr':>10}  {'Time[s]':>9}  {'Score':>10}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*6}"
          f"  {'-'*10}  {'-'*9}  {'-'*10}")
    for rank, r in enumerate(ranked, 1):
        marker = "  ◀ BEST" if rank == 1 else ""
        print(f"  {rank:>4}  {r['q_scale']:>8.1f}  {r['q_pos']:>7.1f}"
              f"  {r['r_val']:>6.2f}  {r['mean_err']:>10.6f}"
              f"  {r['mean_time']:>9.3f}  {r['score']:>10.6f}{marker}")
    print(SEP2)


def print_final_summary(summary):
    print(f"\n\n{'#'*76}")
    print(f"  FINAL SUMMARY — best (Q, R) per (N, h) pair")
    print(f"{'#'*76}")
    print(f"  {'N':>4}  {'h':>6}  {'q_scale':>8}  {'q_pos':>7}  {'r_val':>6}"
          f"  {'FinalErr':>10}  {'Time[s]':>9}  {'Score':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*6}"
          f"  {'-'*10}  {'-'*9}  {'-'*10}")
    for row in summary:
        b = row["best"]
        print(f"  {row['N']:>4}  {row['h']:>6.2f}  {b['q_scale']:>8.1f}"
              f"  {b['q_pos']:>7.1f}  {b['r_val']:>6.2f}"
              f"  {b['mean_err']:>10.6f}  {b['mean_time']:>9.3f}"
              f"  {b['score']:>10.6f}")
    print(f"{'#'*76}")

    # pick overall best across all (N, h) pairs
    overall_best_row = min(summary, key=lambda r: r["best"]["score"])
    ob = overall_best_row["best"]
    print(f"\n  ★  OVERALL BEST  →  N={overall_best_row['N']}"
          f"  h={overall_best_row['h']}")
    print(f"       Q_mat = {ob['q_scale']:.1f} * diag("
          f"[{ob['q_pos']:.1f}]*NQ + [{ob['q_pos']*Q_VEL_RATIO:.1f}]*NV)")
    print(f"       R_mat = diag([{ob['r_val']:.3f}]*NU)")
    print(f"       mean final error : {ob['mean_err']:.6f}  rad")
    print(f"       mean time        : {ob['mean_time']:.3f}  s")
    print(f"       score            : {ob['score']:.6f}")
    print(f"{'#'*76}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    solver_pairs = [(N, h) for N in N_VALUES for h in H_VALUES]
    n_pairs      = len(solver_pairs)

    print(f"\n{'='*76}")
    print(f"  PHASE 1 — Building {n_pairs} ACADOS solvers")
    print(f"{'='*76}")
    for N, h in solver_pairs:
        build_solver(N, h)

    print(f"\n{'='*76}")
    print(f"  PHASE 2 — Q/R grid search  ({len(Q_SCALE_VALUES)}×"
          f"{len(Q_POS_VALUES)}×{len(R_VALUES)} = "
          f"{len(Q_SCALE_VALUES)*len(Q_POS_VALUES)*len(R_VALUES)} candidates)"
          f"  ×  {N_EVAL_SEEDS} seeds  ×  {n_pairs} (N,h) pairs")
    print(f"{'='*76}")

    summary = []

    for N, h in solver_pairs:
        json_file = solver_json_name(N, h)
        print(f"\n  ── Loading solver  N={N}  h={h}  ({json_file})")
        solver = load_solver(json_file)

        print(f"  ── Grid search ...")
        best, ranked = grid_search_QR(solver, N, h)

        print_grid_table(N, h, ranked)

        summary.append({"N": N, "h": h, "best": best})

    print_final_summary(summary)


if __name__ == "__main__":
    main()
