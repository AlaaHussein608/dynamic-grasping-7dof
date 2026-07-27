"""
benchmark_mpc_tracking.py
=========================
Benchmarks the Model Predictive Controller (MPC / ACADOS) on the Franka
Panda using both Quintic and TOPPRA trajectory planners.

50 target configurations are sampled uniformly within the joint limits; every
metric is computed per-seed and the mean over all 50 seeds is printed.

Prerequisite: run build_trajectory_tracking_solver.py once
(→ franka_point_stab.json).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import pinocchio as pin

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_HOME,
                                  DQ_MAX, TAU_MAX, NQ, NV, NX, NU,
                                  sample_targets, script_dir)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       toppra_segment as _toppra_segment,
                                       precompute_feedforward)
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state, set_trajectory_references)
from benchmark_metrics import gather_metrics, mean_metrics, print_results

SCRIPT_DIR = script_dir(__file__)

# ── Configuration ────────────────────────────────────────────────────────────
N_SEEDS     = 50
MASTER_SEED = 42

MPC_H = 0.005
MPC_N = 8

DISTURBANCE_STD   = 2.0    # [N·m]
DISTURBANCE_STEPS = 80
SENSOR_NOISE_STD  = 0.002  # [rad] / [rad/s]

HEADLESS = True

# ── Model setup ──────────────────────────────────────────────────────────────
pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
data.qpos[:7] = np.zeros(7)
mujoco.mj_forward(model, data)
model.opt.gravity[:] = [0, 0, -9.81]

# ── ACADOS solver ────────────────────────────────────────────────────────────
print("Loading pre-compiled ACADOS solver ...")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab.json')
print("Solver loaded.")

Q_mat = 200.0 * np.diag([300.0]*NQ + [30.0]*NV)
R_mat = np.diag([1.000]*NU)
apply_cost_weights(solver, MPC_N, Q_mat, R_mat)

TARGETS = sample_targets(N_SEEDS, MASTER_SEED)


def toppra_segment(q_start, q_end):
    return _toppra_segment(q_start, q_end, pin_model, pin_data)


def generate_quintic(q_start, q_end):
    Tf     = float(np.max(15 * np.abs(q_end - q_start) / (8 * DQ_MAX)))
    Tf     = max(Tf, 0.05)
    N_traj = max(int(Tf / MPC_H), 2)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        q_start, q_end, Tf, N_traj, nq=NQ)
    return thetamatd, dthetamatd, ddthetamatd, N_traj, Tf


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = Q_HOME.copy()
    data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(model, data)


def run_mpc(traj_type, q_target, viewer,
            disturbance_tau=None, sensor_noise_std=0.0):
    """Run one MPC episode to q_target and return raw history arrays."""
    reset()
    rng = np.random.default_rng(0)

    q_start = data.qpos[:7].copy()

    if traj_type == "toppra":
        toppra_traj = toppra_segment(q_start, q_target)
        Tf     = toppra_traj.duration
        N_traj = max(int(Tf / MPC_H), 1)
        t_pts  = np.linspace(0, Tf, N_traj)
        thetamatd   = np.array([toppra_traj(t)    for t in t_pts])
        dthetamatd  = np.array([toppra_traj(t, 1) for t in t_pts])
        ddthetamatd = np.array([toppra_traj(t, 2) for t in t_pts])
    else:
        thetamatd, dthetamatd, ddthetamatd, N_traj, Tf = \
            generate_quintic(q_start, q_target)

    tau_ff = precompute_feedforward(pin_model, pin_data,
                                    thetamatd, dthetamatd, ddthetamatd, nv=NV)

    x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
    init_warm_start(solver, MPC_N, x_curr, nu=NU)

    steps_per_interval = max(int(MPC_H / model.opt.timestep), 1)

    q_desired_hist, q_actual_hist, tau_hist, time_hist, wall_hist = [], [], [], [], []

    for traj_idx in range(N_traj):
        if sensor_noise_std > 0.0:
            x_noisy = x_curr + rng.normal(0, sensor_noise_std, NX)
        # x0 constraint always uses true state
        set_trajectory_references(solver, MPC_N, traj_idx,
                                  thetamatd, dthetamatd, tau_ff)
        pin_initial_state(solver, x_curr)

        t0     = time.perf_counter()
        status = solver.solve()
        wall_hist.append(time.perf_counter() - t0)

        if status not in [0, 2]:
            print(f"  [warn] solver status {status} at traj_idx {traj_idx}")

        tau_raw = solver.get(0, "u").copy()

        if disturbance_tau is not None and traj_idx < len(disturbance_tau):
            tau_raw = tau_raw + disturbance_tau[traj_idx]

        tau_clipped = np.clip(tau_raw, -TAU_MAX, TAU_MAX)

        q_desired_hist.append(thetamatd[traj_idx].copy())
        q_actual_hist.append(data.qpos[:7].copy())
        tau_hist.append(tau_clipped.copy())
        time_hist.append(traj_idx * MPC_H)

        data.ctrl[:7] = tau_clipped
        for _ in range(steps_per_interval):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

        x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
        shift_warm_start(solver, MPC_N)

    return {
        "q_desired":  np.array(q_desired_hist),
        "q_actual":   np.array(q_actual_hist),
        "tau":        np.array(tau_hist),
        "time":       np.array(time_hist),
        "wall_times": np.array(wall_hist),
    }


def main():
    dist_rng    = np.random.default_rng(MASTER_SEED + 1)
    disturbance = dist_rng.normal(0, DISTURBANCE_STD, (DISTURBANCE_STEPS, 7))

    viewer = None
    if not HEADLESS:
        viewer = mujoco.viewer.launch_passive(model, data)

    for traj_type in ("toppra", "quintic"):
        metrics_normal, metrics_dist, metrics_noise = [], [], []

        for seed_idx, q_target in enumerate(TARGETS):
            print(f"  [{traj_type.upper()}] seed {seed_idx+1:02d}/{N_SEEDS} ...", end="\r")

            r = run_mpc(traj_type, q_target, viewer)
            metrics_normal.append(gather_metrics(r, q_target))

            r = run_mpc(traj_type, q_target, viewer, disturbance_tau=disturbance)
            metrics_dist.append(gather_metrics(r, q_target))

            r = run_mpc(traj_type, q_target, viewer, sensor_noise_std=SENSOR_NOISE_STD)
            metrics_noise.append(gather_metrics(r, q_target))

        print(f"  [{traj_type.upper()}] all {N_SEEDS} seeds done.              ")
        print_results("MPC", traj_type, N_SEEDS,
                      mean_metrics(metrics_normal),
                      mean_metrics(metrics_dist),
                      mean_metrics(metrics_noise),
                      DISTURBANCE_STD, DISTURBANCE_STEPS, SENSOR_NOISE_STD,
                      cost_label="ACADOS solver only")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
