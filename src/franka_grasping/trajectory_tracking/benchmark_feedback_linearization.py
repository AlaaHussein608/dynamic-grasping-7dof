"""
benchmark_feedback_linearization.py
===================================
Benchmarks the Feedback Linearization (Computed Torque) controller on the
Franka Panda using both Quintic and TOPPRA trajectory planners.

50 target configurations are sampled uniformly within the joint limits; every
metric is computed per-seed and the mean over all 50 seeds is printed.

Metrics: RMSE (normal / disturbance / sensor-noise), control effort,
smoothness, per-step wall time, constraint violation rates, settling time,
overshoot.
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
                                  DQ_MAX, TAU_MAX, sample_targets)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       toppra_segment as _toppra_segment,
                                       feedback_linearization_control)
from benchmark_metrics import gather_metrics, mean_metrics, print_results

# ── Configuration ────────────────────────────────────────────────────────────
N_SEEDS     = 50
MASTER_SEED = 42

KP = 400.0
KD = 40.0

DISTURBANCE_STD   = 2.0    # [N·m]
DISTURBANCE_STEPS = 80
SENSOR_NOISE_STD  = 0.002  # [rad] / [rad/s]

HEADLESS = True

# ── Model setup ──────────────────────────────────────────────────────────────
pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]
armature = model.dof_armature[:7]

TARGETS = sample_targets(N_SEEDS, MASTER_SEED)


def toppra_segment(q_start, q_end):
    return _toppra_segment(q_start, q_end, pin_model, pin_data)


def generate_quintic(theta_s, theta_end):
    dt = model.opt.timestep
    T  = float(np.max(15 * np.abs(theta_end - theta_s) / (8 * DQ_MAX)))
    T  = max(T, 0.05)
    N  = max(int(T / dt), 2)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_s, theta_end, T, N)
    return thetamatd, dthetamatd, ddthetamatd, N, T


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = Q_HOME.copy()
    data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(model, data)


def run_fl(traj_type, q_target, viewer,
           disturbance_tau=None, sensor_noise_std=0.0):
    """Run one FL episode to q_target and return raw history arrays."""
    reset()
    rng = np.random.default_rng(0)

    theta_start = data.qpos[:7].copy()

    if traj_type == "toppra":
        toppra_traj = toppra_segment(theta_start, q_target)
        Tf = toppra_traj.duration
        def ref(t):
            return toppra_traj(t), toppra_traj(t, 1), toppra_traj(t, 2)
    else:
        thetamatd, dthetamatd, ddthetamatd, N_traj, Tf = \
            generate_quintic(theta_start, q_target)
        def ref(t):
            i = min(int((t / Tf) * (N_traj - 1)), N_traj - 1)
            return thetamatd[i], dthetamatd[i], ddthetamatd[i]

    q_desired_hist, q_actual_hist, tau_hist, time_hist, wall_hist = [], [], [], [], []
    t_start  = data.time
    step_idx = 0

    while True:
        elapsed = data.time - t_start
        t = min(elapsed, Tf)
        if elapsed >= Tf:
            break

        q  = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        if sensor_noise_std > 0.0:
            q  = q  + rng.normal(0, sensor_noise_std, 7)
            qd = qd + rng.normal(0, sensor_noise_std, 7)

        q_d, qd_d, qdd_d = ref(t)

        t0 = time.perf_counter()
        tau = feedback_linearization_control(q, qd, q_d, qd_d, qdd_d,
                                             pin_model, pin_data, armature, KP, KD)
        wall_hist.append(time.perf_counter() - t0)

        if disturbance_tau is not None and step_idx < len(disturbance_tau):
            tau = np.clip(tau + disturbance_tau[step_idx], -TAU_MAX, TAU_MAX)

        q_desired_hist.append(q_d.copy())
        q_actual_hist.append(data.qpos[:7].copy())
        tau_hist.append(tau.copy())
        time_hist.append(t)

        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        step_idx += 1

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

            r = run_fl(traj_type, q_target, viewer)
            metrics_normal.append(gather_metrics(r, q_target))

            r = run_fl(traj_type, q_target, viewer, disturbance_tau=disturbance)
            metrics_dist.append(gather_metrics(r, q_target))

            r = run_fl(traj_type, q_target, viewer, sensor_noise_std=SENSOR_NOISE_STD)
            metrics_noise.append(gather_metrics(r, q_target))

        print(f"  [{traj_type.upper()}] all {N_SEEDS} seeds done.              ")
        print_results("FL", traj_type, N_SEEDS,
                      mean_metrics(metrics_normal),
                      mean_metrics(metrics_dist),
                      mean_metrics(metrics_noise),
                      DISTURBANCE_STD, DISTURBANCE_STEPS, SENSOR_NOISE_STD)

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
