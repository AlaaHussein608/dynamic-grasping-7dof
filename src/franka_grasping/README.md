# Refactored Franka Panda control codebase

Refactor of the scripts in `ALL_CODES/` — no original file was modified.
Duplication is consolidated into shared modules, function signatures are
consistent, files are named by purpose, algorithms are separated from
run/analysis code, and all asset paths resolve relative to the repo (not the
CWD). Plotting / CSV output was removed; the scripts just run.

## Layout

```
refactored/
├── shared/                        ← shared modules (imported by every script)
│   ├── franka_common.py           constants, asset paths, model loading,
│   │                              IK (frantik + Pinocchio DLS), FK,
│   │                              arm_mass_bias (crba + rnea), grasp geometry,
│   │                              workspace / reachable-window checks
│   ├── trajectory_control.py      quintic & TOPP-RA trajectory generation,
│   │                              RNEA feedforward, SMC / FL / PD control laws
│   ├── gripper.py                 shared open_gripper / close_gripper
│   └── acados_mpc.py              parametric ACADOS OCP builder (incl. the
│                                  contraction-constrained variant) + runtime
│                                  helpers (warm-start, shift, refs, weights)
│
├── point_stabilization/           regulate to a fixed joint target
│   ├── build_point_stabilization_solver.py
│   ├── run_point_stabilization_mpc.py     ← MPC
│   ├── run_point_stabilization_fl.py      ← feedback linearization
│   ├── run_point_stabilization_smc.py     ← sliding mode
│   └── tune_point_stabilization_qr_grid.py
│
├── trajectory_tracking/           track a quintic / TOPP-RA reference
│   ├── build_trajectory_tracking_solver.py
│   ├── mpc_pick_and_place_sphere.py       ← MPC pick-and-place
│   ├── track_fl_quintic.py / track_fl_toppra.py     ← FL tracking demos
│   ├── track_smc_quintic.py / track_smc_toppra.py   ← SMC tracking demos
│   ├── benchmark_mpc_tracking.py
│   ├── benchmark_feedback_linearization.py
│   ├── benchmark_sliding_mode.py
│   ├── benchmark_metrics.py               (folder-local shared metrics)
│   └── tune_mpc_trajectory_tracking.py    (merged N × Q/R sweep)
│
└── dynamic_grasping/              intercept and grasp a rolling ball
    ├── dg_common.py               ← grasping helpers over shared: ee_pos,
    │                                toppra_segment, solve_grasp_ik, and the
    │                                shared forward_scan_intercept search
    ├── build_solver.py            (grasping NMPC; --planning for the benchmark)
    ├── build_contractive_solver.py
    ├── intercept_planner_toppra.py / _quintic.py   ← thin wrappers over dg_common
    ├── intercept_planner_croft.py                  ← CROFT rendezvous search
    ├── measure_tp.py              (calibrate CROFT's per-evaluation cost)
    ├── run_grasping_nmpc.py [toppra|croft]         ← NMPC + planner (one script)
    ├── run_grasping_pid_quintic.py                 ← PID + quintic planner
    ├── run_contractive_grasping.py / _v2.py        ← contractive MPC
    ├── contraction_gains.py       (folder-local: alpha selection)
    └── benchmark_grasping_methods.py               (TOPPRA vs CROFT vs contractive)
```

The two intercept planners (`_toppra`, `_quintic`) now share a single
`forward_scan_intercept` in `dg_common.py`, parameterised by a segment-timing
estimator; `_croft` reuses the same `ee_pos` / `toppra_segment` /
`solve_grasp_ik` helpers. The previously duplicated `run_grasping_nmpc_toppra.py`
and `run_grasping_nmpc_croft.py` are merged into `run_grasping_nmpc.py`, which
selects the planner from the command line.

## Shared-module design

* **One dynamics helper.** `arm_mass_bias(pin_model, pin_data, q, v, armature)`
  returns `(M, h)` with `M = crba(q) + diag(armature)` and
  `h = rnea(q, v, 0)` (= `C(q,v)·v + g(q)`). The MuJoCo armature is *always*
  added to `M`; C and g are never formed separately (crba + one rnea is
  faster). Every FL / PD / hold torque uses this. The sliding-mode law is the
  one exception — it needs the Coriolis *matrix* explicitly, so it keeps
  `computeAllTerms`.
* **Two IK solvers**, both seeding from `q0` then `Uniform[Q_MIN, Q_MAX]`:
  `IK_frantik` (closed-chain, used by the planners) and `IK_pinocchio`
  (damped least squares, used by the reactive controllers). No angle wrapping
  is applied — several Panda joint limits exceed ±π.
* **Consistent MPC interface.** `build_franka_ocp_solver(...)` builds every
  solver variant (point-stab, tracking, planning, contractive via
  `contraction=True`); the runtime helpers all take `(solver, N, ...)`.

## File mapping (old → new)

| Original | Refactored |
|---|---|
| `franka_param.py` | `shared/franka_common.py` (class `panda` + constants) |
| `MPC_point_stablization/ACADOS_build_solver.py` | `point_stabilization/build_point_stabilization_solver.py` |
| `MPC_point_stablization/ACADOS_run_mpc.py` | `point_stabilization/run_point_stabilization_mpc.py` |
| `MPC_point_stablization/tune_mpc.py.py` | `point_stabilization/tune_point_stabilization_qr_grid.py` |
| `MPC_point_stablization/plots_and results/test_MPC.py` | *(dropped — was plot/CSV only)* |
| *(new)* | `point_stabilization/run_point_stabilization_fl.py` |
| *(new)* | `point_stabilization/run_point_stabilization_smc.py` |
| `MPC trajectory/ACADOS_build_solver.py` | `trajectory_tracking/build_trajectory_tracking_solver.py` |
| `MPC trajectory/ACADOS_mpc_trajectory_tracking.py` | `trajectory_tracking/mpc_pick_and_place_sphere.py` |
| `MPC trajectory/tune_MPC_sweeps_N.py` + `tune_mpc_sweeps_QR_h0.0005N8.py` | `trajectory_tracking/tune_mpc_trajectory_tracking.py` (merged) |
| `MPC trajectory/codes_plots_toppra+quintic/mpc_traj_quintic2.py` | *(plotting dropped; tracking demo → `track_*`)* |
| `MPC trajectory/benchmark_mpc.py` + `mpc_fl_smc_trajectory_comparizon/benchmark_mpc.py` | `trajectory_tracking/benchmark_mpc_tracking.py` (identical copies merged) |
| `mpc_fl_smc_trajectory_comparizon/benchmark_fl.py` | `trajectory_tracking/benchmark_feedback_linearization.py` |
| `mpc_fl_smc_trajectory_comparizon/benchmark_smc.py` | `trajectory_tracking/benchmark_sliding_mode.py` |
| `smc+fl (live and plots)/SMC_traj_quintic.py` / `_toppra.py` | `trajectory_tracking/track_smc_quintic.py` / `_toppra.py` |
| `smc+fl (live and plots)/feedbacklinearization_traj_quintic.py` / `_toppra.py` | `trajectory_tracking/track_fl_quintic.py` / `_toppra.py` |
| `dynamic grasping/intercept planning/ACADOS_build_solver.py` | `dynamic_grasping/build_solver.py` |
| `dynamic grasping/intercept planning/TOPPRA_Nmpc_intercept_point.py` | `dynamic_grasping/intercept_planner_toppra.py` |
| `dynamic grasping/intercept planning/quintic/intercept_point_code2.py` | `dynamic_grasping/intercept_planner_quintic.py` |
| `dynamic grasping/intercept planning/CROFT.py` | `dynamic_grasping/intercept_planner_croft.py` |
| `dynamic grasping/intercept planning/measure_tp.py` | `dynamic_grasping/measure_tp.py` |
| `dynamic grasping/intercept planning/TOPPRA_Nmpc_moving_ball.py` | `dynamic_grasping/run_grasping_nmpc.py` (`toppra`/`croft` via CLI) |
| `dynamic grasping/intercept planning/TOPPRA_moving_ball.py` | *(folded into the two NMPC runners)* |
| `dynamic grasping/intercept planning/quintic/moving_ball.py` | `dynamic_grasping/run_grasping_pid_quintic.py` |
| `dynamic grasping/benchmark.py` | `dynamic_grasping/benchmark_grasping_methods.py` |
| `dynamic grasping/Contractive MPC/contractive_build_solver.py` | `dynamic_grasping/build_contractive_solver.py` |
| `dynamic grasping/Contractive MPC/contractive_dynamic_graspig.py` | `dynamic_grasping/run_contractive_grasping.py` (+ `contraction_gains.py`) |
| `dynamic grasping/Contractive MPC/version2_contractive_dynamic_graspig.py` | `dynamic_grasping/run_contractive_grasping_v2.py` (+ `contraction_gains.py`) |

## Asset paths

All `.xml` / `.urdf` paths resolve through `shared/franka_common.py`
(`PANDA_URDF`, `SPHERE_SCENE_XML`, `CUBE_SCENE_XML`) to
`ALL_CODES/xml and meshes/`, regardless of the working directory. MuJoCo
resolves `meshdir="assets"` and the URDF's `meshes/…` references relative to
those files, so meshes load unchanged. Pinocchio always builds from
`mjx_panda.urdf` — the MJCF→URDF conversion step is gone. Solver JSONs and
their generated C code are written next to each build script.

## Running

Each script is plain `python <script>.py` from anywhere (it inserts
`franka_grasping/` on `sys.path`). Build the matching ACADOS solver first:

* `point_stabilization/` → `build_point_stabilization_solver.py`
* `trajectory_tracking/` → `build_trajectory_tracking_solver.py`
* `dynamic_grasping/` NMPC runner → `build_solver.py`, then
  `run_grasping_nmpc.py [toppra|croft]`;
  benchmark → `build_solver.py --planning`;
  contractive runners → `build_contractive_solver.py`

External modules `frantik`, `toppra`, `acados_template`, `mujoco`,
`pinocchio`, `modern_robotics` must be installed. `mjx_single_cube.xml` (used
by the point-stabilization scripts) must be present in `xml and meshes/`.

## Verification

All modules compile; all local imports resolve. Verified numerically:
`arm_mass_bias`'s `h` equals `pin.nonLinearEffects` to 1e-18; both IK solvers
converge; TOPP-RA / quintic planners find intercepts; the contractive solver
solves cleanly for the alphas its helpers actually produce. Ran headless
end-to-end: MPC / FL / SMC point-stabilization and trajectory tracking
(sub-mrad RMSE), all three benchmarks, both tuning sweeps, `measure_tp`, and a
full grasping-benchmark catch. Every interactive run script was validated
through its complete setup path (model load → solver load → IK → cost setup)
up to the viewer launch.
