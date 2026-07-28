"""
mpc_pick_and_place_sphere.py
============================
MPC-based pick-and-place of a sphere: IK waypoints (modern-robotics
IKinSpace), quintic references tracked by the ACADOS NMPC, force-based
gripper closing (shared gripper primitives).

Requires the pre-compiled trajectory-tracking solver
(run build_trajectory_tracking_solver.py first).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import pinocchio as pin
from modern_robotics import IKinSpace

from shared.franka_common import (PANDA_URDF, SPHERE_SCENE_XML, Q_HOME,
                                  panda, compute_F_thresh, script_dir)
from shared.trajectory_control import (sample_quintic_trajectory,
                                       precompute_feedforward)
from shared.gripper import open_gripper, close_gripper
from shared.acados_mpc import (load_solver, apply_cost_weights,
                               init_warm_start, shift_warm_start,
                               pin_initial_state, set_trajectory_references)

SCRIPT_DIR = script_dir(__file__)

# ─── MPC parameters (must match the build script) ────────────────────────────
h  = 0.005
N  = 8
nq = 7
nv = 7
nu = nv

# ─── Helpers ─────────────────────────────────────────────────────────────────

def wrap_pi(q):
    """Wrap joint angles to [-pi, pi]."""
    return (q + np.pi) % (2 * np.pi) - np.pi

# ─── Model setup ─────────────────────────────────────────────────────────────
pin_model = pin.buildModelFromUrdf(PANDA_URDF)
pin_data  = pin_model.createData()

model = mujoco.MjModel.from_xml_path(SPHERE_SCENE_XML)
data  = mujoco.MjData(model)
data.qpos[:7] = Q_HOME.copy()
mujoco.mj_forward(model, data)
model.opt.gravity[:] = [0, 0, -9.81]

print(f"MuJoCo timestep: {model.opt.timestep}  |  Steps per MPC interval: {int(h / model.opt.timestep)}")

# ─── Object and gripper properties ───────────────────────────────────────────
sphere_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
sphere_pos  = data.xpos[sphere_id].copy()
sphere_rot  = data.xmat[sphere_id].reshape(3, 3).copy()
sphere_mass = model.body_mass[sphere_id]

sphere_geom_id = next(i for i in range(model.ngeom)
                      if model.geom_bodyid[i] == sphere_id)

left_pad_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
finger_geoms = {left_pad_id, right_pad_id}
finger1_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")

# ─── Inverse kinematics: compute waypoint joint configs ──────────────────────
# Desired EE orientation in world frame: Z pointing down
R_world_target = np.array([
    [ 1.,  0.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0., -1.]
])

# Waypoint 1: above the sphere (10 cm clearance)
T_above_cube = np.eye(4)
T_above_cube[:3, :3] = R_world_target @ sphere_rot
T_above_cube[:3, 3]  = sphere_pos + np.array([0.0, 0.0, 0.1])

q_above_cube, success = IKinSpace(panda.Slist, panda.M, T_above_cube,
                                  np.zeros(7), 0.001, 0.0001)
print(f"IK q_above_cube: {success}")
if not success: exit()
q_above_cube = wrap_pi(q_above_cube)

# Waypoint 2: at the sphere (pick pose)
T_pick = np.eye(4)
T_pick[:3, :3] = R_world_target @ sphere_rot
T_pick[:3, 3]  = sphere_pos

q_pick, success = IKinSpace(panda.Slist, panda.M, T_pick,
                            q_above_cube, 0.001, 0.0001)
print(f"IK q_pick: {success}")
if not success: exit()
q_pick = wrap_pi(q_pick)

# Waypoint 3: above the place location (10 cm clearance)
T_place_rot = np.array([[0, 1, 0, 0], [-1, 0, 0, 0.7], [0, 0, 1, 0.13]])
T_above_place = np.eye(4)
T_above_place[:3, :3] = R_world_target @ T_place_rot[:3, :3]
T_above_place[:3, 3]  = T_place_rot[:3, 3] + np.array([0.0, 0.0, 0.1])

q_above_place, success = IKinSpace(panda.Slist, panda.M, T_above_place,
                                   q_above_cube, 0.001, 0.0001)
print(f"IK q_above_place: {success}")
if not success: exit()
q_above_place = wrap_pi(q_above_place)

# Waypoint 4: at the place location
T_place = T_above_place.copy()
T_place[2, 3] = 0.03

q_place, success = IKinSpace(panda.Slist, panda.M, T_place,
                             q_above_place, 0.001, 0.0001)
print(f"IK q_place: {success}")
if not success: exit()
q_place = wrap_pi(q_place)

# ─── Load pre-compiled ACADOS solver ─────────────────────────────────────────
print("Loading pre-compiled solver...")
solver = load_solver(SCRIPT_DIR / 'franka_point_stab.json')
print("Solver loaded.")

Q = 200 * np.diag([300] * nq + [30] * nv)
R = np.diag([1.] * nu)
apply_cost_weights(solver, N, Q, R)

# ─── Motion primitives ───────────────────────────────────────────────────────

def move(Tf, theta_end):
    """MPC trajectory tracking: drives the arm from current pose to theta_end."""
    theta_start = data.qpos[:7].copy()
    N_traj = int(Tf / h)
    thetamatd, dthetamatd, ddthetamatd = sample_quintic_trajectory(
        theta_start, theta_end, Tf, N_traj, nq=nq)
    tau_ff = precompute_feedforward(pin_model, pin_data,
                                    thetamatd, dthetamatd, ddthetamatd, nv=nv)

    x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
    init_warm_start(solver, N, x_curr, nu=nu)

    steps_per_interval = int(h / model.opt.timestep)

    for traj_idx in range(N_traj):
        set_trajectory_references(solver, N, traj_idx,
                                  thetamatd, dthetamatd, tau_ff)
        pin_initial_state(solver, x_curr)

        status = solver.solve()
        if status not in [0, 2]:
            print(f"WARNING: solver status {status} at traj_idx {traj_idx}")

        data.ctrl[:7] = solver.get(0, 'u')
        for _ in range(steps_per_interval):
            mujoco.mj_step(model, data)
            viewer.sync()

        x_curr = np.concatenate([data.qpos[:7].copy(), data.qvel[:7].copy()])
        shift_warm_start(solver, N)



# ─── Main pick-and-place sequence ────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    F_thresh = compute_F_thresh(model, sphere_geom_id, left_pad_id, sphere_mass,
                                rule="mean")

    move(2, q_above_cube)    # 1. approach above sphere
    open_gripper(model, data, viewer, pin_model, pin_data, finger1_id)  # 2. open
    move(1, q_pick)          # 3. descend to pick pose
    close_gripper(model, data, viewer, pin_model, pin_data,             # 4. grasp
                  sphere_geom_id, finger_geoms, F_thresh,
                  contact_mult=5.0, alpha_step=0.2)
    move(1, q_above_cube)    # 5. lift
    move(2, q_above_place)   # 6. transport to place
    move(1, q_place)         # 7. lower to place pose
    open_gripper(model, data, viewer, pin_model, pin_data, finger1_id)  # 8. release

    while True:
        pass