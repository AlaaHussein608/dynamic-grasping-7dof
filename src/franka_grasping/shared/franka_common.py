"""
franka_common.py
================
Single source of truth for everything Franka-Panda-specific:

  * asset paths (URDF / MJCF / meshes) resolved relative to this repo,
  * joint limits, torque limits, task geometry constants,
  * the ``panda`` parameter class (screw axes, home pose, inertias),
  * Pinocchio / MuJoCo model loading,
  * a module-level IK Pinocchio model and the two IK solvers
    (``IK_frantik`` closed-chain, ``IK_pinocchio`` damped least squares),
  * forward kinematics helpers,
  * the fast arm mass/bias helper (crba + rnea),
  * grasp-pose construction, workspace checks and the analytic
    reachable-time window used by the intercept planners.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import frantik as fk

# ─────────────────────────────────────────────────────────────────────────────
#  Asset paths — resolved relative to the repository layout:
#      src/
#      ├── franka_grasping/shared/franka_common.py   (this file)
#      └── xml and meshes/                           (MJCF, URDF, meshes/, assets/)
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "xml and meshes"

PANDA_URDF       = str(ASSETS_DIR / "mjx_panda.urdf")
PANDA_MJCF       = str(ASSETS_DIR / "mjx_panda.xml")
SPHERE_SCENE_XML = str(ASSETS_DIR / "mjx_single_sphere.xml")

# NOTE: mjx_single_cube.xml is referenced by the point-stabilization scripts
# but is not present in "xml and meshes/" — copy it there before running them
# (e.g. from ~/franka_emika_panda/mjx_single_cube.xml).
CUBE_SCENE_XML = str(ASSETS_DIR / "mjx_single_cube.xml")


def script_dir(file_dunder):
    """Directory of the calling script — used to keep per-project solver
    JSON / code-export paths next to the script instead of depending on CWD."""
    return Path(file_dunder).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
#  Dimensions and limits (Franka Emika Panda, 7 DOF)
# ─────────────────────────────────────────────────────────────────────────────

NQ = 7            # number of joints
NV = 7            # number of velocity DOFs
NX = NQ + NV      # state dimension
NU = NV           # control dimension

Q_MIN   = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX   = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
DQ_MAX  = np.array([ 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# ── Task geometry ─────────────────────────────────────────────────────────────
PANDA_MAX_REACH    = 0.855
PANDA_MIN_REACH    = 0.10
PANDA_BODY_RADIUS  = 0.12
GRIPPER_CLOSE_TIME = 0.2
PREGRASP_OFFSET    = 0.12
FLOOR_MARGIN       = 0.02

LINK_RADII = np.array([0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.05])

# ── Common configurations ─────────────────────────────────────────────────────
Q_HOME = np.array([0., 0., 0., -0.1, 0., 0., 0.])

# Joint-space target used by the demo / tuning scripts
# (IK solution above the demo cube).
Q_TARGET_DEMO = np.array([2.43261247,  0.02842875, -2.42377283, -2.7289824,
                          0.04433423,  2.70720587,  0.75383392])


# ─────────────────────────────────────────────────────────────────────────────
#  Panda kinematic / dynamic parameters (moved verbatim from franka_param.py)
# ─────────────────────────────────────────────────────────────────────────────

class panda:

    T_hand_tcp = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 1.0]
    ])

    Slist = np.array([
        [ 0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
        [ 0.,     1.,     0.,    -1.,     0.,    -1.,     0.   ],
        [ 1.,     0.,     1.,     0.,     1.,     0.,    -1.   ],
        [ 0.,    -0.333,  0.,     0.649,  0.,     1.033,  0.   ],
        [ 0.,     0.,     0.,     0.,     0.,     0.,     0.088],
        [ 0.,     0.,     0.,    -0.0825, 0.,     0.,     0.   ]
    ])

    Blist = np.array([
        [ 0.0,     0.7071,  0.0,    -0.7071,  0.0,    -0.7071,  0.0   ],
        [ 0.0,    -0.7071,  0.0,     0.7071,  0.0,     0.7071,  0.0   ],
        [-1.0,     0.0,    -1.0,     0.0,    -1.0,     0.0,     1.0   ],
        [ 0.0622,  0.3486,  0.0622, -0.1252,  0.0622,  0.1464,  0.0   ],
        [-0.0622,  0.3486, -0.0622, -0.1252, -0.0622,  0.1464,  0.0   ],
        [ 0.0,     0.0880,  0.0,    -0.0055,  0.0,    -0.0880,  0.0   ]
    ])

    M = np.array([
        [ 0.70710681,  0.70710676,  0.,  0.088],
        [ 0.70710676, -0.70710681,  0.,  0.   ],
        [ 0.,          0.,         -1.,  0.826],
        [ 0.,          0.,          0.,  1.   ]
    ])

    Mlist = [
        # M01
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.333],
            [0, 0, 0, 1]
        ]),
        # M12
        np.array([
            [1, 0,  0, 0],
            [0, 0,  1, 0],
            [0, -1, 0, 0],
            [0, 0,  0, 1]
        ]),
        # M23
        np.array([
            [1, 0,  0, 0],
            [0, 0, -1, -0.316],
            [0, 1,  0, 0],
            [0, 0,  0, 1]
        ]),
        # M34
        np.array([
            [1, 0,  0, 0.0825],
            [0, 0, -1, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1]
        ]),
        # M45
        np.array([
            [1, 0,  0, -0.0825],
            [0, 0,  1, 0.384],
            [0, -1, 0, 0],
            [0, 0,  0, 1]
        ]),
        # M56
        np.array([
            [1, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1]
        ]),
        # M67
        np.array([
            [1, 0,  0, 0.088],
            [0, 0, -1, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1]
        ]),
        # M7e (hand frame, NOT TCP)
        np.array([
            [ 0.70710681,  0.70710676, 0, 0],
            [-0.70710676,  0.70710681, 0, 0],
            [ 0,           0,          1, 0.207],
            [ 0,           0,          0, 1]
        ])
    ]

    TAU_MAX = TAU_MAX
    Q_MIN   = Q_MIN
    Q_MAX   = Q_MAX
    DQ_MAX  = DQ_MAX
    PANDA_MAX_REACH    = PANDA_MAX_REACH
    PANDA_MIN_REACH    = PANDA_MIN_REACH
    PANDA_BODY_RADIUS  = PANDA_BODY_RADIUS
    GRIPPER_CLOSE_TIME = GRIPPER_CLOSE_TIME
    PREGRASP_OFFSET    = PREGRASP_OFFSET
    FLOOR_MARGIN       = FLOOR_MARGIN
    LINK_RADII = LINK_RADII

    Glist = np.array([
        # Link 0 (base)
        np.array([
            [3.15000000e-03, 8.29040000e-07, 1.50000000e-04, 0.00000000e+00, -3.14720760e-02, -8.81676600e-05],
            [8.29040000e-07, 3.88000000e-03, 8.22990000e-06, 3.14720760e-02, 0.00000000e+00, 2.58318648e-02],
            [1.50000000e-04, 8.22990000e-06, 4.28500000e-03, 8.81676600e-05, -2.58318648e-02, 0.00000000e+00],
            [0.00000000e+00, 3.14720760e-02, 8.81676600e-05, 6.29769000e-01, 0.00000000e+00, 0.00000000e+00],
            [-3.14720760e-02, 0.00000000e+00, -2.58318648e-02, 0.00000000e+00, 6.29769000e-01, 0.00000000e+00],
            [-8.81676600e-05, 2.58318648e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.29769000e-01]
        ]),
        # Link 1
        np.array([
            [7.03370000e-01, -1.39000000e-04, 6.77200000e-03, 0.00000000e+00, 2.36703972e-01, 1.03439934e-02],
            [-1.39000000e-04, 7.06610000e-01, 1.91690000e-02, -2.36703972e-01, 0.00000000e+00, -1.92614005e-02],
            [6.77200000e-03, 1.91690000e-02, 9.11700000e-03, -1.03439934e-02, 1.92614005e-02, 0.00000000e+00],
            [0.00000000e+00, -2.36703972e-01, -1.03439934e-02, 4.97068400e+00, 0.00000000e+00, 0.00000000e+00],
            [2.36703972e-01, 0.00000000e+00, 1.92614005e-02, 0.00000000e+00, 4.97068400e+00, 0.00000000e+00],
            [1.03439934e-02, -1.92614005e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.97068400e+00]
        ]),
        # Link 2
        np.array([
            [7.96200000e-03, -3.92500000e-03, 1.02540000e-02, 0.00000000e+00, -2.26100637e-03, -1.85797147e-02],
            [-3.92500000e-03, 2.81100000e-02, 7.04000000e-04, 2.26100637e-03, 0.00000000e+00, 2.03199457e-03],
            [1.02540000e-02, 7.04000000e-04, 2.59950000e-02, 1.85797147e-02, -2.03199457e-03, 0.00000000e+00],
            [0.00000000e+00, 2.26100637e-03, 1.85797147e-02, 6.46926000e-01, 0.00000000e+00, 0.00000000e+00],
            [-2.26100637e-03, 0.00000000e+00, -2.03199457e-03, 0.00000000e+00, 6.46926000e-01, 0.00000000e+00],
            [-1.85797147e-02, 2.03199457e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.46926000e-01]
        ]),
        # Link 3
        np.array([
            [3.72420000e-02, -4.76100000e-03, -1.13960000e-02, 0.00000000e+00, 2.14708620e-01, 1.26729160e-01],
            [-4.76100000e-03, 3.61550000e-02, -1.28050000e-02, -2.14708620e-01, 0.00000000e+00, -8.88447200e-02],
            [-1.13960000e-02, -1.28050000e-02, 1.08300000e-02, -1.26729160e-01, 8.88447200e-02, 0.00000000e+00],
            [0.00000000e+00, -2.14708620e-01, -1.26729160e-01, 3.22860400e+00, 0.00000000e+00, 0.00000000e+00],
            [2.14708620e-01, 0.00000000e+00, 8.88447200e-02, 0.00000000e+00, 3.22860400e+00, 0.00000000e+00],
            [1.26729160e-01, -8.88447200e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.22860400e+00]
        ]),
        # Link 4
        np.array([
            [2.58530000e-02, 7.79600000e-03, -1.33200000e-03, 0.00000000e+00, -9.85020700e-02, 3.74644410e-01],
            [7.79600000e-03, 1.95520000e-02, 8.64100000e-03, 9.85020700e-02, 0.00000000e+00, 1.90768380e-01],
            [-1.33200000e-03, 8.64100000e-03, 2.83230000e-02, -3.74644410e-01, -1.90768380e-01, 0.00000000e+00],
            [0.00000000e+00, 9.85020700e-02, -3.74644410e-01, 3.58789500e+00, 0.00000000e+00, 0.00000000e+00],
            [-9.85020700e-02, 0.00000000e+00, -1.90768380e-01, 0.00000000e+00, 3.58789500e+00, 0.00000000e+00],
            [3.74644410e-01, 1.90768380e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.58789500e+00]
        ]),
        # Link 5
        np.array([
            [3.55490000e-02, -2.11700000e-03, -4.03700000e-03, 0.00000000e+00, 4.71216864e-02, 5.03434725e-02],
            [-2.11700000e-03, 2.94740000e-02, 2.29000000e-04, -4.71216864e-02, 0.00000000e+00, 1.46537325e-02],
            [-4.03700000e-03, 2.29000000e-04, 8.62700000e-03, -5.03434725e-02, -1.46537325e-02, 0.00000000e+00],
            [0.00000000e+00, -4.71216864e-02, -5.03434725e-02, 1.22594600e+00, 0.00000000e+00, 0.00000000e+00],
            [4.71216864e-02, 0.00000000e+00, -1.46537325e-02, 0.00000000e+00, 1.22594600e+00, 0.00000000e+00],
            [5.03434725e-02, 1.46537325e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22594600e+00]
        ]),
        # Link 6
        np.array([
            [1.96400000e-03, 1.09000000e-04, -1.15800000e-03, 0.00000000e+00, 1.75271589e-02, -2.35267569e-02],
            [1.09000000e-04, 4.35400000e-03, 3.41000000e-04, -1.75271589e-02, 0.00000000e+00, -1.00241617e-01],
            [-1.15800000e-03, 3.41000000e-04, 5.43300000e-03, 2.35267569e-02, 1.00241617e-01, 0.00000000e+00],
            [0.00000000e+00, -1.75271589e-02, 2.35267569e-02, 1.66655500e+00, 0.00000000e+00, 0.00000000e+00],
            [1.75271589e-02, 0.00000000e+00, 1.00241617e-01, 0.00000000e+00, 1.66655500e+00, 0.00000000e+00],
            [-2.35267569e-02, -1.00241617e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.66655500e+00]
        ]),
        # Link 7
        np.array([
            [1.25160000e-02, -4.28000000e-04, -1.19600000e-03, 0.00000000e+00, -4.53059486e-02, -3.12743954e-03],
            [-4.28000000e-04, 1.00270000e-02, -7.41000000e-04, 4.53059486e-02, 0.00000000e+00, -7.73548487e-03],
            [-1.19600000e-03, -7.41000000e-04, 4.81500000e-03, 3.12743954e-03, 7.73548487e-03, 0.00000000e+00],
            [0.00000000e+00, 4.53059486e-02, 3.12743954e-03, 7.35522000e-01, 0.00000000e+00, 0.00000000e+00],
            [-4.53059486e-02, 0.00000000e+00, 7.73548487e-03, 0.00000000e+00, 7.35522000e-01, 0.00000000e+00],
            [-3.12743954e-03, -7.73548487e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.35522000e-01]
        ]),
        # Hand (link 8)
        np.array([
            [1.00000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.19000000e-02, 0.00000000e+00],
            [0.00000000e+00, 2.50000000e-03, 0.00000000e+00, 2.19000000e-02, 0.00000000e+00, 7.30000000e-03],
            [0.00000000e+00, 0.00000000e+00, 1.70000000e-03, 0.00000000e+00, -7.30000000e-03, 0.00000000e+00],
            [0.00000000e+00, 2.19000000e-02, 0.00000000e+00, 7.30000000e-01, 0.00000000e+00, 0.00000000e+00],
            [-2.19000000e-02, 0.00000000e+00, -7.30000000e-03, 0.00000000e+00, 7.30000000e-01, 0.00000000e+00],
            [0.00000000e+00, 7.30000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.30000000e-01]
        ])
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pinocchio_model(urdf_path=PANDA_URDF):
    """Build a Pinocchio model + data pair from a URDF."""
    model = pin.buildModelFromUrdf(urdf_path)
    return model, model.createData()


def load_mujoco_model(xml_path=SPHERE_SCENE_XML):
    """Build a MuJoCo model + data pair from an MJCF scene."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    return model, mujoco.MjData(model)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level IK model (shared by both IK solvers)
# ─────────────────────────────────────────────────────────────────────────────

ik_model = pin.buildModelFromUrdf(PANDA_URDF)
ik_data  = ik_model.createData()
ik_ee_id = ik_model.getFrameId("gripper")


# ─────────────────────────────────────────────────────────────────────────────
#  Forward kinematics helpers
# ─────────────────────────────────────────────────────────────────────────────

def ee_position(model, data, frame_id, q):
    """End-effector position in world frame."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    return data.oMf[frame_id].translation.copy()


def ee_pose(model, data, frame_id, q):
    """4×4 homogeneous EE transform in world frame."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    return data.oMf[frame_id].homogeneous.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Arm dynamics — fast mass matrix + bias (crba + rnea)
# ─────────────────────────────────────────────────────────────────────────────

def arm_mass_bias(pin_model, pin_data, q, v, armature_diag):
    """
    Return (M, h) for the arm:

        pin.crba(pin_model, pin_data, q)          # M → pin_data.M
        M = pin_data.M.copy() + diag(armature)    # armature always added
        h = pin.rnea(pin_model, pin_data, q, v, 0)   # C·v + g

    crba is fast and C, g never need to be formed separately.
    """
    pin.crba(pin_model, pin_data, q)
    M = pin_data.M.copy() + np.diag(armature_diag)
    h = pin.rnea(pin_model, pin_data, q, v, np.zeros(len(q)))
    return M, h


# ─────────────────────────────────────────────────────────────────────────────
#  Inverse kinematics — frantik closed-chain solver
# ─────────────────────────────────────────────────────────────────────────────

def IK_frantik(T, q0, n_seeds=42):
    """
    Seed convention:
      seed[0]            = q0  (current config)
      seed[1..n_seeds-1] = Uniform[Q_MIN, Q_MAX]  (full 7-DOF draws)
    """
    def valid_q(q):
        return (q is not None and
                np.all(np.isfinite(q)) and
                np.all(q >= Q_MIN) and
                np.all(q <= Q_MAX) and
                q[1] != 0.0)

    best_q, best_score = None, -np.inf

    # seed[0]: q0
    q_sol = fk.cc_ik(T, q0[6], q0)
    if valid_q(q_sol):
        best_q     = q_sol
        best_score = -np.max(np.abs(q_sol - q0) / DQ_MAX)

    # seed[1..]: full Uniform[Q_MIN, Q_MAX] draws
    for q_extra in np.random.uniform(Q_MIN, Q_MAX, (n_seeds - 1, 7)):
        q_sol = fk.cc_ik(T, q_extra[6], q_extra)
        if not valid_q(q_sol):
            continue
        score = -np.max(np.abs(q_sol - q0) / DQ_MAX)
        if score > best_score:
            best_score, best_q = score, q_sol

    return (best_q, True) if best_q is not None else (None, False)


def IK_pinocchio(T_target, q0, n_seeds=14):
    """
    Seed convention:
      seed[0]            = q0  (current config)
      seed[1..n_seeds-1] = Uniform[Q_MIN, Q_MAX]
    """
    eomg_sq, ev_sq    = 1e-8, 1e-8
    max_iter, damping = 25, 1e-4

    T_tgt = pin.SE3(T_target[:3, :3].copy(), T_target[:3, 3].copy())

    # seed[0] = q0, seed[1..] = Uniform[Q_MIN, Q_MAX]
    extra_seeds = np.random.uniform(Q_MIN, Q_MAX, (n_seeds - 1, 7))
    seeds = [q0.copy()] + list(extra_seeds)

    best_score, best_q = -np.inf, None
    for q in seeds:
        for _ in range(max_iter):
            pin.forwardKinematics(ik_model, ik_data, q)
            pin.updateFramePlacement(ik_model, ik_data, ik_ee_id)
            err = pin.log6(ik_data.oMf[ik_ee_id].inverse() * T_tgt).vector
            if err[:3] @ err[:3] < ev_sq and err[3:] @ err[3:] < eomg_sq:
                break
            J  = pin.computeFrameJacobian(ik_model, ik_data, q, ik_ee_id,
                                          pin.ReferenceFrame.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q  = np.clip(q + dq, Q_MIN, Q_MAX)
        else:
            continue

        score = -np.max(np.abs(q - q0) / DQ_MAX)
        in_limits = np.all(q >= Q_MIN - 1e-9) and np.all(q <= Q_MAX + 1e-9)
        if score > best_score and in_limits:
            best_score, best_q = score, q.copy()

    return (best_q, best_q is not None)


# ─────────────────────────────────────────────────────────────────────────────
#  Workspace checks and grasp geometry
# ─────────────────────────────────────────────────────────────────────────────

def check_workspace(p_target, base_pos=np.zeros(3)):
    """Returns True when p_target lies within the Panda's reachable workspace."""
    dist_3d = np.linalg.norm(p_target - base_pos)
    dist_xy = np.linalg.norm(p_target[:2] - base_pos[:2])
    return (PANDA_MIN_REACH <= dist_3d <= PANDA_MAX_REACH
            and dist_xy >= PANDA_BODY_RADIUS)


def _grasp_orientation(v_ball):
    """Rotation matrix: EE X-axis along ball XY velocity, Z pointing down."""
    v_xy    = np.array([v_ball[0], v_ball[1], 0.0])
    v_speed = np.linalg.norm(v_xy)

    flange_x = (v_xy / v_speed) if v_speed > 1e-6 else np.array([1.0, 0.0, 0.0])
    flange_z = np.array([0.0, 0.0, -1.0])
    flange_y = np.cross(flange_z, flange_x)
    flange_y /= np.linalg.norm(flange_y)

    return np.column_stack([flange_x, flange_y, flange_z])


def compute_grasp_pose(p_ball, v_ball, ball_radius=0.03,
                       pregrasp_offset=PREGRASP_OFFSET):
    """(T_pregrasp, T_grasp) pair — the intercept-planner convention."""
    R = _grasp_orientation(v_ball)

    p_grasp    = np.array([p_ball[0], p_ball[1], ball_radius])
    p_pregrasp = p_grasp + np.array([0.0, 0.0, pregrasp_offset])

    T_grasp    = np.eye(4); T_grasp[:3, :3]    = R; T_grasp[:3, 3]    = p_grasp
    T_pregrasp = np.eye(4); T_pregrasp[:3, :3] = R; T_pregrasp[:3, 3] = p_pregrasp

    return T_pregrasp, T_grasp


def single_grasp_pose(p_ball, v_ball, use_pregrasp, ball_radius,
                      pregrasp_offset=None):
    """Single 4×4 EE target — the reactive-MPC / benchmark convention.

    ``pregrasp_offset`` defaults to ``ball_radius`` (as in the originals that
    hover exactly one radius above the ball top).
    """
    if pregrasp_offset is None:
        pregrasp_offset = ball_radius

    R = _grasp_orientation(v_ball)
    p = np.array([p_ball[0], p_ball[1], ball_radius])
    if use_pregrasp:
        p = p + np.array([0.0, 0.0, pregrasp_offset])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T


def reachable_window(p_ball, v_ball, base_pos, t_lo, t_hi, ball_radius):
    """
    Solve |p_grasp(t) - base|^2 <= R_max^2 for the straight-line ball model.
    With d = p_grasp(0) - base and planar velocity v:
        (v·v) t^2 + 2(d·v) t + (d·d - R^2) <= 0
    Returns (t_enter, t_exit) clipped to [t_lo, t_hi], or None if the ball
    never enters (or has permanently left) the outer reach sphere.
    """
    d = np.array([p_ball[0], p_ball[1], ball_radius]) - base_pos
    v = np.array([v_ball[0], v_ball[1], 0.0])

    a = v @ v
    b = 2.0 * (d @ v)
    c = d @ d - PANDA_MAX_REACH**2

    if a < 1e-12:                                  # static ball
        return (t_lo, t_hi) if c <= 0.0 else None

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sq      = np.sqrt(disc)
    t_enter = max((-b - sq) / (2.0 * a), t_lo)
    t_exit  = min((-b + sq) / (2.0 * a), t_hi)

    return (t_enter, t_exit) if t_enter <= t_exit else None


# ─────────────────────────────────────────────────────────────────────────────
#  Motion-time bounds
# ─────────────────────────────────────────────────────────────────────────────

def quintic_min_time(q_from, q_to):
    """
    Minimum time for a rest-to-rest quintic under DQ_MAX.
    Peak velocity of a quintic is 15*dq/(8*T), so |v_peak| <= dq_max gives
        T = max_j 15*|dq_j| / (8*dq_max_j).
    """
    return float(np.max(15.0 * np.abs(q_to - q_from) / (8.0 * DQ_MAX)))


def lower_bound_time(q_start, q_pg, q_g):
    """
    Bang-bang velocity-only lower bound: T = max(|dq| / dq_max).
    No trajectory (TOPP-RA / quintic included) can be faster, so rejection
    on T_lb > t is provably safe — no tuning constant needed.
    """
    T1 = np.max(np.abs(q_pg - q_start) / DQ_MAX)
    T2 = np.max(np.abs(q_g  - q_pg)    / DQ_MAX)
    return T1, T2, GRIPPER_CLOSE_TIME


# ─────────────────────────────────────────────────────────────────────────────
#  Miscellaneous shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def sample_targets(n, seed):
    """n random joint configurations, uniform within the Panda joint limits."""
    rng = np.random.default_rng(seed)
    return rng.uniform(Q_MIN, Q_MAX, size=(n, 7))


def compute_F_thresh(mj_model, sphere_geom_id, pad_geom_id, sphere_mass,
                     rule="mean"):
    """
    Minimum normal force per finger to hold the sphere: F >= m*g / (2*mu_eff).

    ``rule`` selects the effective-friction convention used by the originals:
      "mean" — mu_eff = (mu_sphere + mu_finger) / 2
      "max"  — mu_eff = max(mu_sphere, mu_finger)
    """
    mu_sphere = mj_model.geom_friction[sphere_geom_id][0]
    mu_finger = mj_model.geom_friction[pad_geom_id][0]
    if rule == "max":
        mu_eff = max(mu_sphere, mu_finger)
    else:
        mu_eff = (mu_sphere + mu_finger) / 2.0
    return sphere_mass * 9.81 / (2.0 * mu_eff)
