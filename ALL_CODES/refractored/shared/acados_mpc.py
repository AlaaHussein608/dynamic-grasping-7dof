"""
acados_mpc.py
=============
ACADOS OCP construction and runtime MPC helpers for the Franka Panda.

Previously four near-identical ``build_solver.py`` scripts existed (point
stabilization, trajectory tracking, intercept planning, contractive MPC)
plus per-script copies of the warm-start / shift / reference-setting /
cost-setting boilerplate.  Everything is consolidated here:

  * :func:`build_franka_ocp_solver` — parametric builder covering every
    original variant, including the contraction-constrained one,
  * :func:`load_solver`             — load a pre-compiled solver from JSON,
  * runtime helpers with a single standard signature ``(solver, N, ...)``.
"""

import os

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Solver construction
# ─────────────────────────────────────────────────────────────────────────────

def build_franka_ocp_solver(model_name, h, N, Q, R,
                            urdf_path, mjcf_path,
                            json_file, export_dir,
                            tolerances=None,
                            contraction=False):
    """
    Generate and compile an ACADOS OCP solver for the torque-controlled Panda.

    Dynamics: xdot = [dq, M(q)^{-1}(tau - bias)] via CasADi-Pinocchio, with
    the MuJoCo joint armature added to the mass matrix.  Stage cost is
    LINEAR_LS with weight blkdiag(Q, R); the terminal cost weight is zero
    (references are set at runtime).  Joint-position box constraints are
    hard; velocity bounds are soft (zl=zu=1000, Zl=Zu=100); torque bounds
    are hard.

    Args:
        model_name : ACADOS model name (also used in generated C code).
        h, N       : MPC timestep [s] and horizon length.
        Q, R       : stage cost weights, shapes (nx, nx) and (nu, nu).
        urdf_path  : Panda URDF for the Pinocchio dynamics.
        mjcf_path  : MuJoCo scene, used only to read the joint armature.
        json_file  : output solver description path.
        export_dir : generated-C-code directory.
        tolerances : optional (tol_eq, tol_ineq) pair.
        contraction: add the contraction stage constraint
                     ‖e(k+1)‖² − α²‖e(k)‖² ≤ 0 on the joint-position error,
                     with parameters p = [x_ref (nx), alpha (1)] and the
                     next state approximated by one explicit Euler step.

    Returns the json_file path.
    """
    import casadi as ca
    import mujoco
    import pinocchio as pin
    import pinocchio.casadi as cpin
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    # ── Pinocchio + MuJoCo models ─────────────────────────────────────────────
    pin_model  = pin.buildModelFromUrdf(urdf_path)
    cpin_model = cpin.Model(pin_model)
    cpin_data  = cpin_model.createData()
    mj_model   = mujoco.MjModel.from_xml_path(mjcf_path)

    nq = pin_model.nq
    nv = pin_model.nv
    nx = nq + nv
    nu = nv

    # ── Continuous-time dynamics  xdot = f(x, u) ─────────────────────────────
    cq    = ca.SX.sym('q',   nq)
    cdq   = ca.SX.sym('dq',  nv)
    ctau  = ca.SX.sym('tau', nv)
    x_sym = ca.vertcat(cq, cdq)

    M_sym    = cpin.crba(cpin_model, cpin_data, cq)
    M_sym   += np.diag(mj_model.dof_armature[:nq])
    bias_sym = cpin.rnea(cpin_model, cpin_data, cq, cdq, ca.SX.zeros(nv))
    ddq_sym  = ca.solve(M_sym, ctau - bias_sym)

    f_expl = ca.vertcat(cdq, ddq_sym)   # xdot = [dq, ddq]

    # ── ACADOS model ─────────────────────────────────────────────────────────
    acados_model              = AcadosModel()
    acados_model.name         = model_name
    acados_model.x            = x_sym
    acados_model.u            = ctau
    acados_model.f_expl_expr  = f_expl

    if contraction:
        # p = [x_ref (nx,), alpha (1,)]; constraint at each stage k:
        #     ‖e(k+1)‖² − α² ‖e(k)‖² ≤ 0,   e(k) = x(k)[:nq] − x_ref[:nq]
        # The next state is approximated by one explicit Euler step, which is
        # consistent with the SQP linearisation ACADOS uses internally.
        x_ref_p = ca.SX.sym('x_ref', nx)
        alpha_p = ca.SX.sym('alpha',  1)
        acados_model.p = ca.vertcat(x_ref_p, alpha_p)

        x_next_approx = x_sym + h * f_expl
        e_k   = x_sym[:nq]         - x_ref_p[:nq]
        e_kp1 = x_next_approx[:nq] - x_ref_p[:nq]
        acados_model.con_h_expr = (ca.dot(e_kp1, e_kp1)
                                   - alpha_p**2 * ca.dot(e_k, e_k))

    # ── OCP ──────────────────────────────────────────────────────────────────
    ny = nx + nu
    W          = np.zeros((ny, ny))
    W[:nx, :nx] = Q
    W[nx:, nx:] = R

    ocp       = AcadosOcp()
    ocp.model = acados_model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * h

    if contraction:
        ocp.parameter_values = np.zeros(nx + 1)   # [x_ref (nx), alpha (1)]

    # Stage cost (LINEAR_LS: minimize ||Vx x + Vu u − yref||_W²)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.Vx   = np.vstack([np.eye(nx),         np.zeros((nu, nx))])
    ocp.cost.Vu   = np.vstack([np.zeros((nx, nu)),  np.eye(nu)])
    ocp.cost.W    = W
    ocp.cost.yref = np.zeros(ny)   # updated online

    # Terminal cost (weight zero — set at runtime via cost_set)
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_e        = np.eye(nx)
    ocp.cost.W_e         = np.zeros((nx, nx))
    ocp.cost.yref_e      = np.zeros(nx)

    # State constraints: hard joint-position bounds, huge velocity bounds
    from .franka_common import Q_MIN, Q_MAX
    ocp.constraints.lbx   = np.concatenate([Q_MIN, -1e7 * np.ones(nv)])
    ocp.constraints.ubx   = np.concatenate([Q_MAX,  1e7 * np.ones(nv)])
    ocp.constraints.idxbx = np.arange(nx)

    # Soft velocity constraints
    ocp.constraints.idxsbx = np.arange(nq, nq + nv)
    n_soft = nq
    ocp.cost.zl = 1000 * np.ones(n_soft)
    ocp.cost.zu = 1000 * np.ones(n_soft)
    ocp.cost.Zl = 100  * np.ones(n_soft)
    ocp.cost.Zu = 100  * np.ones(n_soft)

    # Control constraints
    from .franka_common import TAU_MAX
    ocp.constraints.lbu   = -TAU_MAX
    ocp.constraints.ubu   =  TAU_MAX
    ocp.constraints.idxbu = np.arange(nu)

    if contraction:
        # h_contr ≤ 0  →  lh = -inf, uh = 0
        ocp.constraints.lh = np.array([-1e9])
        ocp.constraints.uh = np.array([ 0.0])

    # Initial state equality constraint (updated online)
    ocp.constraints.x0 = np.zeros(nx)

    # Solver options
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4     # RK4
    ocp.solver_options.sim_method_num_steps  = 3
    ocp.solver_options.nlp_solver_type       = 'SQP_RTI'
    ocp.solver_options.print_level           = 0
    if tolerances is not None:
        ocp.solver_options.tol_eq   = tolerances[0]
        ocp.solver_options.tol_ineq = tolerances[1]

    ocp.code_gen_opts.code_export_directory = str(export_dir)

    print(f"Generating and compiling ACADOS solver → {json_file} ...")
    AcadosOcpSolver(ocp, json_file=str(json_file))
    print("Done. Solver saved.")
    return str(json_file)


def load_solver(json_file):
    """Load a pre-compiled ACADOS solver (no code generation / rebuild)."""
    from acados_template import AcadosOcpSolver

    if not os.path.exists(str(json_file)):
        raise FileNotFoundError(
            f"Solver JSON '{json_file}' not found. Run the matching "
            f"build script first."
        )
    return AcadosOcpSolver(
        acados_ocp=None,
        json_file=str(json_file),
        generate=False,
        build=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runtime helpers — standard signatures (solver, N, ...)
# ─────────────────────────────────────────────────────────────────────────────

def apply_cost_weights(solver, N, Q, R):
    """Push blkdiag(Q, R) into every stage and Q into the terminal stage."""
    nx = Q.shape[0]
    nu = R.shape[0]
    W  = np.zeros((nx + nu, nx + nu))
    W[:nx, :nx] = Q
    W[nx:, nx:] = R
    for k in range(N):
        solver.cost_set(k, 'W', W)
    solver.cost_set(N, 'W', Q)   # terminal stage: state only


def init_warm_start(solver, N, x0, nu=7):
    """Initialise every stage of the solver's internal trajectory to (x0, 0)."""
    for k in range(N + 1):
        solver.set(k, 'x', x0)
    for k in range(N):
        solver.set(k, 'u', np.zeros(nu))


def shift_warm_start(solver, N, zero_last_u=False, nu=7):
    """Shift the stored solution forward one step (standard RTI practice).

    ``zero_last_u=True`` reproduces the contractive-MPC variant that resets
    the last control instead of repeating it.
    """
    for k in range(N - 1):
        solver.set(k, 'x', solver.get(k + 1, 'x'))
        solver.set(k, 'u', solver.get(k + 1, 'u'))
    if zero_last_u:
        solver.set(N - 1, 'u', np.zeros(nu))
    else:
        solver.set(N - 1, 'u', solver.get(N - 1, 'u'))
    solver.set(N, 'x', solver.get(N, 'x'))


def pin_initial_state(solver, x):
    """Pin the current state as the stage-0 equality constraint (and guess)."""
    solver.set(0, 'lbx', x)
    solver.set(0, 'ubx', x)
    solver.set(0, 'x',   x)


def set_trajectory_references(solver, N, traj_idx, pos, vel, tau_ff):
    """Set yref for every horizon stage, shifted to the current trajectory
    index.  Stage k: [q_d, dq_d, tau_ff] — terminal: [q_d, dq_d]."""
    n_traj = pos.shape[0]
    for k in range(N):
        idx = min(traj_idx + k, n_traj - 1)
        solver.set(k, 'yref',
                   np.concatenate([pos[idx], vel[idx], tau_ff[idx]]))
    idx_e = min(traj_idx + N, n_traj - 1)
    solver.set(N, 'yref', np.concatenate([pos[idx_e], vel[idx_e]]))


def set_point_reference(solver, N, xs, nu=7):
    """Fixed point-stabilization reference: yref = [xs, 0] on every stage."""
    yref = np.concatenate([xs, np.zeros(nu)])
    for k in range(N):
        solver.set(k, 'yref', yref)
    solver.set(N, 'yref', xs)
