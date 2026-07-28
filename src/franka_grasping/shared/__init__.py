"""Shared modules for the refactored Franka Panda control codebase.

Modules
-------
franka_common       : robot constants, asset paths, model loading, IK/FK,
                      fast arm mass/bias, grasp geometry and workspace checks.
trajectory_control  : quintic / TOPP-RA trajectory generation and the
                      SMC / feedback-linearization / PD control laws.
gripper             : shared open / close gripper primitives.
acados_mpc          : ACADOS OCP solver builder and runtime MPC helpers.
"""
