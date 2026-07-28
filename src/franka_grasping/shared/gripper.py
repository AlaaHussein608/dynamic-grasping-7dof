"""
gripper.py
==========
Shared gripper primitives (open / close) used by the pick-and-place and
dynamic-grasping run scripts, so the finger-control logic lives in one place.

Both hold the arm at a fixed configuration with a gravity-compensated PD law
(fast crba + rnea, armature always added) while driving the finger actuator
``ctrl[7]``.
"""

import numpy as np
import mujoco

from .franka_common import arm_mass_bias


def _arm_hold_torque(model, data, pin_model, pin_data, theta_d, kp, kd):
    q  = data.qpos[:7]
    dq = data.qvel[:7]
    M, h = arm_mass_bias(pin_model, pin_data, q, dq, model.dof_armature[:7])
    return M @ (kp * (theta_d - q) - kd * dq) + h


def open_gripper(model, data, viewer, pin_model, pin_data, finger_joint_id,
                 desired=0.04, kp=500.0, kd=200.0):
    """Open the fingers to ``desired`` while holding the arm at its current
    pose. Steps the sim until the finger reaches the target."""
    theta_d = data.qpos[:7].copy()
    while True:
        error_grip   = desired - data.qpos[finger_joint_id]
        data.ctrl[7] = 200.0 * error_grip
        data.ctrl[:7] = _arm_hold_torque(model, data, pin_model, pin_data,
                                         theta_d, kp, kd)
        mujoco.mj_step(model, data)
        viewer.sync()
        if abs(error_grip) < 0.001:
            break


def close_gripper(model, data, viewer, pin_model, pin_data,
                  sphere_geom_id, finger_geoms, F_thresh,
                  theta_d=None, ramp_time=None, sphere_radius=None,
                  contact_mult=20.0, alpha_step=1.0,
                  kp=500.0, kd=200.0, ramp_kp=500.0, ramp_kd=50.0):
    """
    Close the fingers on the sphere while holding the arm at ``theta_d``.

    If ``ramp_time`` and ``sphere_radius`` are given, the fingers are first
    position-ramped from 0.04 to the sphere surface over ``ramp_time``
    seconds; then a torque squeeze ramps ``alpha`` until the finger–sphere
    contact normal force exceeds ``contact_mult * F_thresh``.

    Returns the grip torque magnitude that secured the ball.
    """
    if theta_d is None:
        theta_d = data.qpos[:7].copy()

    def arm_torque():
        return _arm_hold_torque(model, data, pin_model, pin_data,
                                theta_d, kp, kd)

    def contact_ok():
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == sphere_geom_id and g2 in finger_geoms) or \
               (g2 == sphere_geom_id and g1 in finger_geoms):
                if data.efc_force[c.efc_address] > contact_mult * F_thresh:
                    return True
        return False

    # Position-ramp phase
    if ramp_time is not None and sphere_radius is not None:
        dt           = model.opt.timestep
        n_steps      = int(ramp_time / dt)
        closing_dist = 0.04 - sphere_radius
        for step in range(n_steps):
            t_ratio        = step / n_steps
            gripper_target = 0.04 - t_ratio * closing_dist
            data.ctrl[7]   = (ramp_kp * (gripper_target - data.qpos[7])
                              + ramp_kd * (-data.qvel[7]))
            data.ctrl[:7]  = arm_torque()
            mujoco.mj_step(model, data)
            viewer.sync()

    # Torque squeeze phase
    alpha = 1.0
    while True:
        data.ctrl[7]  = -alpha * F_thresh
        data.ctrl[:7] = arm_torque()
        mujoco.mj_step(model, data)
        viewer.sync()
        if contact_ok():
            return alpha * F_thresh
        alpha += alpha_step
