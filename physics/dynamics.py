"""Dynamics module."""

import torch

from physics.constants import BALL_RADIUS
from .physics import (gravity_acceleration, drag_acceleration, magnus_acceleration)

def ball_dynamics(t, state, params):
    """
    state: (batch, 9)
      [x, y, z, vx, vy, vz, wx, wy, wz]

    params:
      alpha_d
      alpha_l
      k_spin (optional)
    """
    pos = state[:, 0:3]
    vel = state[:, 3:6]
    omega = state[:, 6:9]

    batch = state.shape[0]

    # Accelerations
    a = (
        gravity_acceleration(batch)
        + drag_acceleration(vel, params["alpha_d"])
        + magnus_acceleration(vel, omega, params["alpha_l"])
    )

    # Angular decay (optional)
    domega = -params.get("k_spin", 0.0) * omega

    dpos = vel
    dvel = a

    return torch.cat([dpos, dvel, domega], dim=-1) 

def compute_alphas(delta_rho, Cd, Cl, A, m0, rho0):
    mass = m0 # Decided against using a small change in mass, we should have the ball estimate.
    rho  = rho0 * torch.exp(delta_rho)

    alpha_d = 0.5 * rho * Cd * A / mass
    alpha_l = 0.5 * rho * Cl * A / mass

    return alpha_d, alpha_l

def compute_lift_coefficient(spin_ratio, Cl_max, state):
    """
    spin_ratio: (batch, 1) = (omega * r) / |v|
    returns: (batch, 1) lift coefficient Cl
    """

    vel = torch.linalg.norm(state[:, 3:6], dim=-1, keepdim=True)
    omega = torch.linalg.norm(state[:, 6:9], dim=-1, keepdim=True)
    spin = (omega * BALL_RADIUS) / (vel + 1e-8)  # Add small epsilon to avoid division by zero

    Cl = Cl_max * torch.tanh(spin * spin_ratio)

    return Cl