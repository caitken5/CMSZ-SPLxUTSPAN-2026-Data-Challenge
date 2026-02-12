"""Physics module."""

import torch
from .constants import AIR_DENSITY, BALL_AREA, BALL_MASS, BALL_RADIUS, DRAG_COEFFICIENT, g

# alpha_d: drag acceleration coefficient
# alpha_l: magnus acceleration coefficient
# These are calculated elsewhere so small adjustments can be made to constants.

def drag_acceleration(v, alpha_d):
    """
    v: (batch, 3)
    alpha_d: scalar or (batch, 1)
    """
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)
    return -alpha_d*speed*v

def magnus_acceleration(v, omega, alpha_l):
    """
    v: (batch, 3)
    omega: (batch, 3)
    alpha_l: scalar or (batch, 1)
    """

    return alpha_l*torch.cross(omega, v, dim=-1)

def gravity_acceleration(batch_size):
    """
    returns: (batch_size, 3)
    """
    return g.expand(batch_size, 3)