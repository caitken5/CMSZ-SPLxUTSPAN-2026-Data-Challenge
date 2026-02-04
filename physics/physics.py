"""Physics module."""

import torch
from .constants import AIR_DENSITY, BALL_AREA, BALL_MASS, BALL_RADIUS, DRAG_COEFFICIENT

def drag_acceleration(v):
    """
    v: (batch, 3)
    alpha_d: scalar or (batch, 1)
    """
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)
    return -AIR_DENSITY*DRAG_COEFFICIENT*BALL_AREA*speed*v/(2*BALL_MASS)

def magnus_acceleration(v, omega, c_l):
    """
    v: (batch, 3)
    omega: (batch, 3)
    c_l: (batch, 3) 
    """
    # c_l is the lift coefficient

    ang_speed = torch.linalg.norm(omega, dim=-1, keepdim=True)

    return 0.5*AIR_DENSITY*BALL_AREA*c_l*BALL_RADIUS*torch.cross(omega, v, dim=-1)/ang_speed