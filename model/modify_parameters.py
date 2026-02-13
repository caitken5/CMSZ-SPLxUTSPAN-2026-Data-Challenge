"""Parameter transforms for constrained model outputs.

Includes modified softplus variants useful for smoothly enforcing positivity
(or a chosen lower bound) while retaining tunable curvature.
"""

import math

import torch
import torch.nn.functional as F


def modified_softplus(
    x: torch.Tensor,
    beta: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> torch.Tensor:
    """General modified softplus.

    Equation:
        f(x) = (1 / beta) * log(1 + exp(beta * (x - shift_x))) + shift_y

    Args:
        x: Input tensor.
        beta: Sharpness parameter (> 0).
        shift_x: Horizontal shift c.
        shift_y: Vertical shift d.

    Returns:
        Transformed tensor.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return F.softplus(beta * (x - shift_x)) / beta + shift_y


def zero_centered_softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Modified softplus with f(0)=0.

    Equation:
        f(x) = (1 / beta) * log(1 + exp(beta * x)) - log(2) / beta

    Args:
        x: Input tensor.
        beta: Sharpness parameter (> 0).

    Returns:
        Zero-centered modified softplus transform.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return F.softplus(beta * x) / beta - (math.log(2.0) / beta)


def lower_bounded_softplus(
    x: torch.Tensor,
    lower_bound: float,
    beta: float = 1.0,
) -> torch.Tensor:
    """Soft lower-bound transform.

    Equation:
        f(x) = lower_bound + (1 / beta) * log(1 + exp(beta * x))

    Useful for parameters that must stay above a known minimum.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return lower_bound + F.softplus(beta * x) / beta
