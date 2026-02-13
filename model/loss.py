"""Loss functions for physics-embedded optimization.

Implements nonlinear least squares with embedded ODE integration using the
project's ball dynamics model.
"""

from typing import Iterable, Literal

import torch

from physics.dynamics import ball_dynamics


def nonlinear_least_squares_with_embedded_ode(
    initial_state: torch.Tensor,
    params: dict,
    t_obs: torch.Tensor,
    y_obs: torch.Tensor,
    state_indices: Iterable[int] = (0, 1, 2),
    dt: float = 1e-3,
    reduction: Literal["mean", "sum", "none"] = "mean",
    return_predictions: bool = False,
):
    """Compute nonlinear least-squares loss with ODE-embedded predictions.

    The function integrates the ball ODE from ``initial_state`` to each
    observation time in ``t_obs``, samples the requested state components, and
    minimizes the residuals against ``y_obs``.

    Args:
        initial_state: Tensor of shape (9,)
            [x, y, z, vx, vy, vz, wx, wy, wz] at initial time t=0.
        params: Dynamics parameter dict expected by ``ball_dynamics``
            (e.g., ``alpha_d``, ``alpha_l``, optional ``k_spin``).
        t_obs: Tensor of shape (T,)
            Monotonic non-decreasing observation times in seconds.
        y_obs: Tensor of shape (T, D)
            Observed targets corresponding to ``state_indices``.
        state_indices: Components of state to compare against observations.
            Default compares position (x, y, z).
        dt: Internal integration step size (seconds).
        reduction: "mean", "sum", or "none" over squared residuals.
        return_predictions: If True, also returns predictions and residuals.

    Returns:
        loss, or (loss, y_pred, residuals) if ``return_predictions=True``.
    """
    if initial_state.ndim != 1 or initial_state.shape[0] != 9:
        raise ValueError("initial_state must have shape (9,)")
    if t_obs.ndim != 1:
        raise ValueError("t_obs must have shape (T,)")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    idx = tuple(state_indices)
    if any(i < 0 or i > 8 for i in idx):
        raise ValueError("state_indices must be within [0, 8]")

    if y_obs.ndim != 2:
        raise ValueError("y_obs must have shape (T, D)")
    if y_obs.shape[0] != t_obs.shape[0]:
        raise ValueError("y_obs and t_obs must have matching first dimension")
    if y_obs.shape[1] != len(idx):
        raise ValueError("y_obs second dimension must match len(state_indices)")

    state = initial_state.clone()
    current_t = 0.0
    preds = []

    t_prev = -float("inf")
    for t_target in t_obs:
        t_target_val = float(t_target.detach().cpu().item())
        if t_target_val < t_prev:
            raise ValueError("t_obs must be non-decreasing")
        t_prev = t_target_val

        while current_t + dt < t_target_val:
            deriv = ball_dynamics(current_t, state.unsqueeze(0), params).squeeze(0)
            state = state + dt * deriv
            current_t += dt

        rem = t_target_val - current_t
        if rem > 0:
            deriv = ball_dynamics(current_t, state.unsqueeze(0), params).squeeze(0)
            state = state + rem * deriv
            current_t = t_target_val

        preds.append(state[list(idx)])

    y_pred = torch.stack(preds, dim=0)
    residuals = y_pred - y_obs
    sq = residuals.pow(2)

    if reduction == "mean":
        loss = sq.mean()
    elif reduction == "sum":
        loss = sq.sum()
    elif reduction == "none":
        loss = sq
    else:
        raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

    if return_predictions:
        return loss, y_pred, residuals
    return loss
