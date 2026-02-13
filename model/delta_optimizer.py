"""Delta-parameter optimizer using embedded physics rollout.

This module provides a minimal trainable model (three scalar parameters)
that learns aerodynamic adjustments by minimizing nonlinear least-squares
error between predicted rollout intersections and observed targets.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import physics.constants as const
from physics.constants import AIR_DENSITY, BALL_AREA, BALL_MASS, DRAG_COEFFICIENT, Z_RIM
from physics.dynamics import compute_alphas, compute_lift_coefficient
from physics.rollout import (
    rollout_to_plane,
    scale_targets_fixed_ranges,
    state_at_plane_to_scaled_targets,
    state_at_plane_to_targets,
)


@dataclass
class DeltaOptimizationResult:
    losses: list[float]
    delta_rho: float
    delta_D: float
    delta_k: float


SPIN_FACTOR_DEFAULT = float(getattr(const, "SPIN_FACTOR", 1.2))
CL_MAX_DEFAULT = float(getattr(const, "CL_MAX", 0.4))


class DeltaParameterModel(nn.Module):
    """Small trainable model for aerodynamic deltas.

    Learnable parameters:
      - delta_rho: density adjustment (via exp in compute_alphas)
      - delta_D: drag-coefficient adjustment
      - delta_k: spin-to-lift gain adjustment
    """

    def __init__(self, delta_rho_init: float = 0.0, delta_D_init: float = 0.0, delta_k_init: float = 0.0):
        super().__init__()
        self.delta_rho = nn.Parameter(torch.tensor(float(delta_rho_init), dtype=torch.float32))
        self.delta_D = nn.Parameter(torch.tensor(float(delta_D_init), dtype=torch.float32))
        self.delta_k = nn.Parameter(torch.tensor(float(delta_k_init), dtype=torch.float32))

    def build_params(self, state_batch: torch.Tensor) -> dict:
        """Build dynamics params dict for ``ball_dynamics`` / ``rollout_to_plane``."""
        # Positive drag coefficient around baseline.
        cd = DRAG_COEFFICIENT + F.softplus(self.delta_D)

        # Positive spin gain around baseline.
        spin_gain = SPIN_FACTOR_DEFAULT * torch.exp(self.delta_k)

        cl = compute_lift_coefficient(
            spin_ratio=spin_gain,
            Cl_max=CL_MAX_DEFAULT,
            state=state_batch,
        )

        alpha_d, alpha_l = compute_alphas(
            delta_rho=self.delta_rho,
            Cd=cd,
            Cl=cl,
            A=BALL_AREA,
            m0=BALL_MASS,
            rho0=AIR_DENSITY,
        )

        return {"alpha_d": alpha_d, "alpha_l": alpha_l}


def train_delta_parameters(
    initial_states: torch.Tensor,
    targets_at_plane: torch.Tensor,
    target_indices: Iterable[int] = (0, 1),
    prediction_space: Literal["xy", "scaled_metrics"] = "xy",
    epochs: int = 200,
    lr: float = 1e-2,
    z_plane: float = Z_RIM,
    dt: float = 1e-3,
    t_max: float = 3.0,
    model: Optional[DeltaParameterModel] = None,
    verbose: bool = True,
) -> DeltaOptimizationResult:
    """Train deltas with nonlinear least squares and embedded ODE rollout.

        Objective:
            minimize sum_i || y_hat_i - y_i ||^2 where y_hat_i is obtained by
            integrating the ODE with ``rollout_to_plane``.

    Args:
        initial_states: (N, 9) tensor of launch states.
        targets_at_plane: (N, D) tensor of observed targets at z_plane.
                target_indices: state indices to compare when prediction_space="xy".
                prediction_space:
                    - "xy": compare selected state components at plane crossing.
                    - "scaled_metrics": compare scaled [angle, depth, left_right].
        epochs: optimization epochs.
        lr: learning rate.
        z_plane: plane for rollout intersection.
        dt: integration step size.
        t_max: max integration time.
        model: optional pre-created DeltaParameterModel.
        verbose: print progress.
    """
    if initial_states.ndim != 2 or initial_states.shape[1] != 9:
        raise ValueError("initial_states must have shape (N, 9)")

    idx = tuple(target_indices)
    if targets_at_plane.ndim != 2 or targets_at_plane.shape[0] != initial_states.shape[0]:
        raise ValueError("targets_at_plane must have shape (N, D) with matching N")
    expected_dim = 3 if prediction_space == "scaled_metrics" else len(idx)
    if targets_at_plane.shape[1] != expected_dim:
        raise ValueError("targets_at_plane second dimension does not match prediction_space")

    model = model if model is not None else DeltaParameterModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        sample_losses = []

        for i in range(initial_states.shape[0]):
            s0 = initial_states[i]
            state_batch = s0.unsqueeze(0)
            params = model.build_params(state_batch)

            try:
                _, s_hit = rollout_to_plane(s0=s0, params=params, z_plane=z_plane, dt=dt, t_max=t_max)
            except RuntimeError:
                # Skip trajectories that fail to cross the plane in this step.
                continue

            if prediction_space == "scaled_metrics":
                pred = state_at_plane_to_scaled_targets(s_hit, clamp=True)
            else:
                pred = s_hit[list(idx)]
            target = targets_at_plane[i]
            sample_losses.append(torch.sum((pred - target) ** 2))

        if not sample_losses:
            raise RuntimeError("No valid trajectories crossed the plane during training.")

        loss = torch.stack(sample_losses).mean()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(
                f"epoch={epoch:03d} loss={loss_val:.6f} "
                f"delta_rho={model.delta_rho.item():+.4f} "
                f"delta_D={model.delta_D.item():+.4f} "
                f"delta_k={model.delta_k.item():+.4f}"
            )

    return DeltaOptimizationResult(
        losses=losses,
        delta_rho=float(model.delta_rho.detach().cpu().item()),
        delta_D=float(model.delta_D.detach().cpu().item()),
        delta_k=float(model.delta_k.detach().cpu().item()),
    )
