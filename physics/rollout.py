'''rollout.py'''

import torch
from .constants import METERS_TO_INCHES, Z_RIM, T_WORLD_TO_RIM
from .dynamics import ball_dynamics
from .events import resolve_z_plane_crossing


ANGLE_RANGE = (30.0, 60.0)
DEPTH_RANGE = (-12.0, 30.0)
LEFT_RIGHT_RANGE = (-16.0, 16.0)


def _world_state_to_rim_state(s_world: torch.Tensor) -> torch.Tensor:
    """Convert a 9D state from world frame to rim frame.

    state format: [x, y, z, vx, vy, vz, wx, wy, wz]
    """
    s_rim = s_world.clone()

    # Position: x_rim = R*x_world + t (homogeneous transform)
    pos_h = torch.cat([s_world[0:3], torch.tensor([1.0], dtype=s_world.dtype, device=s_world.device)])
    s_rim[0:3] = (T_WORLD_TO_RIM @ pos_h)[0:3]

    # Velocity and angular velocity: rotate only
    R = T_WORLD_TO_RIM[:3, :3]
    s_rim[3:6] = R @ s_world[3:6]
    s_rim[6:9] = R @ s_world[6:9]

    return s_rim

def rollout_to_plane(
        # convert z_plane to metres
    s0, params, z_plane=Z_RIM, dt = 1e-3, t_max = 3.0):
    """
    Roll out dynamics until the ball DESCENDS through z_plane,
    then return rim-frame state.

    Returns
    -------
    t_hit : float
    s_hit_rim : torch.Tensor, shape (9,)
        State at crossing expressed in rim frame.
    """

    s = s0.clone()
    t = 0.0

    while t < t_max:
        s_next = s + dt * ball_dynamics(t, s.unsqueeze(0), params).squeeze(0)

        # Descending crossing only: above/equal -> below/equal.
        if (s[2] >= z_plane) and (s_next[2] <= z_plane):
            t_hit, s_hit = resolve_z_plane_crossing(
                t0=t,
                s0=s,
                t1=t + dt,
                s1=s_next,
                z_plane=z_plane,
            )
            s_hit_rim = _world_state_to_rim_state(s_hit)
            return t_hit, s_hit_rim

        s = s_next
        t += dt

    raise RuntimeError("Ball never descended through plane")


def state_at_plane_to_targets(s_hit: torch.Tensor) -> torch.Tensor:
    """Convert a rim-frame hit state to native metrics.

    Returns tensor shape (3,):
        [angle_deg, depth_inches, left_right_inches]
    """
    pos_rim = s_hit[0:3]
    vel_rim = s_hit[3:6]

    left_right_inches = pos_rim[0] * METERS_TO_INCHES
    depth_inches = pos_rim[1] * METERS_TO_INCHES

    horiz_speed = torch.linalg.norm(vel_rim[0:2])
    angle_deg = torch.atan2(-vel_rim[2], horiz_speed + 1e-8) * (180.0 / torch.pi)

    return torch.stack([angle_deg, depth_inches, left_right_inches])


def scale_targets_fixed_ranges(targets: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Min-max scale [angle, depth, left_right] using fixed ranges."""
    mins = torch.tensor(
        [ANGLE_RANGE[0], DEPTH_RANGE[0], LEFT_RIGHT_RANGE[0]],
        dtype=targets.dtype,
        device=targets.device,
    )
    maxs = torch.tensor(
        [ANGLE_RANGE[1], DEPTH_RANGE[1], LEFT_RIGHT_RANGE[1]],
        dtype=targets.dtype,
        device=targets.device,
    )
    scaled = (targets - mins) / (maxs - mins)
    return torch.clamp(scaled, 0.0, 1.0) if clamp else scaled


def state_at_plane_to_scaled_targets(s_hit: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Convert rim-frame hit state directly to scaled [angle, depth, left_right]."""
    native = state_at_plane_to_targets(s_hit)
    return scale_targets_fixed_ranges(native, clamp=clamp)