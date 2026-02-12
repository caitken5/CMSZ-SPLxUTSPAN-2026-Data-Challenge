'''rollout.py'''

import torch
from .constants import Z_RIM
from .dynamics import ball_dynamics
from .events import resolve_z_plane_crossing

def rollout_to_plane(
        # convert z_plane to metres
    s0, params, z_plane=Z_RIM, dt = 1e-3, t_max = 3.0):
    """
    Roll out dynamics until the ball cross the z-plane.
    """

    s = s0.clone()
    t = 0.0

    while t < t_max:
        s_next = s + dt * ball_dynamics(t, s.unsqueeze(0), params).squeeze(0)

        if (s[2] - z_plane) * (s_next[2] - z_plane) <= 0:
            t_hit, s_hit = resolve_z_plane_crossing(
                t0=t,
                s0=s,
                t1=t + dt,
                s1=s_next,
                z_plane=z_plane,
            )
            return t_hit, s_hit

        s = s_next
        t += dt

    raise RuntimeError("Ball never crossed plane")