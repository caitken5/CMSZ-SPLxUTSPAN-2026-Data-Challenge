"""Events module."""

import torch

from physics.constants import Z_RIM

def hoop_plane_event(state, hoop_center, hoop_normal):
    """
    Returns signed distance to hoop plane.
    Zero-crossing = intersection.
    """
    pos = state[:, 2]
    return pos - Z_RIM

def resolve_z_plane_crossing(
    t0, s0,
    t1, s1,
    z_plane= Z_RIM,
):
    z0 = s0[2]
    z1 = s1[2]

    # Safety check
    assert (z0 - z_plane) * (z1 - z_plane) <= 0, \
        "No plane crossing in this interval"

    alpha = (z_plane - z0) / (z1 - z0)

    t_cross = t0 + alpha * (t1 - t0)
    s_cross = s0 + alpha * (s1 - s0)

    return t_cross, s_cross