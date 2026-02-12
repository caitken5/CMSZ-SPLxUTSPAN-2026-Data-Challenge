"""Geometry module."""

import torch

def make_hoop_plane(hoop_center, hoop_forward):
    hoop_forward = hoop_forward / torch.linalg.norm(hoop_forward)
    return hoop_center, hoop_forward