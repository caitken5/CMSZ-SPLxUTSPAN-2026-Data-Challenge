"""Physics constants."""

import torch
import math

g = torch.tensor([0.0, 0.0, -9.81]) # gravity (m/s^2)

# Regulation basketball priors
BALL_MASS = 0.6000      # kg
BALL_RADIUS = 0.1194    # m
BALL_AREA = torch.pi * BALL_RADIUS**2

AIR_DENSITY = 1.225     # kg/m^3
DRAG_COEFFICIENT = 0.47 # estimate for sphere


# Length
FEET_TO_METERS = 0.3048
METERS_TO_FEET = 1.0 / FEET_TO_METERS

INCHES_TO_METERS = 0.0254
METERS_TO_INCHES = 1.0 / INCHES_TO_METERS

# Positions of hoop center relative to top left corner of court.
R_HOOP_WORLD = torch.tensor([5.25, -25.0, 10.0], dtype=torch.float32) * FEET_TO_METERS  # vector provided by competition (meters)
HOOP_DIAMETER = 18.0 * INCHES_TO_METERS # 18 inches


Z_RIM = 10*FEET_TO_METERS

# Transformation from WORLD frame to RIM (hoop) frame.
# The rim frame is right-handed and rotated counter-clockwise 90° about +z
# relative to the world frame. If the rim axes are rotated by +theta relative
# to world, the coordinates of a world vector in the rim frame are obtained
# with a rotation by -theta (v_rim = R(-theta) @ (v_world - R_HOOP_WORLD)).
# For theta = +pi/2, use -pi/2 here so +x_world -> [0, -1, 0]_rim as expected.
R_WORLD_TO_RIM = torch.tensor([
	[0.0, 1.0, 0.0],
	[-1.0,  0.0, 0.0],
	[0.0,  0.0, 1.0],
], dtype=torch.float32)

# Homogeneous transform (4x4) from world to rim coordinates: x_rim = R*(x_world) + t
# where t = -R @ R_HOOP_WORLD
T_WORLD_TO_RIM = torch.eye(4, dtype=torch.float32)
T_WORLD_TO_RIM[:3, :3] = R_WORLD_TO_RIM
T_WORLD_TO_RIM[:3, 3] = -R_WORLD_TO_RIM @ R_HOOP_WORLD

# Useful scales (for nondimensionalization if needed)
L_SCALE = 1.0           # meters
T_SCALE = 1.0           # seconds