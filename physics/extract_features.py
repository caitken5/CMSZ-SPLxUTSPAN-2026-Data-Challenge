"""Feature extraction utilities for physics simulations.

Provides utilities to estimate a ball's average linear velocity, rotational
velocity (angular velocity), and position from sets of 3D points associated
with the left and right hands across time. The estimation assumes the points
belong to a rigid body (the ball surface) so a rigid-body velocity fit is
performed per timestep and then averaged.

The primary function is ``estimate_ball_motion``.
"""

from typing import Optional, Tuple

import numpy as np
from physics.constants import X_COURT, Y_COURT, Z_RIM


def _skew(r: np.ndarray) -> np.ndarray:
	"""Return the 3x3 skew-symmetric matrix for vector r.

	r: (..., 3)
	returns: (..., 3, 3)
	"""
	rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]
	mat = np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])
	# If r was batched, broadcast accordingly
	if r.ndim == 1:
		return mat
	# r is (...,3) — create stacked matrices
	mats = np.empty(r.shape[:-1] + (3, 3), dtype=float)
	mats[..., 0, 0] = 0.0
	mats[..., 0, 1] = -r[..., 2]
	mats[..., 0, 2] = r[..., 1]
	mats[..., 1, 0] = r[..., 2]
	mats[..., 1, 1] = 0.0
	mats[..., 1, 2] = -r[..., 0]
	mats[..., 2, 0] = -r[..., 1]
	mats[..., 2, 1] = r[..., 0]
	mats[..., 2, 2] = 0.0
	return mats


def estimate_ball_motion(
	left_points: np.ndarray,
	right_points: Optional[np.ndarray] = None,
	timestamps: Optional[np.ndarray] = None,
	return_per_frame: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Estimate average linear velocity, angular velocity, and position of the ball.

	Parameters
	- left_points: array_like, shape (T, N_left, 3) or (T, P, 3)
		3D points associated with the left hand over T timesteps.
	- right_points: optional, array_like, shape (T, N_right, 3)
		3D points associated with the right hand. If provided, points will be
		concatenated with `left_points` along the point axis.
	- timestamps: optional, array_like, shape (T,)
		Timestamps for frames; if omitted, a unit spacing (dt=1.0) is used.
	- return_per_frame: if True, also return per-timestep estimates as arrays.

	Returns
	- v_avg: (3,) average linear velocity vector (m/s or units/frame if timestamps omitted)
	- omega_avg: (3,) average angular velocity vector (rad/s or rad/frame)
	- pos_avg: (3,) average position (centroid) across frames

	Notes
	The method computes per-timestep rigid-body fits solving v_i = v_cm + omega x r_i
	in a least-squares sense for the observed point velocities, then averages the
	resulting `v_cm` and `omega` over timesteps.
	"""

	# Combine left and right points
	L = np.asarray(left_points)
	if right_points is not None:
		R = np.asarray(right_points)
		points = np.concatenate([L, R], axis=1)
	else:
		points = L

	if points.ndim != 3 or points.shape[2] != 3:
		raise ValueError("points must have shape (T, P, 3)")

	T, P, _ = points.shape

	if timestamps is None:
		timestamps = np.arange(T, dtype=float)
	timestamps = np.asarray(timestamps, dtype=float)
	if timestamps.shape[0] != T:
		raise ValueError("timestamps length must match first dim of points")

	# Per-timestep results
	v_cms = []
	omegas = []
	centers = []

	for t in range(T - 1):
		dt = timestamps[t + 1] - timestamps[t]
		if dt <= 0:
			raise ValueError("timestamps must be strictly increasing")

		pts_t = points[t]      # (P,3)
		pts_t1 = points[t + 1] # (P,3)

		# centroid at time t
		c = pts_t.mean(axis=0)
		centers.append(c)

		# point velocities
		v_pts = (pts_t1 - pts_t) / dt  # (P,3)

		# Build linear system A x = b where x = [v_cm (3), omega (3)]
		# For each point: v_i = v_cm + omega x r_i = [I, -R_i] [v_cm; omega]
		Rmats = _skew(pts_t - c)  # (P,3,3)

		A = np.zeros((3 * P, 6), dtype=float)
		b = v_pts.reshape(3 * P)

		for i in range(P):
			# Row block for point i
			A[3 * i: 3 * i + 3, 0:3] = np.eye(3)
			A[3 * i: 3 * i + 3, 3:6] = -Rmats[i]

		# Solve least squares
		x, *_ = np.linalg.lstsq(A, b, rcond=None)
		v_cm = x[0:3]
		omega = x[3:6]

		v_cms.append(v_cm)
		omegas.append(omega)

	# For position average include last frame centroid as well
	centers.append(points[-1].mean(axis=0))

	v_cms = np.vstack(v_cms) if v_cms else np.zeros((0, 3))
	omegas = np.vstack(omegas) if omegas else np.zeros((0, 3))
	centers = np.vstack(centers)

	v_avg = v_cms.mean(axis=0) if v_cms.size else np.zeros(3)
	omega_avg = omegas.mean(axis=0) if omegas.size else np.zeros(3)
	pos_avg = centers.mean(axis=0)

	if return_per_frame:
		return v_avg, omega_avg, pos_avg, v_cms, omegas, centers

	return v_avg, omega_avg, pos_avg

# TODO: Check if this scaling works well.
def scale_kinematics(
	positions: np.ndarray,
	velocities: Optional[np.ndarray] = None,
	angular_velocities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
	"""Scale kinematic arrays relative to court/rim dimensions.

	Parameters
	- positions: array_like with last dimension 3 (x,y,z)
	- velocities: optional array_like with last dimension 3
	- angular_velocities: optional array_like (returned unchanged)

	Returns (pos_scaled, vel_scaled, ang_scaled)
	- pos_scaled: positions with x/X_COURT, y/Y_COURT, z/Z_RIM
	- vel_scaled: velocities scaled by same per-axis factors (or None)
	- ang_scaled: angular_velocities passed through (or None)

	This is useful to nondimensionalize kinematic data by typical court/rim
	length scales. Inputs may be numpy arrays; they will be converted if not.
	"""
	pos = np.asarray(positions)
	if pos.shape[-1] != 3:
		raise ValueError("positions must have last dimension size 3 (x,y,z)")

	# per-axis scales (divide by these to normalize into [0,1] court-relative units)
	scales = np.array([1.0 / X_COURT, 1.0 / Y_COURT, 1.0 / Z_RIM], dtype=float)

	pos_scaled = pos * scales

	vel_scaled = None
	if velocities is not None:
		vel = np.asarray(velocities)
		if vel.shape[-1] != 3:
			raise ValueError("velocities must have last dimension size 3 (vx,vy,vz)")
		vel_scaled = vel * scales

	ang_scaled = None
	if angular_velocities is not None:
		ang_scaled = np.asarray(angular_velocities)

	return pos_scaled, vel_scaled, ang_scaled

