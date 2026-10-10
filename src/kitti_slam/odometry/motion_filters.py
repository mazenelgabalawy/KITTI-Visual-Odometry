"""Planar-motion rotation refinement and translation/heading plausibility gating.

The refinement blends the closest proper rotation with its best yaw-only
(planar) rotation and re-orthonormalizes. The gates reject a relative motion
whose translation or heading change exceeds the configured limit; the heading
limit is configured as `max_pitch_deg` (it gates the rotation about camera y).
"""

from __future__ import annotations

import numpy as np

from kitti_slam.config.schema import MotionConfig
from kitti_slam.geometry.transforms import rotation_to_euler
from kitti_slam.odometry.base import MotionFilter


def _orthogonalize_to_planar_motion(R: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Pull R toward the closest rotation about camera y (yaw only):
    alpha * R_planar + (1 - alpha) * R_ortho, re-orthonormalized."""
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt  # Closest orthonormal matrix
    if np.linalg.det(R_ortho) < 0:
        R_ortho = U @ np.diag([1, 1, -1]) @ Vt  # Ensure proper rotation (det=1)

    # Extract elements from R_ortho (KITTI: x-right, y-down, z-forward; yaw around y)
    a = R_ortho[0, 0] + R_ortho[2, 2]  # For cos theta
    b = R_ortho[0, 2] - R_ortho[2, 0]  # For sin theta
    theta_opt = np.arctan2(b, a)  # Optimal yaw

    cos_theta = np.cos(theta_opt)
    sin_theta = np.sin(theta_opt)
    R_planar = np.array(
        [
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta],
        ]
    )

    # Blend with original to allow small pitch/roll (soft constraint: alpha=0.8 pulls toward planar)
    R_planar = alpha * R_planar + (1 - alpha) * R_ortho

    # Re-orthonormalize the blended R (optional but recommended)
    U, S, Vt = np.linalg.svd(R_planar)
    R_planar = U @ Vt
    if np.linalg.det(R_planar) < 0:
        R_planar = U @ np.diag([1, 1, -1]) @ Vt

    return R_planar


class PlanarMotionFilter(MotionFilter):
    """Planar rotation refinement plus translation and heading gates."""

    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg

    def refine_rotation(self, R: np.ndarray) -> np.ndarray:
        if self.cfg.orthogonalize_alpha is None:
            # Identity passthrough: no orthogonalization applied.
            return R
        return _orthogonalize_to_planar_motion(R, alpha=self.cfg.orthogonalize_alpha)

    def is_plausible(self, R_world: np.ndarray, t_world: np.ndarray) -> bool:
        return self.rejection_reason(R_world, t_world) is None

    def rejection_reason(self, R_world: np.ndarray, t_world: np.ndarray) -> str | None:
        # Plausible iff norm(t) <= max_translation and |heading| <= max_pitch_deg, where
        # the heading is the second Euler component of rotation_to_euler: the rotation
        # about camera y (the setting's name says "pitch"; see BUG-012).
        if self.cfg.max_translation is not None and np.linalg.norm(t_world) > self.cfg.max_translation:
            return "implausible_translation"

        if self.cfg.max_pitch_deg is not None and abs(rotation_to_euler(R_world)[1]) > np.deg2rad(
            self.cfg.max_pitch_deg
        ):
            return "implausible_heading"

        return None


class NullMotionFilter(MotionFilter):
    """No-op motion filter: passes rotation through unchanged, always plausible."""

    def refine_rotation(self, R: np.ndarray) -> np.ndarray:
        return R

    def is_plausible(self, R_world: np.ndarray, t_world: np.ndarray) -> bool:
        return True

    def rejection_reason(self, R_world: np.ndarray, t_world: np.ndarray) -> str | None:
        return None
