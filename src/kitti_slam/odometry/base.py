"""Abstract interfaces for pose estimation and landmark/motion filtering.

These interfaces decouple the estimation components from the pipeline wiring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from kitti_slam.geometry.calibration import StereoCalibration


@dataclass
class PoseEstimate:
    R: np.ndarray | None
    t: np.ndarray | None
    inliers: np.ndarray | None
    success: bool
    retval: bool | None = None  # the solver's own return value, when a solver ran


class PoseEstimator(ABC):
    """Estimates a camera-frame pose from 3D-2D correspondences."""

    @abstractmethod
    def estimate(self, points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> PoseEstimate: ...


class LandmarkFilter(ABC):
    """Filters triangulated 3D landmarks (e.g. by depth/uncertainty)."""

    @abstractmethod
    def valid_mask(self, points_3d: np.ndarray, calib: StereoCalibration) -> np.ndarray: ...


class MotionFilter(ABC):
    """Constrains/validates the estimated motion (e.g. planar-motion gating)."""

    @abstractmethod
    def refine_rotation(self, R: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def is_plausible(self, R_world: np.ndarray, t_world: np.ndarray) -> bool: ...

    def rejection_reason(self, R_world: np.ndarray, t_world: np.ndarray) -> str | None:
        """None if the motion is plausible, else the diagnostics outcome naming the
        gate that rejected it."""
        return None if self.is_plausible(R_world, t_world) else "implausible_motion"
