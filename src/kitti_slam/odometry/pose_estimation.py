"""PnP pose estimation and depth-uncertainty landmark filtering.

Ports:
- solvePnPRansac call site: VO_constrained.py L158 and VO_base.py L119-120
  (identical kwargs in both: confidence=pnp_confidence,
  iterationsCount=pnp_iterations_count -- no other kwargs differ between the
  two legacy call sites, so PnPConfig's existing `confidence` /
  `iterations_count` fields are sufficient; nothing extra was invented).
- <4 points guard: VO_constrained.py L136 / VO_base.py L106.
- Depth/uncertainty landmark filter: VO_constrained.py L123-128.
"""

from __future__ import annotations

import cv2
import numpy as np

from kitti_slam.config.schema import LandmarkFilterConfig, PnPConfig
from kitti_slam.geometry.calibration import StereoCalibration
from kitti_slam.geometry.triangulation import stereo_depth_sigma
from kitti_slam.odometry.base import LandmarkFilter, PoseEstimate, PoseEstimator


class PnPRansacEstimator(PoseEstimator):
    """Wraps cv2.solvePnPRansac exactly as called in VO_constrained.py L158
    (and identically in VO_base.py L119-120)."""

    def __init__(self, cfg: PnPConfig) -> None:
        self.cfg = cfg

    def estimate(self, points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> PoseEstimate:
        # Legacy guard: VO_constrained.py L136 / VO_base.py L106.
        if points_3d.shape[0] < 4:
            return PoseEstimate(R=None, t=None, inliers=None, success=False)

        points_3d = points_3d.astype(np.float32)
        points_2d = points_2d.astype(np.float32)

        if self.cfg.seed is not None:
            # Set immediately before the call so each estimate() invocation
            # is independently reproducible.
            cv2.setRNGSeed(self.cfg.seed)

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            None,
            confidence=self.cfg.confidence,
            iterationsCount=self.cfg.iterations_count,
        )

        # BUG-002: honor the solver's return value and validate its outputs. A
        # rejected solve is a failed registration; `inliers` is kept either way.
        retval = bool(retval)
        if not retval or rvec is None or tvec is None:
            return PoseEstimate(R=None, t=None, inliers=inliers, success=False, retval=retval)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        num_inliers = 0 if inliers is None else len(inliers)
        if not (np.isfinite(R).all() and np.isfinite(t).all()) or num_inliers < self.cfg.min_inliers:
            return PoseEstimate(R=None, t=None, inliers=inliers, success=False, retval=retval)

        return PoseEstimate(R=R, t=t, inliers=inliers, success=True, retval=retval)


class DepthUncertaintyFilter(LandmarkFilter):
    """Ports VO_constrained.py L123-128 exactly."""

    def __init__(self, cfg: LandmarkFilterConfig) -> None:
        self.cfg = cfg

    def valid_mask(self, points_3d: np.ndarray, calib: StereoCalibration) -> np.ndarray:
        depths = points_3d[:, 2]
        sigma_z = stereo_depth_sigma(calib, depths)

        valid_idx = points_3d[:, 2] > self.cfg.min_depth_multiplier * calib.baseline
        valid_idx = valid_idx & (sigma_z / depths < self.cfg.relative_depth_uncertainty_threshold)
        return valid_idx
