"""Stereo triangulation, landmark validity, and stereo depth uncertainty."""

from __future__ import annotations

import cv2
import numpy as np

from kitti_slam.geometry.calibration import StereoCalibration


def triangulate_stereo_homogeneous(calib: StereoCalibration, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    """(N, 4) homogeneous points from (N, 2) left/right pixel arrays.

    cv2.triangulatePoints returns the unit-norm DLT solution, so |w| is about
    1/range (1/m) for a point at range >> 1 m.
    """
    return cv2.triangulatePoints(calib.P0, calib.P1, pts_left.T, pts_right.T).T


def triangulate_stereo(calib: StereoCalibration, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    """(N, 3) Euclidean points. No validity check."""
    points_h = triangulate_stereo_homogeneous(calib, pts_left, pts_right)
    return points_h[:, :3] / points_h[:, 3:4]


def valid_landmarks(calib: StereoCalibration, points_h: np.ndarray, min_abs_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Landmark validity check (BUG-003) on (N, 4) homogeneous points.

    Returns (mask, points_3d): points_3d is the Euclidean division of every
    point (nonfinite where w is 0), and mask keeps a point only if its
    homogeneous and Euclidean coordinates are finite, |w| >= min_abs_w and
    w != 0, and its depth is positive in both cameras. Order is preserved.
    """
    w = points_h[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        points_3d = points_h[:, :3] / points_h[:, 3:4]
        points_h1 = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        depth_left = points_h1 @ calib.P0[2]
        depth_right = points_h1 @ calib.P1[2]
        mask = np.isfinite(points_h).all(axis=1) & np.isfinite(points_3d).all(axis=1)
        mask &= (w != 0) & (np.abs(w) >= min_abs_w)
        mask &= (depth_left > 0) & (depth_right > 0)
    return mask, points_3d


def stereo_depth_sigma(calib: StereoCalibration, depths: np.ndarray) -> np.ndarray:
    """Depth standard deviation for 1 px of disparity error: depths**2 / (focal * baseline)."""
    return np.square(depths) / (calib.focal_length * calib.baseline)
