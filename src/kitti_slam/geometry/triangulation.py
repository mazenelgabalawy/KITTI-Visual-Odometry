"""Stereo triangulation and depth-uncertainty helpers ported from VO_constrained.py."""

from __future__ import annotations

import cv2
import numpy as np

from kitti_slam.geometry.calibration import StereoCalibration


def triangulate_stereo(calib: StereoCalibration, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    """Replicates VO_constrained.py L120-121 exactly.

    ``pts_left``/``pts_right`` are (N, 2) pixel coordinate arrays.
    Returns an (N, 3) array of triangulated 3D points.
    """
    points_4d = cv2.triangulatePoints(calib.P0, calib.P1, pts_left.T, pts_right.T)
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T
    return points_3d


def stereo_depth_sigma(calib: StereoCalibration, depths: np.ndarray) -> np.ndarray:
    """Replicates VO_constrained.py L124-126: sigma_z = depths**2 / (focal * baseline)."""
    return np.square(depths) / (calib.focal_length * calib.baseline)
