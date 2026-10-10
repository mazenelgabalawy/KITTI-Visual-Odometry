"""Tests for kitti_slam.geometry.triangulation."""

import numpy as np

from kitti_slam.geometry.calibration import StereoCalibration
from kitti_slam.geometry.triangulation import stereo_depth_sigma, triangulate_stereo

# Same KITTI seq-00 P0/P1 values used in test_calibration.py.
P0 = np.array(
    [
        [718.856, 0.0, 607.1928, 0.0],
        [0.0, 718.856, 185.2157, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
P1 = np.array(
    [
        [718.856, 0.0, 607.1928, -386.1448],
        [0.0, 718.856, 185.2157, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_h = np.append(X, 1.0)
    p_h = P @ X_h
    return p_h[:2] / p_h[2]


def test_triangulate_known_points():
    calib = StereoCalibration.from_projection_matrices(P0, P1)

    rng = np.random.default_rng(42)
    n = 10
    depths = np.linspace(5.0, 50.0, n)
    xs = rng.uniform(-5.0, 5.0, size=n)
    ys = rng.uniform(-2.0, 2.0, size=n)
    points_3d = np.stack([xs, ys, depths], axis=1)

    pts_left = np.array([_project(P0, X) for X in points_3d])
    pts_right = np.array([_project(P1, X) for X in points_3d])

    recovered = triangulate_stereo(calib, pts_left, pts_right)

    np.testing.assert_allclose(recovered, points_3d, atol=1e-6)


def test_depth_sigma_formula():
    calib = StereoCalibration.from_projection_matrices(P0, P1)
    depths = np.array([5.0, 10.0, 25.0, 50.0])

    expected = np.square(depths) / (calib.focal_length * calib.baseline)
    actual = stereo_depth_sigma(calib, depths)

    np.testing.assert_array_equal(actual, expected)
