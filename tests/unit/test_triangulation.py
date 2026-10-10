"""Tests for kitti_slam.geometry.triangulation."""

import numpy as np
import pytest

from kitti_slam.geometry.calibration import StereoCalibration
from kitti_slam.geometry.triangulation import (
    stereo_depth_sigma,
    triangulate_stereo,
    triangulate_stereo_homogeneous,
    valid_landmarks,
)

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


# ---------------------------------------------------------------------------
# Landmark validity (BUG-003)
# ---------------------------------------------------------------------------

_F, _B, _CX, _CY = 700.0, 0.5, 320.0, 240.0
_P0 = np.array([[_F, 0.0, _CX, 0.0], [0.0, _F, _CY, 0.0], [0.0, 0.0, 1.0, 0.0]])
_P1 = np.array([[_F, 0.0, _CX, -_F * _B], [0.0, _F, _CY, 0.0], [0.0, 0.0, 1.0, 0.0]])
_CALIB = StereoCalibration.from_projection_matrices(_P0, _P1)


def _pixels(disparities, u=400.0, v=250.0):
    left = np.array([[u, v]] * len(disparities), dtype=np.float64)
    right = left - np.c_[np.asarray(disparities, dtype=np.float64), np.zeros(len(disparities))]
    return left, right


def test_homogeneous_points_are_unit_norm_with_w_about_one_over_range():
    left, right = _pixels([35.0, 3.5, 0.35])  # depths 10, 100, 1000 m
    points_h = triangulate_stereo_homogeneous(_CALIB, left, right)
    np.testing.assert_allclose(np.linalg.norm(points_h, axis=1), 1.0, atol=1e-9)
    ranges = np.linalg.norm(triangulate_stereo(_CALIB, left, right), axis=1)
    np.testing.assert_allclose(np.abs(points_h[:, 3]), 1.0 / np.sqrt(ranges**2 + 1.0), rtol=1e-6)


def test_valid_points_pass_in_order():
    left, right = _pixels([35.0, 7.0, 3.5])
    mask, points_3d = valid_landmarks(_CALIB, triangulate_stereo_homogeneous(_CALIB, left, right), 1e-4)
    assert mask.tolist() == [True, True, True]
    np.testing.assert_allclose(points_3d[:, 2], [10.0, 50.0, 100.0], rtol=1e-6)


@pytest.mark.parametrize(
    ("disparity", "reason"),
    [(-5.0, "behind both cameras"), (0.0, "zero disparity: point at infinity"), (0.01, "beyond 1/min_abs_w")],
)
def test_degenerate_points_are_rejected(disparity, reason):
    left, right = _pixels([35.0, disparity])
    mask, _ = valid_landmarks(_CALIB, triangulate_stereo_homogeneous(_CALIB, left, right), 1e-4)
    assert mask.tolist() == [True, False], reason


def test_nonfinite_homogeneous_points_are_rejected():
    points_h = np.array([[0.0, 0.0, 0.5, 0.05], [np.nan, 0.0, 1.0, 0.1], [0.0, 0.0, 1.0, np.inf], [0.0, 0.0, 1.0, 0.0]])
    mask, _ = valid_landmarks(_CALIB, points_h, 0.0)
    assert mask.tolist() == [True, False, False, False]


def test_depth_must_be_positive_in_both_cameras():
    # A right camera 20 m ahead of the left one: a point 10 m in front of the
    # left camera is behind the right one.
    P1_ahead = np.array([[_F, 0.0, _CX, -_F * _B], [0.0, _F, _CY, 0.0], [0.0, 0.0, 1.0, -20.0]])
    calib = StereoCalibration(P0=_P0, P1=P1_ahead, K=_P0[:, :3], baseline=_B, focal_length=_F)
    points_h = np.array([[0.0, 0.0, 10.0, 1.0], [0.0, 0.0, 30.0, 1.0]])
    points_h /= np.linalg.norm(points_h, axis=1, keepdims=True)
    mask, _ = valid_landmarks(calib, points_h, 1e-4)
    assert mask.tolist() == [False, True]
