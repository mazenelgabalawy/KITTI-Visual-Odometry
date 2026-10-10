"""Tests for kitti_slam.odometry.pose_estimation."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kitti_slam.config.schema import LandmarkFilterConfig, PnPConfig
from kitti_slam.geometry.calibration import StereoCalibration
from kitti_slam.odometry.pose_estimation import DepthUncertaintyFilter, PnPRansacEstimator

# Real KITTI seq-00 P0/P1 (same as test_calibration.py / test_triangulation.py).
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
K = P0[:, :3]


def _project(K, points_3d_cam):
    """Projects camera-frame 3D points with intrinsic matrix K (3x3)."""
    p = points_3d_cam @ K.T
    return p[:, :2] / p[:, 2:3]


def test_pnp_recovers_known_pose():
    rng = np.random.default_rng(42)
    n = 100
    xs = rng.uniform(-5, 5, size=n)
    ys = rng.uniform(-2, 2, size=n)
    depths = rng.uniform(5, 50, size=n)
    points_3d_world = np.stack([xs, ys, depths], axis=1)

    # Known small-yaw rotation and translation (camera frame convention:
    # X_cam = R @ X_world + t).
    R_true = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    t_true = np.array([0.5, -0.1, 1.2])

    points_cam = (R_true @ points_3d_world.T).T + t_true
    points_2d = _project(K, points_cam)

    cfg = PnPConfig(confidence=0.999, iterations_count=1000, min_inliers=6, seed=1234)
    estimator = PnPRansacEstimator(cfg)

    result = estimator.estimate(points_3d_world, points_2d, K)

    assert result.success
    rvec_true, _ = cv2.Rodrigues(R_true)
    rvec_recovered, _ = cv2.Rodrigues(result.R)
    np.testing.assert_allclose(rvec_recovered.reshape(3), rvec_true.reshape(3), atol=1e-4)
    np.testing.assert_allclose(result.t, t_true, atol=1e-4)


def test_pnp_deterministic_with_seed():
    rng = np.random.default_rng(7)
    n = 50
    xs = rng.uniform(-5, 5, size=n)
    ys = rng.uniform(-2, 2, size=n)
    depths = rng.uniform(5, 50, size=n)
    points_3d = np.stack([xs, ys, depths], axis=1)

    R_true = Rotation.from_euler("y", -3, degrees=True).as_matrix()
    t_true = np.array([0.2, 0.05, 0.8])
    points_cam = (R_true @ points_3d.T).T + t_true
    points_2d = _project(K, points_cam)

    cfg = PnPConfig(confidence=0.999, iterations_count=500, min_inliers=6, seed=99)
    estimator = PnPRansacEstimator(cfg)

    result_a = estimator.estimate(points_3d.copy(), points_2d.copy(), K)
    result_b = estimator.estimate(points_3d.copy(), points_2d.copy(), K)

    assert result_a.success and result_b.success
    np.testing.assert_array_equal(result_a.R, result_b.R)
    np.testing.assert_array_equal(result_a.t, result_b.t)


def test_pnp_too_few_points_fails():
    points_3d = np.array([[1.0, 2.0, 10.0], [0.5, -1.0, 15.0], [-1.0, 0.0, 20.0]])
    points_2d = _project(K, points_3d)

    cfg = PnPConfig(confidence=0.999, iterations_count=100, min_inliers=6, seed=1)
    estimator = PnPRansacEstimator(cfg)

    result = estimator.estimate(points_3d, points_2d, K)

    assert result.success is False
    assert result.R is None
    assert result.t is None
    assert result.inliers is None


def test_depth_filter_specification():
    """Kept iff z > min_depth_multiplier * b and sigma_z / z < tau, with
    sigma_z = z**2 / (f * b) the stereo depth standard deviation for 1 px of
    disparity error."""
    calib = StereoCalibration.from_projection_matrices(P0, P1)

    rng = np.random.default_rng(3)
    n = 30
    depths = rng.uniform(0.1, 100.0, size=n)
    xs = rng.uniform(-5, 5, size=n)
    ys = rng.uniform(-2, 2, size=n)
    points_3d = np.stack([xs, ys, depths], axis=1)

    cfg = LandmarkFilterConfig(min_depth_multiplier=2.0, relative_depth_uncertainty_threshold=0.1)
    landmark_filter = DepthUncertaintyFilter(cfg)
    actual_mask = landmark_filter.valid_mask(points_3d, calib)

    z = points_3d[:, 2]
    sigma_z = z**2 / (calib.focal_length * calib.baseline)
    expected_mask = (z > cfg.min_depth_multiplier * calib.baseline) & (
        sigma_z / z < cfg.relative_depth_uncertainty_threshold
    )

    np.testing.assert_array_equal(actual_mask, expected_mask)

    # Cross-check semantics: a very close point should fail the min-depth
    # multiplier gate; a very far point should fail the relative-uncertainty
    # gate for a tight enough threshold.
    close_point = np.array([[0.0, 0.0, 0.01 * calib.baseline]])
    assert not landmark_filter.valid_mask(close_point, calib)[0]

    tight_cfg = LandmarkFilterConfig(min_depth_multiplier=0.0, relative_depth_uncertainty_threshold=1e-9)
    tight_filter = DepthUncertaintyFilter(tight_cfg)
    far_point = np.array([[0.0, 0.0, 1000.0]])
    assert not tight_filter.valid_mask(far_point, calib)[0]


def _synthetic_pnp_problem(n=40, seed=11):
    rng = np.random.default_rng(seed)
    points_3d = np.stack([rng.uniform(-5, 5, n), rng.uniform(-2, 2, n), rng.uniform(5, 50, n)], axis=1)
    t_true = np.array([0.1, 0.0, 1.0])
    return points_3d, _project(K, points_3d + t_true)


def _estimator(min_inliers=6):
    return PnPRansacEstimator(PnPConfig(confidence=0.999, iterations_count=200, min_inliers=min_inliers, seed=1))


def test_failed_solve_is_a_failure(monkeypatch):
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **k: (False, None, None, None))
    result = _estimator().estimate(*_synthetic_pnp_problem(), K)
    assert (result.success, result.retval, result.R, result.t) == (False, False, None, None)


def test_nonfinite_solve_is_a_failure(monkeypatch):
    inliers = np.arange(40).reshape(-1, 1)
    bad = (True, np.array([[np.nan], [0.0], [0.0]]), np.zeros((3, 1)), inliers)
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **k: bad)
    result = _estimator().estimate(*_synthetic_pnp_problem(), K)
    assert result.success is False and result.retval is True
    assert result.inliers is inliers


def test_too_few_inliers_is_a_failure():
    points_3d, points_2d = _synthetic_pnp_problem()
    ok = _estimator(min_inliers=6).estimate(points_3d, points_2d, K)
    assert ok.success and len(ok.inliers) == 40
    rejected = _estimator(min_inliers=41).estimate(points_3d, points_2d, K)
    assert rejected.success is False and rejected.retval is True and len(rejected.inliers) == 40
