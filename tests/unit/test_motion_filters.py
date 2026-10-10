"""Tests for kitti_slam.odometry.motion_filters."""

import numpy as np
from scipy.spatial.transform import Rotation

from kitti_slam.config.schema import MotionConfig
from kitti_slam.odometry.motion_filters import (
    NullMotionFilter,
    PlanarMotionFilter,
    _orthogonalize_to_planar_motion,
)

# Hand-checkable golden input/output pairs (the larger frozen comparison
# lives in the goldens fixture).
GOLDEN_PAIRS = [
    (
        np.eye(3),
        0.8,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ),
    (
        # 90 deg rotation about Y
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        0.8,
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    ),
    (
        # Small rotation about X (roll) with alpha=1.0 should fully snap to planar (yaw-only)
        Rotation.from_euler("x", 10, degrees=True).as_matrix(),
        1.0,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ),
]


def test_orthogonalize_matches_golden(goldens):
    """Vs. the frozen output on 20 random and 4 near-planar rotations, for
    alpha in (0.8, 0.5, 1.0)."""
    for R, expected_per_alpha in zip(goldens["ortho_in"], goldens["ortho_out"], strict=True):
        for alpha, expected in zip(goldens["ortho_alphas"], expected_per_alpha, strict=True):
            actual = _orthogonalize_to_planar_motion(R, alpha=float(alpha))
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_orthogonalize_golden():
    for R, alpha, expected in GOLDEN_PAIRS:
        actual = _orthogonalize_to_planar_motion(R, alpha=alpha)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_planar_gate():
    R_identity = np.eye(3)

    # Within limits: plausible.
    cfg = MotionConfig(max_translation=5.0, max_pitch_deg=10.0, orthogonalize_alpha=0.8)
    motion_filter = PlanarMotionFilter(cfg)
    t_small = np.array([1.0, 0.0, 2.0])
    assert motion_filter.is_plausible(R_identity, t_small) is True

    # Translation exceeds limit: implausible.
    t_large = np.array([10.0, 0.0, 0.0])
    assert motion_filter.is_plausible(R_identity, t_large) is False

    # Pitch exceeds limit: implausible.
    R_big_pitch = Rotation.from_euler("y", 45, degrees=True).as_matrix()
    assert motion_filter.is_plausible(R_big_pitch, t_small) is False

    # None limits => passthrough / always True for that axis.
    cfg_no_translation_limit = MotionConfig(max_translation=None, max_pitch_deg=10.0, orthogonalize_alpha=0.8)
    filter_no_translation_limit = PlanarMotionFilter(cfg_no_translation_limit)
    assert filter_no_translation_limit.is_plausible(R_identity, t_large) is True

    cfg_no_pitch_limit = MotionConfig(max_translation=5.0, max_pitch_deg=None, orthogonalize_alpha=0.8)
    filter_no_pitch_limit = PlanarMotionFilter(cfg_no_pitch_limit)
    assert filter_no_pitch_limit.is_plausible(R_big_pitch, t_small) is True

    cfg_no_limits = MotionConfig(max_translation=None, max_pitch_deg=None)
    filter_no_limits = PlanarMotionFilter(cfg_no_limits)
    assert filter_no_limits.is_plausible(R_big_pitch, t_large) is True

    # refine_rotation: alpha None => identity passthrough (unchanged R).
    cfg_no_alpha = MotionConfig(orthogonalize_alpha=None)
    filter_no_alpha = PlanarMotionFilter(cfg_no_alpha)
    R_random = Rotation.random(random_state=np.random.default_rng(5)).as_matrix()
    np.testing.assert_array_equal(filter_no_alpha.refine_rotation(R_random), R_random)

    # refine_rotation: alpha set => delegates to _orthogonalize_to_planar_motion.
    cfg_alpha = MotionConfig(orthogonalize_alpha=0.8)
    filter_alpha = PlanarMotionFilter(cfg_alpha)
    expected = _orthogonalize_to_planar_motion(R_random, alpha=0.8)
    np.testing.assert_array_equal(filter_alpha.refine_rotation(R_random), expected)


def test_null_filter():
    null_filter = NullMotionFilter()
    R_random = Rotation.random(random_state=np.random.default_rng(6)).as_matrix()
    t_random = np.array([1000.0, 1000.0, 1000.0])

    np.testing.assert_array_equal(null_filter.refine_rotation(R_random), R_random)
    assert null_filter.is_plausible(R_random, t_random) is True
