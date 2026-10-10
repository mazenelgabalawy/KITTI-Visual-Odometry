"""Tests for kitti_slam.geometry.transforms."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kitti_slam.geometry.transforms import (
    compose_se3,
    invert_pose,
    rotation_to_euler,
    to_homogeneous,
)

# Hand-checkable golden input/output pairs (the larger frozen comparison
# lives in the goldens fixture).
GOLDEN_PAIRS = [
    (np.eye(3), np.array([0.0, 0.0, 0.0])),
    # 90 deg rotation about Z: [[0,-1,0],[1,0,0],[0,0,1]]
    (
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([0.0, 0.0, np.pi / 2]),
    ),
    # 90 deg rotation about X: [[1,0,0],[0,0,-1],[0,1,0]]
    (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        np.array([np.pi / 2, 0.0, 0.0]),
    ),
]


def test_euler_matches_golden(goldens):
    """Vs. the frozen angles of 20 random rotations."""
    for R, expected in zip(goldens["euler_in"][:20], goldens["euler_out"][:20], strict=True):
        np.testing.assert_allclose(rotation_to_euler(R), expected, atol=1e-12)


def test_euler_golden():
    for R, expected in GOLDEN_PAIRS:
        actual = rotation_to_euler(R)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_euler_gimbal_lock_branch(goldens):
    """Singular branch (rotation about Y by +/-90 deg) vs. the frozen output."""
    for R, expected in zip(goldens["euler_in"][20:], goldens["euler_out"][20:], strict=True):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        assert sy < 1e-6  # confirm we're exercising the singular branch
        np.testing.assert_allclose(rotation_to_euler(R), expected, atol=1e-12)
    np.testing.assert_allclose(goldens["euler_out"][20], [0.0, np.pi / 2, 0.0], atol=1e-12)


def test_invert_pose_roundtrip():
    rng = np.random.default_rng(1)
    R = Rotation.random(random_state=rng).as_matrix()
    t = rng.uniform(-10, 10, size=3)

    R_inv, t_inv = invert_pose(R, t)

    np.testing.assert_allclose(R_inv @ R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(R @ R_inv, np.eye(3), atol=1e-10)

    x = rng.uniform(-5, 5, size=3)
    y = R @ x + t
    x_recovered = R_inv @ y + t_inv
    np.testing.assert_allclose(x_recovered, x, atol=1e-10)


def test_compose_se3_accepts_3x4_and_4x4():
    rng = np.random.default_rng(2)
    R = Rotation.random(random_state=rng).as_matrix()
    t = rng.uniform(-5, 5, size=3)
    T34 = np.hstack((R, t.reshape(3, 1)))
    T44 = np.vstack((T34, np.array([0.0, 0.0, 0.0, 1.0])))

    C = np.eye(4)
    C[:3, 3] = rng.uniform(-2, 2, size=3)

    result_34 = compose_se3(C, T34)
    result_44 = compose_se3(C, T44)

    expected = C @ T44
    np.testing.assert_allclose(result_34, expected)
    np.testing.assert_allclose(result_44, expected)

    with pytest.raises(ValueError):
        compose_se3(C, np.eye(2))


def test_to_homogeneous():
    rng = np.random.default_rng(3)
    T34 = rng.uniform(-1, 1, size=(3, 4))
    T44 = to_homogeneous(T34)

    assert T44.shape == (4, 4)
    np.testing.assert_allclose(T44[:3, :4], T34)
    np.testing.assert_allclose(T44[3, :], np.array([0.0, 0.0, 0.0, 1.0]))
