"""Tests for kitti_slam.evaluation.trajectory_io."""

import numpy as np
from scipy.spatial.transform import Rotation

from kitti_slam.evaluation.trajectory_io import TrajectoryWriter, load_kitti_poses


def _random_pose(rng):
    R = Rotation.random(random_state=rng).as_matrix()
    t = rng.uniform(-10, 10, size=3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_writer_byte_identical_to_golden(tmp_path, goldens):
    """Vs. the frozen bytes written for 5 random poses."""
    writer = TrajectoryWriter()
    for C in goldens["writer_poses"]:
        writer.append(C)
    path = tmp_path / "ported.txt"
    writer.write(path)

    assert path.read_text() == str(goldens["writer_text"])


def test_writer_golden_format(tmp_path):
    C = np.array(
        [
            [1.0, 0.0, 0.0, 1.5],
            [0.0, 1.0, 0.0, -2.25],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected = (
        "1.000000e+00 0.000000e+00 0.000000e+00 1.500000e+00 "
        "0.000000e+00 1.000000e+00 0.000000e+00 -2.250000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 3.000000e+00\n"
    )

    writer = TrajectoryWriter()
    writer.append(C)
    path = tmp_path / "golden.txt"
    writer.write(path)

    assert path.read_text() == expected


def test_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    matrices = [_random_pose(rng) for _ in range(4)]

    writer = TrajectoryWriter()
    for C in matrices:
        writer.append(C)
    path = tmp_path / "traj.txt"
    writer.write(path)

    loaded = load_kitti_poses(path)
    expected = np.stack([C[:3, :4] for C in matrices], axis=0)

    np.testing.assert_allclose(loaded, expected, atol=1e-6)


def test_as_array_shape():
    rng = np.random.default_rng(2)
    matrices = [_random_pose(rng) for _ in range(3)]

    writer = TrajectoryWriter()
    for C in matrices:
        writer.append(C)

    arr = writer.as_array()
    assert arr.shape == (3, 3, 4)
    expected = np.stack([C[:3, :4] for C in matrices], axis=0)
    np.testing.assert_allclose(arr, expected)
