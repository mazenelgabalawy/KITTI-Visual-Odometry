"""Tests for kitti_slam.datasets.kitti.KittiSequence."""

import cv2
import numpy as np
import pytest

from kitti_slam.datasets import KittiSequence, StereoFrame

P0_LINE = (
    "P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
)
P1_LINE = (
    "P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 "
    "0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
)


@pytest.fixture
def dataset_root(tmp_path):
    seq_dir = tmp_path / "05"
    (seq_dir / "image_0").mkdir(parents=True)
    (seq_dir / "image_1").mkdir(parents=True)

    rng = np.random.default_rng(7)
    left_images = []
    right_images = []
    for idx in range(2):
        left = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
        right = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "image_0" / f"{idx:06d}.png"), left)
        cv2.imwrite(str(seq_dir / "image_1" / f"{idx:06d}.png"), right)
        left_images.append(left)
        right_images.append(right)

    (seq_dir / "calib.txt").write_text(P0_LINE + "\n" + P1_LINE + "\n")

    return tmp_path, left_images, right_images


def test_len(dataset_root):
    root, _, _ = dataset_root
    seq = KittiSequence(root, 5)
    assert len(seq) == 2


def test_getitem_returns_matching_pixels(dataset_root):
    root, left_images, right_images = dataset_root
    seq = KittiSequence(root, 5)

    for idx in range(2):
        frame = seq[idx]
        assert isinstance(frame, StereoFrame)
        assert frame.index == idx
        np.testing.assert_array_equal(frame.left, left_images[idx])
        np.testing.assert_array_equal(frame.right, right_images[idx])


def test_getitem_out_of_range_raises_index_error(dataset_root):
    root, _, _ = dataset_root
    seq = KittiSequence(root, 5)
    with pytest.raises(IndexError):
        seq[2]


def test_missing_root_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        KittiSequence(tmp_path, 99)


def test_missing_calib_file_raises_file_not_found_error(dataset_root):
    root, _, _ = dataset_root
    (root / "05" / "calib.txt").unlink()
    with pytest.raises(FileNotFoundError, match="calib"):
        KittiSequence(root, 5)


def test_calibration_property(dataset_root):
    root, _, _ = dataset_root
    seq = KittiSequence(root, 5)
    calib = seq.calibration

    assert calib.K.shape == (3, 3)
    assert calib.K[0, 0] == pytest.approx(718.856, rel=1e-6)
    assert calib.baseline == pytest.approx(0.53716, rel=1e-3)
    # Property should be cached / stable across accesses.
    assert seq.calibration is calib


def test_iteration_yields_all_frames(dataset_root):
    root, left_images, right_images = dataset_root
    seq = KittiSequence(root, 5)

    frames = list(seq)
    assert len(frames) == 2
    for idx, frame in enumerate(frames):
        assert frame.index == idx
        np.testing.assert_array_equal(frame.left, left_images[idx])
        np.testing.assert_array_equal(frame.right, right_images[idx])
