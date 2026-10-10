"""Tests for kitti_slam.features.association.intersect_temporal_stereo."""

import cv2
import numpy as np

from kitti_slam.features.association import intersect_temporal_stereo
from kitti_slam.features.base import Features


def _build_case():
    left_prev_kps = [cv2.KeyPoint(x=float(i), y=float(i * 2), size=5) for i in range(5)]
    right_prev_kps = [cv2.KeyPoint(x=float(i + 0.5), y=float(i * 2), size=5) for i in range(5)]
    left_curr_kps = [cv2.KeyPoint(x=float(i + 1), y=float(i * 2 + 1), size=5) for i in range(5)]

    # Temporal matches: left_prev idx i -> left_curr idx i, for all 5.
    temporal_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(5)]
    # Stereo matches only exist for left_prev indices 0, 1, 3.
    stereo_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in (0, 1, 3)]

    return (
        temporal_matches,
        stereo_matches,
        Features(left_prev_kps, None),
        Features(right_prev_kps, None),
        Features(left_curr_kps, None),
    )


def test_matches_golden_on_non_empty_case(goldens):
    """Vs. the frozen output (values and dtype)."""
    temporal_matches, stereo_matches, fl_prev, fr_prev, fl_curr = _build_case()

    result = intersect_temporal_stereo(temporal_matches, stereo_matches, fl_prev, fr_prev, fl_curr)

    for actual, key in (
        (result.left_prev, "assoc_left_prev"),
        (result.right_prev, "assoc_right_prev"),
        (result.left_curr, "assoc_left_curr"),
    ):
        assert actual.dtype == goldens[key].dtype
        assert np.array_equal(actual, goldens[key])


def test_known_intersection_exact_expected_arrays():
    temporal_matches, stereo_matches, fl_prev, fr_prev, fl_curr = _build_case()

    result = intersect_temporal_stereo(temporal_matches, stereo_matches, fl_prev, fr_prev, fl_curr)

    # Only left_prev indices 0, 1, 3 survive (those with a stereo match).
    expected_left_prev = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    expected_right_prev = np.array([[0.5, 0.0], [1.5, 2.0], [3.5, 6.0]], dtype=np.float32)
    expected_left_curr = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 7.0]], dtype=np.float32)

    assert np.array_equal(result.left_prev, expected_left_prev)
    assert np.array_equal(result.right_prev, expected_right_prev)
    assert np.array_equal(result.left_curr, expected_left_curr)
    assert result.left_prev.dtype == np.float32
    assert result.right_prev.dtype == np.float32
    assert result.left_curr.dtype == np.float32


def test_empty_returns_empty_arrays_not_raise():
    _, _, fl_prev, fr_prev, fl_curr = _build_case()

    # No overlap between temporal and stereo matches at all.
    temporal_matches = [cv2.DMatch(_queryIdx=4, _trainIdx=4, _distance=0.0)]
    stereo_matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0)]

    result = intersect_temporal_stereo(temporal_matches, stereo_matches, fl_prev, fr_prev, fl_curr)

    assert result.left_prev.shape == (0, 2)
    assert result.right_prev.shape == (0, 2)
    assert result.left_curr.shape == (0, 2)
    assert result.left_prev.dtype == np.float32
    assert result.right_prev.dtype == np.float32
    assert result.left_curr.dtype == np.float32


def test_empty_inputs_return_empty_arrays():
    _, _, fl_prev, fr_prev, fl_curr = _build_case()

    result = intersect_temporal_stereo([], [], fl_prev, fr_prev, fl_curr)

    assert result.left_prev.shape == (0, 2)
    assert result.right_prev.shape == (0, 2)
    assert result.left_curr.shape == (0, 2)
