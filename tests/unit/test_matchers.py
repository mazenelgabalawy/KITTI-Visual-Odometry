"""Tests for kitti_slam.features.matchers (BruteForceMatcher, RectifiedStereoMatcher)."""

import cv2
import numpy as np
import pytest

from kitti_slam.config.schema import DetectorConfig, GridFilterConfig, StereoMatcherConfig
from kitti_slam.features.base import Features
from kitti_slam.features.grid_filter import GridKeypointFilter
from kitti_slam.features.matchers import BruteForceMatcher, RectifiedStereoMatcher
from kitti_slam.features.orb import OrbDetector


def _stereo_pair(seed: int = 1, shift: int = 8):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, : 320 - shift] = left[:, shift:]
    right[:, 320 - shift :] = rng.integers(0, 255, size=(240, shift), dtype=np.uint8)
    return left, right


def _detect_pair(left: np.ndarray, right: np.ndarray):
    dcfg = DetectorConfig(nfeatures=500, scale_factor=1.2, nlevels=8)
    gcfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=5)
    detector = OrbDetector(dcfg, GridKeypointFilter(gcfg))
    return detector.detect(left), detector.detect(right)


def _triples(matches):
    return [(m.queryIdx, m.trainIdx, m.distance) for m in matches]


def _golden_features(goldens, side: str) -> Features:
    """Stereo features frozen alongside the matcher outputs (matchers read only kp.pt)."""
    kps = [cv2.KeyPoint(x=float(x), y=float(y), size=10.0) for x, y in goldens[f"match_{side}_pts"]]
    return Features(kps, goldens[f"match_{side}_desc"])


def _golden_triples(goldens, key: str):
    return [(int(q), int(t), float(d)) for q, t, d in goldens[key]]


def test_bf_matcher_ratio_path_matches_golden(goldens):
    """Fraction-kept path (ratio_threshold=0.5) vs. the frozen matches."""
    fl, fr = _golden_features(goldens, "left"), _golden_features(goldens, "right")
    new_matches = BruteForceMatcher(norm="NORM_HAMMING", cross_check=False, ratio_threshold=0.5).match(fl, fr)
    assert _triples(new_matches) == _golden_triples(goldens, "bf_ratio_matches")


def test_bf_matcher_plain_path_matches_raw_bf_match():
    """ratio_threshold=None must reproduce the raw bf.match(...) path used by VO_base."""
    left, right = _stereo_pair()
    fl, fr = _detect_pair(left, right)

    new_matcher = BruteForceMatcher(norm="NORM_HAMMING", cross_check=False, ratio_threshold=None)
    new_matches = new_matcher.match(fl, fr)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.match(fl.descriptors, fr.descriptors)

    assert _triples(new_matches) == _triples(raw_matches)


def test_rectified_matches_golden(goldens):
    """Vs. the frozen matches on the same frozen features."""
    fl, fr = _golden_features(goldens, "left"), _golden_features(goldens, "right")

    cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=True,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.7,
        max_disparity=64.0,
    )
    matcher = RectifiedStereoMatcher(cfg)
    new_matches = matcher.match(fl, fr)
    assert _triples(new_matches) == _golden_triples(goldens, "rectified_matches")


def test_golden_features_match_current_detector(goldens):
    """The frozen matcher inputs are what the current detector produces (pinned OpenCV)."""
    fl, fr = _detect_pair(*_stereo_pair())
    np.testing.assert_array_equal([kp.pt for kp in fl.keypoints], goldens["match_left_pts"])
    np.testing.assert_array_equal(fr.descriptors, goldens["match_right_desc"])


def test_rectified_golden_match_count_and_first_ten():
    """Golden recorded output for seed=1 stereo pair, cross_check=True, default thresholds."""
    left, right = _stereo_pair()
    fl, fr = _detect_pair(left, right)

    cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=True,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.7,
        max_disparity=64.0,
    )
    matcher = RectifiedStereoMatcher(cfg)
    matches = matcher.match(fl, fr)

    expected_first_ten = [
        (0, 62, 0.0),
        (1, 64, 0.0),
        (2, 97, 0.0),
        (5, 151, 0.0),
        (8, 23, 0.0),
        (9, 25, 0.0),
        (10, 26, 17.0),
        (12, 104, 0.0),
        (13, 103, 0.0),
        (14, 106, 0.0),
    ]

    assert len(matches) == 112
    assert _triples(matches)[:10] == expected_first_ten


def test_rectified_epipolar_y_window_rejection():
    left_kps = [cv2.KeyPoint(x=50.0, y=10.0, size=5)]
    left_desc = np.zeros((1, 32), dtype=np.uint8)
    # One candidate within the y-window (y=10), one far outside (y=50).
    right_kps = [cv2.KeyPoint(x=45.0, y=10.0, size=5), cv2.KeyPoint(x=45.0, y=50.0, size=5)]
    right_desc = np.zeros((2, 32), dtype=np.uint8)

    cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=False,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.99,
        max_disparity=64.0,
    )
    matcher = RectifiedStereoMatcher(cfg)
    matches = matcher.match(Features(left_kps, left_desc), Features(right_kps, right_desc))

    assert [(m.queryIdx, m.trainIdx) for m in matches] == [(0, 0)]


def test_rectified_disparity_cap():
    left_kps = [cv2.KeyPoint(x=50.0, y=10.0, size=5)]
    left_desc = np.zeros((1, 32), dtype=np.uint8)
    right_kps = [cv2.KeyPoint(x=45.0, y=10.0, size=5)]  # disparity = 5
    right_desc = np.zeros((1, 32), dtype=np.uint8)

    tight_cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=False,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.99,
        max_disparity=3.0,
    )
    matches_tight = RectifiedStereoMatcher(tight_cfg).match(
        Features(left_kps, left_desc), Features(right_kps, right_desc)
    )
    assert matches_tight == []

    loose_cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=False,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.99,
        max_disparity=10.0,
    )
    matches_loose = RectifiedStereoMatcher(loose_cfg).match(
        Features(left_kps, left_desc), Features(right_kps, right_desc)
    )
    assert [(m.queryIdx, m.trainIdx) for m in matches_loose] == [(0, 0)]


def test_rectified_cross_check_toggling():
    left, right = _stereo_pair()
    fl, fr = _detect_pair(left, right)

    no_cross_cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=False,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.7,
        max_disparity=64.0,
    )
    cross_cfg = StereoMatcherConfig(
        kind="rectified",
        cross_check=True,
        pixel_threshold=2.0,
        max_hamming_distance=30,
        ratio_test=0.7,
        max_disparity=64.0,
    )

    matches_no_cross = RectifiedStereoMatcher(no_cross_cfg).match(fl, fr)
    matches_cross = RectifiedStereoMatcher(cross_cfg).match(fl, fr)

    # Cross-checked matches must be a subset of the l2r-only matches.
    no_cross_pairs = {(m.queryIdx, m.trainIdx) for m in matches_no_cross}
    cross_pairs = {(m.queryIdx, m.trainIdx) for m in matches_cross}
    assert cross_pairs.issubset(no_cross_pairs)


def _one_row_features(xs, descs):
    return Features([cv2.KeyPoint(x=float(x), y=50.0, size=5.0) for x in xs], np.array(descs, dtype=np.uint8))


def _flip_bits(desc, n):
    out = desc.copy()
    for i in range(n):
        out[i // 8] ^= 1 << (i % 8)
    return out


@pytest.mark.parametrize("cross_check", [False, True])
def test_rectified_invalid_geometry_does_not_suppress_valid_match(cross_check):
    """BUG-005: the descriptor-best right keypoint has negative disparity (u = 120 >
    100); the valid one (disparity 10 px) is 3 bits further. Geometry is checked
    first, so the valid candidate is matched instead of nothing."""
    d = np.random.default_rng(0).integers(0, 256, size=32, dtype=np.uint8)
    left = _one_row_features([100.0], [d])
    right = _one_row_features([120.0, 90.0], [d, _flip_bits(d, 3)])
    cfg = StereoMatcherConfig(
        kind="rectified", cross_check=cross_check, pixel_threshold=2.0, max_hamming_distance=30, ratio_test=0.8
    )
    assert _triples(RectifiedStereoMatcher(cfg).match(left, right)) == [(0, 1, 3.0)]


def test_rectified_invalid_geometry_does_not_enter_ratio_test():
    """A geometrically invalid near-duplicate (4 bits) must not make the only valid
    candidate (3 bits) fail the ratio test (3/4 >= 0.7)."""
    d = np.random.default_rng(1).integers(0, 256, size=32, dtype=np.uint8)
    left = _one_row_features([100.0], [d])
    right = _one_row_features([90.0, 30.0], [_flip_bits(d, 3), _flip_bits(d, 4)])  # disparities 10 and 70 px
    cfg = StereoMatcherConfig(
        kind="rectified", cross_check=True, pixel_threshold=2.0, max_hamming_distance=30, ratio_test=0.7
    )
    assert _triples(RectifiedStereoMatcher(cfg).match(left, right)) == [(0, 0, 3.0)]
