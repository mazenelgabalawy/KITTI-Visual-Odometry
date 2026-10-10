"""Randomized tests: vectorized RectifiedStereoMatcher.match() vs the plain
per-keypoint specification _spec_match() below.

These stress the tie-break-sensitive edges of the vectorization: argmin
first-occurrence semantics, ratio-test band-size gating, searchsorted band
boundaries with duplicate y coordinates, and disparity inequality directions.
"""

import bisect

import cv2
import numpy as np
import pytest

from kitti_slam.config.schema import StereoMatcherConfig
from kitti_slam.features.base import Features
from kitti_slam.features.matchers import RectifiedStereoMatcher


def _spec_match(cfg: StereoMatcherConfig, feat_a: Features, feat_b: Features) -> list:
    """Per-keypoint specification of RectifiedStereoMatcher.match().

    For each query keypoint, the candidates are the other image's keypoints within
    pixel_threshold rows. The best candidate (first minimum Hamming distance in
    row-sorted order) must be within max_hamming_distance and, with two or more
    candidates, pass the ratio test against the second-best distance. Its
    disparity must lie in [1, max_disparity] px. With cross_check, only mutual
    best matches survive.
    """
    feat_left = feat_a.keypoints
    desc_left = feat_a.descriptors
    feat_right = feat_b.keypoints
    desc_right = feat_b.descriptors
    cross_check = cfg.cross_check
    pixel_threshold = cfg.pixel_threshold
    max_hamming_distance = cfg.max_hamming_distance
    ratio_test = cfg.ratio_test
    max_disparity = cfg.max_disparity
    min_disparity = 1.0

    if len(feat_left) == 0 or len(feat_right) == 0:
        return []

    # Precompute popcount LUT for fast Hamming distance
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=int)

    # Pre-sort right keypoints by y for l2r
    y_right = np.array([kp.pt[1] for kp in feat_right])
    idx_right_sorted = np.argsort(y_right)
    y_right_sorted = y_right[idx_right_sorted]
    feat_right_sorted = np.array(feat_right)[idx_right_sorted]
    desc_right_sorted = desc_right[idx_right_sorted]

    # Left-to-right matches
    l2r_matches = []
    for query_idx, kp_query in enumerate(feat_left):
        desc_query = desc_left[query_idx].reshape(1, -1)  # (1, 32)
        row = kp_query.pt[1]
        low = bisect.bisect_left(y_right_sorted, row - pixel_threshold)
        high = bisect.bisect_right(y_right_sorted, row + pixel_threshold)
        if low == high:
            continue

        desc_train_subset = desc_right_sorted[low:high]  # (c, 32)
        feat_train_subset = feat_right_sorted[low:high]

        # Vectorized Hamming distance using LUT
        xor = desc_query ^ desc_train_subset  # (c, 32) uint8
        bit_counts = popcount[xor]  # (c, 32) int
        dists = np.sum(bit_counts, axis=1)  # (c,)

        if len(dists) == 0:
            continue

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        # Ratio test if >=2 candidates
        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / (second_dist + 1e-9) >= ratio_test:
                continue
        # For len==1, no ratio, but already checked max_dist

        # Disparity check
        u_query = kp_query.pt[0]
        u_train = feat_train_subset[min_idx].pt[0]
        disparity = u_query - u_train
        if disparity < min_disparity or disparity > max_disparity:
            continue

        # Match with original train idx (fix: cast min_dist to float for cv2.DMatch)
        train_idx_original = idx_right_sorted[low + min_idx]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=float(min_dist))
        l2r_matches.append(match)

    if not cross_check:
        return l2r_matches

    # Pre-sort left for r2l
    y_left = np.array([kp.pt[1] for kp in feat_left])
    idx_left_sorted = np.argsort(y_left)
    y_left_sorted = y_left[idx_left_sorted]
    feat_left_sorted = np.array(feat_left)[idx_left_sorted]
    desc_left_sorted = desc_left[idx_left_sorted]

    # Right-to-left matches
    r2l_matches = []
    for query_idx, kp_query in enumerate(feat_right):
        desc_query = desc_right[query_idx].reshape(1, -1)
        row = kp_query.pt[1]
        low = bisect.bisect_left(y_left_sorted, row - pixel_threshold)
        high = bisect.bisect_right(y_left_sorted, row + pixel_threshold)
        if low == high:
            continue

        desc_train_subset = desc_left_sorted[low:high]
        feat_train_subset = feat_left_sorted[low:high]

        # Vectorized Hamming distance using LUT (optimized)
        xor = desc_query ^ desc_train_subset
        bit_counts = popcount[xor]
        dists = np.sum(bit_counts, axis=1)

        if len(dists) == 0:
            continue

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / (second_dist + 1e-9) >= ratio_test:
                continue

        u_query = kp_query.pt[0]
        u_train = feat_train_subset[min_idx].pt[0]
        disparity = u_train - u_query  # Reversed for r2l
        if disparity < min_disparity or disparity > max_disparity:
            continue

        # Match with original train idx (left) (fix: cast min_dist to float for cv2.DMatch)
        train_idx_original = idx_left_sorted[low + min_idx]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=float(min_dist))
        r2l_matches.append(match)

    # Mutual best matches (cross-check)
    r2l_backpointer = {m.queryIdx: m.trainIdx for m in r2l_matches}  # right -> left
    mutual_matches = [
        m for m in l2r_matches if m.trainIdx in r2l_backpointer and r2l_backpointer[m.trainIdx] == m.queryIdx
    ]

    return mutual_matches


def _triples(matches):
    return [(m.queryIdx, m.trainIdx, m.distance) for m in matches]


def _make_features(rng, n, y_choices=None, x_range=(0.0, 640.0), desc_mode="random"):
    """Build a Features instance with n synthetic cv2.KeyPoints + uint8 descriptors.

    desc_mode:
      - "random": full-entropy random bytes.
      - "low_entropy": descriptors drawn from a small pool of distinct byte
        patterns, so many pairs share identical/near-identical Hamming distances
        (stresses argmin/partition tie-breaks).
    """
    if n == 0:
        return Features([], np.zeros((0, 32), dtype=np.uint8))

    ys = rng.choice(y_choices, size=n) if y_choices is not None else rng.uniform(0.0, 240.0, size=n)
    xs = rng.uniform(x_range[0], x_range[1], size=n)

    keypoints = [cv2.KeyPoint(x=float(xs[i]), y=float(ys[i]), size=5.0) for i in range(n)]

    if desc_mode == "random":
        desc = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    elif desc_mode == "low_entropy":
        pool_size = 4
        pool = rng.integers(0, 256, size=(pool_size, 32), dtype=np.uint8)
        choice = rng.integers(0, pool_size, size=n)
        desc = pool[choice]
    else:
        raise ValueError(desc_mode)

    return Features(keypoints, desc)


def _run_case(seed, n_left, n_right, cfg_kwargs, y_choices=None, desc_mode="random"):
    rng = np.random.default_rng(seed)
    fl = _make_features(rng, n_left, y_choices=y_choices, desc_mode=desc_mode)
    fr = _make_features(rng, n_right, y_choices=y_choices, desc_mode=desc_mode)

    cfg = StereoMatcherConfig(kind="rectified", **cfg_kwargs)
    matcher = RectifiedStereoMatcher(cfg)

    expected = _spec_match(cfg, fl, fr)
    actual = matcher.match(fl, fr)

    assert _triples(actual) == _triples(expected), (
        f"seed={seed} n_left={n_left} n_right={n_right} cfg={cfg_kwargs} y_choices={y_choices} desc_mode={desc_mode}"
    )


DEFAULT_CFG = dict(
    cross_check=True,
    pixel_threshold=2.0,
    max_hamming_distance=30,
    ratio_test=0.7,
    max_disparity=64.0,
)


@pytest.mark.parametrize("seed", range(30))
def test_random_descriptors_and_keypoints(seed):
    _run_case(seed, n_left=80, n_right=90, cfg_kwargs=DEFAULT_CFG)


@pytest.mark.parametrize("seed", range(20))
def test_random_descriptors_cross_check_false(seed):
    cfg = dict(DEFAULT_CFG)
    cfg["cross_check"] = False
    _run_case(seed, n_left=70, n_right=65, cfg_kwargs=cfg)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("cross_check", [True, False])
def test_low_entropy_descriptors_tie_heavy(seed, cross_check):
    cfg = dict(DEFAULT_CFG)
    cfg["cross_check"] = cross_check
    cfg["ratio_test"] = 0.99  # loosen ratio gate so ties matter more for argmin
    _run_case(seed, n_left=60, n_right=55, cfg_kwargs=cfg, desc_mode="low_entropy")


@pytest.mark.parametrize("seed", range(20))
def test_duplicate_y_coordinates(seed):
    """Many keypoints share identical y -> stresses searchsorted band boundaries."""
    cfg = dict(DEFAULT_CFG)
    y_choices = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 30.0])  # heavy duplication
    _run_case(seed, n_left=50, n_right=45, cfg_kwargs=cfg, y_choices=y_choices)


@pytest.mark.parametrize(
    "n_left,n_right",
    [(0, 0), (0, 5), (5, 0)],
)
def test_empty_inputs(n_left, n_right):
    _run_case(seed=0, n_left=n_left, n_right=n_right, cfg_kwargs=DEFAULT_CFG)
    # Both directions must return [] for empty inputs.
    rng = np.random.default_rng(0)
    fl = _make_features(rng, n_left)
    fr = _make_features(rng, n_right)
    cfg = StereoMatcherConfig(kind="rectified", **DEFAULT_CFG)
    matcher = RectifiedStereoMatcher(cfg)
    assert matcher.match(fl, fr) == []


@pytest.mark.parametrize("seed", range(15))
def test_single_candidate_bands_tiny_pixel_threshold(seed):
    """pixel_threshold tiny -> bands are size 0 or 1, so the ratio test never fires."""
    cfg = dict(DEFAULT_CFG)
    cfg["pixel_threshold"] = 0.05
    y_choices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    _run_case(seed, n_left=40, n_right=40, cfg_kwargs=cfg, y_choices=y_choices)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_disparity", [0.0, 1.0, 5.0, 500.0])
def test_max_disparity_boundaries(seed, max_disparity):
    cfg = dict(DEFAULT_CFG)
    cfg["max_disparity"] = max_disparity
    _run_case(seed, n_left=40, n_right=40, cfg_kwargs=cfg)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_hamming", [0, 1, 30, 256])
def test_max_hamming_boundaries(seed, max_hamming):
    cfg = dict(DEFAULT_CFG)
    cfg["max_hamming_distance"] = max_hamming
    _run_case(seed, n_left=40, n_right=40, cfg_kwargs=cfg)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ratio_test", [0.0, 0.5, 1.0, 1.5])
def test_ratio_test_boundaries(seed, ratio_test):
    cfg = dict(DEFAULT_CFG)
    cfg["ratio_test"] = ratio_test
    _run_case(seed, n_left=40, n_right=40, cfg_kwargs=cfg, desc_mode="low_entropy")


@pytest.mark.parametrize("seed", range(15))
def test_large_random_case(seed):
    """Larger population closer to real ORB feature counts."""
    _run_case(seed, n_left=400, n_right=380, cfg_kwargs=DEFAULT_CFG)
