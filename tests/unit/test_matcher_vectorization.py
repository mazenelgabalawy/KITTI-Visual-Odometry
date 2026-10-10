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


def _spec_direction(feat_q, desc_q, feat_t, desc_t, cfg: StereoMatcherConfig, reverse: bool) -> list:
    """One matching direction. The candidates for a query keypoint are the train
    keypoints within pixel_threshold rows whose disparity lies in [1, max_disparity]
    px (left u minus right u). The best candidate is the first minimum Hamming
    distance in row-sorted order. It must be within max_hamming_distance and, with
    two or more candidates, pass the ratio test against the second-best candidate
    distance."""
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=int)
    y_train = np.array([kp.pt[1] for kp in feat_t])
    order = np.argsort(y_train)
    y_sorted = y_train[order]
    matches = []
    for qi, kp in enumerate(feat_q):
        low = bisect.bisect_left(y_sorted, kp.pt[1] - cfg.pixel_threshold)
        high = bisect.bisect_right(y_sorted, kp.pt[1] + cfg.pixel_threshold)
        candidates = []  # (Hamming distance, train index) in row-sorted order
        for j in order[low:high]:
            disparity = feat_t[j].pt[0] - kp.pt[0] if reverse else kp.pt[0] - feat_t[j].pt[0]
            if 1.0 <= disparity <= cfg.max_disparity:
                candidates.append((int(popcount[desc_q[qi] ^ desc_t[j]].sum()), int(j)))
        if not candidates:
            continue
        best = min(candidates, key=lambda c: c[0])
        if best[0] > cfg.max_hamming_distance:
            continue
        if len(candidates) >= 2:
            second = sorted(c[0] for c in candidates)[1]
            if best[0] / (second + 1e-9) >= cfg.ratio_test:
                continue
        matches.append(cv2.DMatch(_queryIdx=qi, _trainIdx=best[1], _distance=float(best[0])))
    return matches


def _spec_match(cfg: StereoMatcherConfig, feat_a: Features, feat_b: Features) -> list:
    """Specification of RectifiedStereoMatcher.match(): left-to-right matches,
    kept only if mutual with the right-to-left matches when cross_check is set."""
    if len(feat_a.keypoints) == 0 or len(feat_b.keypoints) == 0:
        return []
    l2r = _spec_direction(feat_a.keypoints, feat_a.descriptors, feat_b.keypoints, feat_b.descriptors, cfg, False)
    if not cfg.cross_check:
        return l2r
    r2l = _spec_direction(feat_b.keypoints, feat_b.descriptors, feat_a.keypoints, feat_a.descriptors, cfg, True)
    backpointer = {m.queryIdx: m.trainIdx for m in r2l}
    return [m for m in l2r if backpointer.get(m.trainIdx) == m.queryIdx]


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
