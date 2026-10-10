"""Randomized parity tests: vectorized RectifiedStereoMatcher.match() vs the
verbatim per-keypoint reference implementation _match_reference().

These stress the tie-break-sensitive edges of the vectorization: argmin
first-occurrence semantics, ratio-test band-size gating, searchsorted band
boundaries with duplicate y coordinates, and disparity inequality directions.
"""

import cv2
import numpy as np
import pytest

from kitti_slam.config.schema import StereoMatcherConfig
from kitti_slam.features.base import Features
from kitti_slam.features.matchers import RectifiedStereoMatcher


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

    expected = matcher._match_reference(fl, fr)
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
