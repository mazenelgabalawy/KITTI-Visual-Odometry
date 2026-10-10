"""Tests for kitti_slam.features.grid_filter.GridKeypointFilter."""

import cv2
import numpy as np
import pytest

from kitti_slam.config.schema import GridFilterConfig
from kitti_slam.features.base import Features
from kitti_slam.features.grid_filter import GridKeypointFilter


def _synthetic_keypoints(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kps = []
    for _ in range(n):
        x = rng.uniform(0, 320)
        y = rng.uniform(0, 240)
        resp = rng.uniform(0, 1)
        kps.append(cv2.KeyPoint(x=float(x), y=float(y), size=10.0, response=float(resp)))
    descs = np.arange(n * 32, dtype=np.uint8).reshape(n, 32)
    return kps, descs


def test_matches_golden_on_200_keypoints(goldens):
    """Vs. the frozen selection of the same 200 keypoints."""
    kps, descs = _synthetic_keypoints(200, seed=0)
    np.testing.assert_array_equal([kp.pt for kp in kps], goldens["grid_in_pts"])
    np.testing.assert_array_equal([kp.response for kp in kps], goldens["grid_in_responses"])

    cfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=3)
    out = GridKeypointFilter(cfg)(Features(keypoints=kps, descriptors=descs), (240, 320))

    np.testing.assert_array_equal([kp.pt for kp in out.keypoints], goldens["grid_out_pts"])
    np.testing.assert_array_equal([kp.response for kp in out.keypoints], goldens["grid_out_responses"])
    assert np.array_equal(out.descriptors, goldens["grid_out_desc"])


def test_golden_selection():
    """Fixed small input with hand-recorded expected selection order/indices.

    Two grid cells (grid_size=(10, 10) on a 20x20 image): cell (0,0) contains
    keypoint indices 0,1,2 and cell (0,1) contains indices 3,4,5. With
    max_per_grid=2, the top-2-by-response per cell survive, in per-cell
    descending-response order, cells visited in first-insertion order.
    """
    pts = [(1, 1), (2, 2), (3, 3), (11, 1), (12, 2), (13, 3)]
    responses = [0.5, 0.9, 0.1, 0.3, 0.8, 0.6]
    kps = [
        cv2.KeyPoint(x=float(x), y=float(y), size=5.0, response=float(r))
        for (x, y), r in zip(pts, responses, strict=True)
    ]
    descs = np.arange(6 * 32, dtype=np.uint8).reshape(6, 32)

    cfg = GridFilterConfig(grid_size=(10, 10), max_per_grid=2)
    grid_filter = GridKeypointFilter(cfg)
    out = grid_filter(Features(keypoints=kps, descriptors=descs), (20, 20))

    # Recorded expected selection: original indices [1, 0, 4, 5].
    expected_pts = [(2.0, 2.0), (1.0, 1.0), (12.0, 2.0), (13.0, 3.0)]
    expected_responses = [
        pytest.approx(0.8999999761581421),
        pytest.approx(0.5),
        pytest.approx(0.800000011920929),
        pytest.approx(0.6000000238418579),
    ]
    expected_desc_col0 = [32, 0, 128, 160]

    assert len(out.keypoints) == 4
    for kp, expected_pt, expected_resp in zip(out.keypoints, expected_pts, expected_responses, strict=True):
        assert kp.pt == expected_pt
        assert kp.response == expected_resp
    assert list(out.descriptors[:, 0]) == expected_desc_col0


def test_caps_per_cell():
    kps, descs = _synthetic_keypoints(200, seed=0)
    cfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=3)
    grid_filter = GridKeypointFilter(cfg)
    out = grid_filter(Features(keypoints=kps, descriptors=descs), (240, 320))

    counts = {}
    for kp in out.keypoints:
        row = int(np.clip(kp.pt[1] // 30, 0, (240 // 30) - 1))
        col = int(np.clip(kp.pt[0] // 20, 0, (320 // 20) - 1))
        counts[(row, col)] = counts.get((row, col), 0) + 1

    assert max(counts.values()) <= 3


def test_descriptor_alignment_preserved():
    n = 50
    kps, _ = _synthetic_keypoints(n, seed=3)
    # Encode the original index directly in column 0 (fits in uint8 for n < 256).
    descs = np.zeros((n, 32), dtype=np.uint8)
    descs[:, 0] = np.arange(n, dtype=np.uint8)

    cfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=2)
    grid_filter = GridKeypointFilter(cfg)
    out = grid_filter(Features(keypoints=kps, descriptors=descs), (240, 320))

    # Each selected descriptor's first column encodes its original index;
    # verify it still matches the keypoint it is paired with.
    for kp, desc in zip(out.keypoints, out.descriptors, strict=True):
        original_idx = int(desc[0])
        assert kps[original_idx].pt == kp.pt
