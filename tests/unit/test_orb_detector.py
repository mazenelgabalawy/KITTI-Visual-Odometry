"""Tests for kitti_slam.features.orb.OrbDetector."""

import numpy as np

from kitti_slam.config.schema import DetectorConfig, GridFilterConfig
from kitti_slam.features.grid_filter import GridKeypointFilter
from kitti_slam.features.orb import OrbDetector


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(240, 320), dtype=np.uint8)


def test_detect_matches_golden(goldens):
    """Vs. the frozen ORB + grid output (pinned OpenCV 4.11.0 ORB)."""
    image = _synthetic_image(0)
    assert np.array_equal(image, goldens["orb_image"])
    dcfg = DetectorConfig(nfeatures=500, scale_factor=1.2, nlevels=8)
    gcfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=5)

    features = OrbDetector(dcfg, GridKeypointFilter(gcfg)).detect(image)

    np.testing.assert_array_equal([kp.pt for kp in features.keypoints], goldens["orb_pts"])
    np.testing.assert_array_equal([kp.response for kp in features.keypoints], goldens["orb_responses"])
    assert np.array_equal(features.descriptors, goldens["orb_desc"])


def test_detector_reuse_deterministic():
    image = _synthetic_image(0)
    dcfg = DetectorConfig(nfeatures=500, scale_factor=1.2, nlevels=8)
    gcfg = GridFilterConfig(grid_size=(30, 20), max_per_grid=5)

    detector = OrbDetector(dcfg, GridKeypointFilter(gcfg))
    first = detector.detect(image)
    second = detector.detect(image)

    assert len(first.keypoints) == len(second.keypoints)
    for a, b in zip(first.keypoints, second.keypoints, strict=True):
        assert a.pt == b.pt
        assert a.response == b.response
    assert np.array_equal(first.descriptors, second.descriptors)


def test_detect_without_grid_filter_matches_raw_orb():
    image = _synthetic_image(1)
    dcfg = DetectorConfig(nfeatures=300, scale_factor=1.2, nlevels=8)

    detector = OrbDetector(dcfg, keypoint_filter=None)
    features = detector.detect(image)

    import cv2

    orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.2, nlevels=8)
    raw_kps, raw_descs = orb.detectAndCompute(image, None)

    assert len(features.keypoints) == len(raw_kps)
    assert np.array_equal(features.descriptors, raw_descs)
