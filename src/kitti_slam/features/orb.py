"""ORB detector, optionally followed by grid selection. The cv2.ORB object is
created once and reused; detection is deterministic for a given image and config.
"""

from __future__ import annotations

import cv2
import numpy as np

from kitti_slam.config.schema import DetectorConfig
from kitti_slam.features.base import FeatureDetector, Features
from kitti_slam.features.grid_filter import GridKeypointFilter


class OrbDetector(FeatureDetector):
    def __init__(self, cfg: DetectorConfig, keypoint_filter: GridKeypointFilter | None = None):
        self.cfg = cfg
        self.keypoint_filter = keypoint_filter
        self._orb = cv2.ORB_create(
            nfeatures=cfg.nfeatures,
            scaleFactor=cfg.scale_factor,
            nlevels=cfg.nlevels,
        )

    def detect(self, image: np.ndarray) -> Features:
        keypoints, descriptors = self._orb.detectAndCompute(image, None)
        features = Features(keypoints=keypoints, descriptors=descriptors)
        if self.keypoint_filter is not None:
            features = self.keypoint_filter(features, image.shape[:2])
        return features
