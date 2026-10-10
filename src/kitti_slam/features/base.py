"""Shared feature/matcher interfaces used by the detector and matcher implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Features:
    """Container for detected keypoints and their descriptors."""

    keypoints: list
    descriptors: np.ndarray | None


class FeatureDetector(ABC):
    """Detects keypoints/descriptors in a single image."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> Features: ...


class Matcher(ABC):
    """Matches descriptors between two sets of Features."""

    @abstractmethod
    def match(self, feat_a: Features, feat_b: Features) -> list: ...
