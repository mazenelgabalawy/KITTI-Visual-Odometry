"""Abstract stereo sequence interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from kitti_slam.geometry.calibration import StereoCalibration


@dataclass(frozen=True)
class StereoFrame:
    index: int
    left: np.ndarray
    right: np.ndarray


class StereoSequence(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> StereoFrame: ...

    @property
    @abstractmethod
    def calibration(self) -> StereoCalibration: ...

    def __iter__(self) -> Iterator[StereoFrame]:
        for idx in range(len(self)):
            yield self[idx]
