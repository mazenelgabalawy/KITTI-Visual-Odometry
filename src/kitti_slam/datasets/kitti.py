"""KITTI odometry benchmark stereo sequence loader.

Ports utils.load_stereo_pair and the sequence-length/calib-file discovery
logic from VO_base.py / VO_constrained.py (L41-61 in the latter).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import cv2

from kitti_slam.datasets.base import StereoFrame, StereoSequence
from kitti_slam.geometry.calibration import StereoCalibration


class KittiSequence(StereoSequence):
    def __init__(self, dataset_root: str | Path, sequence: int) -> None:
        self._sequence_dir = Path(dataset_root) / f"{sequence:02d}"
        if not self._sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {self._sequence_dir}")

        self._calib_path = self._sequence_dir / "calib.txt"
        if not self._calib_path.is_file():
            raise FileNotFoundError(f"KITTI calibration file not found: {self._calib_path}")

        self._length = len(glob(str(self._sequence_dir / "image_0" / "*.png")))
        self._calibration: StereoCalibration | None = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> StereoFrame:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Frame index {idx} out of range for sequence of length {self._length}")

        left_path = self._sequence_dir / "image_0" / f"{idx:06d}.png"
        right_path = self._sequence_dir / "image_1" / f"{idx:06d}.png"

        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

        if left is None:
            raise OSError(f"Failed to read left image: {left_path}")
        if right is None:
            raise OSError(f"Failed to read right image: {right_path}")

        return StereoFrame(index=idx, left=left, right=right)

    @property
    def calibration(self) -> StereoCalibration:
        if self._calibration is None:
            self._calibration = StereoCalibration.from_kitti_calib_file(self._calib_path)
        return self._calibration
