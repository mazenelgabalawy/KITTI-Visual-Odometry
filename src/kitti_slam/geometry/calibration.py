"""Stereo camera calibration parsing (KITTI calib.txt format).

Ports the calibration handling from VO_constrained.py (L57-77) and
VO_base.py (L44-60): parse the first two lines of a KITTI ``calib.txt``
into 3x4 projection matrices, decompose them with
``cv2.decomposeProjectionMatrix``, normalize the resulting homogeneous
translation vector, and derive the stereo baseline and focal length.

Note: the legacy code also computes the essential/fundamental matrices
from these values, but never uses them (VO_constrained.py L78-83,
VO_base.py L62-67). That computation is intentionally NOT ported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    P0: np.ndarray
    P1: np.ndarray
    K: np.ndarray
    baseline: float
    focal_length: float

    @classmethod
    def from_kitti_calib_file(cls, path: str | Path) -> StereoCalibration:
        """Parse a KITTI calib.txt file (first two lines: P0, P1)."""
        path = Path(path)
        with open(path) as f:
            lines = f.readlines()

        P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4)
        P1 = np.array([float(x) for x in lines[1].strip().split()[1:]]).reshape(3, 4)

        return cls.from_projection_matrices(P0, P1)

    @classmethod
    def from_projection_matrices(cls, P0: np.ndarray, P1: np.ndarray) -> StereoCalibration:
        """Replicates VO_constrained.py L63-76 exactly (minus the unused E/F)."""
        K0, R0, t0 = cv2.decomposeProjectionMatrix(P0)[:3]
        K1, R1, t1 = cv2.decomposeProjectionMatrix(P1)[:3]

        # Normalize translation vector (t0 is zero vector for the left camera).
        t0 = (t0[:3] / t0[3]).reshape(3)
        t1 = (t1[:3] / t1[3]).reshape(3)

        baseline = float(np.linalg.norm(t1 - t0))
        focal_length = float(K0[0, 0])

        return cls(P0=P0, P1=P1, K=K0, baseline=baseline, focal_length=focal_length)
