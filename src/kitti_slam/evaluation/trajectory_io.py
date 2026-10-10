"""Trajectory writing/loading, byte-identical to utils.save_transformation_matrix.

Ports utils.py L11-23 (save_transformation_matrix): the top-left 3x4 block of
each pose, flattened row-major, formatted as ``' '.join(f"{x:.6e}" for x in
row)``, one line per pose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _format_pose_line(C: np.ndarray) -> str:
    mat = np.asarray(C)
    row = mat[:3, :4].reshape(-1)
    return " ".join(f"{x:.6e}" for x in row)


class TrajectoryWriter:
    """Accumulates 4x4 (or 3x4) poses and writes them in the legacy KITTI
    pose-file format."""

    def __init__(self) -> None:
        self._poses: list[np.ndarray] = []

    def append(self, C: np.ndarray) -> None:
        self._poses.append(np.asarray(C))

    def as_array(self) -> np.ndarray:
        """Returns an (N, 3, 4) array of the accumulated poses."""
        return np.stack([np.asarray(C)[:3, :4] for C in self._poses], axis=0)

    def write(self, path: str | Path) -> None:
        lines = [_format_pose_line(C) for C in self._poses]
        content = "".join(line + "\n" for line in lines)
        with open(path, "w") as f:
            f.write(content)


def load_kitti_poses(path: str | Path) -> np.ndarray:
    """Loads a KITTI-format pose file into an (N, 3, 4) array."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = np.array([float(x) for x in line.split()])
            poses.append(values.reshape(3, 4))
    return np.stack(poses, axis=0)
