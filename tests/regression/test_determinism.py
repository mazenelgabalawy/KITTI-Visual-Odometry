"""Two in-process runs of the constrained pipeline on sequence 04 produce identical bytes.

Scope: this environment's OpenCV/NumPy build, thread settings, and the current
solvePnPRansac path (whose RANSAC uses a fixed internal RNG state); it says
nothing about other builds or future stochastic solvers.
"""

from __future__ import annotations

from pathlib import Path

from kitti_slam.config.loader import load_config
from kitti_slam.datasets.kitti import KittiSequence
from kitti_slam.evaluation.trajectory_io import TrajectoryWriter
from kitti_slam.odometry.factory import build_pipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(kitti_root: Path, out_path: Path) -> bytes:
    cfg = load_config(_REPO_ROOT / "configs" / "constrained.yaml")
    writer = TrajectoryWriter()
    for pose in build_pipeline(cfg).run(KittiSequence(kitti_root, 4), progress=False):
        writer.append(pose)
    writer.write(out_path)
    return out_path.read_bytes()


def test_determinism_seq04(kitti_root, tmp_path):
    assert _run(kitti_root, tmp_path / "a.txt") == _run(kitti_root, tmp_path / "b.txt")
