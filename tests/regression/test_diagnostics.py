"""`run_vo.py --diagnostics` writes one row per frame and leaves poses.txt unchanged."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SEQUENCE = 4


def _run(variant: str, kitti_root: Path, output: Path, diagnostics: bool) -> Path:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_vo.py"),
        "--config",
        str(_REPO_ROOT / "configs" / f"{variant}.yaml"),
        "--sequence",
        str(_SEQUENCE),
        "--dataset-path",
        str(kitti_root),
        "--output",
        str(output),
        "--no-progress",
    ]
    if diagnostics:
        cmd.append("--diagnostics")
    subprocess.run(cmd, check=True, capture_output=True)
    return output / f"{_SEQUENCE:02d}"


@pytest.mark.parametrize("variant", ["base", "constrained"])
def test_diagnostics_leave_poses_unchanged(variant, kitti_root, tmp_path):
    plain = _run(variant, kitti_root, tmp_path / "plain", diagnostics=False)
    diag = _run(variant, kitti_root, tmp_path / "diag", diagnostics=True)

    assert (plain / "poses.txt").read_bytes() == (diag / "poses.txt").read_bytes()
    assert not (plain / "diagnostics.csv").exists()

    num_poses = len((plain / "poses.txt").read_text().splitlines())
    with open(diag / "diagnostics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["frame"]) for r in rows] == list(range(1, num_poses))
