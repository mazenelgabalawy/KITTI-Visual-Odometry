"""Historical metric fixtures: the `legacy_frames` profile must reproduce every
committed `output_base/XX` and `output_constrained/XX` evo CLI result.

The historical files came from `create_results.sh`, i.e. `evo_ape kitti gt est`
and `evo_rpe kitti gt est --delta 100` with evo's default *frame* interval
(their header prints the literal "{self.delta}" placeholder). Only the
ground-truth poses need to be on disk; the estimates are the committed
historical trajectories, so no pipeline run is involved. Expected values are
the historical numbers, never regenerated.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from kitti_slam.evaluation.metrics import LEGACY_FRAMES, compute_ape, compute_rpe
from kitti_slam.evaluation.trajectory_io import load_kitti_poses

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORICAL = ("output_base", "output_constrained")
_SEQUENCES = [f"{i:02d}" for i in range(11)]
_STATS = ("max", "mean", "median", "min", "rmse", "sse", "std")
# The historical text prints 6 decimals: allow half a unit in the last place,
# plus float64 summation-order slack proportional to the magnitude (sse of
# long sequences reaches ~5e7 m^2).
_ROUNDING = 5e-7
_REL_SLACK = 1e-9

# output_base/01/poses.txt was truncated to the identity pose in commit b8796f7
# (the legacy runner truncates the next sequence's file, then crashes on its
# second sequence); its committed metric files describe the earlier 1101-pose
# trajectory, Git blob bc8aed40... (`git show 9f15e7c:output_base/01/poses.txt`).
_PRE_TRUNCATION_BLOBS = {("output_base", "01"): "bc8aed404dcc8f08d2057c9160847303663eff65"}


def _parse_evo_result_file(path: Path) -> dict[str, float]:
    """'       rmse\\t108.783267' lines -> {"rmse": 108.783267, ...}."""
    pattern = re.compile(r"^\s*(max|mean|median|min|rmse|sse|std)\s+([0-9.eE+-]+)\s*$")
    stats = {}
    for line in path.read_text().splitlines():
        if match := pattern.match(line):
            stats[match.group(1)] = float(match.group(2))
    return stats


def _assert_matches(result, expected: dict[str, float], label: str) -> None:
    assert set(expected) == set(_STATS), f"{label}: historical file lacks statistics {set(_STATS) - set(expected)}"
    for key in _STATS:
        actual = getattr(result.statistics, key)
        tol = _ROUNDING + _REL_SLACK * abs(expected[key])
        assert abs(actual - expected[key]) <= tol, f"{label} {key}: {actual!r} vs historical {expected[key]!r}"


def _load(gt_dir: Path, tree: str, seq: str, tmp_path: Path):
    est_file = _REPO_ROOT / tree / seq / "poses.txt"
    if (tree, seq) in _PRE_TRUNCATION_BLOBS:
        blob = _PRE_TRUNCATION_BLOBS[(tree, seq)]
        proc = subprocess.run(["git", "cat-file", "-p", blob], cwd=_REPO_ROOT, capture_output=True)
        if proc.returncode != 0:
            pytest.skip(f"{tree}/{seq}: committed poses.txt is truncated and Git blob {blob} is unavailable")
        est_file = tmp_path / "poses.txt"
        est_file.write_bytes(proc.stdout)
    if not est_file.is_file():
        pytest.skip(f"{est_file} not present")
    return load_kitti_poses(gt_dir / f"{seq}.txt"), load_kitti_poses(est_file)


def test_truncated_historical_trajectory_is_still_truncated():
    """Documents the known historical defect; if this fails, the frozen tree changed."""
    identity = " ".join(f"{v:.6e}" for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))
    assert (_REPO_ROOT / "output_base" / "01" / "poses.txt").read_text() == identity + "\n"


@pytest.mark.parametrize("tree", _HISTORICAL)
@pytest.mark.parametrize("seq", _SEQUENCES)
def test_legacy_frames_ape_matches_historical(gt_dir, tree, seq, tmp_path):
    gt, est = _load(gt_dir, tree, seq, tmp_path)
    result = compute_ape(gt, est, "translation")
    assert result.unit == "m"
    _assert_matches(result, _parse_evo_result_file(_REPO_ROOT / tree / seq / "ape_results.txt"), f"{tree}/{seq} APE")


@pytest.mark.parametrize("tree", _HISTORICAL)
@pytest.mark.parametrize("seq", _SEQUENCES)
def test_legacy_frames_rpe_matches_historical(gt_dir, tree, seq, tmp_path):
    gt, est = _load(gt_dir, tree, seq, tmp_path)
    result = compute_rpe(gt, est, LEGACY_FRAMES.rpe, "translation")
    assert result.unit == "m"
    expected = _parse_evo_result_file(_REPO_ROOT / tree / seq / "rpe_results.txt")
    _assert_matches(result, expected, f"{tree}/{seq} RPE")
    # The historical pair count is implied by sse / rmse^2: consecutive 100-frame pairs.
    assert result.num_pairs == round(expected["sse"] / expected["rmse"] ** 2) == (len(gt) - 1) // 100
