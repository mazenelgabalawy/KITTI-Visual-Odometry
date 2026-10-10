"""Strict pose-file validation and full-pose comparison (plan task A2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kitti_slam.evaluation.pose_checks import check_pose_file, check_poses, compare_pose_files
from kitti_slam.evaluation.trajectory_io import TrajectoryWriter


def _trajectory(n: int = 30, seed: int = 0) -> np.ndarray:
    """Plausible accumulated VO poses starting at the exact identity."""
    rng = np.random.default_rng(seed)
    C = np.eye(4)
    poses = [C.copy()]
    for _ in range(n - 1):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix()
        T[:3, 3] = [rng.normal(0, 0.05), rng.normal(0, 0.01), 1.0 + rng.normal(0, 0.1)]
        C = C @ T
        poses.append(C.copy())
    return np.stack(poses)


def _write(path, poses) -> None:
    writer = TrajectoryWriter()
    for p in poses:
        writer.append(p)
    writer.write(path)


def _translation_only_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """The previous parity gate: translation-only RMSE over min(lengths)."""
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean(np.sum((a[:n, :3, 3] - b[:n, :3, 3]) ** 2, axis=1))))


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.txt"
    _write(path, _trajectory())
    return path


def test_valid_file_passes(reference):
    report = check_pose_file(reference, expected_poses=30, require_identity_start=True)
    assert report.ok, report.issues
    assert report.num_poses == 30


def test_identical_files_compare_identical(reference, tmp_path):
    copy = tmp_path / "copy.txt"
    copy.write_bytes(reference.read_bytes())
    c = compare_pose_files(reference, copy, expected_poses=30, require_identity_start=True)
    assert c.identical
    assert c.issues == ()
    assert c.max_abs_element_diff == 0.0
    assert c.first_differing_pose is None


def test_rotation_only_corruption_is_detected(reference, tmp_path):
    """A rotation change with untouched translations passes the old translation-only
    gate but fails byte comparison and is quantified in radians."""
    poses = _trajectory()
    corrupted = poses.copy()
    corrupted[12, :3, :3] = poses[12, :3, :3] @ Rotation.from_rotvec([0, 0.02, 0]).as_matrix()
    path = tmp_path / "rotated.txt"
    _write(path, corrupted)

    assert _translation_only_rmse(poses, corrupted) == 0.0
    c = compare_pose_files(reference, path, expected_poses=30)
    assert not c.identical
    assert c.candidate.ok  # still a valid rotation, so only the comparison catches it
    assert c.differing_poses == 1
    assert c.first_differing_pose == 12
    assert c.max_translation_diff_m == pytest.approx(0.0, abs=1e-6)
    assert c.max_rotation_diff_rad == pytest.approx(0.02, abs=1e-5)


def test_non_orthonormal_rotation_is_invalid(tmp_path):
    poses = _trajectory()
    poses[7, 0, 0] += 1e-3
    path = tmp_path / "bad_rotation.txt"
    _write(path, poses)
    report = check_pose_file(path)
    assert not report.ok
    assert any("invalid rotation at pose 7" in i for i in report.issues)


def test_truncated_file_missing_poses(reference, tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("".join(reference.read_text().splitlines(keepends=True)[:20]))
    assert _translation_only_rmse(_trajectory(), _trajectory()[:20]) == 0.0  # old gate: passes
    c = compare_pose_files(reference, path, expected_poses=30)
    assert not c.identical
    assert any("expected 30 poses, found 20" in i for i in c.candidate.issues)
    assert any("pose count differs" in i for i in c.issues)
    assert c.first_differing_pose == 20


def test_truncated_mid_line(reference, tmp_path):
    path = tmp_path / "cut.txt"
    data = reference.read_bytes()
    path.write_bytes(data[: len(data) - 40])
    report = check_pose_file(path, expected_poses=30)
    assert not report.ok
    assert any("missing trailing newline" in i for i in report.issues)
    assert any("malformed line" in i for i in report.issues)


def test_nonfinite_pose_is_invalid(reference, tmp_path):
    lines = reference.read_text().splitlines()
    values = lines[5].split()
    values[3] = "nan"
    lines[5] = " ".join(values)
    path = tmp_path / "nan.txt"
    path.write_text("\n".join(lines) + "\n")
    report = check_pose_file(path)
    assert any("nonfinite values at pose 5" in i for i in report.issues)

    values[3] = "inf"
    lines[5] = " ".join(values)
    path.write_text("\n".join(lines) + "\n")
    assert any("nonfinite" in i for i in check_pose_file(path).issues)


def test_wrong_column_count_and_garbage(tmp_path):
    path = tmp_path / "garbage.txt"
    path.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0\nfoo bar\n")
    report = check_pose_file(path)
    assert any("malformed line" in i and "pose 1, 2" in i for i in report.issues)


def test_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert check_pose_file(empty).issues == ("empty file",)
    assert "unreadable" in check_pose_file(tmp_path / "absent.txt").issues[0]


def test_identity_start_required(tmp_path):
    poses = _trajectory()
    poses[0, 0, 3] = 0.5
    path = tmp_path / "offset.txt"
    _write(path, poses)
    assert check_pose_file(path).ok
    assert any("identity" in i for i in check_pose_file(path, require_identity_start=True).issues)


def test_serialization_rounding_stays_within_rotation_tolerance(tmp_path):
    """%.6e rounding of long accumulated trajectories must not trip the validity check."""
    poses = _trajectory(n=3000, seed=3)
    path = tmp_path / "long.txt"
    _write(path, poses)
    assert check_pose_file(path, expected_poses=3000, require_identity_start=True).ok


def test_check_poses_shape():
    assert "expected an (N, 3, 4)" in check_poses(np.zeros((3, 4)))[0]
