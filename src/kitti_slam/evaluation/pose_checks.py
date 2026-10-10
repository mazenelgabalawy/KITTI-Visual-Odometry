"""Strict validation and full-pose comparison of KITTI pose files.

Used as the gate for reference parity: a candidate trajectory passes only when
both files are well formed and their bytes are identical. Numeric difference
summaries (element, translation in m, rotation in rad) are diagnostics for
explaining a mismatch, never a substitute for the byte comparison.

Validity rules for one file:

- non-empty, newline-terminated (a missing final newline indicates a truncated
  or partially written file), ASCII text;
- every line holds exactly 12 numeric values (the row-major 3x4 pose);
- every value is finite;
- every rotation block is orthonormal with determinant +1 within
  `ROTATION_TOLERANCE`, which comfortably covers `%.6e` serialization
  (<= 5e-7 absolute per element for |r| <= 1) but rejects corrupted rotations;
- optionally, an exact expected pose count and an exact identity first pose
  (the VO initialization pose).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitti_slam.geometry.transforms import rotation_angle_between

ROTATION_TOLERANCE = 1e-5
_MAX_LISTED = 5


@dataclass(frozen=True)
class PoseFileReport:
    path: str
    num_poses: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ParsedPoseFile:
    report: PoseFileReport
    lines: tuple[str, ...]
    poses: np.ndarray | None  # (N, 3, 4); None when any line is malformed


@dataclass(frozen=True)
class PoseFileComparison:
    reference: PoseFileReport
    candidate: PoseFileReport
    bytes_identical: bool
    compared_poses: int
    differing_poses: int
    first_differing_pose: int | None
    max_abs_element_diff: float | None
    max_translation_diff_m: float | None
    max_rotation_diff_rad: float | None

    @property
    def identical(self) -> bool:
        return self.bytes_identical and self.reference.ok and self.candidate.ok

    @property
    def issues(self) -> tuple[str, ...]:
        out = [f"reference: {i}" for i in self.reference.issues]
        out += [f"candidate: {i}" for i in self.candidate.issues]
        if self.reference.num_poses != self.candidate.num_poses:
            out.append(
                f"pose count differs: reference {self.reference.num_poses}, candidate {self.candidate.num_poses}"
            )
        if not self.bytes_identical:
            out.append(
                f"bytes differ: {self.differing_poses} of {self.compared_poses} compared poses differ, "
                f"first at pose {self.first_differing_pose}"
            )
        return tuple(out)


def _summarize(kind: str, indices: list[int]) -> str:
    listed = ", ".join(str(i) for i in indices[:_MAX_LISTED])
    more = f" (+{len(indices) - _MAX_LISTED} more)" if len(indices) > _MAX_LISTED else ""
    return f"{kind} at pose {listed}{more}"


def check_poses(
    poses: np.ndarray,
    expected_poses: int | None = None,
    require_identity_start: bool = False,
) -> list[str]:
    """Validity issues of an (N, 3, 4) pose array (empty list when valid)."""
    poses = np.asarray(poses, dtype=float)
    issues: list[str] = []
    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        return [f"expected an (N, 3, 4) pose array, got shape {poses.shape}"]
    n = poses.shape[0]
    if n == 0:
        issues.append("no poses")
    if expected_poses is not None and n != expected_poses:
        issues.append(f"expected {expected_poses} poses, found {n}")

    finite = np.isfinite(poses).all(axis=(1, 2))
    if not finite.all():
        issues.append(_summarize("nonfinite values", np.flatnonzero(~finite).tolist()))

    if finite.any():
        R = poses[finite, :, :3]
        ortho_err = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max(axis=(1, 2))
        det_err = np.abs(np.linalg.det(R) - 1.0)
        bad = (ortho_err > ROTATION_TOLERANCE) | (det_err > ROTATION_TOLERANCE)
        if bad.any():
            idx = np.flatnonzero(finite)[bad]
            worst = float(max(ortho_err[bad].max(), det_err[bad].max()))
            issues.append(
                _summarize("invalid rotation", idx.tolist())
                + f" (max orthonormality/determinant error {worst:.3e} > {ROTATION_TOLERANCE:.0e})"
            )

    if require_identity_start and n > 0 and not np.array_equal(poses[0], np.eye(3, 4)):
        issues.append("first pose is not the exact identity initialization pose")
    return issues


def read_pose_file(
    path: str | Path,
    expected_poses: int | None = None,
    require_identity_start: bool = False,
) -> ParsedPoseFile:
    """Strictly parses and validates one KITTI pose file."""
    path = Path(path)
    issues: list[str] = []
    try:
        data = path.read_bytes()
    except OSError as exc:
        report = PoseFileReport(str(path), 0, (f"unreadable: {exc}",))
        return ParsedPoseFile(report, (), None)

    if not data:
        return ParsedPoseFile(PoseFileReport(str(path), 0, ("empty file",)), (), None)
    if not data.endswith(b"\n"):
        issues.append("missing trailing newline (truncated or partially written file)")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return ParsedPoseFile(PoseFileReport(str(path), 0, ("not ASCII text",)), (), None)

    lines = tuple(text.split("\n"))
    if lines and lines[-1] == "":
        lines = lines[:-1]

    malformed: list[int] = []
    rows: list[list[float]] = []
    for i, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) != 12:
            malformed.append(i)
            continue
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError:
            malformed.append(i)

    poses = None
    if malformed:
        issues.append(_summarize("malformed line (expected 12 numeric values)", malformed))
    else:
        poses = np.asarray(rows, dtype=float).reshape(-1, 3, 4)
        issues.extend(check_poses(poses, expected_poses, require_identity_start))

    if malformed and expected_poses is not None and len(lines) != expected_poses:
        issues.append(f"expected {expected_poses} poses, found {len(lines)} lines")

    return ParsedPoseFile(PoseFileReport(str(path), len(lines), tuple(issues)), lines, poses)


def check_pose_file(
    path: str | Path,
    expected_poses: int | None = None,
    require_identity_start: bool = False,
) -> PoseFileReport:
    return read_pose_file(path, expected_poses, require_identity_start).report


def compare_pose_files(
    reference: str | Path,
    candidate: str | Path,
    expected_poses: int | None = None,
    require_identity_start: bool = False,
) -> PoseFileComparison:
    """Validates both files and compares them byte-for-byte, with numeric
    difference diagnostics over the common prefix of well-formed poses."""
    ref = read_pose_file(reference, expected_poses, require_identity_start)
    cand = read_pose_file(candidate, expected_poses, require_identity_start)

    try:
        bytes_identical = Path(reference).read_bytes() == Path(candidate).read_bytes()
    except OSError:
        bytes_identical = False

    n = min(len(ref.lines), len(cand.lines))
    differing = [i for i in range(n) if ref.lines[i] != cand.lines[i]]
    first = differing[0] if differing else (n if len(ref.lines) != len(cand.lines) else None)
    if bytes_identical:
        first = None

    max_elem = max_trans = max_rot = None
    if ref.poses is not None and cand.poses is not None and n > 0:
        a, b = ref.poses[:n], cand.poses[:n]
        with np.errstate(invalid="ignore"):
            max_elem = float(np.max(np.abs(a - b)))
            max_trans = float(np.max(np.linalg.norm(a[:, :, 3] - b[:, :, 3], axis=1)))
            max_rot = float(np.max(rotation_angle_between(a[:, :, :3], b[:, :, :3])))

    return PoseFileComparison(
        reference=ref.report,
        candidate=cand.report,
        bytes_identical=bytes_identical,
        compared_poses=n,
        differing_poses=len(differing),
        first_differing_pose=first,
        max_abs_element_diff=max_elem,
        max_translation_diff_m=max_trans,
        max_rotation_diff_rad=max_rot,
    )
