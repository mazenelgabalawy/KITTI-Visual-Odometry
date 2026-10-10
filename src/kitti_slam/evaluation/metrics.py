"""Trajectory error metrics under explicit evaluation profiles, in SI units.

Units (SI throughout -- results are never converted to degrees or percent):

- translation error: m
- rotation error: rad (geodesic angle of the rotation error, in [0, pi])
- KITTI segment translational drift: m/m (error length per traveled length)
- KITTI segment rotational drift: rad/m

Profiles (see `PROFILES`):

``metric_distance`` (the default; RPE defaults to 100 m)
    For comparing experiments: 100 m RPE whose pairs are chosen on the
    reference (ground-truth) path, so every estimate is scored on the same
    pairs. For every start frame i, the end frame j is the one whose reference
    path length from i is closest to 100 m; the pair is kept when that length
    is within 10 m (relative tolerance 0.1). No alignment.
``kitti_segments``
    KITTI odometry devkit drift: segments start every 10 frames; for each
    length L in 100, 200, ..., 800 m the end frame is the first whose
    reference path length from the start exceeds L. Per segment,
    E = inv(inv(P_i) P_j) (inv(Q_i) Q_j) with estimate P and reference Q;
    translational drift ||t(E)|| / L (m/m), rotational drift angle(R(E)) / L
    (rad/m). The KITTI score is the mean over all segments. Implemented here
    independently of evo, in float64.

`evaluate()` defaults to ``metric_distance`` and `compute_rpe()` to its 100 m
reference-selected pairs. Other pair selections (for example a frame interval)
can be passed to `compute_rpe()` explicitly.

Every result records its unit, pose relation, alignment, pair selection,
pose and pair counts, and frame coverage. A metric with no eligible pairs is
returned as unavailable with a reason, never as zero. Trajectories of
different lengths are rejected rather than truncated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from evo.core import filters
from evo.core import metrics as evo_metrics
from evo.core.trajectory import PosePath3D

from kitti_slam.evaluation.pose_checks import check_poses

UNIT_TRANSLATION = "m"
UNIT_ROTATION = "rad"
UNIT_TRANSLATION_DRIFT = "m/m"
UNIT_ROTATION_DRIFT = "rad/m"
ALIGNMENT_NONE = "none"


class TrajectoryInputError(ValueError):
    """Ground truth and estimate cannot be evaluated together."""


# ---------------------------------------------------------------------------
# Result / selection records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorStatistics:
    """Summary of an error array, all in the owning metric's unit (sse in unit^2)."""

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    sse: float

    @classmethod
    def from_errors(cls, errors: np.ndarray) -> ErrorStatistics:
        e = np.asarray(errors, dtype=float)
        sq = np.power(e, 2)
        return cls(
            rmse=float(np.sqrt(np.mean(sq))),
            mean=float(np.mean(e)),
            median=float(np.median(e)),
            std=float(np.std(e)),
            min=float(np.min(e)),
            max=float(np.max(e)),
            sse=float(np.sum(sq)),
        )


@dataclass(frozen=True)
class PairSelection:
    """How RPE pose pairs are chosen.

    pairs_from: "index" (frame interval; identical for both trajectories),
    "reference", or "estimate" (path used to measure metric intervals).
    rel_tolerance applies to metric intervals with all_pairs=True only.
    """

    interval: float
    interval_unit: str  # "frames" | "m"
    all_pairs: bool
    pairs_from: str
    rel_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.interval_unit not in ("frames", "m"):
            raise ValueError(f"interval_unit must be 'frames' or 'm', got {self.interval_unit!r}")
        if self.interval_unit == "frames" and self.pairs_from != "index":
            raise ValueError("frame intervals select pairs by index (pairs_from='index')")
        if self.interval_unit == "m" and self.pairs_from not in ("reference", "estimate"):
            raise ValueError("metric intervals need pairs_from='reference' or 'estimate'")


@dataclass(frozen=True)
class SegmentSelection:
    lengths_m: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
    step_frames: int = 10


@dataclass(frozen=True)
class MetricResult:
    name: str
    unit: str
    pose_relation: str
    alignment: str
    num_poses: int
    num_pairs: int | None  # None for pose-wise (APE) metrics
    frame_coverage: float | None  # fraction of frame intervals spanned by evaluated pairs/poses
    statistics: ErrorStatistics | None
    selection: PairSelection | SegmentSelection | None = None
    unavailable_reason: str | None = None
    breakdown: dict[str, ErrorStatistics] = field(default_factory=dict)  # e.g. per segment length
    # Raw per-pose / per-pair errors (in `unit`) and the (i, j) pairs; omitted from to_dict().
    errors: np.ndarray | None = field(default=None, compare=False, repr=False)
    pairs: tuple[tuple[int, int], ...] | None = field(default=None, compare=False, repr=False)

    @property
    def available(self) -> bool:
        return self.statistics is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("errors")
        d.pop("pairs")
        d["selection"] = None if self.selection is None else {"kind": type(self.selection).__name__, **d["selection"]}
        return d


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    description: str
    include_ape: bool
    rpe: PairSelection | None = None
    segments: SegmentSelection | None = None


@dataclass(frozen=True)
class ProfileEvaluation:
    profile: EvaluationProfile
    num_poses: int
    metrics: tuple[MetricResult, ...]

    def get(self, name: str) -> MetricResult:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.name,
            "description": self.profile.description,
            "units": "SI (m, rad, m/m, rad/m)",
            "num_poses": self.num_poses,
            "metrics": {m.name: m.to_dict() for m in self.metrics},
        }


METRIC_DISTANCE = EvaluationProfile(
    name="metric_distance",
    description=(
        "Experiment comparison: unaligned APE; RPE over 100 m pairs selected on the reference path "
        "(every start frame, nearest end frame, kept within +/-10 m)."
    ),
    include_ape=True,
    rpe=PairSelection(interval=100.0, interval_unit="m", all_pairs=True, pairs_from="reference", rel_tolerance=0.1),
)
KITTI_SEGMENTS = EvaluationProfile(
    name="kitti_segments",
    description=(
        "KITTI odometry devkit drift over 100-800 m reference-path segments starting every 10 frames, in m/m and rad/m."
    ),
    include_ape=False,
    segments=SegmentSelection(),
)
PROFILES: dict[str, EvaluationProfile] = {p.name: p for p in (METRIC_DISTANCE, KITTI_SEGMENTS)}
DEFAULT_PROFILE = METRIC_DISTANCE.name
DEFAULT_RPE_SELECTION = METRIC_DISTANCE.rpe  # 100 m pairs selected on the reference path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_poses(poses: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(poses, dtype=float)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        arr = arr[:, :3, :]
    issues = check_poses(arr)
    if issues:
        raise TrajectoryInputError(f"{label} trajectory is invalid: {'; '.join(issues)}")
    return arr


def validate_pair(gt: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validates and returns (gt, est) as (N, 3, 4) float arrays of equal length."""
    gt_arr = _as_poses(gt, "ground-truth")
    est_arr = _as_poses(est, "estimated")
    if gt_arr.shape[0] != est_arr.shape[0]:
        raise TrajectoryInputError(
            f"Trajectory length mismatch: ground truth has {gt_arr.shape[0]} poses, estimate has "
            f"{est_arr.shape[0]}. Metrics require one estimate per ground-truth pose; nothing is truncated."
        )
    return gt_arr, est_arr


def homogeneous(poses: np.ndarray) -> np.ndarray:
    """(N, 3, 4) -> (N, 4, 4)."""
    poses = np.asarray(poses, dtype=float)
    out = np.zeros((poses.shape[0], 4, 4))
    out[:, :3, :] = poses[:, :3, :4]
    out[:, 3, 3] = 1.0
    return out


def poses_to_trajectory(poses: np.ndarray) -> PosePath3D:
    """(N, 3, 4) pose array -> evo PosePath3D."""
    return PosePath3D(poses_se3=list(homogeneous(poses)))


def path_distances(poses: np.ndarray) -> np.ndarray:
    """Accumulated path length (m) at each pose, starting at 0."""
    t = np.asarray(poses)[:, :3, 3]
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def frame_coverage(pairs: list[tuple[int, int]], num_poses: int) -> float | None:
    """Fraction of the num_poses - 1 frame intervals covered by any pair [i, j)."""
    if num_poses < 2:
        return None
    covered = np.zeros(num_poses - 1, dtype=bool)
    for i, j in pairs:
        covered[min(i, j) : max(i, j)] = True
    return float(covered.mean())


def _unavailable(name, unit, relation, num_poses, selection, reason) -> MetricResult:
    return MetricResult(
        name=name,
        unit=unit,
        pose_relation=relation,
        alignment=ALIGNMENT_NONE,
        num_poses=num_poses,
        num_pairs=0,
        frame_coverage=0.0 if num_poses >= 2 else None,
        statistics=None,
        selection=selection,
        unavailable_reason=reason,
    )


# ---------------------------------------------------------------------------
# APE / RPE (evo)
# ---------------------------------------------------------------------------

_RELATIONS = {
    "translation": (evo_metrics.PoseRelation.translation_part, UNIT_TRANSLATION),
    "rotation": (evo_metrics.PoseRelation.rotation_angle_rad, UNIT_ROTATION),
}


def compute_ape(gt: np.ndarray, est: np.ndarray, quantity: str = "translation") -> MetricResult:
    """Unaligned absolute pose error (evo APE) for "translation" (m) or "rotation" (rad)."""
    gt_arr, est_arr = validate_pair(gt, est)
    relation, unit = _RELATIONS[quantity]
    ape = evo_metrics.APE(relation)
    ape.process_data((poses_to_trajectory(gt_arr), poses_to_trajectory(est_arr)))
    n = gt_arr.shape[0]
    return MetricResult(
        name=f"ape_{quantity}",
        unit=unit,
        pose_relation=relation.name,
        alignment=ALIGNMENT_NONE,
        num_poses=n,
        num_pairs=None,
        frame_coverage=1.0 if n >= 2 else None,
        statistics=ErrorStatistics.from_errors(ape.error),
        errors=np.asarray(ape.error, dtype=float),
    )


def select_pairs(gt: np.ndarray, est: np.ndarray, selection: PairSelection) -> list[tuple[int, int]]:
    """RPE pose pairs per `selection`, via evo's own pair filter (empty if none qualify)."""
    source = est if selection.pairs_from == "estimate" else gt
    poses = list(homogeneous(source))
    if selection.interval_unit == "frames":
        return filters.filter_pairs_by_index(poses, int(selection.interval), selection.all_pairs)
    tol = selection.interval * (selection.rel_tolerance or 0.0)
    return filters.filter_pairs_by_path(poses, float(selection.interval), tol, selection.all_pairs)


def compute_rpe(
    gt: np.ndarray, est: np.ndarray, selection: PairSelection | None = None, quantity: str = "translation"
) -> MetricResult:
    """Unaligned relative pose error (evo RPE) on the pairs chosen by `selection`
    (default: 100 m pairs selected on the reference path)."""
    selection = DEFAULT_RPE_SELECTION if selection is None else selection
    gt_arr, est_arr = validate_pair(gt, est)
    relation, unit = _RELATIONS[quantity]
    n = gt_arr.shape[0]
    name = f"rpe_{quantity}"
    pairs = select_pairs(gt_arr, est_arr, selection)
    if not pairs:
        reason = (
            f"no pose pairs {selection.interval:g} {selection.interval_unit} apart "
            f"(selected on the {selection.pairs_from} path) in a {n}-pose trajectory"
        )
        return _unavailable(name, unit, relation.name, n, selection, reason)

    rpe = evo_metrics.RPE(
        pose_relation=relation,
        delta=selection.interval,
        delta_unit=evo_metrics.Unit.frames if selection.interval_unit == "frames" else evo_metrics.Unit.meters,
        rel_delta_tol=selection.rel_tolerance if selection.rel_tolerance is not None else 0.1,
        all_pairs=selection.all_pairs,
        pairs_from_reference=selection.pairs_from == "reference",
    )
    rpe.process_data((poses_to_trajectory(gt_arr), poses_to_trajectory(est_arr)))
    if len(rpe.error) != len(pairs):  # evo re-derives the pairs; they must be ours
        raise RuntimeError(f"evo used {len(rpe.error)} pairs, expected {len(pairs)}")
    return MetricResult(
        name=name,
        unit=unit,
        pose_relation=relation.name,
        alignment=ALIGNMENT_NONE,
        num_poses=n,
        num_pairs=len(pairs),
        frame_coverage=frame_coverage(pairs, n),
        statistics=ErrorStatistics.from_errors(rpe.error),
        selection=selection,
        errors=np.asarray(rpe.error, dtype=float),
        pairs=tuple((int(i), int(j)) for i, j in pairs),
    )


# ---------------------------------------------------------------------------
# KITTI segment drift (independent of evo)
# ---------------------------------------------------------------------------


def _rotation_error_kitti(R: np.ndarray) -> np.ndarray:
    """Devkit rotationError: acos(clamp((trace(R) - 1) / 2)) in rad, vectorized."""
    d = 0.5 * (np.trace(R, axis1=-2, axis2=-1) - 1.0)
    return np.arccos(np.clip(d, -1.0, 1.0))


def kitti_segments(gt: np.ndarray, selection: SegmentSelection | None = None) -> list[tuple[int, int, float]]:
    """(first, last, length_m) segments in devkit order, chosen on the reference path:
    `last` is the first frame whose path length from `first` exceeds the length."""
    selection = selection or SegmentSelection()
    dist = path_distances(gt)
    segments = []
    for first in range(0, dist.size, selection.step_frames):
        for length in selection.lengths_m:
            # dist is non-decreasing: side="right" gives the first index with dist > target.
            last = int(np.searchsorted(dist, dist[first] + length, side="right"))
            if last < dist.size:
                segments.append((first, last, float(length)))
    return segments


def compute_segment_drift(
    gt: np.ndarray, est: np.ndarray, selection: SegmentSelection | None = None
) -> tuple[MetricResult, MetricResult]:
    """KITTI devkit segment drift: (translation m/m, rotation rad/m)."""
    selection = selection or SegmentSelection()
    gt_arr, est_arr = validate_pair(gt, est)
    n = gt_arr.shape[0]
    names = ("segment_translation_drift", "segment_rotation_drift")
    relations = ("kitti_segment_translation_per_length", "kitti_segment_rotation_per_length")
    units = (UNIT_TRANSLATION_DRIFT, UNIT_ROTATION_DRIFT)

    segments = kitti_segments(gt_arr, selection)
    if not segments:
        reason = f"reference path shorter than the shortest segment ({min(selection.lengths_m):g} m)"
        return tuple(
            _unavailable(nm, u, r, n, selection, reason) for nm, u, r in zip(names, units, relations, strict=True)
        )

    Q, P = homogeneous(gt_arr), homogeneous(est_arr)
    first = np.array([s[0] for s in segments])
    last = np.array([s[1] for s in segments])
    lengths = np.array([s[2] for s in segments])
    delta_gt = np.linalg.inv(Q[first]) @ Q[last]
    delta_est = np.linalg.inv(P[first]) @ P[last]
    error = np.linalg.inv(delta_est) @ delta_gt
    t_drift = np.linalg.norm(error[:, :3, 3], axis=1) / lengths
    r_drift = _rotation_error_kitti(error[:, :3, :3]) / lengths

    coverage = frame_coverage(list(zip(first.tolist(), last.tolist(), strict=True)), n)
    results = []
    for nm, u, rel, values in zip(names, units, relations, (t_drift, r_drift), strict=True):
        breakdown = {
            f"{L:g}": ErrorStatistics.from_errors(values[lengths == L])
            for L in selection.lengths_m
            if (lengths == L).any()
        }
        results.append(
            MetricResult(
                name=nm,
                unit=u,
                pose_relation=rel,
                alignment=ALIGNMENT_NONE,
                num_poses=n,
                num_pairs=len(segments),
                frame_coverage=coverage,
                statistics=ErrorStatistics.from_errors(values),
                selection=selection,
                breakdown=breakdown,
                errors=values,
                pairs=tuple((int(i), int(j)) for i, j in zip(first, last, strict=True)),
            )
        )
    return results[0], results[1]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def evaluate(gt: np.ndarray, est: np.ndarray, profile: str | EvaluationProfile = DEFAULT_PROFILE) -> ProfileEvaluation:
    """Evaluates `est` against `gt` with every metric the profile defines (default: metric_distance)."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    gt_arr, est_arr = validate_pair(gt, est)
    results: list[MetricResult] = []
    if prof.include_ape:
        results += [compute_ape(gt_arr, est_arr, "translation"), compute_ape(gt_arr, est_arr, "rotation")]
    if prof.rpe is not None:
        results += [
            compute_rpe(gt_arr, est_arr, prof.rpe, "translation"),
            compute_rpe(gt_arr, est_arr, prof.rpe, "rotation"),
        ]
    if prof.segments is not None:
        results += list(compute_segment_drift(gt_arr, est_arr, prof.segments))
    return ProfileEvaluation(profile=prof, num_poses=gt_arr.shape[0], metrics=tuple(results))
