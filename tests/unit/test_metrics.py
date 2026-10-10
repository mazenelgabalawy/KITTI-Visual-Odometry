"""Synthetic known-motion tests for kitti_slam.evaluation.metrics.

No dataset or ground-truth files: every expected value is derived in closed
form from the constructed trajectories. All metrics are in SI units.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from evo.core import metrics as evo_metrics

from kitti_slam.evaluation.metrics import (
    KITTI_SEGMENTS,
    METRIC_DISTANCE,
    PROFILES,
    ErrorStatistics,
    PairSelection,
    SegmentSelection,
    TrajectoryInputError,
    compute_ape,
    compute_rpe,
    compute_segment_drift,
    evaluate,
    kitti_segments,
    poses_to_trajectory,
)

SI_UNITS = {"m", "rad", "m/m", "rad/m"}


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _straight_line(n: int, step: float = 1.0, axis: int = 2) -> np.ndarray:
    """Identity rotations, moving `step` m per frame along `axis` (z = forward)."""
    poses = np.zeros((n, 3, 4))
    poses[:, :3, :3] = np.eye(3)
    poses[:, axis, 3] = step * np.arange(n)
    return poses


# ---------------------------------------------------------------------------
# Profiles and units
# ---------------------------------------------------------------------------


def test_rpe_defaults_to_100m_reference_pairs():
    gt = _straight_line(400, step=1.0)
    est = _straight_line(400, step=1.1)
    default = compute_rpe(gt, est)
    explicit = compute_rpe(gt, est, METRIC_DISTANCE.rpe)
    assert default.selection == PairSelection(100.0, "m", all_pairs=True, pairs_from="reference", rel_tolerance=0.1)
    assert default.pairs == explicit.pairs
    assert default.errors[0] == pytest.approx(10.0, rel=1e-12)  # 10% of 100 m
    assert evaluate(gt, est).profile.name == "metric_distance"


def test_profiles_are_registered_with_explicit_settings():
    assert set(PROFILES) == {"metric_distance", "kitti_segments"}
    assert METRIC_DISTANCE.rpe.interval_unit == "m"
    assert METRIC_DISTANCE.rpe.pairs_from == "reference"
    assert KITTI_SEGMENTS.segments == SegmentSelection(tuple(float(x) for x in range(100, 900, 100)), 10)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_metric_reports_an_si_unit(profile):
    gt = _straight_line(1200, step=1.0)
    result = evaluate(gt, gt, profile)
    assert result.metrics
    for m in result.metrics:
        assert m.unit in SI_UNITS, (m.name, m.unit)
        assert m.alignment == "none"
        expected = {"translation": "m", "rotation": "rad"}
        for word, unit in expected.items():
            if m.name.startswith(("ape_", "rpe_")) and word in m.name:
                assert m.unit == unit
        if m.name == "segment_translation_drift":
            assert m.unit == "m/m"
        if m.name == "segment_rotation_drift":
            assert m.unit == "rad/m"


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_identical_trajectories_have_zero_error(profile):
    gt = _straight_line(1200, step=1.0)
    for m in evaluate(gt, gt, profile).metrics:
        assert m.available
        assert m.statistics.max == pytest.approx(0.0, abs=1e-12), m.name


def test_to_dict_is_json_serializable_and_complete():
    gt = _straight_line(300, step=1.0)
    d = evaluate(gt, gt, "metric_distance").to_dict()
    text = json.dumps(d)
    assert "errors" not in d["metrics"]["rpe_translation"]
    rpe = json.loads(text)["metrics"]["rpe_translation"]
    assert rpe["unit"] == "m"
    assert rpe["selection"] == {
        "kind": "PairSelection",
        "interval": 100.0,
        "interval_unit": "m",
        "all_pairs": True,
        "pairs_from": "reference",
        "rel_tolerance": 0.1,
    }
    assert rpe["num_poses"] == 300 and rpe["num_pairs"] > 0


def test_error_statistics_match_evo_definitions():
    errors = np.array([0.5, 1.0, 2.5, 4.0])
    stats = ErrorStatistics.from_errors(errors)
    pe = evo_metrics.APE()
    pe.error = errors
    evo_stats = pe.get_all_statistics()
    for key, value in evo_stats.items():
        assert getattr(stats, key) == pytest.approx(value, rel=1e-15, abs=0.0)


# ---------------------------------------------------------------------------
# APE
# ---------------------------------------------------------------------------


def test_ape_translation_constant_offset_in_meters():
    gt = _straight_line(20)
    est = gt.copy()
    est[:, 1, 3] += 1.5
    result = compute_ape(gt, est, "translation")
    assert result.unit == "m"
    assert result.statistics.rmse == pytest.approx(1.5, abs=1e-12)
    assert result.statistics.max == pytest.approx(1.5, abs=1e-12)
    assert result.num_poses == 20 and result.num_pairs is None
    assert compute_ape(gt, est, "rotation").statistics.max == pytest.approx(0.0, abs=1e-12)


def test_ape_rotation_constant_offset_in_radians():
    gt = _straight_line(20)
    est = gt.copy()
    est[:, :3, :3] = _rot_y(0.05)
    result = compute_ape(gt, est, "rotation")
    assert result.unit == "rad"
    assert result.statistics.mean == pytest.approx(0.05, abs=1e-12)
    assert result.statistics.std == pytest.approx(0.0, abs=1e-12)
    assert compute_ape(gt, est, "translation").statistics.max == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# RPE
# ---------------------------------------------------------------------------


# Consecutive 100-frame pairs selected by index: a frame-interval RPE.
FRAMES_100 = PairSelection(interval=100, interval_unit="frames", all_pairs=False, pairs_from="index")


def test_frame_interval_rpe_uses_consecutive_pairs():
    n = 451
    gt = _straight_line(n, step=1.0)
    est = _straight_line(n, step=1.1)  # 10% scale error -> 10 m per 100-frame pair
    result = compute_rpe(gt, est, FRAMES_100, "translation")
    assert result.pairs == ((0, 100), (100, 200), (200, 300), (300, 400))
    assert result.num_pairs == 4
    assert result.frame_coverage == pytest.approx(400 / 450)
    assert result.statistics.rmse == pytest.approx(10.0, rel=1e-12)
    assert compute_rpe(gt, est, FRAMES_100, "rotation").statistics.max == pytest.approx(0.0, abs=1e-12)


def test_frame_interval_is_not_a_distance():
    """At 2 m per frame, 100 frames span 200 m, not 100 m."""
    gt = _straight_line(301, step=2.0)
    est = _straight_line(301, step=2.2)
    frames = compute_rpe(gt, est, FRAMES_100, "translation")
    meters = compute_rpe(gt, est, METRIC_DISTANCE.rpe, "translation")
    assert frames.statistics.mean == pytest.approx(20.0, rel=1e-12)  # 10% of 200 m
    assert meters.errors[0] == pytest.approx(10.0, rel=1e-12)  # 10% of 100 m
    assert meters.statistics.max == pytest.approx(10.0, rel=1e-12)


def test_metric_distance_pairs_come_from_reference_for_every_estimate():
    gt = _straight_line(400, step=1.0)
    slow = _straight_line(400, step=0.5)
    fast = _straight_line(400, step=1.7)
    a = compute_rpe(gt, slow, METRIC_DISTANCE.rpe)
    b = compute_rpe(gt, fast, METRIC_DISTANCE.rpe)
    assert a.pairs == b.pairs
    # Every start frame pairs with the end frame nearest 100 m of reference path,
    # kept within the +/-10 m tolerance: exact 100 m spans for starts 0..299,
    # then shorter tail pairs ending at the last frame for starts 300..309.
    spans = [j - i for i, j in a.pairs]
    assert a.pairs[:300] == tuple((i, i + 100) for i in range(300))
    assert a.pairs[300:] == tuple((i, 399) for i in range(300, 310))
    assert min(spans) == 90 and max(spans) == 100
    assert a.errors[:300] == pytest.approx(np.full(300, 50.0), rel=1e-12)  # 50% short
    assert b.errors[:300] == pytest.approx(np.full(300, 70.0), rel=1e-12)  # 70% long


def test_rpe_rotation_known_yaw_drift():
    """A constant extra yaw of d rad per frame gives 100*d rad over a 100-frame pair."""
    n, d = 301, 1e-3
    gt = _straight_line(n)
    est = gt.copy()
    for i in range(n):
        est[i, :3, :3] = _rot_y(i * d)
    result = compute_rpe(gt, est, FRAMES_100, "rotation")
    assert result.unit == "rad"
    assert result.statistics.mean == pytest.approx(100 * d, rel=1e-9)


def test_rpe_without_eligible_pairs_is_unavailable_with_reason():
    gt = _straight_line(50, step=1.0)  # 49 m and 49 frame intervals
    for selection in (FRAMES_100, METRIC_DISTANCE.rpe):
        result = compute_rpe(gt, gt, selection)
        assert not result.available
        assert result.statistics is None
        assert result.num_pairs == 0
        assert "no pose pairs" in result.unavailable_reason


# ---------------------------------------------------------------------------
# KITTI segments
# ---------------------------------------------------------------------------


def test_kitti_segments_follow_devkit_selection():
    gt = _straight_line(1000, step=1.0)
    segments = kitti_segments(gt)
    # dist[i] = i, so the first frame strictly beyond L metres is first + L + 1.
    assert segments[0] == (0, 101, 100.0)
    assert all(last == first + int(length) + 1 for first, last, length in segments)
    assert {first for first, _, _ in segments} == set(range(0, 1000 - 101, 10))
    # Start frames step by 10 and lengths cycle in devkit order.
    assert [s[2] for s in segments[:8]] == [float(x) for x in range(100, 900, 100)]


def test_segment_translation_drift_equals_scale_error():
    gt = _straight_line(1000, step=1.0)
    est = _straight_line(1000, step=1.03)
    t_drift, r_drift = compute_segment_drift(gt, est)
    assert t_drift.unit == "m/m" and r_drift.unit == "rad/m"
    # Each segment spans L + 1 m of reference path and 1.03 * (L + 1) m of estimate.
    lengths = np.array([s[2] for s in kitti_segments(gt)])
    expected = 0.03 * (lengths + 1) / lengths
    assert t_drift.statistics.mean == pytest.approx(float(np.mean(expected)), rel=1e-12)
    assert r_drift.statistics.max == pytest.approx(0.0, abs=1e-12)
    assert set(t_drift.breakdown) == {f"{L}" for L in range(100, 900, 100)}
    assert t_drift.breakdown["100"].mean == pytest.approx(0.03 * 101 / 100, rel=1e-12)


def test_segment_rotation_drift_in_radians_per_meter():
    n, d = 1000, 2e-4
    gt = _straight_line(n)
    est = gt.copy()
    for i in range(n):
        est[i, :3, :3] = _rot_y(i * d)
    _, r_drift = compute_segment_drift(gt, est)
    lengths = np.array([s[2] for s in kitti_segments(gt)])
    expected = d * (lengths + 1) / lengths  # (last - first) * d rad over L m
    np.testing.assert_allclose(r_drift.errors, expected, rtol=1e-6)
    assert r_drift.breakdown["800"].mean == pytest.approx(d * 801 / 800, rel=1e-6)


def test_segments_unavailable_for_short_reference_path():
    gt = _straight_line(90, step=1.0)
    t_drift, r_drift = compute_segment_drift(gt, gt)
    for m in (t_drift, r_drift):
        assert not m.available
        assert "shorter than the shortest segment" in m.unavailable_reason


def test_segment_selection_is_configurable():
    gt = _straight_line(100, step=1.0)
    t_drift, _ = compute_segment_drift(gt, gt, SegmentSelection(lengths_m=(10.0, 20.0), step_frames=5))
    assert t_drift.available
    assert set(t_drift.breakdown) == {"10", "20"}


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_length_mismatch_is_rejected_not_truncated():
    gt = _straight_line(300)
    est = _straight_line(299)
    with pytest.raises(TrajectoryInputError, match="length mismatch.*nothing is truncated"):
        evaluate(gt, est, "metric_distance")


def test_nonfinite_estimate_is_rejected():
    gt = _straight_line(300)
    est = gt.copy()
    est[10, 0, 3] = np.nan
    with pytest.raises(TrajectoryInputError, match="nonfinite"):
        compute_ape(gt, est)


def test_invalid_rotation_is_rejected():
    gt = _straight_line(300)
    est = gt.copy()
    est[5, 0, 0] = 1.2
    with pytest.raises(TrajectoryInputError, match="invalid rotation"):
        compute_ape(gt, est)


def test_accepts_homogeneous_4x4_input():
    gt = _straight_line(150)
    gt4 = np.zeros((150, 4, 4))
    gt4[:, :3, :] = gt
    gt4[:, 3, 3] = 1.0
    assert compute_ape(gt4, gt).statistics.max == pytest.approx(0.0, abs=1e-12)


def test_poses_to_trajectory_roundtrip():
    poses = _straight_line(10, step=2.0, axis=0)
    traj = poses_to_trajectory(poses)
    assert traj.num_poses == 10
    for mat in traj.poses_se3:
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat[3, :], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(traj.positions_xyz[:, 0], [i * 2.0 for i in range(10)])
