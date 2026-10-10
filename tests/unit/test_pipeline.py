"""Tests for kitti_slam.odometry.pipeline / factory (Module 5).

All tests here are synthetic and always run (no KITTI dataset needed).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys

import cv2
import numpy as np
import pytest

from kitti_slam.config import ConfigError, load_config
from kitti_slam.datasets.base import StereoFrame, StereoSequence
from kitti_slam.features.base import FeatureDetector, Features, Matcher
from kitti_slam.features.grid_filter import GridKeypointFilter
from kitti_slam.features.matchers import BruteForceMatcher, RectifiedStereoMatcher
from kitti_slam.features.orb import OrbDetector
from kitti_slam.geometry.calibration import StereoCalibration
from kitti_slam.odometry.base import LandmarkFilter, MotionFilter, PoseEstimate, PoseEstimator
from kitti_slam.odometry.factory import build_pipeline
from kitti_slam.odometry.motion_filters import NullMotionFilter, PlanarMotionFilter
from kitti_slam.odometry.pipeline import StereoVOPipeline
from kitti_slam.odometry.pose_estimation import DepthUncertaintyFilter, PnPRansacEstimator

# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_base_variant(repo_root):
    cfg = load_config(repo_root / "configs" / "base.yaml")
    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline, StereoVOPipeline)
    assert isinstance(pipeline.detector, OrbDetector)
    assert pipeline.detector.keypoint_filter is None
    assert isinstance(pipeline.temporal_matcher, BruteForceMatcher)
    assert isinstance(pipeline.stereo_matcher, BruteForceMatcher)
    assert pipeline.stereo_matcher.ratio_threshold is None
    assert isinstance(pipeline.motion_filter, NullMotionFilter)
    assert pipeline.landmark_filter is None
    assert isinstance(pipeline.pose_estimator, PnPRansacEstimator)


def test_factory_constrained_variant(repo_root):
    cfg = load_config(repo_root / "configs" / "constrained.yaml")
    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.detector, OrbDetector)
    assert isinstance(pipeline.detector.keypoint_filter, GridKeypointFilter)
    assert isinstance(pipeline.temporal_matcher, BruteForceMatcher)
    assert pipeline.temporal_matcher.ratio_threshold == 0.5
    assert isinstance(pipeline.stereo_matcher, RectifiedStereoMatcher)
    assert isinstance(pipeline.motion_filter, PlanarMotionFilter)
    assert isinstance(pipeline.landmark_filter, DepthUncertaintyFilter)


@pytest.mark.parametrize(("temporal", "stereo"), [(True, False), (False, True)])
def test_factory_bruteforce_stereo_uses_its_own_cross_check(repo_root, temporal, stereo):
    import dataclasses

    cfg = load_config(repo_root / "configs" / "base.yaml")
    cfg = dataclasses.replace(
        cfg,
        temporal_matcher=dataclasses.replace(cfg.temporal_matcher, cross_check=temporal),
        stereo_matcher=dataclasses.replace(cfg.stereo_matcher, cross_check=stereo),
    )
    pipeline = build_pipeline(cfg)
    assert pipeline.temporal_matcher.cross_check is temporal
    assert pipeline.stereo_matcher.cross_check is stereo


# ---------------------------------------------------------------------------
# Synthetic in-memory stereo sequence
# ---------------------------------------------------------------------------


class InMemoryStereoSequence(StereoSequence):
    def __init__(self, frames: list[StereoFrame], calibration: StereoCalibration):
        self._frames = frames
        self._calibration = calibration

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> StereoFrame:
        return self._frames[idx]

    @property
    def calibration(self) -> StereoCalibration:
        return self._calibration


def _make_synthetic_sequence(n_frames: int = 5, step_px: int = 6, depth: float = 20.0):
    """Builds a synthetic stereo sequence approximating a fronto-parallel
    textured plane at fixed depth, viewed by a camera translating laterally.

    A wide random-texture canvas is generated once; each frame's left/right
    images are pixel-exact crops of that canvas, offset frame-to-frame by
    step_px (simulating lateral translation) and offset left-to-right by the
    disparity implied by (focal_length, baseline, depth).
    """
    rng = np.random.default_rng(1234)

    height, width = 240, 320
    focal = 700.0
    baseline = 0.54
    cx, cy = width / 2.0, height / 2.0
    disp_px = int(round(focal * baseline / depth))

    margin = 40
    canvas_width = margin + n_frames * step_px + width + disp_px + margin
    canvas = rng.integers(0, 256, size=(height, canvas_width), dtype=np.uint8)

    P0 = np.array([[focal, 0.0, cx, 0.0], [0.0, focal, cy, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P1 = np.array([[focal, 0.0, cx, -focal * baseline], [0.0, focal, cy, 0.0], [0.0, 0.0, 1.0, 0.0]])
    calib = StereoCalibration.from_projection_matrices(P0, P1)

    frames = []
    for k in range(n_frames):
        left_start = margin + k * step_px
        right_start = left_start + disp_px
        left_img = canvas[:, left_start : left_start + width]
        right_img = canvas[:, right_start : right_start + width]
        frames.append(StereoFrame(index=k, left=left_img.copy(), right=right_img.copy()))

    expected_translation_per_frame = step_px * depth / focal
    return InMemoryStereoSequence(frames, calib), expected_translation_per_frame


def test_pipeline_runs_synthetic_sequence(repo_root):
    cfg = load_config(repo_root / "configs" / "base.yaml")
    # Keep PnP RANSAC deterministic and iteration count modest for test speed.
    cfg = dataclasses.replace(cfg, pnp=dataclasses.replace(cfg.pnp, confidence=0.999, iterations_count=500))
    pipeline = build_pipeline(cfg)

    sequence, expected_t = _make_synthetic_sequence(n_frames=5, step_px=6, depth=20.0)

    poses = pipeline.run(sequence, progress=False)

    assert poses.shape == (5, 3, 4)
    np.testing.assert_allclose(poses[0], np.eye(3, 4), atol=1e-9)

    translations = poses[:, :, 3]
    # Non-identity motion recovered: later frames should have moved from the origin.
    assert np.linalg.norm(translations[-1]) > 1e-3

    # Loose sanity check on magnitude: per-frame translation should be in the
    # right ballpark for the planar approximation (within a generous factor).
    per_frame_deltas = np.diff(translations, axis=0)
    per_frame_norms = np.linalg.norm(per_frame_deltas, axis=1)
    nonzero = per_frame_norms[per_frame_norms > 0]
    if len(nonzero) > 0:
        assert np.median(nonzero) < expected_t * 5 + 1.0

    # Monotonic drift: x-translation should not oscillate in sign once motion starts.
    xs = translations[:, 0]
    nonzero_xs = xs[np.abs(xs) > 1e-6]
    if len(nonzero_xs) > 1:
        signs = np.sign(nonzero_xs)
        assert len(set(signs.tolist())) == 1


# ---------------------------------------------------------------------------
# Skip-policy tests
# ---------------------------------------------------------------------------


class _EmptyDetector(FeatureDetector):
    def detect(self, image: np.ndarray) -> Features:
        return Features(keypoints=[], descriptors=np.empty((0, 32), dtype=np.uint8))


class _EmptyMatcher(Matcher):
    def match(self, feat_a: Features, feat_b: Features) -> list:
        return []


class _AlwaysFailEstimator(PoseEstimator):
    def estimate(self, points_3d, points_2d, K) -> PoseEstimate:
        return PoseEstimate(R=None, t=None, inliers=None, success=False)


def _make_trivial_sequence(n_frames: int = 4) -> InMemoryStereoSequence:
    P0 = np.array([[700.0, 0.0, 160.0, 0.0], [0.0, 700.0, 120.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P1 = np.array([[700.0, 0.0, 160.0, -700.0 * 0.54], [0.0, 700.0, 120.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    calib = StereoCalibration.from_projection_matrices(P0, P1)
    frames = [
        StereoFrame(index=k, left=np.zeros((10, 10), dtype=np.uint8), right=np.zeros((10, 10), dtype=np.uint8))
        for k in range(n_frames)
    ]
    return InMemoryStereoSequence(frames, calib)


_N_SCRIPTED = 8


class _StereoPairDetector(FeatureDetector):
    """The same 8 keypoints in every left image; the right image sees them 10 px
    to the left (disparity 10 px, about 38 m deep with the trivial calibration)."""

    def __init__(self):
        self.calls = 0

    def detect(self, image: np.ndarray) -> Features:
        shift = 10.0 if self.calls % 2 else 0.0  # the pipeline detects left, then right
        self.calls += 1
        kps = [cv2.KeyPoint(x=100.0 + 20.0 * i - shift, y=60.0 + 15.0 * i, size=5.0) for i in range(_N_SCRIPTED)]
        return Features(kps, np.arange(_N_SCRIPTED * 32, dtype=np.uint8).reshape(_N_SCRIPTED, 32))


class _IdentityMatcher(Matcher):
    """Matches i -> i, except on the listed (1-based) calls, where it returns nothing."""

    def __init__(self, empty_calls=()):
        self.calls, self.empty_calls = 0, set(empty_calls)

    def match(self, feat_a: Features, feat_b: Features) -> list:
        self.calls += 1
        if self.calls in self.empty_calls:
            return []
        return [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(_N_SCRIPTED)]


class _RejectingLandmarkFilter(LandmarkFilter):
    def __init__(self, reject_calls=()):
        self.calls, self.reject_calls = 0, set(reject_calls)

    def valid_mask(self, points_3d, calib):
        self.calls += 1
        return np.full(len(points_3d), self.calls not in self.reject_calls)


class _RejectingMotionFilter(MotionFilter):
    def __init__(self, reject_calls=()):
        self.calls, self.reject_calls = 0, set(reject_calls)

    def refine_rotation(self, R):
        return R

    def is_plausible(self, R_world, t_world):
        self.calls += 1
        return self.calls not in self.reject_calls


class _ScriptedEstimator(PoseEstimator):
    """Returns the scripted camera-frame translation per call (identity rotation);
    None scripts a failed solve."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def estimate(self, points_3d, points_2d, K) -> PoseEstimate:
        t = self.script[self.calls]
        self.calls += 1
        if t is None:
            return PoseEstimate(R=None, t=None, inliers=None, success=False, retval=False)
        inliers = np.arange(len(points_3d)).reshape(-1, 1)
        return PoseEstimate(R=np.eye(3), t=np.asarray(t, dtype=float), inliers=inliers, success=True, retval=True)


# Camera-frame PnP translations: (0, 0, -1) moves the camera 1 m forward in the
# world, (-2, 0, 0) moves it 2 m along world x.
_FORWARD = (0.0, 0.0, -1.0)
_SIDEWAYS = (-2.0, 0.0, 0.0)
_IMPLAUSIBLE = (0.0, 0.0, -50.0)


def _scripted_pipeline(kind: str) -> StereoVOPipeline:
    """Frame 1 registers 1 m forward, frames 2-3 fail in the given way, frame 4
    registers 2 m sideways."""
    fail = {2, 3}
    temporal = _IdentityMatcher(empty_calls=fail if kind == "no_correspondences" else ())
    landmarks = _RejectingLandmarkFilter(reject_calls=fail if kind == "too_few_points" else ())
    motion = _RejectingMotionFilter(reject_calls=fail if kind == "implausible_motion" else ())
    script = {
        "no_correspondences": [_FORWARD, _SIDEWAYS],
        "too_few_points": [_FORWARD, _SIDEWAYS],
        "pnp_failed": [_FORWARD, None, None, _SIDEWAYS],
        "implausible_motion": [_FORWARD, _IMPLAUSIBLE, _IMPLAUSIBLE, _SIDEWAYS],
    }[kind]
    return StereoVOPipeline(
        detector=_StereoPairDetector(),
        temporal_matcher=temporal,
        stereo_matcher=_IdentityMatcher(),
        landmark_filter=landmarks,
        pose_estimator=_ScriptedEstimator(script),
        motion_filter=motion,
    )


@pytest.mark.parametrize("kind", ["no_correspondences", "too_few_points", "pnp_failed", "implausible_motion"])
def test_failed_registration_propagates_constant_velocity(kind):
    from kitti_slam.odometry.diagnostics import FrameDiagnostics

    diagnostics = FrameDiagnostics()
    poses = _scripted_pipeline(kind).run(_make_trivial_sequence(n_frames=5), progress=False, diagnostics=diagnostics)

    assert [r["outcome"] for r in diagnostics.rows] == ["registered", kind, kind, "registered"]
    assert [r["consecutive_failures"] for r in diagnostics.rows] == [0, 1, 2, 0]
    for pose in poses:
        np.testing.assert_allclose(pose[:, :3], np.eye(3), atol=1e-12)
    # Frames 2-3 repeat frame 1's 1 m forward step; frame 4's success replaces it.
    np.testing.assert_allclose(poses[:, :, 3], [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [2, 0, 3]], atol=1e-12)


def test_failure_before_first_success_holds_identity():
    pipeline = StereoVOPipeline(
        detector=_StereoPairDetector(),
        temporal_matcher=_IdentityMatcher(),
        stereo_matcher=_IdentityMatcher(),
        landmark_filter=None,
        pose_estimator=_ScriptedEstimator([None, None, _FORWARD]),
        motion_filter=NullMotionFilter(),
    )
    poses = pipeline.run(_make_trivial_sequence(n_frames=4), progress=False)
    np.testing.assert_allclose(poses[:, :, 3], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]], atol=1e-12)


def test_interpolate_on_skip_is_not_a_config_key(repo_root, tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text(
        (repo_root / "configs" / "constrained.yaml")
        .read_text()
        .replace("orthogonalize_alpha: 0.8\n", "orthogonalize_alpha: 0.8\n  interpolate_on_skip: true\n")
    )
    with pytest.raises(ConfigError, match="interpolate_on_skip"):
        load_config(old)


# ---------------------------------------------------------------------------
# run_vo.py safety guard
# ---------------------------------------------------------------------------


def _load_run_vo_module(repo_root):
    module_path = repo_root / "scripts" / "run_vo.py"
    spec = importlib.util.spec_from_file_location("run_vo_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_vo_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_run_vo_refuses_reference_dirs(repo_root):
    run_vo = _load_run_vo_module(repo_root)

    with pytest.raises(SystemExit):
        run_vo.guard_output_dir("reference", repo_root)

    with pytest.raises(SystemExit):
        run_vo.guard_output_dir("reference/base/04", repo_root)

    # A sibling directory that merely shares a prefix must NOT be refused.
    resolved = run_vo.guard_output_dir("reference_experiment", repo_root)
    assert resolved == (repo_root / "reference_experiment").resolve()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_do_not_change_poses(repo_root):
    from kitti_slam.odometry.diagnostics import FIELDS, FrameDiagnostics

    cfg = load_config(repo_root / "configs" / "constrained.yaml")
    sequence, _ = _make_synthetic_sequence(n_frames=5, step_px=6, depth=20.0)

    plain = build_pipeline(cfg).run(sequence, progress=False)
    diagnostics = FrameDiagnostics()
    with_diag = build_pipeline(cfg).run(sequence, progress=False, diagnostics=diagnostics)

    assert np.array_equal(plain, with_diag)
    assert [row["frame"] for row in diagnostics.rows] == [1, 2, 3, 4]
    for row in diagnostics.rows:
        assert set(row) == set(FIELDS)
        assert row["associated"] <= min(row["temporal_matches"], row["stereo_matches"])
        if row["outcome"] == "registered":
            assert row["consecutive_failures"] == 0
            assert row["pnp_retval"] == 1 and row["pnp_inliers"] >= 4


def test_diagnostics_record_failures_and_streaks(tmp_path):
    from kitti_slam.odometry.diagnostics import FrameDiagnostics

    pipeline = StereoVOPipeline(
        detector=_EmptyDetector(),
        temporal_matcher=_EmptyMatcher(),
        stereo_matcher=_EmptyMatcher(),
        landmark_filter=None,
        pose_estimator=_AlwaysFailEstimator(),
        motion_filter=NullMotionFilter(),
    )
    diagnostics = FrameDiagnostics()
    pipeline.run(_make_trivial_sequence(n_frames=4), progress=False, diagnostics=diagnostics)

    assert [r["outcome"] for r in diagnostics.rows] == ["no_correspondences"] * 3
    assert [r["consecutive_failures"] for r in diagnostics.rows] == [1, 2, 3]
    assert all(r["pnp_retval"] == "" and r["landmarks"] == "" for r in diagnostics.rows)

    path = tmp_path / "diagnostics.csv"
    diagnostics.write_csv(path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("frame,outcome,temporal_matches")
    assert lines[1] == "1,no_correspondences,0,0,0,,,,,,,1"


def test_diagnostics_reject_unknown_outcome():
    from kitti_slam.odometry.diagnostics import FrameDiagnostics

    with pytest.raises(ValueError):
        FrameDiagnostics().append({"frame": 1, "outcome": "skipped"})


class _MixedDepthDetector(_StereoPairDetector):
    """Keypoints 0-3 at +10 px disparity (in front), 4-7 at -10 px (behind)."""

    def detect(self, image: np.ndarray) -> Features:
        right = self.calls % 2
        self.calls += 1
        kps = [
            cv2.KeyPoint(x=100.0 + 20.0 * i - (10.0 if i < 4 else -10.0) * right, y=60.0 + 15.0 * i, size=5.0)
            for i in range(_N_SCRIPTED)
        ]
        return Features(kps, np.arange(_N_SCRIPTED * 32, dtype=np.uint8).reshape(_N_SCRIPTED, 32))


def test_points_behind_the_cameras_do_not_reach_pnp():
    from kitti_slam.odometry.diagnostics import FrameDiagnostics

    seen = []

    class _Recorder(_ScriptedEstimator):
        def estimate(self, points_3d, points_2d, K):
            seen.append(points_3d.copy())
            return super().estimate(points_3d, points_2d, K)

    pipeline = StereoVOPipeline(
        detector=_MixedDepthDetector(),
        temporal_matcher=_IdentityMatcher(),
        stereo_matcher=_IdentityMatcher(),
        landmark_filter=None,
        pose_estimator=_Recorder([_FORWARD]),
        motion_filter=NullMotionFilter(),
        min_abs_w=1e-4,
    )
    diagnostics = FrameDiagnostics()
    pipeline.run(_make_trivial_sequence(n_frames=2), progress=False, diagnostics=diagnostics)
    (row,) = diagnostics.rows
    assert (row["associated"], row["nonpositive_depth_points"], row["invalid_points"], row["landmarks"]) == (8, 4, 4, 4)
    assert len(seen) == 1 and seen[0].shape == (4, 3) and (seen[0][:, 2] > 0).all()
