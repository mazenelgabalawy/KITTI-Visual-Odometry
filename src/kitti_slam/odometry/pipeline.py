"""Stereo visual-odometry pipeline: a near-transcription of the shared loop body
in VO_constrained.py (L96-211) and VO_base.py (L76-139).

Both legacy scripts run the *same* algorithm skeleton; they differ only in which
concrete components are plugged in:
  - detector:        OrbDetector alone (base) vs. OrbDetector + GridKeypointFilter (constrained)
  - temporal matcher: raw bf.match (base) vs. BF_matcher with ratio_threshold (constrained)
  - stereo matcher:   raw bf.match reused as a BruteForceMatcher (base) vs.
                      RectifiedStereoMatcher with epipolar+disparity gating (constrained)
  - landmark filter:  none (base) vs. DepthUncertaintyFilter (constrained)
  - motion filter:    NullMotionFilter (base, always plausible, no orthogonalization)
                      vs. PlanarMotionFilter (constrained)

StereoVOPipeline.step()/run() implement that single skeleton; base-vs-constrained
behavior falls out purely from which components are injected (see factory.py),
never from a branch on "which variant am I".

Transcription notes (see also module docstrings on the components used here):

1. Feature re-detection every step, never a persisted descriptor cache. VO_base.py
   L86-88 re-runs orb.detectAndCompute on left_prev/right_prev *every iteration*,
   even though left_prev/right_prev were already featured in a prior iteration —
   it never caches descriptors across iterations, only the raw images. VO_constrained.py
   instead detects the "prev" pair once (L98-99) and carries the Features objects
   forward across iterations (L142-146, L181-185, L204-208), only re-detecting
   "curr". Because ORB detection (+ grid filtering) is a deterministic pure function
   of (image, config) with no RNG involved, carrying the previously-detected
   Features forward (as VO_constrained.py does) is numerically identical to
   VO_base.py's redetection of the same prev images. This pipeline therefore
   carries prev features forward in FrameState and detects only the current
   pair each step — reproducing both variants' numerics at half the detection
   cost of redetecting prev every iteration.

2. Failed registration assumes constant velocity (BUG-013). Every failure --
   no correspondences, too few points, a failed or rejected PnP solve, or
   implausible motion -- propagates C_k = C_{k-1} @ T_last, where T_last is the
   last accepted relative transform (identity before the first). Propagation
   does not change T_last, and prev image/features advance to curr, so the
   next registration runs from frame k. Legacy base held the pose instead
   (zero velocity); legacy constrained already repeated the last motion.
3. The <4-points guard is checked exactly once, AFTER any landmark filtering
   (VO_constrained.py L136, VO_base.py L106 -- base has no landmark filter so
   there the check is trivially "right after triangulation").
4. N==0 correspondences (intersect_temporal_stereo returns empty arrays rather
   than raising, Module 3 change) are a failed registration (note 2) rather than
   calling cv2.triangulatePoints with empty input.
5. Motion gate operates on WORLD-frame R, t: refine_rotation() is applied to the
   camera-frame R from PnP *first* (VO_constrained.py L164), then invert_pose()
   converts to world frame (R_cam.T, -R_cam.T @ t, mirroring L167-168), and
   is_plausible() is evaluated on that world-frame pair (L174).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from kitti_slam.datasets.base import StereoSequence
from kitti_slam.evaluation.trajectory_io import TrajectoryWriter
from kitti_slam.features.association import intersect_temporal_stereo
from kitti_slam.features.base import FeatureDetector, Features, Matcher
from kitti_slam.geometry.transforms import compose_se3, invert_pose
from kitti_slam.geometry.triangulation import triangulate_stereo_homogeneous, valid_landmarks
from kitti_slam.odometry.base import LandmarkFilter, MotionFilter, PoseEstimator
from kitti_slam.odometry.diagnostics import REGISTERED, FrameDiagnostics
from kitti_slam.odometry.timing import NullStageTimer, StageTimer


@dataclass
class FrameState:
    """Everything the loop body carries from one iteration to the next."""

    left_image: np.ndarray
    right_image: np.ndarray
    left_features: Features
    right_features: Features
    T_last: np.ndarray  # last accepted 3x4 relative transform (the constant-velocity model)
    C: np.ndarray  # accumulated 4x4 global pose


class StereoVOPipeline:
    def __init__(
        self,
        detector: FeatureDetector,
        temporal_matcher: Matcher,
        stereo_matcher: Matcher,
        landmark_filter: LandmarkFilter | None,
        pose_estimator: PoseEstimator,
        motion_filter: MotionFilter,
        min_points: int = 4,
        min_abs_w: float = 0.0,
    ) -> None:
        self.detector = detector
        self.temporal_matcher = temporal_matcher
        self.stereo_matcher = stereo_matcher
        self.landmark_filter = landmark_filter
        self.pose_estimator = pose_estimator
        self.motion_filter = motion_filter
        self.min_points = min_points
        self.min_abs_w = min_abs_w

    def _propagate(
        self,
        state: FrameState,
        left_curr_img,
        right_curr_img,
        left_feat_curr,
        right_feat_curr,
        diag: dict | None = None,
        outcome: str = "",
    ) -> FrameState:
        """Failed registration (note 2): propagate C_k with the constant-velocity
        model, advance prev <- curr, and leave T_last unchanged."""
        if diag is not None:
            diag["outcome"] = outcome
        C = compose_se3(state.C, state.T_last)
        return FrameState(
            left_image=left_curr_img,
            right_image=right_curr_img,
            left_features=left_feat_curr,
            right_features=right_feat_curr,
            T_last=state.T_last,
            C=C,
        )

    def step(
        self,
        state: FrameState,
        left_curr_img: np.ndarray,
        right_curr_img: np.ndarray,
        calib,
        K,
        timer: StageTimer | NullStageTimer | None = None,
        diag: dict | None = None,
    ) -> FrameState:
        """One iteration of the shared loop body (VO_constrained.py L102-211).

        When `diag` is a dict, the frame's outcome and counts are written into it
        (see kitti_slam.odometry.diagnostics); the returned state is the same."""
        if timer is None:
            timer = NullStageTimer()

        with timer.stage("detect"):
            left_feat_curr = self.detector.detect(left_curr_img)
            right_feat_curr = self.detector.detect(right_curr_img)

        # Note 1: prev features are carried forward in FrameState, matching
        # VO_constrained.py L142-146/L204-208; see module docstring for why
        # this is numerically identical to VO_base.py's literal redetection.
        left_feat_prev = state.left_features
        right_feat_prev = state.right_features

        with timer.stage("temporal_match"):
            temporal_matches = self.temporal_matcher.match(left_feat_prev, left_feat_curr)
        with timer.stage("stereo_match"):
            stereo_matches_prev = self.stereo_matcher.match(left_feat_prev, right_feat_prev)

        with timer.stage("associate"):
            corr = intersect_temporal_stereo(
                temporal_matches, stereo_matches_prev, left_feat_prev, right_feat_prev, left_feat_curr
            )

        if diag is not None:
            diag["temporal_matches"] = len(temporal_matches)
            diag["stereo_matches"] = len(stereo_matches_prev)
            diag["associated"] = corr.left_prev.shape[0]

        # Note 4: N==0 correspondences -> failed registration, no triangulation call.
        if corr.left_prev.shape[0] == 0:
            return self._propagate(
                state, left_curr_img, right_curr_img, left_feat_curr, right_feat_curr, diag, "no_correspondences"
            )

        with timer.stage("triangulate"):
            points_h = triangulate_stereo_homogeneous(calib, corr.left_prev, corr.right_prev)
            # BUG-003: only finite points in front of both cameras reach PnP.
            valid, points_3d = valid_landmarks(calib, points_h, self.min_abs_w)

        if diag is not None:
            finite = np.isfinite(points_3d).all(axis=1)
            diag["nonfinite_points"] = int(np.count_nonzero(~finite))
            diag["nonpositive_depth_points"] = int(np.count_nonzero(finite & (points_3d[:, 2] <= 0)))
            diag["invalid_points"] = int(np.count_nonzero(~valid))

        points_3d = points_3d[valid]
        left_curr_pts = corr.left_curr[valid]

        if self.landmark_filter is not None:
            mask = self.landmark_filter.valid_mask(points_3d, calib)
            points_3d = points_3d[mask]
            left_curr_pts = left_curr_pts[mask]

        if diag is not None:
            diag["landmarks"] = points_3d.shape[0]

        # Note 3: the <4 guard is checked once, after any landmark filtering.
        if points_3d.shape[0] < self.min_points:
            return self._propagate(
                state, left_curr_img, right_curr_img, left_feat_curr, right_feat_curr, diag, "too_few_points"
            )

        with timer.stage("pnp"):
            estimate = self.pose_estimator.estimate(points_3d, left_curr_pts, K)
        if diag is not None:
            diag["pnp_retval"] = "" if estimate.retval is None else int(estimate.retval)
            diag["pnp_inliers"] = 0 if estimate.inliers is None else len(estimate.inliers)
        if not estimate.success:
            return self._propagate(
                state, left_curr_img, right_curr_img, left_feat_curr, right_feat_curr, diag, "pnp_failed"
            )

        with timer.stage("motion"):
            # Note 5: orthogonalize the camera-frame R first, then invert to world frame.
            R_ortho = self.motion_filter.refine_rotation(estimate.R)
            R_world, t_world = invert_pose(R_ortho, estimate.t)
            rejection = self.motion_filter.rejection_reason(R_world, t_world)

        if rejection is not None:
            return self._propagate(
                state, left_curr_img, right_curr_img, left_feat_curr, right_feat_curr, diag, rejection
            )

        if diag is not None:
            diag["outcome"] = REGISTERED
        T = np.hstack((R_world, t_world.reshape(3, 1)))
        C = compose_se3(state.C, T)

        return FrameState(
            left_image=left_curr_img,
            right_image=right_curr_img,
            left_features=left_feat_curr,
            right_features=right_feat_curr,
            T_last=T,
            C=C,
        )

    def run(
        self,
        sequence: StereoSequence,
        progress: bool = False,
        timer: StageTimer | NullStageTimer | None = None,
        diagnostics: FrameDiagnostics | None = None,
    ) -> np.ndarray:
        if timer is None:
            timer = NullStageTimer()

        calib = sequence.calibration
        K = calib.K

        writer = TrajectoryWriter()

        frame0 = sequence[0]
        state = FrameState(
            left_image=frame0.left,
            right_image=frame0.right,
            left_features=self.detector.detect(frame0.left),
            right_features=self.detector.detect(frame0.right),
            T_last=np.eye(3, 4),
            C=np.eye(4),
        )
        # Frame-0 pose = identity, saved first (both variants).
        writer.append(state.C)

        indices = range(1, len(sequence))
        if progress:
            indices = tqdm(indices)

        for k in indices:
            with timer.frame():
                with timer.stage("io"):
                    frame = sequence[k]
                diag = {"frame": k} if diagnostics is not None else None
                state = self.step(state, frame.left, frame.right, calib, K, timer=timer, diag=diag)
            if diagnostics is not None:
                diagnostics.append(diag)
            writer.append(state.C)

        return writer.as_array()
