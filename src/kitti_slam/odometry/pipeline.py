"""Stereo visual-odometry pipeline: one loop shared by every configuration.

Configurations differ only in the components factory.py injects:
  - detector:         OrbDetector, optionally with a GridKeypointFilter
  - temporal matcher: BruteForceMatcher, optionally keeping the best fraction of matches
  - stereo matcher:   BruteForceMatcher, or RectifiedStereoMatcher (row band and
                      disparity range checked before ranking)
  - landmark filter:  none, or DepthUncertaintyFilter
  - motion filter:    NullMotionFilter (no refinement, always plausible), or
                      PlanarMotionFilter (planar rotation refinement, translation
                      and heading gates)
step() and run() never branch on which configuration is running.

Notes:

1. Each image is detected once: the previous frame's features are carried
   forward in FrameState and only the current pair is detected. Detection is a
   deterministic function of (image, config), so this equals re-detecting the
   previous frame, at half the cost.
2. A failed registration assumes constant velocity (BUG-013). Every failure --
   no correspondences, too few points, a failed or rejected PnP solve, or
   implausible motion -- propagates C_k = C_{k-1} @ T_last, where T_last is the
   last accepted relative transform (identity before the first). Propagation
   does not change T_last, and the current frame becomes the previous one, so
   the next registration runs from frame k.
3. Triangulated points pass the landmark validity check (finite, positive depth
   in both cameras, |w| >= min_abs_w; BUG-003) and then the optional landmark
   filter. The min_points guard is checked once, after both.
4. Zero correspondences are a failed registration (note 2); triangulation is
   never called with empty input.
5. The motion filter refines the camera-frame rotation from PnP first; the pose
   is then inverted to the world frame (R.T, -R.T @ t), and the gates evaluate
   that world-frame motion.
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
        """Registers the current frame against the previous one (module notes 1-5).

        When `diag` is a dict, the frame's outcome and counts are written into it
        (see kitti_slam.odometry.diagnostics); the returned state is the same."""
        if timer is None:
            timer = NullStageTimer()

        with timer.stage("detect"):
            left_feat_curr = self.detector.detect(left_curr_img)
            right_feat_curr = self.detector.detect(right_curr_img)

        # Note 1: the previous frame's features are carried forward in FrameState.
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

        # Note 3: the min_points guard is checked once, after validity and filtering.
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
