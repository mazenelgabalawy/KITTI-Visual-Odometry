"""Builds a StereoVOPipeline from a PipelineConfig.

Maps config -> concrete components exactly as VO_base.py / VO_constrained.py
wire them up (see module docstrings on pipeline.py, motion_filters.py,
pose_estimation.py, and matchers.py for the specific transcription notes).
"""

from __future__ import annotations

from kitti_slam.config.schema import PipelineConfig
from kitti_slam.features.grid_filter import GridKeypointFilter
from kitti_slam.features.matchers import BruteForceMatcher, RectifiedStereoMatcher
from kitti_slam.features.orb import OrbDetector
from kitti_slam.odometry.motion_filters import NullMotionFilter, PlanarMotionFilter
from kitti_slam.odometry.pipeline import StereoVOPipeline
from kitti_slam.odometry.pose_estimation import DepthUncertaintyFilter, PnPRansacEstimator


def build_pipeline(cfg: PipelineConfig) -> StereoVOPipeline:
    keypoint_filter = GridKeypointFilter(cfg.grid_filter) if cfg.grid_filter is not None else None
    detector = OrbDetector(cfg.detector, keypoint_filter)

    temporal_matcher = BruteForceMatcher(
        cfg.temporal_matcher.norm, cfg.temporal_matcher.cross_check, cfg.temporal_matcher.ratio_threshold
    )

    if cfg.stereo_matcher.kind == "rectified":
        stereo_matcher = RectifiedStereoMatcher(cfg.stereo_matcher)
    else:
        # VO_base.py reuses the same BFMatcher object (raw bf.match, no ratio
        # test) for both temporal and stereo matching -- built from the
        # temporal matcher's norm/cross_check since StereoMatcherConfig(kind=
        # "bruteforce") carries no norm of its own.
        stereo_matcher = BruteForceMatcher(
            cfg.temporal_matcher.norm, cfg.temporal_matcher.cross_check, ratio_threshold=None
        )

    landmark_filter = DepthUncertaintyFilter(cfg.landmark_filter) if cfg.landmark_filter is not None else None

    pose_estimator = PnPRansacEstimator(cfg.pnp)

    motion_cfg = cfg.motion
    if (
        motion_cfg.max_translation is not None
        or motion_cfg.max_pitch_deg is not None
        or (motion_cfg.orthogonalize_alpha is not None)
    ):
        motion_filter = PlanarMotionFilter(motion_cfg)
    else:
        motion_filter = NullMotionFilter()

    return StereoVOPipeline(
        detector=detector,
        temporal_matcher=temporal_matcher,
        stereo_matcher=stereo_matcher,
        landmark_filter=landmark_filter,
        pose_estimator=pose_estimator,
        motion_filter=motion_filter,
        min_abs_w=cfg.triangulation.min_abs_w,
    )
