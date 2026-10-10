"""Temporal/stereo match intersection: the previous left keypoints that have both a
temporal match (into the current left image) and a stereo match (into the previous
right image). When none survive, the result holds empty (0, 2) float32 arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kitti_slam.features.base import Features


@dataclass
class StereoTemporalCorrespondences:
    left_prev: np.ndarray  # (N, 2) float32
    right_prev: np.ndarray  # (N, 2) float32
    left_curr: np.ndarray  # (N, 2) float32


def intersect_temporal_stereo(
    temporal_matches,
    stereo_matches,
    feat_left_prev: Features,
    feat_right_prev: Features,
    feat_left_curr: Features,
) -> StereoTemporalCorrespondences:
    """
    Filters temporal matches to those with corresponding stereo matches in the previous frame.

    Args:
    - temporal_matches: List of cv2.DMatch objects (left_prev_desc to left_curr_desc).
    - stereo_matches: List of cv2.DMatch objects (left_prev_desc to right_prev_desc).
    - feat_left_prev: Features for left_prev.
    - feat_right_prev: Features for right_prev.
    - feat_left_curr: Features for left_curr.

    Returns:
    - StereoTemporalCorrespondences of left_prev, right_prev, left_curr point arrays.
    """
    left_feat_prev = feat_left_prev.keypoints
    right_feat_prev = feat_right_prev.keypoints
    left_feat_curr = feat_left_curr.keypoints
    stereo_matches_prev = stereo_matches

    # Map left_prev indices to right_prev indices via stereo matches
    stereo_map = {m.queryIdx: m.trainIdx for m in stereo_matches_prev}

    # Filter temporal matches where left_prev (queryIdx) has a stereo match
    filtered_temporal = [m for m in temporal_matches if m.queryIdx in stereo_map]

    if not filtered_temporal:
        empty = np.empty((0, 2), dtype=np.float32)
        return StereoTemporalCorrespondences(left_prev=empty, right_prev=empty.copy(), left_curr=empty.copy())

    # Extract indices as NumPy arrays
    query_indices = np.array([m.queryIdx for m in filtered_temporal])
    train_indices = np.array([m.trainIdx for m in filtered_temporal])
    right_indices = np.array([stereo_map[q] for q in query_indices])

    # Pre-extract all points as NumPy arrays (dtype float32 for cv2 compatibility)
    left_prev_all_pts = np.array([kp.pt for kp in left_feat_prev], dtype=np.float32)
    right_prev_all_pts = np.array([kp.pt for kp in right_feat_prev], dtype=np.float32)
    left_curr_all_pts = np.array([kp.pt for kp in left_feat_curr], dtype=np.float32)

    # Select points via vectorized indexing
    left_prev_pts = left_prev_all_pts[query_indices]
    right_prev_pts = right_prev_all_pts[right_indices]
    left_curr_pts = left_curr_all_pts[train_indices]

    return StereoTemporalCorrespondences(left_prev=left_prev_pts, right_prev=right_prev_pts, left_curr=left_curr_pts)
