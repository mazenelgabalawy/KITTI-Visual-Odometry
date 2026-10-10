"""Descriptor matchers, ported from legacy filter.BF_matcher and filter.rectified_stereo_matcher."""

from __future__ import annotations

import cv2
import numpy as np

from kitti_slam.config.schema import StereoMatcherConfig
from kitti_slam.features.base import Features, Matcher


class BruteForceMatcher(Matcher):
    """Brute-force descriptor matcher.

    With ratio_threshold=None this reproduces the plain `bf.match(...)` path used by
    VO_base.py (raw matches, no sorting/truncation). With ratio_threshold set, this
    reproduces filter.BF_matcher's `inliner_threshold` behavior: sort by distance and
    keep the top `int(len(matches) * ratio_threshold)` matches.
    """

    def __init__(self, norm: str, cross_check: bool, ratio_threshold: float | None = None):
        self.norm_type = getattr(cv2, norm)
        self.cross_check = cross_check
        self.ratio_threshold = ratio_threshold

    def match(self, feat_a: Features, feat_b: Features) -> list:
        desc1 = feat_a.descriptors
        desc2 = feat_b.descriptors

        # Defensive dtype casting based on norm type (copied from BF_matcher)
        if self.norm_type in (cv2.NORM_HAMMING, cv2.NORM_HAMMING2):
            if desc1.dtype != np.uint8:
                desc1 = desc1.astype(np.uint8)
            if desc2.dtype != np.uint8:
                desc2 = desc2.astype(np.uint8)
        elif self.norm_type in (cv2.NORM_L2, cv2.NORM_L1):
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
            if desc2.dtype != np.float32:
                desc2 = desc2.astype(np.float32)

        bf = cv2.BFMatcher(self.norm_type, crossCheck=self.cross_check)
        matches = bf.match(desc1, desc2)

        if self.ratio_threshold is None:
            return matches

        # Filter good matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        num_good_matches = int(len(matches) * self.ratio_threshold)
        matches = matches[:num_good_matches]

        return matches


class RectifiedStereoMatcher(Matcher):
    """Epipolar-constrained stereo matcher, ported from filter.rectified_stereo_matcher."""

    def __init__(self, cfg: StereoMatcherConfig):
        self.cfg = cfg

    # Precomputed once at import time: popcount[b] = number of set bits in byte b.
    _POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    @classmethod
    def _directional_match(
        cls,
        feat_query: list,
        desc_query: np.ndarray,
        feat_train: list,
        desc_train: np.ndarray,
        pixel_threshold: float,
        max_hamming_distance: int,
        ratio_test: float,
        max_disparity: float,
        min_disparity: float,
        reverse_disparity: bool,
    ) -> list:
        """Vectorized one-directional epipolar-constrained matching pass.

        Matches the per-keypoint specification in
        tests/unit/test_matcher_vectorization.py exactly: candidates are the
        train keypoints within the row band whose disparity lies in
        [min_disparity, max_disparity]; the best is the first minimum Hamming
        distance in row-sorted order; with two or more candidates it must pass
        the ratio test against the second-best candidate distance.

        Instead of computing a dense (n_query, n_train) Hamming distance matrix
        (which wastes work: the true epipolar bands are typically tiny relative to
        n_train), this gathers, for every query, only the columns within its own
        [low, high) band into a padded (n_query, max_band, 32) tensor and computes
        Hamming distances there with the same LUT-popcount approach as the
        reference. Padding columns are masked with a sentinel so they never win
        argmin/partition, preserving the reference's tie-break semantics exactly.
        """
        n_query = len(feat_query)
        n_train = len(feat_train)

        # Pre-sort train keypoints by y (stable, matches np.argsort default 'quicksort'
        # behavior used by the reference -- np.argsort ties are resolved identically
        # regardless of algorithm here because we replicate the exact same call).
        y_train = np.array([kp.pt[1] for kp in feat_train])
        idx_train_sorted = np.argsort(y_train)
        y_train_sorted = y_train[idx_train_sorted]
        desc_train_sorted = desc_train[idx_train_sorted]
        # Extract x-coords once as a numeric array and reorder by the sort permutation,
        # avoiding an expensive object-array construction + Python-loop over sorted objects.
        u_train_sorted = np.array([kp.pt[0] for kp in feat_train], dtype=np.float64)[idx_train_sorted]

        rows = np.array([kp.pt[1] for kp in feat_query], dtype=np.float64)
        u_query = np.array([kp.pt[0] for kp in feat_query], dtype=np.float64)

        low = np.searchsorted(y_train_sorted, rows - pixel_threshold, side="left")
        high = np.searchsorted(y_train_sorted, rows + pixel_threshold, side="right")
        band_size = high - low

        max_band = int(band_size.max()) if n_query > 0 else 0
        if max_band == 0:
            # Every query's band is empty -- nothing can match.
            return []

        SENTINEL = 1 << 30

        # Gather, per query, the candidate descriptors in its band (padded to max_band
        # with clipped, later-masked indices so the gather never goes out of bounds).
        offsets = np.arange(max_band)
        cols = low[:, np.newaxis] + offsets[np.newaxis, :]  # (n_query, max_band)
        valid = cols < high[:, np.newaxis]  # cols is already >= low by construction
        clipped_cols = np.minimum(cols, n_train - 1)

        gathered = desc_train_sorted[clipped_cols]  # (n_query, max_band, 32) uint8
        xor = desc_query[:, np.newaxis, :] ^ gathered  # (n_query, max_band, 32) uint8
        dists = cls._POPCOUNT_LUT[xor].sum(axis=2, dtype=np.int64)  # (n_query, max_band)
        dists = np.where(valid, dists, SENTINEL)

        # Geometry before ranking (BUG-005): only candidates inside the row band
        # AND the disparity range compete for best and second-best, so a closer
        # descriptor with impossible geometry cannot suppress a valid match.
        u_band = u_train_sorted[clipped_cols]  # (n_query, max_band)
        disparity = u_band - u_query[:, np.newaxis] if reverse_disparity else u_query[:, np.newaxis] - u_band
        candidate = valid & (disparity >= min_disparity) & (disparity <= max_disparity)
        dists = np.where(candidate, dists, SENTINEL)
        n_candidates = candidate.sum(axis=1)

        min_idx_local = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(n_query), min_idx_local]

        if max_band >= 2:
            second_dist = np.partition(dists, 1, axis=1)[:, 1]
        else:
            # n_candidates <= max_band < 2 here, so the ratio test never applies;
            # this value is unused but must exist for the boolean expression below.
            second_dist = np.full(n_query, SENTINEL, dtype=dists.dtype)

        min_idx_col = np.minimum(low + min_idx_local, n_train - 1)

        keep = n_candidates > 0
        keep &= min_dist <= max_hamming_distance
        ratio_gate = (n_candidates >= 2) & (min_dist / (second_dist + 1e-9) >= ratio_test)
        keep &= ~ratio_gate

        matches = []
        train_idx_original_all = idx_train_sorted[min_idx_col]
        keep_query_indices = np.nonzero(keep)[0]
        for query_idx in keep_query_indices:
            matches.append(
                cv2.DMatch(
                    _queryIdx=int(query_idx),
                    _trainIdx=int(train_idx_original_all[query_idx]),
                    _distance=float(min_dist[query_idx]),
                )
            )
        return matches

    def match(self, feat_a: Features, feat_b: Features) -> list:
        feat_left = feat_a.keypoints
        desc_left = feat_a.descriptors
        feat_right = feat_b.keypoints
        desc_right = feat_b.descriptors
        cross_check = self.cfg.cross_check
        pixel_threshold = self.cfg.pixel_threshold
        max_hamming_distance = self.cfg.max_hamming_distance
        ratio_test = self.cfg.ratio_test
        max_disparity = self.cfg.max_disparity
        min_disparity = 1.0

        if len(feat_left) == 0 or len(feat_right) == 0:
            return []

        l2r_matches = self._directional_match(
            feat_left,
            desc_left,
            feat_right,
            desc_right,
            pixel_threshold,
            max_hamming_distance,
            ratio_test,
            max_disparity,
            min_disparity,
            reverse_disparity=False,
        )

        if not cross_check:
            return l2r_matches

        r2l_matches = self._directional_match(
            feat_right,
            desc_right,
            feat_left,
            desc_left,
            pixel_threshold,
            max_hamming_distance,
            ratio_test,
            max_disparity,
            min_disparity,
            reverse_disparity=True,
        )

        r2l_backpointer = {m.queryIdx: m.trainIdx for m in r2l_matches}  # right -> left
        mutual_matches = [
            m for m in l2r_matches if m.trainIdx in r2l_backpointer and r2l_backpointer[m.trainIdx] == m.queryIdx
        ]

        return mutual_matches
