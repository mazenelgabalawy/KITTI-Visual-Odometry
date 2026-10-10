"""Grid-based keypoint downsampling: at most max_per_grid keypoints per grid cell,
the strongest by response. Cells are visited in first-occurrence order and keep
the input order among equal responses.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from kitti_slam.config.schema import GridFilterConfig
from kitti_slam.features.base import Features


class GridKeypointFilter:
    """Downsamples keypoints to at most max_per_grid per grid cell, by response."""

    def __init__(self, cfg: GridFilterConfig):
        self.cfg = cfg

    def __call__(self, features: Features, image_shape: tuple[int, int]) -> Features:
        keypoints = features.keypoints
        descriptors = features.descriptors
        grid_size = self.cfg.grid_size
        max_per_grid = self.cfg.max_per_grid

        if len(keypoints) == 0:
            return Features(keypoints=[], descriptors=np.array([]))

        height, width = image_shape[:2]

        # Extract positions
        positions = np.array([kp.pt for kp in keypoints])

        # Compute grid indices (row = y // cell_height, col = x // cell_width). The
        # cell counts round up, so a partial row or column at the image remainder is
        # its own cell instead of being merged into its neighbour (BUG-006).
        n_rows = -(-height // grid_size[0])
        n_cols = -(-width // grid_size[1])
        rows = np.clip(positions[:, 1] // grid_size[0], 0, n_rows - 1)
        cols = np.clip(positions[:, 0] // grid_size[1], 0, n_cols - 1)

        # Group by grid cell
        grid_groups = defaultdict(list)
        for idx, (row, col) in enumerate(zip(rows, cols, strict=True)):
            grid_groups[(row, col)].append(idx)

        # Select top max_per_grid by response per cell
        selected_indices = []
        for cell_indices in grid_groups.values():
            if len(cell_indices) > max_per_grid:
                sorted_indices = sorted(cell_indices, key=lambda i: keypoints[i].response, reverse=True)
                selected_indices.extend(sorted_indices[:max_per_grid])
            else:
                selected_indices.extend(cell_indices)

        # Extract selected
        selected_kps = [keypoints[i] for i in selected_indices]
        selected_descs = descriptors[selected_indices]

        return Features(keypoints=selected_kps, descriptors=selected_descs)
