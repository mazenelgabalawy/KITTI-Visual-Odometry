"""Opt-in per-frame diagnostics for StereoVOPipeline.run().

One row per registered or failed frame: every frame after the first (frame 0 is
the identity pose and is never registered). Each row records the registration
outcome and the counts behind it. Collecting diagnostics never changes the
estimated poses.

Columns:
  frame                     frame index k (registration of k-1 -> k)
  outcome                   one of OUTCOMES
  temporal_matches          left(k-1) -> left(k) matches
  stereo_matches            left(k-1) -> right(k-1) matches
  associated                correspondences present in both match sets
  nonfinite_points          triangulated points with a nonfinite coordinate
  nonpositive_depth_points  finite triangulated points with z <= 0
  invalid_points            points rejected by the landmark validity check (BUG-003)
  landmarks                 points passed to PnP (after any filtering)
  pnp_retval                cv2.solvePnPRansac return value (1/0); empty if PnP did not run
  pnp_inliers               RANSAC inlier count; empty if PnP did not run
  consecutive_failures      failed registrations in a row, ending at this frame
"""

from __future__ import annotations

import csv
from pathlib import Path

REGISTERED = "registered"
OUTCOMES = (
    REGISTERED,
    "no_correspondences",
    "too_few_points",
    "pnp_failed",
    "implausible_translation",
    "implausible_heading",
    "implausible_motion",  # a motion filter that does not name its gate
)
FIELDS = (
    "frame",
    "outcome",
    "temporal_matches",
    "stereo_matches",
    "associated",
    "nonfinite_points",
    "nonpositive_depth_points",
    "invalid_points",
    "landmarks",
    "pnp_retval",
    "pnp_inliers",
    "consecutive_failures",
)


class FrameDiagnostics:
    """Collects one diagnostics row per frame."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def append(self, row: dict) -> None:
        if row.get("outcome") not in OUTCOMES:
            raise ValueError(f"unknown outcome {row.get('outcome')!r}")
        previous = self.rows[-1]["consecutive_failures"] if self.rows else 0
        row = {field: row.get(field, "") for field in FIELDS}
        row["consecutive_failures"] = 0 if row["outcome"] == REGISTERED else previous + 1
        self.rows.append(row)

    def write_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
