# Errata for `VO_Seminar.pdf`

`VO_Seminar.pdf` is kept unchanged as a historical record of the experiment.
This page corrects the statements in it that do not match the code that produced
its results: `VO_base.py` and `VO_constrained.py` at commit `b8796f7`. The items
come from a 2026-09-06 code audit and were rechecked against the PDF text and
the code. The code on this branch has since been corrected (see the readme);
where a correction changes the behavior described below, it is noted. All error
values are in SI units: m and rad.

## Abstract, Section 3.8: "monocular" and bundle adjustment

- **Claim:** the abstract describes "monocular-to-stereo" estimation and three
  approaches: "baseline monocular VO, constrained optimization …, and bundle
  adjustment refinement". Section 3.8 describes BA as applied "when enabled".
- **Correction:** both working pipelines are **stereo**. They triangulate from
  the left/right pair and estimate pose with 3D–2D PnP. No monocular pipeline
  exists. Bundle adjustment was never run: `VO_BA.py` never called it, and the
  `BundleAdjustment` class in `bundle_adjustment.py` had unimplemented methods
  and failed on construction (undefined `pose_noise` / `landmark_noise`). Both
  files are removed on this branch. The conclusion's description of BA as future
  work is accurate.

## Section 3.4.1: baseline matching

- **Claim:** the baseline uses brute-force matching "and a distance threshold".
- **Correction:** the baseline keeps every cross-checked brute-force match
  (`bf.match`) with **no** descriptor-distance threshold. The constrained
  temporal matcher (`BF_matcher`) keeps the best fraction of matches by distance
  (`temporal_inlier_threshold: 0.5`). This is not a Lowe nearest/second-nearest ratio test. The
  ratio test (`stereo_ratio_test: 0.8`) exists only in the constrained stereo matcher.

## Sections 3.6 and 3.9: what happens on a skipped frame

- **Claim:** "the update for that frame is skipped and the previous pose is
  reused". Algorithm 1 line 19 says "keep previous pose".
- **Correction:** this held for the **base** pipeline only. The **constrained**
  pipeline applied the last successful relative motion again,
  `C_k ← C_{k−1} · T_prev`. This is constant-motion **extrapolation**. The code comment and
  README called it "interpolation", but no interpolation between known poses takes place.
  The constrained pipeline also skipped frames that failed its motion gate, not only frames
  with too few points.
- **On this branch:** both pipelines assume constant velocity on every skipped frame.
  With the default configurations, no frame of the 11 labeled sequences is skipped.

## Section 3.7.3: planar-motion prior and the motion gate

- **Yaw extraction:** the report gives `θ = atan2(R₀₂, R₀₀)`. The code
  (`orthogonalize_to_planar_motion`) uses `θ = atan2(R₀₂ − R₂₀, R₀₀ + R₂₂)`
  after SVD orthonormalization. This is the least-squares yaw of the nearest
  rotation about the camera y axis. The two formulas agree for a pure y rotation and differ
  in general.
- **"Extreme pitch/roll" rejection:** the code gates one Euler component only. That is the
  second component of `rotationMatrixToEulerAngles`, the rotation about the
  camera **y** axis. With x right, y down, and z forward, y is the vehicle's heading
  (yaw) axis, so `max_pitch_deg: 30` rejects frames whose heading change exceeds 30°.
  Roll and pitch are not gated. A pure 45° rotation about y is rejected, and a pure 45° rotation
  about x passes. The translation gate (`max_translation: 15` m) is as described.

## Section 4.4: alignment "across all sequences"

- **Claim:** strong yaw consistency explains "the good alignment of the
  estimated x–z trajectories with the ground truth across all sequences".
- **Correction:** this conflicts with the report's own Table 3. On highway
  sequence 01 the constrained APE RMSE is 346.56 m, and the trajectory falls far
  short of the ground truth. Orientation quality needs a declared convention
  and quantitative angular errors. `create_results.sh` now reports the rotation
  APE and RPE in rad. Euler-angle plots alone do not show it.

## Section 4.5, Table 3: the RPE interval is 100 frames, not 100 m

- **Claim:** "RPE is computed over 100 m segments."
- **Correction:** the values come from `evo_rpe kitti gt est --delta 100` with
  evo's default interval unit, **frames**. They are consecutive 100-frame pairs. Both the
  APE and RPE values are translation errors in meters. Only the pair interval is in
  frames. Recomputed from the committed `output_constrained` trajectories:

  | Seq | Table 3 RPE RMSE / max (m) | 100 frames, recomputed (m) | 100 m, pairs chosen on the ground truth (m) |
  |---|---|---|---|
  | 00 | 1.71 / 4.29 | 1.716797 / 4.295980 | 2.130465 / 11.542421 |
  | 01 | 61.03 / 151.40 | 61.033702 / 151.403500 | 26.406932 / 89.473644 |
  | 05 | 1.74 / 3.83 | 1.746188 / 3.830967 | 1.989421 / 5.436592 |

  The 100 m column is what `create_results.sh` computes on this branch
  (`--delta 100 --delta_unit m --all_pairs --pairs_from_reference`).
  The Table 3 APE values (39.01 / 87.59, 346.55 / 531.25, and 11.81 / 36.18 m) are
  correct as printed. They are truncated, not rounded, from 39.015891 / 87.597984,
  346.558673 / 531.252296, and 11.807016 / 36.182707 m. Neither interval
  corresponds to the official KITTI score. That score is the mean segment drift over
  100–800 m, computed by the `kitti_slam` package on the `main` branch (`kitti_segments`).
  For these trajectories it is 0.023135, 0.213890, and 0.019072 m/m translational drift
  and 1.842e-04, 2.441e-04, and 1.563e-04 rad/m rotational drift.

## Committed `output_base/01/poses.txt` is truncated

Not a seminar statement, but it affects any reuse of the historical results.
Since commit `b8796f7`, `output_base/01/poses.txt` has held a single identity
pose. That commit introduced `config/base.yaml` with all 11 sequences. The original
runner truncated the next sequence's file and then crashed on its second
sequence (`getattr(cv2, <int>)`). The committed `output_base/01/ape_results.txt`
and `rpe_results.txt` were computed from the earlier 1101-pose trajectory,
Git blob `bc8aed40` (`git show 9f15e7c:output_base/01/poses.txt`), which
reproduces them exactly. On this branch, the runners run every configured
sequence and write to `results/` by default, so the committed results are not
overwritten.
