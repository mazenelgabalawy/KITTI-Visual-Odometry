# Capabilities, configuration, and contracts

Status as of 2026-10-10. **Implemented** means code exists. **Validated** means a
recorded check passed.

The package began as a port of the original seminar scripts and reproduced their
trajectories byte for byte on all 11 labeled sequences in both configurations,
until the corrections in §4. The original scripts are in Git history at revision
`461c8fe`.

## 1. Capability status

| Capability | Status | Entry point |
|---|---|---|
| Stereo VO, `base` configuration | Implemented; validated on all 11 labeled sequences (§4) | `scripts/run_vo.py --config configs/base.yaml` |
| Stereo VO, `constrained` configuration | Implemented; validated on all 11 labeled sequences (§4) | `scripts/run_vo.py --config configs/constrained.yaml` |
| Trajectory evaluation, SI units: `metric_distance` (100 m RPE) and `kitti_segments` | Implemented; checked against closed-form synthetic trajectories | `scripts/evaluate.py` |
| Feature/match figures (keypoints, grid selection, BF vs. rectified stereo, temporal, ORB vs. SIFT) | Implemented | `scripts/visualize_features.py` |
| Per-stage timing | Implemented (opt-in) | `run_vo.py --timing` |
| Per-frame diagnostics (registration outcome, match/point/inlier counts) | Implemented (opt-in); poses unchanged | `run_vo.py --diagnostics` |
| Bundle adjustment | **Not implemented** (Phase E) | — |
| Mapping, loop closure, pose graph, visual SLAM | Not implemented (Phase E) | — |
| Monocular VO, 3D reconstruction | Not implemented (Phase F) | — |

`scripts/` contains checkout tools that run from a repository clone. They are
not installed as console commands by the wheel.

## 2. Configuration reference (`configs/*.yaml`)

Unknown keys are rejected, so a misspelled or removed setting fails loudly.

| Key | Meaning |
|---|---|
| `run.sequences`, `run.dataset_path`, `run.output_dir` | Paths resolve against the repository root. `run_vo.py --sequence/--output/--dataset-path` override them. The default output is `results/<variant>`. |
| `detector.nfeatures`, `detector.scale_factor`, `detector.nlevels` | ORB parameters |
| `grid_filter.grid_size`, `grid_filter.max_per_grid` | Grid cell size `[rows, columns]` (px) and the number of strongest keypoints kept per cell (constrained only) |
| `temporal_matcher.norm` | OpenCV norm name. The brute-force stereo matcher reuses it. |
| `temporal_matcher.cross_check`, `stereo_matcher.cross_check` | Mutual-best filtering, set separately for each matcher |
| `temporal_matcher.ratio_threshold` | The **fraction of matches kept** after sorting by distance (0.5 keeps the best half). It is not a Lowe ratio test. |
| `stereo_matcher.kind` | `bruteforce`, or `rectified` with `pixel_threshold` (row band, px), `max_hamming_distance`, `ratio_test` (Lowe ratio among geometrically valid candidates), and `max_disparity` (px; the minimum is 1 px) |
| `triangulation.min_abs_w` | Landmark validity: a triangulated point reaches PnP only if it is finite, in front of both cameras, and its unit-norm homogeneous `\|w\|` ≥ `min_abs_w`. `\|w\|` ≈ 1/range (1/m), so `1.0e-4` rejects points beyond about 10 km. |
| `landmark_filter.min_depth_multiplier`, `landmark_filter.relative_depth_uncertainty_threshold` | Minimum depth = multiplier × baseline (m). The uncertainty rule `σ_z/z < τ` is equivalent to disparity > 1/τ px. |
| `pnp.confidence`, `pnp.iterations_count` | `cv2.solvePnPRansac` arguments. The reprojection threshold is OpenCV's default of 8 px. |
| `pnp.min_inliers` | A solve with fewer RANSAC inliers is a failed registration. `6` rejects exactly the solves with no inlier beyond OpenCV's 5-point minimal sample. |
| `pnp.seed` | Sets `cv2.setRNGSeed`. It has **no effect** on the current solver path, whose RANSAC uses a fixed internal RNG. |
| `motion.max_translation` | Rejects a relative motion whose translation exceeds this value (m) |
| `motion.max_pitch_deg` | **Misnamed:** gates the second Euler component, the rotation about camera *y*, which is the vehicle's heading (yaw) axis, not pitch. Never fires on KITTI. |
| `motion.orthogonalize_alpha` | Blend weight toward the planar (yaw-only) rotation |

A failed registration (no correspondences, too few points, a failed or rejected
PnP solve, implausible motion) always propagates the last accepted relative
motion: constant velocity, in every configuration.

## 3. File and output contracts

- **Pose files:** one line per frame. Each line holds the top 3×4 block of the
  camera-to-world pose, row-major, 12 values formatted with `%.6e` and separated by spaces,
  followed by a newline. The first line is the exact identity. A complete run of an N-frame
  sequence writes exactly N lines. `kitti_slam.evaluation.pose_checks` enforces this contract,
  plus finite values and valid rotations.
- **Run outputs:** `run_vo.py` writes `<output>/<XX>/poses.txt` after the sequence
  completes, and `diagnostics.csv` next to it with `--diagnostics`. Each sequence
  runs independently. Paths that resolve into `output_base/`, `output_constrained/`,
  or `reference/` are refused, including through symlinks at any level
  (`kitti_slam.protected_paths`).
- **Metrics:** `evaluate.py` writes `<XX>/metrics/<profile>.{json,txt}` and,
  with `--plots`, `<XX>/plots/*.png`. Definitions: [evaluation.md](evaluation.md).

## 4. Corrections (Phase B)

Each correction is its own commit with tests. Effects are measured on all 11
labeled sequences × 2 configurations against the package before the correction,
with the default profiles (`metric_distance`, `kitti_segments`) in SI units.

| ID | Before | Corrected behavior | Measured effect on KITTI |
|---|---|---|---|
| BUG-013 | A failed registration held the pose in base (zero velocity) and repeated the last relative motion in constrained (`interpolate_on_skip`) | Every failed registration (no correspondences, too few points, failed or rejected PnP, implausible motion) propagates the last accepted relative transform: constant velocity, in both configurations. Propagation does not change the transform, and the next registration starts from the current frame | None on the shipped configs: no registration failed on any sequence before this change, so all 22 trajectories are byte-identical |
| BUG-003 | Every triangulated point reached PnP, including points behind the cameras and points with near-zero homogeneous `w` (zero disparity, z up to 1e25 m) | One validity check in both configurations: finite coordinates, positive depth in both cameras, `\|w\|` ≥ `triangulation.min_abs_w` | Base only (all 11 base trajectories change, no constrained one): it rejects 2.8% (seq 04) to 20.1% (seq 01) of associated points; on seq 01, 11.3% are behind the cameras and 8.7% lie beyond 10 km. APE improves on 7 of 11 sequences (seq 01: 522.8 → 417.8 m; seq 06: 14.71 → 8.85 m) and worsens on 4 (largest: seq 08, 101.4 → 112.4 m). Constrained is unchanged: its matcher bounds disparity to [1, 64] px |
| BUG-002 | The `solvePnPRansac` return value was ignored and its outputs were composed into the trajectory unchecked | A solve is accepted only if the solver succeeds, its rotation and translation are finite, and it has at least `pnp.min_inliers` inliers; otherwise it is a failed registration (BUG-013). The inliers are kept on the estimate | The solver never failed and no output was nonfinite. One frame on all of KITTI had fewer than 6 inliers: constrained seq 01 frame 435 (5 inliers of 8 landmarks), propagated by this correction; after BUG-005 the frame has 7 inliers and registers |
| BUG-006 | Grid cell indices were clipped to `size // cell − 1`, so the partial bottom row and right column at the image remainder shared a cell (and its `max_per_grid` cap) with their neighbours | Cell counts round up: the remainder row and column are cells of their own. The response tie order is kept | None on the shipped configs: ORB's 31 px border keeps every keypoint at least 32 px from the right and bottom edges, while the remainder strips are 11–26 px wide, so no keypoint ever fell in a merged cell. Grids whose remainder is wider than the border (for example 100 px rows) are affected |
| BUG-005 | The rectified matcher chose the best and second-best descriptor among all keypoints in the row band, then rejected the best if its disparity was outside [1, `max_disparity`] px. A closer descriptor with impossible geometry suppressed a valid match, and entered the ratio test | Only candidates inside the row band **and** the disparity range compete for best and second-best; the ratio test compares valid candidates only; cross-check keeps mutual best matches under the same rule | Matches are only added, never removed: +0.4% (seq 04), +0.6% (seq 00), +5.7% (seq 01) stereo matches. On highway seq 01 the added matches are mostly far (median disparity 1.8 px, 79% below 4 px) and less distinctive; rotation error falls and translation error rises there: APE 345.7 → 365.0 m, segment drift 0.214 → 0.231 m/m and 2.44e-4 → 2.32e-4 rad/m. Over the 11 constrained sequences APE changes by −0.9 to +1.2 m except seq 01, improving on 4 and worsening on 7 |
| BUG-004 | The base (brute-force) stereo matcher was built with `temporal_matcher.cross_check`; `stereo_matcher.cross_check` was ignored | The stereo matcher uses `stereo_matcher.cross_check` | None on the shipped configs, which set both to `true`; experiments that vary the stereo setting now get it |

Net effect of all corrections (before → after all corrections), number of
sequences out of 11 where each metric improved / worsened:

| Configuration | APE t (m) | RPE t, 100 m (m) | RPE r, 100 m (rad) | Segment t (m/m) | Segment r (rad/m) |
|---|---|---|---|---|---|
| base | 7 / 4 | 7 / 4 | 6 / 5 | 8 / 3 | 5 / 6 |
| constrained | 4 / 7 | 4 / 7 | 5 / 6 | 5 / 6 | 6 / 5 |

A correction is kept because it removes a defect, not because it lowers
average error; the regressions above are reported, not hidden. No
registration failed on any sequence after the corrections.
