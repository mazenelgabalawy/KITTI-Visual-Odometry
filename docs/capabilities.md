# Capabilities, configuration, and contracts

Status as of 2026-10-10. **Implemented** means code exists. **Validated** means a
recorded check passed.

The package began as a port of the seminar scripts and reproduced their
trajectories byte for byte on all 11 labeled sequences in both configurations.
Later corrections were applied to both; the seminar scripts, corrected, are on
the `legacy` branch and still produce byte-identical trajectories.

## 1. Capability status

| Capability | Status | Entry point |
|---|---|---|
| Stereo VO, `base` configuration | Implemented; validated on all 11 labeled sequences | `scripts/run_vo.py --config configs/base.yaml` |
| Stereo VO, `constrained` configuration | Implemented; validated on all 11 labeled sequences | `scripts/run_vo.py --config configs/constrained.yaml` |
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
  runs independently. Paths that resolve into `reference/` are refused,
  including through symlinks at any level (`kitti_slam.protected_paths`).
- **Metrics:** `evaluate.py` writes `<XX>/metrics/<profile>.{json,txt}` and,
  with `--plots`, `<XX>/plots/*.png`. Definitions: [evaluation.md](evaluation.md).
