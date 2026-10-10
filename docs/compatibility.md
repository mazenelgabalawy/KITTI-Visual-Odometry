# Capabilities, compatibility contracts, and legacy retirement

Status as of 2026-10-08. **Implemented** means code exists. **Validated**
means a recorded check passed. The legacy root scripts were removed after the
port was verified (§5).

## 1. Capability status

| Capability | Status | Entry point |
|---|---|---|
| Stereo VO, `base` configuration | Implemented; validated byte-identical to fresh legacy on all 11 labeled sequences | `scripts/run_vo.py --config configs/base.yaml` |
| Stereo VO, `constrained` configuration | Implemented; validated byte-identical to fresh legacy on all 11 labeled sequences | `scripts/run_vo.py --config configs/constrained.yaml` |
| Trajectory evaluation, SI units: `metric_distance` (100 m RPE, default), `kitti_segments` (default), `legacy_frames` (opt-in) | Implemented; `legacy_frames` reproduces all 44 historical metric files | `scripts/evaluate.py` |
| Feature/match figures (keypoints, grid selection, BF vs. rectified stereo, temporal, ORB vs. SIFT) | Implemented; replaces legacy `features.py` / `detectors.py` | `scripts/visualize_features.py` |
| Per-stage timing | Implemented (opt-in) | `run_vo.py --timing` |
| Bundle adjustment | **Not implemented** (Phase E). The non-functional legacy skeleton was removed (§5) | — |
| Mapping, loop closure, pose graph, visual SLAM | Not implemented (Phase E) | — |
| Monocular VO, 3D reconstruction | Not implemented (Phase F) | — |

`scripts/` contains checkout tools that run from a repository clone. They are
not installed as console commands by the wheel.

## 2. Configuration mapping (removed legacy `config/` → modern `configs/`)

Each modern key has one meaning. A setting that was shared in legacy code is now
a separate key.

| Legacy key (`config/*.yaml`) | Modern key (`configs/*.yaml`) | Meaning and notes |
|---|---|---|
| `sequences`, `dataset_path`, `output_dir` | `run.sequences`, `run.dataset_path`, `run.output_dir` | Paths resolve against the repository root. `run_vo.py --sequence/--output/--dataset-path` override them. The modern default output is `results/<variant>`, never `output_*`. |
| `nfeatures`, `scale`, `nlevels` | `detector.nfeatures`, `detector.scale_factor`, `detector.nlevels` | ORB parameters |
| `grid_size`, `max_per_grid` | `grid_filter.grid_size`, `grid_filter.max_per_grid` | Grid cell size (px) and the number of strongest keypoints kept per cell (constrained only). |
| `matching_norm` | `temporal_matcher.norm` | OpenCV norm name. The base stereo brute-force matcher reuses it. |
| `cross_check` (base) | `temporal_matcher.cross_check` and `stereo_matcher.cross_check` | Legacy base used one value for both. **Known defect BUG-004:** the factory currently builds the base stereo matcher from `temporal_matcher.cross_check` and ignores `stereo_matcher.cross_check`. The shipped configs set both to `true`. |
| `temporal_cross_check`, `temporal_inlier_threshold` | `temporal_matcher.cross_check`, `temporal_matcher.ratio_threshold` | `ratio_threshold` is the **fraction of matches kept** after sorting by distance (0.5 keeps the best half). It is not a Lowe ratio test. |
| `stereo_cross_check`, `stereo_pixel_threshold`, `stereo_max_hamming_distance`, `stereo_ratio_test`, `stereo_max_disparity` | `stereo_matcher.{cross_check, pixel_threshold, max_hamming_distance, ratio_test, max_disparity}` with `kind: rectified` | Rectified matcher: y-band (px), Hamming cap, Lowe ratio, and disparity cap (px). `kind: bruteforce` is the base matcher. |
| `min_depth_multiplier`, `relative_depth_uncertainty_threshold` | `landmark_filter.*` | Minimum depth = multiplier × baseline (m). The uncertainty rule `σ_z/z < τ` is equivalent to disparity > 1/τ px. |
| — | `triangulation.min_abs_w` | Landmark validity (BUG-003, §4b): a triangulated point reaches PnP only if it is finite, in front of both cameras, and its unit-norm homogeneous `\|w\|` ≥ `min_abs_w`. `\|w\|` ≈ 1/range (1/m), so `1.0e-4` rejects points beyond about 10 km. |
| `pnp_confidence`, `pnp_iterations_count` | `pnp.confidence`, `pnp.iterations_count` | `cv2.solvePnPRansac` arguments. The reprojection threshold is OpenCV's default of 8 px. |
| — | `pnp.min_inliers` | A solve with fewer RANSAC inliers is rejected as a failed registration (BUG-002, §4b). `6` rejects exactly the solves with no inlier beyond OpenCV's 5-point minimal sample. |
| — | `pnp.seed` | Sets `cv2.setRNGSeed`. It has **no effect** on the current solver path, whose RANSAC uses a fixed internal RNG (BUG-014). |
| `max_translation` | `motion.max_translation` | Rejects a frame whose relative translation exceeds this value (m). |
| `max_pitch_deg` | `motion.max_pitch_deg` | **Misnamed (BUG-012):** this setting gates the second Euler component, the rotation about camera *y*. That axis is the vehicle's heading (yaw) axis. It is not pitch. |
| `orthogonalize_alpha` | `motion.orthogonalize_alpha` | Blend weight toward the planar (yaw-only) rotation |
| — (behavior differed by script) | removed (was `motion.interpolate_on_skip`) | Every configuration now assumes constant velocity on a failed registration (§4b, BUG-013). Configs that still set the key are rejected as unknown. |

## 3. File and output contracts

- **Pose files:** one line per frame. Each line holds the top 3×4 block of the
  camera-to-world pose, row-major, 12 values formatted with `%.6e` and separated by spaces,
  followed by a newline. The first line is the exact identity. A complete run of an N-frame
  sequence writes exactly N lines. `kitti_slam.evaluation.pose_checks` enforces this contract,
  plus finite values and valid rotations.
- **Run outputs:** `run_vo.py` writes `<output>/<XX>/poses.txt`. Paths that
  resolve into `output_base/`, `output_constrained/`, or `reference/` are refused,
  including through symlinks at any level (`kitti_slam.protected_paths`).
- **Metrics:** `evaluate.py` writes `<XX>/metrics/<profile>.{json,txt}` and,
  with `--plots`, `<XX>/plots/*.png`. The historical `ape_results.txt` and `rpe_results.txt`
  layouts are not reproduced. Their statistics are reproduced and compared numerically.

## 4. Intentional differences from legacy

| Area | Legacy behavior | Modern behavior |
|---|---|---|
| Multiple sequences per run | Crashes on the second sequence after truncating its output file | Each sequence runs independently |
| No correspondences | `cv2.triangulatePoints` on empty input | A failed registration, handled like every other failure (§4b) |
| Base feature re-detection | Re-detects the previous frame's features every iteration | Carries features forward. The results are numerically identical and detection costs half as much. |
| Output timing | Appends each pose as it goes, so a crash leaves a partial file | Writes after the sequence completes. Atomic writes and resume are deferred: they do not affect estimates (Phase B scope) |
| Evaluation script | `create_results.sh`: broken sequence selection; `>` truncates the header | `scripts/evaluate.py` with explicit profiles and an exit status |
| RPE interval | 100 frames (implicit evo default) | **100 m by default** (`metric_distance`, pairs on the reference path). `legacy_frames` keeps 100 frames, opt-in. |

Before the corrections in §4b, pose computation was unchanged: the 11×2 sweep
showed byte-identical trajectories (§1).

## 4b. Corrections since the port (Phase B)

Each correction is its own commit with tests. Effects are measured on all 11
labeled sequences against the package before the correction, with the default
profiles in SI units.

| ID | Legacy and port behavior | Corrected behavior | Measured effect on KITTI |
|---|---|---|---|
| BUG-013 | A failed registration held the pose in base (zero velocity) and repeated the last relative motion in constrained (`interpolate_on_skip`) | Every failed registration (no correspondences, too few points, failed or rejected PnP, implausible motion) propagates the last accepted relative transform: constant velocity, in both configurations. Propagation does not change the transform, and the next registration starts from the current frame | None on the shipped configs: no registration failed on any sequence before this change, so all 22 trajectories are byte-identical |
| BUG-003 | Every triangulated point reached PnP, including points behind the cameras and points with near-zero homogeneous `w` (zero disparity, z up to 1e25 m) | One validity check in both configurations: finite coordinates, positive depth in both cameras, `\|w\|` ≥ `triangulation.min_abs_w` | Base only: it rejects 2.8% (seq 04) to 20.1% (seq 01) of associated points; on seq 01, 11.3% are behind the cameras and 8.7% lie beyond 10 km. Constrained is unchanged: its matcher bounds disparity to [1, 64] px |
| BUG-002 | The `solvePnPRansac` return value was ignored and its outputs were composed into the trajectory unchecked | A solve is accepted only if the solver succeeds, its rotation and translation are finite, and it has at least `pnp.min_inliers` inliers; otherwise it is a failed registration (BUG-013). The inliers are kept on the estimate | The solver never failed and no output was nonfinite. One frame on all of KITTI had fewer than 6 inliers: constrained seq 01 frame 435 (5 inliers of 8 landmarks), now propagated |

## 5. Legacy retirement (plan task B7, executed 2026-10-08)

The owner asked to remove the legacy code once its migration was verified.
Prerequisites that held at removal: all 11 labeled sequences × 2 configurations
byte-identical between the modern pipeline and fresh legacy runs; the legacy
trajectories and logs retained outside the repository; every legacy-oracle unit
test switched to frozen goldens. Anything removed is recoverable from Git revision `461c8fe`
(`git show 461c8fe:<path>`).

| Removed | What it was | Replacement / disposition |
|---|---|---|
| `VO_base.py`, `VO_constrained.py`, `utils.py`, `filter.py` (pipeline parts), `config/*.yaml` | Original pipelines and their configs | `kitti_slam` + `configs/*.yaml` + `scripts/run_vo.py`, byte-identical. The sources and configs are in Git at `461c8fe` |
| `create_results.sh` | evo CLI wrapper (broken sequence selection, truncated headers) | `scripts/evaluate.py` |
| `features.py`, `detectors.py` | One-off seminar figure scripts | `scripts/visualize_features.py`, driven by a pipeline config, headless |
| `filter.py`: `planar_2pt_pnp`, `RANSAC_pnp_two_point`, `is_valid_keyframe` | Unused, unvalidated research helpers (the custom RANSAC never shortened its loop) | Not migrated: never part of either pipeline |
| `VO_BA.py`, `bundle_adjustment.py` | BA skeleton: no BA call, constructor bug, unimplemented methods; `VO_BA.py` would have truncated `output_constrained/` | Not migrated: never functional. BA is Phase E work |
| `VO_code.zip` | Duplicate archive of the four sources and the old README | Git history |

Kept: `output_base/`, `output_constrained/` (historical results, write-protected;
`output_base/01/poses.txt` is truncated, see the
[errata](errata/vo-seminar-errata.md)), and the historical figures
(`output_features/`, `keypoints/`, root `*.png`), the seminar PDFs, and
`imageas.zip`. These are not code.

### Tests that used legacy modules as oracles

These nine tests previously called `filter.py` / `utils.py` through
`pytest.importorskip`, so deleting the modules would have silently skipped
them. At retirement they were switched to frozen legacy outputs. Since
2026-10-10 they compare against package outputs instead:
`tests/unit/data/goldens.npz`, written by `tests/unit/data/generate_goldens.py`,
which calls only package code. Before the switch, all 25 package arrays were
checked equal (dtype, shape, values) to the frozen legacy outputs. The legacy
goldens and their generator stay local and are not committed. ORB-dependent
entries are tied to the pinned `opencv-python-headless` 4.11.0.86.

| Test | Legacy function it replaced |
|---|---|
| `test_matchers.py::test_bf_matcher_ratio_path_matches_golden` | `filter.BF_matcher` |
| `test_matchers.py::test_rectified_matches_golden` | `filter.rectified_stereo_matcher` |
| `test_association.py::test_matches_golden_on_non_empty_case` | `filter.filter_temporal_and_stereo_features` |
| `test_grid_filter.py::test_matches_golden_on_200_keypoints` | `filter.cluster_keypoints` |
| `test_orb_detector.py::test_detect_matches_golden` | `filter.process_image` |
| `test_motion_filters.py::test_orthogonalize_matches_golden` | `filter.orthogonalize_to_planar_motion` |
| `test_trajectory_io.py::test_writer_byte_identical_to_golden` | `utils.save_transformation_matrix` |
| `test_transforms.py::test_euler_matches_golden` | `utils.rotationMatrixToEulerAngles` |
| `test_transforms.py::test_euler_gimbal_lock_branch` | `utils.rotationMatrixToEulerAngles` |

`test_matchers.py::test_golden_features_match_current_detector` checks that the
frozen matcher inputs are still what the current detector produces.
