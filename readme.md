# KITTI stereo visual odometry

Stereo visual odometry on the KITTI odometry benchmark, packaged as
`kitti_slam` (`src/kitti_slam/`), with command-line tools in `scripts/` and
pipeline configurations in `configs/`. The `base` and `constrained` pipelines
are two configurations of one shared pipeline.

What works today is stereo VO, trajectory evaluation, and feature/match
visualization. Bundle adjustment, mapping, loop closure, monocular VO, and
reconstruction are **not implemented** yet. See
[docs/compatibility.md](docs/compatibility.md) for the capability status.

## Layout

```text
src/kitti_slam/
  config/       frozen-dataclass configs + YAML loader
  datasets/     KITTI stereo sequences, prefetching
  features/     ORB, grid selection, brute-force / rectified stereo matching, association
  geometry/     calibration, triangulation, SE(3)/SO(3) helpers
  odometry/     pipeline, factory, PnP, motion filters, stage timing
  evaluation/   pose-file I/O and validation, metrics (SI units), plots
  protected_paths.py, provenance.py   output protection, input hashing
scripts/        run_vo, evaluate, visualize_features
configs/        base.yaml, constrained.yaml
tests/          unit/ (CI) and regression/ (needs KITTI data)
docs/           evaluation definitions, compatibility, errata
```

## Setup

Python 3.12:

```bash
python3.12 -m venv .visual_odometry
source .visual_odometry/bin/activate
pip install -e ".[dev,eval]"      # package + tests/lint + evo metrics
```

| Extra | Provides | Needed for |
|---|---|---|
| (none) | numpy, OpenCV (`opencv-python-headless`), PyYAML, tqdm, scipy, matplotlib | running VO |
| `eval` | `evo` | `scripts/evaluate.py`, the metric tests |
| `dev` | pytest, ruff, pre-commit | development |
| `slam` | `gtsam==4.2.1` (the only release with Python 3.12 wheels) | future BA/graph work; nothing uses it yet |

Core VO imports work without `evo` and `gtsam`. The package depends on the
headless OpenCV wheel, because it never opens windows. Do not install `opencv-python` next to it:
both wheels provide `cv2`. `requirements.txt` pins a complete development
environment that matches `pyproject.toml`.

## Data

Expected layout (all paths are gitignored):

```text
sequences/XX/image_0/*.png   # left grayscale
sequences/XX/image_1/*.png   # right grayscale
sequences/XX/calib.txt
poses/XX.txt                 # ground truth for 00–10 (ground_truth/poses/ also works)
```

`run.dataset_path` in the configs is resolved against the repository root.
Override it with `--dataset-path`. Sequences 11–21 have no public ground truth.

## Run VO

```bash
python scripts/run_vo.py --config configs/constrained.yaml --sequence 4 --output results/constrained
python scripts/run_vo.py --config configs/base.yaml          # all sequences in the config
```

Each sequence is written to `<output>/XX/poses.txt` in KITTI format: 3×4
camera-to-world pose, row-major, `%.6e`, one line per frame. The first line is the identity.
Add `--timing` for a per-stage timing table. Add `--diagnostics` to also write
`<output>/XX/diagnostics.csv`: one row per frame with the registration outcome
(registered, or why it failed) and the match, point, and PnP inlier counts behind
it. Diagnostics do not change the poses. Outputs that resolve into
`output_base/`, `output_constrained/`, or `reference/` are refused.

## Evaluate

```bash
python scripts/evaluate.py --results-dir results/constrained --plots
```

**All error metrics are in SI units**: m, rad, m/m, rad/m. **RPE defaults to
100 m segments.** By default the script computes:

- `metric_distance`: APE, plus RPE over 100 m pairs chosen on the ground-truth path, so every estimate is scored on the same pairs.
- `kitti_segments`: KITTI devkit drift over 100–800 m segments, in m/m and rad/m.

`--profiles legacy_frames` reproduces the historical numbers, which use RPE over 100
**frames**. Results go to `XX/metrics/<profile>.{json,txt}` and, with `--plots`,
`XX/plots/`. See [docs/evaluation.md](docs/evaluation.md) for the exact definitions.

## Visualize features and matches

```bash
python scripts/visualize_features.py --config configs/constrained.yaml --sequence 0 --frame 0 \
    --output results/figures/00_000000 --compare-sift
```

This draws one frame using the config's own detector and matchers. It writes:

- `keypoints.png`: ORB keypoints before and after grid selection.
- `stereo_matches.png`: brute-force matches vs. the configured stereo matcher.
- `temporal_matches.png`: matches from frame k to frame k+1.
- `orb_vs_sift.png`: optional, with `--compare-sift`.

## Tests and lint

```bash
pytest tests -m "not regression" -q   # unit tier (what CI runs)
pytest tests -q                       # + regression tier (needs KITTI images and ground truth)
ruff check . && ruff format --check .
```

Regression tests skip, rather than pass, when their data is missing.

## Known issues

This is a research package evaluated on KITTI. The current phase fixes only defects
that affect the estimates; corrections already made are listed in
[docs/compatibility.md](docs/compatibility.md) §4b. Still open:

- The rectified stereo matcher picks the best descriptor before checking its geometry (BUG-005).
- `stereo_matcher.cross_check` is ignored for the base (brute-force) stereo
  matcher, which uses the temporal value instead (BUG-004).

Known limitations, not planned because they do not affect KITTI estimates:

- A frame without ORB descriptors (for example a blank image) would crash the base pipeline.
- There is no input preflight, and no atomic or resumable output.
- `pnp.seed` has no effect.
- `motion.max_pitch_deg` gates rotation about camera *y*. That is the heading axis, not pitch.

## Historical material

- `output_base/` and `output_constrained/` hold the original results. They are
  write-protected, and `output_base/01/poses.txt` is truncated.
- `output_features/`, `keypoints/`, and the root `*.png` files are figures from the original scripts.
- `VO_Seminar.pdf` is the original report. Its corrections, including that its
  "100 m" RPE is a 100-frame RPE, are in
  [docs/errata/vo-seminar-errata.md](docs/errata/vo-seminar-errata.md).
- The original root scripts (`VO_*.py`, `filter.py`, `utils.py`,
  `create_results.sh`, `config/`, …) were removed after the port was verified.
  They remain in Git at revision `461c8fe`.
