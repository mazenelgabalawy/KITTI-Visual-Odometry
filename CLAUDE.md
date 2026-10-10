# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Stereo visual odometry on the KITTI odometry benchmark. All code lives in the
`kitti_slam` package (`src/kitti_slam/`), command-line tools in `scripts/`, and
configs in `configs/`. There is no root-level code. `base` and `constrained` are
configurations of one shared pipeline. Only stereo VO, evaluation, and
feature visualization work today. Bundle adjustment, mapping, loop closure,
GTSAM pose graphs, monocular VO, and reconstruction are planned, not
implemented. Keep new code modular so later components reuse the existing
features/geometry/evaluation modules.

## Public repository

The repository is public. Commit only code that serves the pipeline: package
code, its scripts, configs, tests, CI, and user-facing docs. Until Phase C,
commits contain pipeline and structure changes only. Reference snapshots,
reference-comparison tooling, plans, reports, and audit records stay local and
are never committed; their paths are listed in `.git/info/exclude`. When
`CLAUDE.local.md` exists, it describes the local roadmap and tooling.

## Scope: research package, KITTI only

This is a research package, currently evaluated only on KITTI (good images,
calibration, and ground truth). Fix defects that change, or can change, trajectory
or map estimates. Do **not** add production hardening unless asked or unless a
result depends on it: input preflight, atomic/checkpointed outputs, solver seeding,
config domain validation, resource-lifecycle handling, or guards for failures that
cannot occur on KITTI.

## Environment

- Python **3.12** venv at `.visual_odometry/` (gitignored):
  `.visual_odometry/bin/python -m pip install -e ".[dev,eval]"`.
- Extras: `eval` = evo (metrics and their tests), `dev` = pytest/ruff/pre-commit,
  `slam` = `gtsam==4.2.1` (the only release with Python 3.12 wheels). Core VO must
  import without `evo`/`gtsam` (`tests/unit/test_optional_dependencies.py`).
- OpenCV is `opencv-python-headless==4.11.0.86`. Never install `opencv-python`
  next to it (both provide `cv2`). Keep `requirements.txt` and `pyproject.toml`
  pins in agreement.

## Dataset

Present in this workspace: `sequences -> dataset/sequences` (00–21) and ground
truth 00–10 in both `poses/` and `ground_truth/poses/` (identical). Configs use
`run.dataset_path: ./sequences`, resolved against the repo root. Override it with
`--dataset-path`. `evaluate.py` looks for GT in `./poses`, then `./ground_truth/poses`.
Never commit dataset files, archives, or large binaries.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/kitti_slam/{config,datasets,features,geometry,odometry}` | VO pipeline: frozen-dataclass configs, KITTI loading/prefetch, ORB/grid/matching/association, calibration/triangulation/transforms, pipeline/factory/PnP/motion filters/timing |
| `src/kitti_slam/evaluation/` | trajectory I/O, `pose_checks` (strict file validation and comparison), `metrics` (profiles, SI units), `plots` |
| `src/kitti_slam/{protected_paths,provenance}.py` | output write-protection, file hashing and provenance |
| `scripts/run_vo.py`, `scripts/evaluate.py`, `scripts/visualize_features.py` | run VO; evaluate trajectories; draw one frame's keypoints and matches |
| `configs/*.yaml` | the only configs (all tunable parameters) |
| `tests/unit/` (CI), `tests/regression/` (needs data) | `tests/unit/data/goldens.npz` freezes package outputs (`generate_goldens.py`) |
| `output_base/`, `output_constrained/` | historical results. **Read-only.** `output_base/01/poses.txt` is truncated (see errata) |
| `output_features/`, `keypoints/`, root `*.png`, `*.pdf` | historical figures and reports, not code |
| `docs/` | `capabilities.md` (capabilities, configuration reference, corrections), `evaluation.md`, `errata/` |

The original seminar scripts are in Git history at revision `461c8fe`.

## Running & Evaluating

```bash
.visual_odometry/bin/python scripts/run_vo.py --config configs/constrained.yaml --sequence 4 --output results/smoke --no-progress
.visual_odometry/bin/python scripts/evaluate.py --results-dir results/smoke --sequences 04
```

- During iteration use sequence 04 (short) and 01 (hard highway case). Run the
  full 11×2 sweep only at milestone boundaries.
- `run_vo.py --diagnostics` writes a per-frame `diagnostics.csv` (registration
  outcome and match/point/inlier counts) without changing the poses. Use it to
  explain every trajectory change.
- `evaluate.py` computes the `metric_distance` (100 m RPE) and `kitti_segments`
  profiles.

## Conventions & Behaviors

- **Units:** every error metric is reported in SI units: m, rad, m/m, rad/m.
  Never use degrees, percent, or deg/m in stored or printed metrics. Carry the unit
  with the value.
- **Evaluation:** RPE defaults to 100 m segments: pairs are selected on the
  ground-truth path, and every start frame is paired with the frame nearest
  100 m ahead, within ±10 m. State the profile with every result
  (`docs/evaluation.md`). Never truncate unequal GT/estimate lengths.
- **Configuration:** never hardcode parameters. Add them to the config schema
  (`src/kitti_slam/config/schema.py`) and `configs/*.yaml`, and document them.
- **Poses/transforms:** 4x4 homogeneous camera-to-world poses internally.
  PnP gives the previous-to-current camera transform, which is inverted before
  composition. Camera axes: x right, y down, z forward. Serialize the top 3×4
  block row-major with `%.6e` via `kitti_slam.evaluation.trajectory_io`. Put
  shared geometry in `src/kitti_slam/geometry/`.
- **Parity and goldens:** controlled numerical parity means byte-identical
  complete pose files in the same recorded environment. Translation-only or
  tolerance checks are diagnostics only. Never refresh a golden or historical
  output to make a test pass; regenerate `goldens.npz` only together with an
  intended, explained behavior change.
- **Outputs:** each experiment gets its own `results/...` directory, which is gitignored.
  Writes into `output_*` or `reference/` are refused by `kitti_slam.protected_paths`.
- **Style:** plain procedural Python with NumPy/OpenCV, formatted and linted with
  Ruff. New code goes in the package or in `scripts/`, never at the repo root.
  Scripts use `if __name__ == "__main__":` and argparse.
- **Git:** do not commit datasets, venvs, caches, `results/`, or anything listed
  in `.git/info/exclude`.

## Development

- Tests: `.visual_odometry/bin/python -m pytest tests -m "not regression" -q`
  (the unit tier, as in CI). The regression tier needs KITTI images/GT and skips
  when they are absent. A skip is never a pass. Report the actual
  pass/fail/skip counts.
- Lint/format: `ruff check .` and `ruff format --check .` (config in
  `pyproject.toml`). All committed code is in scope. Only the
  result/reference/historical trees, `docs/`, and the venv are excluded.
- Pre-commit: `pre-commit install`. It runs Ruff and hygiene hooks and excludes the
  historical and reference trees.
- CI (`.github/workflows/ci.yml`): installs `.[dev,eval]`, runs `pip check`, Ruff,
  and the unit tier. A separate job builds the wheel and imports the core modules
  from a clean venv outside the checkout.

## Future-Work Notes

- BA starts from scratch in Phase E. The old skeleton was non-functional and was
  removed. It is in Git at `461c8fe` if needed for reference.
- GTSAM idioms: `NonlinearFactorGraph`, `Values`, `noiseModel`,
  `GenericProjectionFactorCal3_S2` for BA, `BetweenFactorPose3` for pose graphs,
  and `ISAM2` for incremental SLAM. Keep `gtsam` imports lazy and out of core VO.
