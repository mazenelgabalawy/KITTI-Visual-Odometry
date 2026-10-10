# Trajectory evaluation

`scripts/evaluate.py` and `kitti_slam.evaluation.metrics` score estimated KITTI
trajectories under named **profiles**. Every result states its unit, pose
relation, alignment, pair selection, pose and pair counts, and frame coverage.

## Units

All error metrics are reported in SI units. Nothing is converted to degrees or
percent.

| Quantity | Unit |
|---|---|
| Translation error (APE, RPE) | m |
| Rotation error (APE, RPE) | rad, the geodesic angle of the rotation error in [0, π] |
| KITTI segment translational drift | m/m, the error length per traveled length |
| KITTI segment rotational drift | rad/m |

The RPE *interval* is a selection setting, not an error unit. It is given in
`frames` or `m`. In both cases the translation error is in meters.

## Profiles

| Profile | Use | APE | RPE pairs | Segments |
|---|---|---|---|---|
| `metric_distance` (**default**) | Compare experiments; the default RPE | translation (m), rotation (rad); no alignment | for **every** start frame *i*, the end frame *j* whose **reference** path length from *i* is closest to 100 m; kept within ±10 m (relative tolerance 0.1) | — |
| `kitti_segments` (**default**) | KITTI devkit-style drift | — | — | lengths 100, 200, …, 800 m on the reference path; a segment starts every 10 frames |
| `legacy_frames` (opt-in) | Reproduce historical `create_results.sh` numbers | translation (m), rotation (rad); no alignment | consecutive 100-**frame** pairs `(0,100), (100,200), …`; selected by index | — |

**RPE defaults to 100 m segments.** `scripts/evaluate.py` computes
`metric_distance` and `kitti_segments` unless `--profiles` says otherwise, and in
the Python API `evaluate()` defaults to `metric_distance` and `compute_rpe()` to
its 100 m reference-selected pairs. The 100-frame interval is used only when
`legacy_frames` is requested explicitly.

Notes:

- **Historical interval.** The removed `create_results.sh` called `evo_rpe kitti gt est --delta 100`
  without `--delta_unit`, so evo used its default unit, frames. The committed
  `rpe_results.txt` files print the literal `{self.delta}` placeholder (an evo
  formatting bug), and the seminar report mislabels the interval as 100 m.
  `legacy_frames` reproduces all 44 committed historical metric files
  (`tests/regression/test_evaluation.py`).
- **Reference-selected pairs.** The previous modern evaluator chose 100 m pairs
  on the *estimate's* path (evo's `pairs_from_reference=False`). Two estimates
  were then scored on different pairs, and an estimate with a wrong scale was
  scored on a different stretch of road. `metric_distance` chooses pairs on the
  ground truth only, so every estimate of a sequence uses identical pairs. Near the end of a
  trajectory, start frames whose remaining reference path is 90–100 m pair
  with the last frame. This is a consequence of the ±10 m tolerance. The
  equivalent evo CLI call is
  `evo_rpe kitti gt est --delta 100 --delta_unit m --all_pairs --pairs_from_reference`.
- **KITTI segments** follow the KITTI odometry devkit (`evaluate_odometry.cpp`).
  For each start frame *i* (step 10) and length *L*, the end *j* is the first frame whose
  reference path length from *i* exceeds *L*. The segment error is
  `E = inv(inv(P_i) P_j) · inv(Q_i) Q_j`, where *P* is the estimate and *Q* the reference.
  The translational drift is `‖t(E)‖ / L` in m/m. The rotational drift is
  `acos((tr R(E) − 1)/2) / L` in rad/m. The KITTI score is the **mean** over all segments.
  The per-length breakdown is in the JSON output. The implementation is
  independent of evo, runs in float64, and is checked against closed-form synthetic
  trajectories (`tests/unit/test_metrics.py`). The official devkit prints `t_err × 100` (%)
  and `r_err` in deg/m. Here the raw SI values are reported instead.
- **No alignment** is applied in any profile. The VO trajectories start at the
  identity, which is the ground-truth frame-0 pose.
- **Frame coverage** is the fraction of the `N − 1` frame intervals that are
  spanned by at least one evaluated pair or segment. It is 1.0 for APE.
- **Unavailable metrics.** If no pair or segment qualifies, for example on a
  trajectory shorter than the interval, the metric is reported as unavailable
  with a reason. It is never reported as zero.
- **Input contract.** The ground truth and the estimate must have the same number of poses,
  finite values, and valid rotations. A mismatch is an error. Trajectories are
  never truncated to the shorter length.

## Running

```bash
python scripts/evaluate.py --results-dir results/constrained                 # metric_distance + kitti_segments
python scripts/evaluate.py --results-dir results/constrained --sequences 00 05 --plots
python scripts/evaluate.py --results-dir results/constrained --profiles legacy_frames   # historical 100-frame RPE
```

The ground-truth directory defaults to the first existing one of `./poses` and
`./ground_truth/poses`. Use `--gt-dir` to choose another. For each sequence the script writes
`XX/metrics/<profile>.json`, which holds the complete settings, counts, coverage, statistics,
and input sha256 values, and `XX/metrics/<profile>.txt`. With `--plots` it also writes
`XX/plots/`. The exit status is non-zero if any requested sequence cannot be evaluated.
Writing into `output_base/`, `output_constrained/`, or `reference/` is refused.

## Seminar Table 3 under each definition

These are the committed `output_constrained` trajectories. All values are in m.

| Seq | APE RMSE | RPE RMSE, 100 frames (`legacy_frames`; the seminar's value) | RPE RMSE, 100 m on reference pairs (`metric_distance`) | RPE RMSE, 100 m on estimate pairs (previous modern evaluator) |
|---|---:|---:|---:|---:|
| 00 | 39.015891 | 1.716797 (45 pairs) | 2.130465 (4458 pairs) | 2.181183 (35 pairs) |
| 01 | 346.558673 | 61.033702 (11 pairs) | 26.406932 (1041 pairs) | 41.378789 (19 pairs) |
| 05 | 11.807016 | 1.746188 (27 pairs) | 1.989421 (2677 pairs) | 1.926522 (21 pairs) |

KITTI segment drift for the same trajectories (`kitti_segments`, mean):

| Seq | Translational drift (m/m) | Rotational drift (rad/m) |
|---|---:|---:|
| 00 | 0.023135 | 1.842e-04 |
| 01 | 0.213890 | 2.441e-04 |
| 05 | 0.019072 | 1.563e-04 |
