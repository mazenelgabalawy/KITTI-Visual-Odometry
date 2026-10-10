# Regression tier

These tests need data that is not committed. Each fixture skips with a reason
when its input is absent. A skip means "not checked". It never means passed.

| Fixture | Needs |
|---|---|
| `kitti_root` | KITTI images at `./sequences/XX/` |
| `gt_dir` | ground truth in `./poses/` or `./ground_truth/poses/` |

## `test_evaluation.py`: historical metric fixtures

The `legacy_frames` profile must reproduce every committed
`output_*/XX/{ape,rpe}_results.txt`. That is 44 files, all 7 statistics, to the 6 printed
decimals, plus the implied pair count. These fixtures require only ground truth.
The truncated `output_base/01/poses.txt` is replaced by its pre-truncation Git
blob, which produced the committed metric files.
