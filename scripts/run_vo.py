#!/usr/bin/env python
"""Run the restructured stereo VO pipeline over one or more KITTI sequences.

Usage:
    python scripts/run_vo.py --config configs/base.yaml
    python scripts/run_vo.py --config configs/constrained.yaml --sequence 4 --output /tmp/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kitti_slam.config.loader import ConfigError, load_config  # noqa: E402
from kitti_slam.datasets.kitti import KittiSequence  # noqa: E402
from kitti_slam.datasets.prefetch import PrefetchingSequence  # noqa: E402
from kitti_slam.evaluation.trajectory_io import TrajectoryWriter  # noqa: E402
from kitti_slam.odometry.diagnostics import FrameDiagnostics  # noqa: E402
from kitti_slam.odometry.factory import build_pipeline  # noqa: E402
from kitti_slam.odometry.timing import StageTimer  # noqa: E402
from kitti_slam.protected_paths import ProtectedPathError, check_writable  # noqa: E402


def guard_output_dir(output_dir: str | Path, repo_root: str | Path | None = None) -> Path:
    """Refuses (via SystemExit) if output_dir resolves to, or inside, a protected
    tree (frozen reference/ snapshots). Returns the resolved path otherwise. The
    same check applies to individual output files, so symlinks cannot bypass it."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    try:
        return check_writable(output_dir, root)
    except ProtectedPathError as exc:
        sys.exit(f"{exc} Choose a different --output / run.output_dir.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stereo VO pipeline over KITTI sequences.")
    parser.add_argument("--config", required=True, help="Path to a pipeline config YAML (e.g. configs/base.yaml).")
    parser.add_argument(
        "--sequence",
        type=int,
        default=None,
        help="Run only this sequence number (overrides config's sequence list).",
    )
    parser.add_argument("--output", default=None, help="Output directory (overrides config's run.output_dir).")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="KITTI sequences root containing XX/ directories (overrides config's run.dataset_path).",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the tqdm progress bar (default on).",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        default=False,
        help="Enable per-stage wall-time instrumentation and print a summary table per sequence (default off).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help="Write a per-frame diagnostics.csv next to poses.txt (default off; poses are unchanged).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        sys.exit(f"Invalid config {args.config!r}: {exc}")
    except FileNotFoundError as exc:
        sys.exit(f"Config file not found: {exc}")

    output_dir = args.output if args.output is not None else cfg.run.output_dir
    resolved_output = guard_output_dir(output_dir)

    sequences = [args.sequence] if args.sequence is not None else list(cfg.run.sequences)

    pipeline = build_pipeline(cfg)

    dataset_root = Path(args.dataset_path if args.dataset_path is not None else cfg.run.dataset_path)
    if not dataset_root.is_absolute():
        dataset_root = (_REPO_ROOT / dataset_root).resolve()

    for seq in sequences:
        print(f"Running pipeline on sequence {seq:02d}")
        try:
            sequence = PrefetchingSequence(KittiSequence(dataset_root, seq))
        except FileNotFoundError as exc:
            sys.exit(
                f"KITTI dataset not found for sequence {seq:02d}: {exc}\n"
                f"Expected a directory at {dataset_root / f'{seq:02d}'} containing "
                "image_0/, image_1/, and calib.txt. Download the KITTI odometry "
                "dataset and point run.dataset_path (or --dataset-path) at it."
            )

        try:
            timer = StageTimer() if args.timing else None
            diagnostics = FrameDiagnostics() if args.diagnostics else None
            poses = pipeline.run(sequence, progress=args.progress, timer=timer, diagnostics=diagnostics)

            seq_output_dir = guard_output_dir(resolved_output / f"{seq:02d}")
            seq_output_dir.mkdir(parents=True, exist_ok=True)
            # Re-check the final file: the sequence dir or poses.txt itself may
            # be a symlink into a protected tree.
            poses_path = guard_output_dir(seq_output_dir / "poses.txt")

            writer = TrajectoryWriter()
            for pose in poses:
                writer.append(pose)
            writer.write(poses_path)
            print(f"Wrote {len(poses)} poses to {poses_path}")

            if diagnostics is not None:
                diagnostics_path = guard_output_dir(seq_output_dir / "diagnostics.csv")
                diagnostics.write_csv(diagnostics_path)
                print(f"Wrote {len(diagnostics.rows)} diagnostics rows to {diagnostics_path}")

            if timer is not None:
                timer.print_summary(label=f"sequence {seq:02d}")
                if timer.loop_total > 0:
                    print(f"wall time: {timer.loop_total:.3f}s, {timer.frame_count / timer.loop_total:.2f} it/s")
        finally:
            sequence.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
