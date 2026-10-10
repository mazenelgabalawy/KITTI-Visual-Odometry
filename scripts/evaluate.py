#!/usr/bin/env python
"""Evaluate KITTI trajectories under explicit metric profiles (SI units).

For every sequence and profile, writes `<results-dir>/<XX>/metrics/<profile>.json`
(complete settings, counts, coverage, statistics) and a readable `.txt`
alongside, then prints a summary table. Units are SI throughout: m, rad, m/m,
rad/m. Profiles (see kitti_slam.evaluation.metrics):

    metric_distance  APE + RPE over 100 m pairs selected on the reference path   [default]
    kitti_segments   KITTI devkit 100-800 m segment drift (m/m, rad/m)            [default]
    legacy_frames    APE + RPE over 100 frames: reproduces historical numbers     [opt-in]

Usage:
    python scripts/evaluate.py --results-dir results/base
    python scripts/evaluate.py --results-dir results/base --sequences 00 04 --plots
    python scripts/evaluate.py --results-dir results/base --profiles legacy_frames

Exits non-zero if any requested sequence cannot be evaluated (missing or
invalid files, length mismatch). A metric without eligible pairs is reported
as unavailable, not as a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kitti_slam.evaluation.metrics import (  # noqa: E402
    PROFILES,
    MetricResult,
    ProfileEvaluation,
    TrajectoryInputError,
    evaluate,
)
from kitti_slam.evaluation.pose_checks import read_pose_file  # noqa: E402
from kitti_slam.protected_paths import ProtectedPathError, check_writable  # noqa: E402
from kitti_slam.provenance import sha256_file  # noqa: E402

DEFAULT_GT_DIRS = ("poses", "ground_truth/poses")
# RPE defaults to 100 m segments (metric_distance); the 100-frame legacy_frames
# profile is opt-in, for reproducing historical numbers only.
DEFAULT_PROFILES = ("metric_distance", "kitti_segments")
# Statistic shown per metric in the summary table: the KITTI score is the mean
# segment drift; everything else reports RMSE.
HEADLINE = {"segment_translation_drift": "mean", "segment_rotation_drift": "mean"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI trajectories under explicit metric profiles.")
    parser.add_argument("--results-dir", required=True, help="Directory with per-sequence XX/poses.txt.")
    parser.add_argument(
        "--gt-dir",
        default=None,
        help=f"Ground-truth pose directory (XX.txt). Default: first existing of {', '.join(DEFAULT_GT_DIRS)}.",
    )
    parser.add_argument("--sequences", nargs="+", default=None, help="Sequence ids (e.g. 00 04). Default: all found.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=list(DEFAULT_PROFILES),
        help=f"Metric profiles to compute (default: {' '.join(DEFAULT_PROFILES)}).",
    )
    parser.add_argument("--plots", action="store_true", help="Also save trajectory and error plots per sequence.")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else _REPO_ROOT / p).resolve()


def _resolve_gt_dir(arg: str | None) -> Path | None:
    if arg is not None:
        return _resolve(arg)
    for candidate in DEFAULT_GT_DIRS:
        if _resolve(candidate).is_dir():
            return _resolve(candidate)
    return None


def _discover_sequences(results_dir: Path) -> list[str]:
    return sorted(p.name for p in results_dir.iterdir() if p.is_dir() and len(p.name) == 2 and p.name.isdigit())


def _selection_text(m: MetricResult) -> str:
    sel = m.selection
    if sel is None:
        return ""
    if type(sel).__name__ == "SegmentSelection":
        lengths = ", ".join(f"{L:g}" for L in sel.lengths_m)
        return f"  segments: lengths {lengths} m on the reference path, a start every {sel.step_frames} frames\n"
    mode = "all pairs" if sel.all_pairs else "consecutive pairs"
    tol = f", tolerance +/-{sel.interval * sel.rel_tolerance:g} m" if sel.rel_tolerance and sel.all_pairs else ""
    return f"  pairs: {sel.interval:g} {sel.interval_unit}, {mode}, selected by {sel.pairs_from}{tol}\n"


def format_text(evaluation: ProfileEvaluation, gt_file: Path, est_file: Path) -> str:
    lines = [
        f"profile: {evaluation.profile.name}",
        f"description: {evaluation.profile.description}",
        f"reference: {gt_file}",
        f"estimate: {est_file}",
        "units: SI (m, rad, m/m, rad/m)",
        "",
    ]
    text = "\n".join(lines) + "\n"
    for m in evaluation.metrics:
        pairs = "" if m.num_pairs is None else f" pairs={m.num_pairs}"
        cov = "n/a" if m.frame_coverage is None else f"{m.frame_coverage:.4f}"
        text += (
            f"{m.name} [{m.unit}]  relation={m.pose_relation} alignment={m.alignment} "
            f"poses={m.num_poses}{pairs} frame_coverage={cov}\n"
        )
        text += _selection_text(m)
        if m.statistics is None:
            text += f"  unavailable: {m.unavailable_reason}\n\n"
            continue
        for key in ("max", "mean", "median", "min", "rmse", "sse", "std"):
            unit = f"{m.unit}^2" if key == "sse" else m.unit
            text += f"  {key:>7}  {getattr(m.statistics, key):.9g} {unit}\n"
        text += "\n"
    return text


def _write_outputs(seq_dir: Path, evaluation: ProfileEvaluation, gt_file: Path, est_file: Path) -> None:
    out_dir = seq_dir / "metrics"
    out_dir.mkdir(exist_ok=True)
    record = {
        **evaluation.to_dict(),
        "reference_file": str(gt_file),
        "reference_sha256": sha256_file(gt_file),
        "estimate_file": str(est_file),
        "estimate_sha256": sha256_file(est_file),
    }
    for name, content in (
        (f"{evaluation.profile.name}.json", json.dumps(record, indent=2) + "\n"),
        (f"{evaluation.profile.name}.txt", format_text(evaluation, gt_file, est_file)),
    ):
        check_writable(out_dir / name, _REPO_ROOT)
        (out_dir / name).write_text(content)


def _save_plots(seq_dir: Path, gt, est, evaluations: list[ProfileEvaluation]) -> None:
    from kitti_slam.evaluation.plots import save_error_plots, save_trajectory_plot

    plot_dir = seq_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_trajectory_plot(gt, est, plot_dir / "trajectory.png")
    done: set[str] = set()
    for ev in evaluations:
        for m in ev.metrics:
            # APE is profile-independent; plot it once.
            key = m.name if m.name.startswith("ape_") else f"{ev.profile.name}_{m.name}"
            if key not in done:
                save_error_plots(gt, est, m, plot_dir / key)
                done.add(key)


def _print_summary(rows: dict[str, list[tuple[str, ProfileEvaluation]]]) -> None:
    for profile, entries in rows.items():
        if not entries:
            continue
        metrics = entries[0][1].metrics
        headers = [f"{m.name} {HEADLINE.get(m.name, 'rmse')} [{m.unit}]" for m in metrics]
        widths = [max(len(h), 12) for h in headers]
        print(f"\n{profile}")
        print(" seq | " + " | ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True)))
        print("-" * (7 + sum(w + 3 for w in widths)))
        for seq, ev in entries:
            cells = []
            for m, w in zip(ev.metrics, widths, strict=True):
                value = "n/a" if m.statistics is None else f"{getattr(m.statistics, HEADLINE.get(m.name, 'rmse')):.6g}"
                cells.append(value.rjust(w))
            print(f"  {seq} | " + " | ".join(cells))


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = _resolve(args.results_dir)
    try:
        check_writable(results_dir, _REPO_ROOT)
    except ProtectedPathError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    gt_dir = _resolve_gt_dir(args.gt_dir)
    if gt_dir is None or not gt_dir.is_dir():
        print(f"Ground-truth directory not found (tried {args.gt_dir or DEFAULT_GT_DIRS}).", file=sys.stderr)
        return 1
    print(f"ground truth: {gt_dir}")

    sequences = args.sequences or _discover_sequences(results_dir)
    if not sequences:
        print(f"No XX/ sequence directories under {results_dir}.", file=sys.stderr)
        return 1

    failures: list[str] = []
    rows: dict[str, list[tuple[str, ProfileEvaluation]]] = {p: [] for p in args.profiles}
    for seq in sequences:
        gt_file = gt_dir / f"{seq}.txt"
        est_file = results_dir / seq / "poses.txt"
        gt_parsed = read_pose_file(gt_file)
        est_parsed = read_pose_file(est_file, expected_poses=gt_parsed.report.num_poses or None)
        problems = [f"ground truth {i}" for i in gt_parsed.report.issues]
        problems += [f"estimate {i}" for i in est_parsed.report.issues]
        if problems:
            failures.append(f"{seq}: " + "; ".join(problems))
            continue
        try:
            evaluations = [evaluate(gt_parsed.poses, est_parsed.poses, p) for p in args.profiles]
        except TrajectoryInputError as exc:
            failures.append(f"{seq}: {exc}")
            continue
        for ev in evaluations:
            _write_outputs(results_dir / seq, ev, gt_file, est_file)
            rows[ev.profile.name].append((seq, ev))
        if args.plots:
            _save_plots(results_dir / seq, gt_parsed.poses, est_parsed.poses, evaluations)

    _print_summary(rows)
    if failures:
        print("\nFailed sequences:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except ProtectedPathError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
