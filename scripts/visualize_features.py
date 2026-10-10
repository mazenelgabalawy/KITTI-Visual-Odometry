#!/usr/bin/env python
"""Draw a pipeline config's features and matches for one KITTI frame.

Replaces the legacy figure scripts (features.py, detectors.py). It uses the
same detector and matchers the pipeline builds from the config, so the images
show what VO actually sees. Images written to --output:

    keypoints.png         ORB keypoints before (top) and after (bottom) grid selection
    stereo_matches.png    left/right matches: raw cross-checked brute force (top)
                          vs. the config's stereo matcher (bottom)
    temporal_matches.png  frame k -> k+1 left-image matches from the config's temporal matcher
    orb_vs_sift.png       with --compare-sift: SIFT (top) vs. ORB (bottom) keypoints,
                          both with the config's nfeatures

Usage:
    python scripts/visualize_features.py --config configs/constrained.yaml --sequence 0 --frame 0 \\
        --output results/figures/00_000000 --compare-sift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kitti_slam.config.loader import ConfigError, load_config  # noqa: E402
from kitti_slam.datasets.kitti import KittiSequence  # noqa: E402
from kitti_slam.features.matchers import BruteForceMatcher  # noqa: E402
from kitti_slam.features.orb import OrbDetector  # noqa: E402
from kitti_slam.odometry.factory import build_pipeline  # noqa: E402
from kitti_slam.protected_paths import ProtectedPathError, check_writable  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw one frame's features and matches for a pipeline config.")
    p.add_argument("--config", required=True, help="Pipeline config, e.g. configs/constrained.yaml.")
    p.add_argument("--sequence", type=int, required=True)
    p.add_argument("--frame", type=int, default=0, help="Left/right frame k; temporal matches use k -> k+1.")
    p.add_argument("--output", required=True, help="Directory for the PNG files.")
    p.add_argument("--dataset-path", default=None, help="Override the config's run.dataset_path.")
    p.add_argument("--compare-sift", action="store_true", help="Also draw SIFT vs. ORB keypoints.")
    return p.parse_args(argv)


def _abs(path: str) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else _REPO_ROOT / p).resolve()


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    return cv2.drawKeypoints(
        image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def _draw_matches(img_a, feat_a, img_b, feat_b, matches) -> np.ndarray:
    return cv2.drawMatches(img_a, feat_a.keypoints, img_b, feat_b.keypoints, matches, None)


def _write(path: Path, image: np.ndarray) -> None:
    check_writable(path, _REPO_ROOT)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(_abs(args.config))
        out_dir = check_writable(args.output, _REPO_ROOT)
    except (ConfigError, FileNotFoundError, ProtectedPathError) as exc:
        print(exc, file=sys.stderr)
        return 1

    dataset_root = _abs(args.dataset_path or cfg.run.dataset_path)
    sequence = KittiSequence(dataset_root, args.sequence)
    if not 0 <= args.frame < len(sequence) - 1:
        print(f"--frame must be in [0, {len(sequence) - 2}] for sequence {args.sequence:02d}", file=sys.stderr)
        return 1
    frame, nxt = sequence[args.frame], sequence[args.frame + 1]
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(cfg)
    raw_detector = OrbDetector(cfg.detector)  # same ORB settings, no grid selection
    left, right, left_next = (pipeline.detector.detect(img) for img in (frame.left, frame.right, nxt.left))

    raw_left = raw_detector.detect(frame.left)
    _write(
        out_dir / "keypoints.png",
        np.vstack(
            [
                _label(_draw_keypoints(frame.left, raw_left.keypoints), f"ORB: {len(raw_left.keypoints)}"),
                _label(_draw_keypoints(frame.left, left.keypoints), f"selected: {len(left.keypoints)}"),
            ]
        ),
    )

    raw_stereo = BruteForceMatcher(cfg.temporal_matcher.norm, cross_check=True).match(left, right)
    stereo = pipeline.stereo_matcher.match(left, right)
    _write(
        out_dir / "stereo_matches.png",
        np.vstack(
            [
                _label(
                    _draw_matches(frame.left, left, frame.right, right, raw_stereo), f"brute force: {len(raw_stereo)}"
                ),
                _label(
                    _draw_matches(frame.left, left, frame.right, right, stereo),
                    f"{cfg.stereo_matcher.kind}: {len(stereo)}",
                ),
            ]
        ),
    )

    temporal = pipeline.temporal_matcher.match(left, left_next)
    _write(
        out_dir / "temporal_matches.png",
        _label(_draw_matches(frame.left, left, nxt.left, left_next, temporal), f"temporal: {len(temporal)}"),
    )

    if args.compare_sift:
        sift_kps = cv2.SIFT_create(nfeatures=cfg.detector.nfeatures).detect(frame.left, None)
        orb_kps = cv2.ORB_create(nfeatures=cfg.detector.nfeatures).detect(frame.left, None)
        _write(
            out_dir / "orb_vs_sift.png",
            np.vstack(
                [
                    _label(_draw_keypoints(frame.left, sift_kps), f"SIFT: {len(sift_kps)}"),
                    _label(_draw_keypoints(frame.left, orb_kps), f"ORB: {len(orb_kps)}"),
                ]
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
