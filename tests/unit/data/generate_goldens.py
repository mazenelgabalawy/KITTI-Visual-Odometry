"""Regenerate tests/unit/data/goldens.npz from the package.

Each entry stores the inputs together with the package outputs, so the tests
compare against frozen values instead of recomputing them. Two entry groups
depend on ORB detection (`orb_*`, and the keypoints and descriptors of the
matcher inputs): they are tied to the pinned opencv-python-headless 4.11.0.86
build.

Regenerating changes the expectations of every test that reads this file. Do it
only together with an intended, explained behavior change, never to make a
failing test pass.

Usage (from the repository root):
    python tests/unit/data/generate_goldens.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from kitti_slam.config.schema import DetectorConfig, GridFilterConfig, StereoMatcherConfig
from kitti_slam.evaluation.trajectory_io import TrajectoryWriter
from kitti_slam.features.association import intersect_temporal_stereo
from kitti_slam.features.base import Features
from kitti_slam.features.grid_filter import GridKeypointFilter
from kitti_slam.features.matchers import BruteForceMatcher, RectifiedStereoMatcher
from kitti_slam.features.orb import OrbDetector
from kitti_slam.geometry.transforms import rotation_to_euler
from kitti_slam.odometry.motion_filters import _orthogonalize_to_planar_motion

OUTPUT = Path(__file__).with_name("goldens.npz")


def _stereo_pair(seed: int = 1, shift: int = 8):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, : 320 - shift] = left[:, shift:]
    right[:, 320 - shift :] = rng.integers(0, 255, size=(240, shift), dtype=np.uint8)
    return left, right


def _triples(matches) -> np.ndarray:
    return np.array([(m.queryIdx, m.trainIdx, m.distance) for m in matches], dtype=np.float64).reshape(-1, 3)


def _pts(kps) -> np.ndarray:
    return np.array([kp.pt for kp in kps], dtype=np.float64).reshape(-1, 2)


def build() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # Matchers: stereo features from a synthetic shifted pair.
    detector = OrbDetector(
        DetectorConfig(nfeatures=500, scale_factor=1.2, nlevels=8),
        GridKeypointFilter(GridFilterConfig(grid_size=(30, 20), max_per_grid=5)),
    )
    left, right = _stereo_pair()
    fl, fr = detector.detect(left), detector.detect(right)
    out["match_left_pts"], out["match_left_desc"] = _pts(fl.keypoints), fl.descriptors
    out["match_right_pts"], out["match_right_desc"] = _pts(fr.keypoints), fr.descriptors
    out["bf_ratio_matches"] = _triples(
        BruteForceMatcher(norm="NORM_HAMMING", cross_check=False, ratio_threshold=0.5).match(fl, fr)
    )
    out["rectified_matches"] = _triples(
        RectifiedStereoMatcher(
            StereoMatcherConfig(
                kind="rectified",
                cross_check=True,
                pixel_threshold=2.0,
                max_hamming_distance=30,
                ratio_test=0.7,
                max_disparity=64.0,
            )
        ).match(fl, fr)
    )

    # Association: 5 features, stereo matches only for left_prev 0, 1, 3.
    lp = [cv2.KeyPoint(x=float(i), y=float(i * 2), size=5) for i in range(5)]
    rp = [cv2.KeyPoint(x=float(i + 0.5), y=float(i * 2), size=5) for i in range(5)]
    lc = [cv2.KeyPoint(x=float(i + 1), y=float(i * 2 + 1), size=5) for i in range(5)]
    temporal = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(5)]
    stereo = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in (0, 1, 3)]
    corr = intersect_temporal_stereo(temporal, stereo, Features(lp, None), Features(rp, None), Features(lc, None))
    out["assoc_left_prev"], out["assoc_right_prev"], out["assoc_left_curr"] = (
        corr.left_prev,
        corr.right_prev,
        corr.left_curr,
    )

    # Grid selection of 200 synthetic keypoints.
    rng = np.random.default_rng(0)
    kps = []
    for _ in range(200):
        x, y, resp = rng.uniform(0, 320), rng.uniform(0, 240), rng.uniform(0, 1)
        kps.append(cv2.KeyPoint(x=float(x), y=float(y), size=10.0, response=float(resp)))
    descs = np.arange(200 * 32, dtype=np.uint8).reshape(200, 32)
    selected = GridKeypointFilter(GridFilterConfig(grid_size=(30, 20), max_per_grid=3))(
        Features(kps, descs), (240, 320)
    )
    out["grid_in_pts"] = _pts(kps)
    out["grid_in_responses"] = np.array([kp.response for kp in kps], dtype=np.float64)
    out["grid_out_pts"] = _pts(selected.keypoints)
    out["grid_out_responses"] = np.array([kp.response for kp in selected.keypoints], dtype=np.float64)
    out["grid_out_desc"] = selected.descriptors

    # ORB + grid on a synthetic image.
    image = np.random.default_rng(0).integers(0, 255, size=(240, 320), dtype=np.uint8)
    features = detector.detect(image)
    out["orb_image"] = image
    out["orb_pts"] = _pts(features.keypoints)
    out["orb_responses"] = np.array([kp.response for kp in features.keypoints], dtype=np.float64)
    out["orb_desc"] = features.descriptors

    # Planar orthogonalization: random + near-planar rotations x 3 alphas.
    rotations = [r.as_matrix() for r in Rotation.random(20, random_state=np.random.default_rng(0))]
    for yaw_deg in (0.0, 5.0, -12.0, 30.0):
        R_yaw = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
        rotations.append(R_yaw @ Rotation.from_euler("xz", [0.5, -0.3], degrees=True).as_matrix())
    alphas = np.array([0.8, 0.5, 1.0])
    out["ortho_in"] = np.stack(rotations)
    out["ortho_alphas"] = alphas
    out["ortho_out"] = np.stack(
        [[_orthogonalize_to_planar_motion(R, alpha=float(a)) for a in alphas] for R in rotations]
    )

    # Euler angles: random rotations and both gimbal-lock (singular) cases.
    euler_in = [r.as_matrix() for r in Rotation.random(20, random_state=np.random.default_rng(0))]
    euler_in.append(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    euler_in.append(np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    out["euler_in"] = np.stack(euler_in)
    out["euler_out"] = np.stack([rotation_to_euler(R) for R in euler_in])

    # Pose-file writer: 5 random poses.
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(5):
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        T[:3, 3] = rng.uniform(-10, 10, size=3)
        poses.append(T)
    writer = TrajectoryWriter()
    for T in poses:
        writer.append(T)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "poses.txt"
        writer.write(path)
        out["writer_text"] = np.array(path.read_text())
    out["writer_poses"] = np.stack(poses)

    return out


def main() -> None:
    out = build()
    np.savez_compressed(OUTPUT, **out)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(out)} arrays)")


if __name__ == "__main__":
    main()
