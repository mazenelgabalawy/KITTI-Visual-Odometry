"""Tests for kitti_slam.geometry.calibration.StereoCalibration.

Golden values below are the real KITTI odometry sequence-00 calib.txt
P0/P1 lines: fx=fy=718.856, cx=607.1928, cy=185.2157, with P1's x-translation
term (P1[0][3]) = -386.1448, giving baseline = 386.1448 / 718.856 ~= 0.53716 m.
"""

from pathlib import Path

import numpy as np
import pytest

from kitti_slam.geometry.calibration import StereoCalibration

P0_LINE = (
    "P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
)
P1_LINE = (
    "P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 "
    "0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 "
    "0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
)

EXPECTED_FOCAL = 718.856
EXPECTED_BASELINE = 386.1448 / 718.856  # ~= 0.53716


def _write_calib(tmp_path: Path) -> Path:
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text(P0_LINE + "\n" + P1_LINE + "\n")
    return calib_path


def test_parse_kitti_calib_file(tmp_path):
    calib_path = _write_calib(tmp_path)
    calib = StereoCalibration.from_kitti_calib_file(calib_path)

    assert calib.P0.shape == (3, 4)
    assert calib.P1.shape == (3, 4)
    assert calib.K.shape == (3, 3)

    assert calib.K[0, 0] == pytest.approx(EXPECTED_FOCAL, rel=1e-6)
    assert calib.focal_length == pytest.approx(EXPECTED_FOCAL, rel=1e-6)
    assert calib.baseline == pytest.approx(EXPECTED_BASELINE, rel=1e-3)
    assert calib.baseline == pytest.approx(0.53716, rel=1e-3)

    np.testing.assert_allclose(calib.P0[0, 0], 718.856)
    np.testing.assert_allclose(calib.P1[0, 3], -386.1448)


def test_baseline_positive(tmp_path):
    calib_path = _write_calib(tmp_path)
    calib = StereoCalibration.from_kitti_calib_file(calib_path)
    assert calib.baseline > 0
