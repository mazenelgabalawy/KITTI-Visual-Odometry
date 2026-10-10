"""End-to-end checks of scripts/evaluate.py on synthetic inputs: output schema,
units, plots, and exit statuses."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from kitti_slam.evaluation.trajectory_io import TrajectoryWriter


def _load_script(repo_root, name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", repo_root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_line_trajectory(path, n, step=1.0, yaw_rate=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = TrajectoryWriter()
    for i in range(n):
        C = np.eye(4)
        a = i * yaw_rate
        C[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        C[2, 3] = i * step
        writer.append(C)
    writer.write(path)


@pytest.fixture
def evaluate_inputs(tmp_path):
    gt_dir = tmp_path / "gt"
    _write_line_trajectory(gt_dir / "04.txt", 300)
    results = tmp_path / "results"
    _write_line_trajectory(results / "04" / "poses.txt", 300, step=1.05, yaw_rate=1e-4)
    return gt_dir, results


def _check_si_record(results, profile):
    record = json.loads((results / "04" / "metrics" / f"{profile}.json").read_text())
    assert record["profile"] == profile
    assert record["units"].startswith("SI")
    assert len(record["estimate_sha256"]) == 64 and len(record["reference_sha256"]) == 64
    for name, m in record["metrics"].items():
        assert m["unit"] in {"m", "rad", "m/m", "rad/m"}, name
        assert m["alignment"] == "none"
        assert m["num_poses"] == 300
        assert m["statistics"] is not None, name
    text = (results / "04" / "metrics" / f"{profile}.txt").read_text()
    assert "units: SI" in text
    assert "deg" not in text and "%" not in text
    return record["metrics"]


def test_evaluate_defaults_to_100m_rpe(repo_root, evaluate_inputs, capsys):
    gt_dir, results = evaluate_inputs
    evaluate = _load_script(repo_root, "evaluate")
    assert evaluate.main(["--results-dir", str(results), "--gt-dir", str(gt_dir)]) == 0

    md = _check_si_record(results, "metric_distance")
    rpe = md["rpe_translation"]["selection"]
    assert (rpe["interval"], rpe["interval_unit"], rpe["pairs_from"]) == (100.0, "m", "reference")
    # ~5% of 100 m; yaw coupling and the shorter +/-10 m tail pairs spread it (exact
    # values are covered in test_metrics.py).
    assert md["rpe_translation"]["statistics"]["median"] == pytest.approx(5.0, rel=0.05)
    ks = _check_si_record(results, "kitti_segments")
    assert ks["segment_rotation_drift"]["unit"] == "rad/m"
    assert ks["segment_rotation_drift"]["statistics"]["mean"] == pytest.approx(1e-4, rel=0.02)  # 1e-4 rad per m
    assert sorted(p.name for p in (results / "04" / "metrics").glob("*.json")) == [
        "kitti_segments.json",
        "metric_distance.json",
    ]

    out = capsys.readouterr().out
    assert "metric_distance" in out
    assert "rpe_translation rmse [m]" in out and "segment_translation_drift mean [m/m]" in out


def test_evaluate_plots_are_valid_images(repo_root, evaluate_inputs):
    import matplotlib.image as mpimg

    gt_dir, results = evaluate_inputs
    evaluate = _load_script(repo_root, "evaluate")
    args = ["--results-dir", str(results), "--gt-dir", str(gt_dir), "--profiles", "metric_distance", "--plots"]
    assert evaluate.main(args) == 0
    plots = sorted((results / "04" / "plots").glob("*.png"))
    names = {p.name for p in plots}
    assert {"trajectory.png", "ape_translation_map.png", "metric_distance_rpe_rotation_raw.png"} <= names
    for p in plots:
        image = mpimg.imread(p)
        assert image.ndim == 3 and image.shape[0] > 100 and image.shape[1] > 100
        assert float(image[..., :3].std()) > 0.01, f"{p.name} looks blank"


def test_evaluate_fails_on_length_mismatch_and_reports_it(repo_root, tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    _write_line_trajectory(gt_dir / "04.txt", 300)
    results = tmp_path / "results"
    _write_line_trajectory(results / "04" / "poses.txt", 250)
    evaluate = _load_script(repo_root, "evaluate")
    assert evaluate.main(["--results-dir", str(results), "--gt-dir", str(gt_dir)]) == 1
    assert "expected 300 poses, found 250" in capsys.readouterr().err
    assert not (results / "04" / "metrics").exists()


def test_evaluate_refuses_protected_results_dir(repo_root, capsys):
    evaluate = _load_script(repo_root, "evaluate")
    assert evaluate.main(["--results-dir", "output_constrained"]) == 1
    assert "Refusing to write" in capsys.readouterr().err
