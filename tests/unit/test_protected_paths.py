"""Output protection: writes into the historical baselines or reference/ are refused."""

from __future__ import annotations

import os

import pytest

from kitti_slam.protected_paths import ProtectedPathError, check_writable


@pytest.mark.parametrize(
    "path",
    ["output_base", "output_constrained/04", "reference", "reference/snapshot/base/04/poses.txt"],
)
def test_protected_trees_are_refused(tmp_path, path):
    with pytest.raises(ProtectedPathError):
        check_writable(path, tmp_path)


def test_prefix_sibling_is_allowed(tmp_path):
    assert check_writable("output_base_experiment/04", tmp_path) == (tmp_path / "output_base_experiment/04").resolve()
    assert check_writable("references", tmp_path) == (tmp_path / "references").resolve()


def test_nested_directory_symlink_into_reference_is_refused(tmp_path):
    (tmp_path / "reference" / "snapshot" / "base").mkdir(parents=True)
    (tmp_path / "results" / "run").mkdir(parents=True)
    os.symlink(tmp_path / "reference" / "snapshot" / "base", tmp_path / "results" / "run" / "04")
    with pytest.raises(ProtectedPathError):
        check_writable("results/run/04/poses.txt", tmp_path)


def test_final_file_symlink_into_baseline_is_refused(tmp_path):
    (tmp_path / "output_base" / "04").mkdir(parents=True)
    target = tmp_path / "output_base" / "04" / "poses.txt"
    target.write_text("frozen\n")
    out = tmp_path / "results" / "04"
    out.mkdir(parents=True)
    os.symlink(target, out / "poses.txt")
    with pytest.raises(ProtectedPathError):
        check_writable(out / "poses.txt", tmp_path)
    # A dangling link into a protected tree is refused too.
    os.symlink(tmp_path / "reference" / "new.txt", out / "dangling.txt")
    with pytest.raises(ProtectedPathError):
        check_writable(out / "dangling.txt", tmp_path)


def test_run_vo_guard_covers_reference(repo_root):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_vo_ref_guard", repo_root / "scripts" / "run_vo.py")
    run_vo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_vo)
    with pytest.raises(SystemExit):
        run_vo.guard_output_dir("reference/snapshot", repo_root)
