"""Core VO must import and run its CLI wiring without the optional backends.

`evo` (the `eval` extra) and `gtsam` (the `slam` extra) are blocked with a
meta-path finder in a fresh interpreter, so an accidental top-level import of
either one from a core module fails this test even when both are installed in
the development environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

CORE_MODULES = (
    "kitti_slam",
    "kitti_slam.config.loader",
    "kitti_slam.datasets.kitti",
    "kitti_slam.datasets.prefetch",
    "kitti_slam.features.matchers",
    "kitti_slam.geometry.transforms",
    "kitti_slam.odometry.factory",
    "kitti_slam.odometry.pipeline",
    "kitti_slam.evaluation",
    "kitti_slam.evaluation.trajectory_io",
    "kitti_slam.evaluation.pose_checks",
    "kitti_slam.protected_paths",
    "kitti_slam.provenance",
)

_BLOCKER = """
import importlib, importlib.abc, sys

BLOCKED = {"evo", "gtsam"}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, _Blocker())
"""


def _run_isolated(body: str, repo_root) -> subprocess.CompletedProcess:
    code = textwrap.dedent(_BLOCKER) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_core_modules_import_without_optional_backends(repo_root):
    body = f"""
    for name in {CORE_MODULES!r}:
        importlib.import_module(name)
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("ok")
    """
    proc = _run_isolated(body, repo_root)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_run_vo_script_imports_without_optional_backends(repo_root):
    body = f"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_vo_isolated", {str(repo_root / "scripts" / "run_vo.py")!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    print("ok")
    """
    proc = _run_isolated(body, repo_root)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_blocker_actually_blocks(repo_root):
    """Guards the guard: if the blocker silently stopped working, the two tests
    above would pass vacuously."""
    proc = _run_isolated("import evo\n", repo_root)
    assert proc.returncode != 0
    assert "blocked optional dependency" in proc.stderr
