"""Fixtures for the regression tier.

Tests here need KITTI images, which are not committed (`kitti_root`). The
fixture skips with a clear reason when the data is absent, so a missing input
shows as SKIPPED -- never as a pass and never as an ERROR. Every test in this
package is marked `regression`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/regression" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.regression)


@pytest.fixture
def kitti_root() -> Path:
    root = _REPO_ROOT / "sequences"
    if not (root / "00").is_dir():
        pytest.skip("KITTI dataset not present (expected ./sequences/00) -- regression tests skipped")
    return root
