from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDENS = Path(__file__).resolve().parent / "unit" / "data" / "goldens.npz"


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def goldens() -> dict[str, np.ndarray]:
    """Frozen package outputs (with their inputs), written by
    tests/unit/data/generate_goldens.py."""
    with np.load(_GOLDENS) as data:
        return {key: data[key] for key in data.files}
