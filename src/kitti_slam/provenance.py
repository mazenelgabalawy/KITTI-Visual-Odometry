"""Provenance records for reference snapshots and comparison runs.

A Git SHA alone does not identify the code that produced a result when the
working tree is dirty, and a trajectory is only reproducible together with its
inputs and numerical environment. These helpers capture:

- source identity: commit, dirty flag, and a hash of the tracked diff, plus
  per-file hashes of any explicitly listed source files;
- environment: interpreter, platform/CPU, installed distributions, OpenCV build
  information, and thread-related settings;
- input identity: per-sequence calibration hash, left/right image counts, and
  an image-content digest (see `sequence_identity` for the exact granularity).

Everything returned is plain JSON-serializable data.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

_CHUNK = 1 << 20

# Environment variables that can change thread counts (and therefore, for some
# libraries, floating-point reduction order) in OpenCV / BLAS / OpenMP.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=False)


def git_state(repo_root: str | Path) -> dict:
    """Commit, branch, dirty flag, and a hash of `git diff HEAD` (tracked changes).

    Untracked files are listed by path only; hash any that matter to a result
    through `file_hashes`.
    """
    root = Path(repo_root)
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"available": False, "reason": head.stderr.decode(errors="replace").strip()}
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
    diff = _git(root, "diff", "HEAD", "--binary").stdout
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.decode()
    lines = [line for line in status.splitlines() if line]
    return {
        "available": True,
        "commit": head.stdout.decode().strip(),
        "branch": branch,
        "dirty": bool(lines),
        "tracked_diff_sha256": sha256_bytes(diff),
        "tracked_diff_bytes": len(diff),
        "modified": sorted(line[3:] for line in lines if not line.startswith("??")),
        "untracked": sorted(line[3:] for line in lines if line.startswith("??")),
    }


def file_hashes(paths: Iterable[str | Path], base: str | Path) -> dict[str, str]:
    """{path relative to `base` (POSIX): sha256}. Paths may be absolute (inside
    `base`) or relative to `base`."""
    base = Path(base)
    out: dict[str, str] = {}
    for p in map(Path, paths):
        full = p if p.is_absolute() else base / p
        out[full.relative_to(base).as_posix()] = sha256_file(full)
    return dict(sorted(out.items()))


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _distributions() -> dict[str, str]:
    dists = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            dists[name.lower()] = dist.version
    return dict(sorted(dists.items()))


def environment_info() -> dict:
    """Interpreter, platform, CPU, packages, OpenCV build, and thread settings."""
    import cv2
    import numpy as np

    build_info = cv2.getBuildInformation()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_build_information_sha256": sha256_bytes(build_info.encode()),
        "opencv_num_threads": cv2.getNumThreads(),
        "opencv_num_cpus": cv2.getNumberOfCPUs(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "distributions": _distributions(),
    }


def opencv_build_information() -> str:
    import cv2

    return cv2.getBuildInformation()


def sequence_identity(sequence_dir: str | Path, workers: int = 8) -> dict:
    """Identity of one KITTI sequence directory.

    Granularity: `calib.txt` is hashed directly; images are hashed per file and
    folded into one digest per sequence, `sha256` over the sorted lines
    ``"<camera>/<file name> <file sha256>\\n"`` for every PNG in `image_0/` and
    `image_1/`. Any changed, added, removed, or renamed image changes the
    digest; recomputing per-file hashes localizes the change.
    """
    seq = Path(sequence_dir)
    files = sorted(
        (camera, p) for camera in ("image_0", "image_1") for p in (seq / camera).glob("*.png") if p.is_file()
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(lambda item: sha256_file(item[1]), files))
    lines = "".join(f"{camera}/{p.name} {h}\n" for (camera, p), h in zip(files, hashes, strict=True))
    calib = seq / "calib.txt"
    return {
        "path": str(seq.resolve()),
        "calib_sha256": sha256_file(calib) if calib.is_file() else None,
        "image_0_count": sum(1 for camera, _ in files if camera == "image_0"),
        "image_1_count": sum(1 for camera, _ in files if camera == "image_1"),
        "images_digest_sha256": sha256_bytes(lines.encode()),
    }
