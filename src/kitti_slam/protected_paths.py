"""Write protection for frozen reference snapshots.

`reference/` holds local reference snapshots. Runners and evaluators must
never write into it.

Checks operate on fully resolved paths, so a symlink anywhere along the output
path -- including a final `poses.txt` that is itself a link -- cannot redirect
a write into a protected tree.
"""

from __future__ import annotations

from pathlib import Path

PROTECTED_DIRS: tuple[str, ...] = ("reference",)


class ProtectedPathError(ValueError):
    """Raised when an output path resolves into a protected tree."""


def _resolve(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    # resolve() follows every existing symlink component, including a dangling
    # final link, and keeps not-yet-existing tail components verbatim.
    return candidate.resolve()


def protected_roots(repo_root: str | Path) -> tuple[Path, ...]:
    root = Path(repo_root).resolve()
    return tuple((root / name).resolve() for name in PROTECTED_DIRS)


def check_writable(path: str | Path, repo_root: str | Path) -> Path:
    """Returns the resolved `path`, or raises ProtectedPathError if it is, or is
    nested inside, a protected tree. Relative paths resolve against repo_root."""
    root = Path(repo_root).resolve()
    resolved = _resolve(path, root)
    for protected in protected_roots(root):
        if resolved == protected or protected in resolved.parents:
            raise ProtectedPathError(
                f"Refusing to write to {resolved} (matches or is inside protected directory {protected}). "
                "reference/ holds frozen reference snapshots; choose a different output location."
            )
    return resolved
