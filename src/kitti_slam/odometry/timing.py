"""Opt-in per-stage wall-time instrumentation for StereoVOPipeline.

StageTimer accumulates elapsed wall time per named stage across frames, plus
the total wall time of each frame's loop body, so a summary can report an
implicit "other" bucket (loop_total - sum(named stages)) without needing to
know every stage name in advance. Stage names are created on first use --
nothing is hardcoded -- so future stages (e.g. "ba", "graph_opt") work with
no changes here.

Zero-overhead-when-disabled: NullStageTimer is a no-op with the same
interface (context-manager `stage()`, `frame()`, `mark_frame()`). Its
`stage()`/`frame()` return a single shared module-level no-op context
manager instance (no per-call allocation, no perf_counter calls), so callers
can unconditionally write `with timer.stage("detect"): ...` in the hot path
and pay nothing when timing is off.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager


class _NullContext(AbstractContextManager):
    """Shared no-op context manager singleton: __enter__/__exit__ do nothing,
    no perf_counter calls, no allocation per use."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL_CONTEXT = _NullContext()


class StageTimer:
    """Accumulates per-stage wall time and per-frame loop-total wall time.

    Usage:
        timer = StageTimer()
        for frame in frames:
            with timer.frame():
                with timer.stage("detect"):
                    ...
                with timer.stage("pnp"):
                    ...
        timer.print_summary(label="sequence 04")
    """

    def __init__(self) -> None:
        self._stage_totals: dict[str, float] = defaultdict(float)
        self._loop_total: float = 0.0
        self._frame_count: int = 0

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stage_totals[name] += time.perf_counter() - start

    @contextmanager
    def frame(self):
        """Wraps one full frame iteration; also counts the frame. Elapsed
        time here becomes `loop_total`, used to derive the "other" bucket."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._loop_total += time.perf_counter() - start
            self._frame_count += 1

    def mark_frame(self) -> None:
        """Alternative to `frame()` for callers that don't want to wrap the
        whole iteration: just increments the frame count."""
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def loop_total(self) -> float:
        return self._loop_total

    def stage_total(self, name: str) -> float:
        return self._stage_totals.get(name, 0.0)

    def summary_rows(self) -> list[tuple[str, float, float, float]]:
        """Returns (stage, total_s, ms_per_frame, pct_of_loop) rows, sorted
        by total_s descending, plus a final "other" row (loop_total minus
        the sum of named stages) when loop_total was measured via frame()."""
        frames = max(self._frame_count, 1)
        named_total = sum(self._stage_totals.values())

        rows = [
            (name, total, (total / frames) * 1000.0, (total / self._loop_total * 100.0) if self._loop_total else 0.0)
            for name, total in self._stage_totals.items()
        ]
        rows.sort(key=lambda row: row[1], reverse=True)

        if self._loop_total > 0.0:
            other_total = max(self._loop_total - named_total, 0.0)
            other_pct = (other_total / self._loop_total * 100.0) if self._loop_total else 0.0
            rows.append(("other", other_total, (other_total / frames) * 1000.0, other_pct))

        return rows

    def format_summary(self, label: str | None = None) -> str:
        header = f"Stage timing summary ({label})" if label else "Stage timing summary"
        rows = self.summary_rows()

        lines = [header, f"{'stage':<16}{'total_s':>12}{'ms/frame':>12}{'% of loop':>12}"]
        for name, total_s, ms_per_frame, pct in rows:
            lines.append(f"{name:<16}{total_s:>12.4f}{ms_per_frame:>12.3f}{pct:>12.1f}")

        if self._loop_total > 0.0:
            lines.append(f"{'TOTAL (loop)':<16}{self._loop_total:>12.4f}{'':>12}{'100.0':>12}")
        lines.append(f"frames: {self._frame_count}")
        return "\n".join(lines)

    def print_summary(self, label: str | None = None) -> None:
        print(self.format_summary(label))


class NullStageTimer:
    """No-op drop-in replacement for StageTimer: `stage()`/`frame()` return a
    shared singleton no-op context manager, `mark_frame()` does nothing.
    Guarantees zero perf_counter calls and zero per-call allocation on the
    hot path when timing is disabled."""

    __slots__ = ()

    def stage(self, name: str):  # noqa: ARG002 - name unused by design (no-op)
        return _NULL_CONTEXT

    def frame(self):
        return _NULL_CONTEXT

    def mark_frame(self) -> None:
        return None
