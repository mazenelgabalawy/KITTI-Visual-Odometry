"""Background-thread prefetching wrapper for :class:`StereoSequence`.

``KittiSequence.__getitem__`` decodes two PNGs per call via ``cv2.imread``.
OpenCV's image codecs release the GIL while decoding, and each call touches
only local state (distinct file paths on disk, a freshly-constructed
:class:`~kitti_slam.datasets.base.StereoFrame`) -- there is no shared mutable
state between calls. That makes it safe to run one background reader thread
concurrently with the main thread's compute: the two threads never touch the
same Python objects, so there is nothing to race on, and the GIL release
during ``cv2.imread`` lets decode of frame ``k+1`` genuinely overlap with
odometry compute on frame ``k``.

The one piece of *lazy* shared state on ``KittiSequence`` is its
``calibration`` property, which builds and caches a ``StereoCalibration`` on
first access. ``PrefetchingSequence`` eagerly reads ``wrapped.calibration``
in ``__init__`` (before the background thread starts), so that cache is
always populated single-threaded and never touched concurrently.

Usage pattern this is optimized for -- and the only pattern the odometry
pipeline uses -- is strictly ascending sequential access, ``sequence[0]``,
``sequence[1]``, ..., ``sequence[N-1]``, with no gaps or repeats. That fast
path is served entirely from a bounded prefetch queue. Any other access
pattern (repeats, out-of-order, gaps) falls back to a direct synchronous read
through the wrapped sequence, which is always correct but forfeits the
overlap benefit for that particular read.
"""

from __future__ import annotations

import queue
import threading
from types import TracebackType

from kitti_slam.datasets.base import StereoFrame, StereoSequence
from kitti_slam.geometry.calibration import StereoCalibration

# Sentinel placed on the queue once the reader thread has produced every
# index in [0, len(wrapped)); lets the consumer/close() logic tell "done"
# apart from "exception" without extra bookkeeping.
_DONE = object()


class PrefetchingSequence(StereoSequence):
    """Wraps a :class:`StereoSequence` and prefetches frames ahead of demand.

    A single daemon background thread reads frames ``0, 1, 2, ...`` from the
    wrapped sequence in order and pushes them onto a bounded
    ``queue.Queue(maxsize=max_prefetch)``. The bounded size provides
    backpressure: the reader thread blocks once it is ``max_prefetch`` frames
    ahead of the consumer, so memory use stays flat regardless of sequence
    length.

    ``__getitem__(k)`` takes the fast path (pop from the queue) only when
    ``k`` is exactly the next index the reader thread is expected to
    produce; any other index -- a repeat read, a rewind, a skip-ahead --
    falls back to a direct synchronous call on the wrapped sequence, which
    is always correct regardless of prefetch state.
    """

    def __init__(self, wrapped: StereoSequence, max_prefetch: int = 2) -> None:
        self._wrapped = wrapped
        self._length = len(wrapped)
        # Eagerly populate the wrapped sequence's lazily-cached calibration
        # single-threaded, before the background thread can touch it.
        self._calibration: StereoCalibration = wrapped.calibration

        self._queue: queue.Queue = queue.Queue(maxsize=max_prefetch)
        self._next_expected = 0
        self._fallback_count = 0
        self._closed = False

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="PrefetchingSequence-reader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        for idx in range(self._length):
            try:
                frame = self._wrapped[idx]
            except BaseException as exc:  # noqa: BLE001 - propagate to consumer
                self._queue.put((idx, exc))
                return
            self._queue.put((idx, frame))
        self._queue.put(_DONE)

    def __len__(self) -> int:
        return self._length

    @property
    def calibration(self) -> StereoCalibration:
        return self._calibration

    def __getitem__(self, idx: int) -> StereoFrame:
        if idx == self._next_expected:
            item = self._queue.get()
            if item is _DONE:
                # Reader thread believes it is finished but the requested
                # index was never produced (shouldn't happen given _length
                # bookkeeping); fall back for correctness.
                self._fallback_count += 1
                return self._wrapped[idx]

            got_idx, payload = item
            # got_idx should always equal idx here since the reader produces
            # strictly ascending indices and idx == self._next_expected, but
            # guard defensively in case of an unexpected mismatch.
            if got_idx != idx:
                self._fallback_count += 1
                return self._wrapped[idx]

            self._next_expected = idx + 1
            if isinstance(payload, BaseException):
                raise payload
            return payload

        # Out-of-order / repeat / skip access: direct synchronous read.
        self._fallback_count += 1
        return self._wrapped[idx]

    def close(self) -> None:
        """Join the background thread, draining any queued items first.

        Safe to call multiple times. After close(), no background thread
        remains associated with this sequence.
        """
        if self._closed:
            return
        self._closed = True
        # Drain the queue so the reader thread (if blocked on a full queue)
        # can make progress and reach the end of its loop / put _DONE.
        while self._thread.is_alive():
            try:
                self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
        self._thread.join(timeout=5.0)

    def __enter__(self) -> PrefetchingSequence:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
