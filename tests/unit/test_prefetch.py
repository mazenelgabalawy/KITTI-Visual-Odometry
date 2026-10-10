"""Unit tests for kitti_slam.datasets.prefetch.PrefetchingSequence."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from kitti_slam.datasets.base import StereoFrame, StereoSequence
from kitti_slam.datasets.prefetch import PrefetchingSequence
from kitti_slam.geometry.calibration import StereoCalibration


def _make_calibration() -> StereoCalibration:
    P0 = np.zeros((3, 4))
    P1 = np.zeros((3, 4))
    K = np.eye(3)
    return StereoCalibration(P0=P0, P1=P1, K=K, baseline=1.0, focal_length=1.0)


class FakeSequence(StereoSequence):
    """In-memory StereoSequence with deterministic frames and call tracking."""

    def __init__(self, length: int, raise_at: int | None = None) -> None:
        self._length = length
        self._raise_at = raise_at
        self._calibration = _make_calibration()
        self.calls: list[int] = []
        self.calibration_accesses = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> StereoFrame:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"index {idx} out of range for length {self._length}")
        with self._lock:
            self.calls.append(idx)
        if self._raise_at is not None and idx == self._raise_at:
            raise RuntimeError(f"boom at {idx}")
        left = np.full((2, 2), idx, dtype=np.uint8)
        right = np.full((2, 2), idx + 100, dtype=np.uint8)
        return StereoFrame(index=idx, left=left, right=right)

    @property
    def calibration(self) -> StereoCalibration:
        self.calibration_accesses += 1
        return self._calibration


def _active_thread_count() -> int:
    return threading.active_count()


def test_ordered_access_matches_wrapped() -> None:
    fake = FakeSequence(10)
    reference = FakeSequence(10)
    prefetching = PrefetchingSequence(fake)
    try:
        for k in range(len(prefetching)):
            got = prefetching[k]
            expected = reference[k]
            assert got.index == expected.index
            assert np.array_equal(got.left, expected.left)
            assert np.array_equal(got.right, expected.right)
    finally:
        prefetching.close()


def test_ordered_access_served_from_queue_not_fallback() -> None:
    fake = FakeSequence(20)
    prefetching = PrefetchingSequence(fake)
    try:
        for k in range(len(prefetching)):
            prefetching[k]
        assert prefetching._fallback_count == 0
    finally:
        prefetching.close()


def test_random_access_correctness() -> None:
    fake = FakeSequence(10)
    reference = FakeSequence(10)
    prefetching = PrefetchingSequence(fake)
    try:
        for k in (3, 0, 3):
            got = prefetching[k]
            expected = reference[k]
            assert got.index == expected.index
            assert np.array_equal(got.left, expected.left)
            assert np.array_equal(got.right, expected.right)
        # These reads were not the sequential fast path, so fallback should
        # have been used at least once.
        assert prefetching._fallback_count > 0
    finally:
        prefetching.close()


def test_len_and_calibration_passthrough() -> None:
    fake = FakeSequence(7)
    prefetching = PrefetchingSequence(fake)
    try:
        assert len(prefetching) == 7
        assert prefetching.calibration is fake._calibration
        # Eagerly read once in __init__; property access afterwards must not
        # trigger additional calls into the wrapped sequence.
        assert prefetching.calibration is fake._calibration
        assert fake.calibration_accesses == 1
    finally:
        prefetching.close()


def test_exception_propagates() -> None:
    fake = FakeSequence(10, raise_at=4)
    prefetching = PrefetchingSequence(fake)
    try:
        for k in range(4):
            prefetching[k]
        with pytest.raises(RuntimeError, match="boom at 4"):
            prefetching[4]
    finally:
        prefetching.close()


def test_no_thread_leak_with_context_manager() -> None:
    baseline = _active_thread_count()
    fake = FakeSequence(15)
    with PrefetchingSequence(fake) as prefetching:
        for k in range(len(prefetching)):
            prefetching[k]
    assert _active_thread_count() == baseline


def test_no_thread_leak_explicit_close() -> None:
    baseline = _active_thread_count()
    fake = FakeSequence(15)
    prefetching = PrefetchingSequence(fake)
    prefetching[0]
    prefetching[1]
    prefetching.close()
    assert _active_thread_count() == baseline

    # close() should be idempotent.
    prefetching.close()
    assert _active_thread_count() == baseline
