"""Tests for kitti_slam.odometry.timing (StageTimer / NullStageTimer).

All tests are synthetic and always run (no KITTI dataset needed).
"""

from __future__ import annotations

import time

from kitti_slam.odometry.timing import NullStageTimer, StageTimer


def test_accumulates_single_stage_across_multiple_calls():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("detect"):
            time.sleep(0.001)

    assert timer.stage_total("detect") > 0.0
    rows = {name: total for name, total, _, _ in timer.summary_rows()}
    assert rows["detect"] == timer.stage_total("detect")


def test_multiple_stages_tracked_independently():
    timer = StageTimer()
    with timer.stage("detect"):
        time.sleep(0.001)
    with timer.stage("pnp"):
        time.sleep(0.002)

    assert timer.stage_total("detect") > 0.0
    assert timer.stage_total("pnp") > 0.0
    names = {name for name, *_ in timer.summary_rows()}
    assert "detect" in names
    assert "pnp" in names


def test_ms_per_frame_and_pct_of_loop():
    timer = StageTimer()
    for _ in range(4):
        with timer.frame(), timer.stage("detect"):
            time.sleep(0.001)

    assert timer.frame_count == 4
    rows = {name: (total_s, ms_per_frame, pct) for name, total_s, ms_per_frame, pct in timer.summary_rows()}
    total_s, ms_per_frame, pct = rows["detect"]

    assert ms_per_frame > 0.0
    # ms_per_frame should equal total_s / frame_count * 1000 within fp tolerance.
    assert abs(ms_per_frame - (total_s / timer.frame_count) * 1000.0) < 1e-9
    assert 0.0 <= pct <= 100.0


def test_other_bucket_is_loop_total_minus_named_stages():
    timer = StageTimer()
    with timer.frame():
        with timer.stage("detect"):
            time.sleep(0.001)
        time.sleep(0.001)  # unaccounted time inside the frame, not in any named stage

    rows = {name: total for name, total, _, _ in timer.summary_rows()}
    assert "other" in rows
    expected_other = timer.loop_total - timer.stage_total("detect")
    assert abs(rows["other"] - expected_other) < 1e-9
    assert rows["other"] >= 0.0


def test_unknown_stage_names_are_created_on_first_use():
    timer = StageTimer()
    with timer.stage("some_future_stage_ba"):
        pass
    with timer.stage("graph_opt"):
        pass

    assert timer.stage_total("some_future_stage_ba") >= 0.0
    assert timer.stage_total("graph_opt") >= 0.0
    names = {name for name, *_ in timer.summary_rows()}
    assert "some_future_stage_ba" in names
    assert "graph_opt" in names


def test_stage_total_of_unused_stage_is_zero():
    timer = StageTimer()
    assert timer.stage_total("never_used") == 0.0


def test_format_summary_and_print_summary_do_not_raise():
    timer = StageTimer()
    with timer.frame(), timer.stage("detect"):
        pass
    text = timer.format_summary(label="sequence 04")
    assert "sequence 04" in text
    assert "detect" in text
    timer.print_summary(label="sequence 04")  # smoke test: must not raise


def test_null_stage_timer_is_noop_and_never_raises():
    timer = NullStageTimer()

    for _ in range(3):
        with timer.frame():
            with timer.stage("detect"):
                pass
            with timer.stage("pnp"):
                raise_free = 1
                assert raise_free == 1
    timer.mark_frame()

    # NullStageTimer exposes no accumulation state to assert on -- the point
    # is simply that using it end-to-end never raises and never measures
    # anything (no attributes like frame_count/stage_total are promised).
    assert not hasattr(timer, "frame_count")


def test_null_stage_timer_stage_context_returns_shared_singleton():
    timer = NullStageTimer()
    ctx1 = timer.stage("a")
    ctx2 = timer.stage("b")
    # Zero-allocation guarantee: the no-op context manager is a shared
    # singleton regardless of stage name.
    assert ctx1 is ctx2
