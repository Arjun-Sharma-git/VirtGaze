"""Tests for the FPS counter, stage timer, and latency profiler.

The clock is faked so the assertions are exact rather than "greater than zero".
"""

from __future__ import annotations

import pytest

import gaze_estimation.utils.timing as timing
from gaze_estimation.utils.timing import FPSCounter, LatencyProfiler, StageTimer


class _FakeClock:
    """Stand-in for the stdlib ``time`` module with a settable reading."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(timing, "time", fake)
    return fake


# ── FPSCounter ────────────────────────────────────────────────────────────────


def test_fps_is_zero_without_enough_samples(clock):
    counter = FPSCounter()
    assert counter.get() == 0.0

    counter.tick()

    assert counter.get() == 0.0


def test_fps_uses_the_sample_window(clock):
    counter = FPSCounter(window=10)
    for _ in range(5):
        counter.tick()
        clock.advance(0.05)

    # 4 intervals of 50 ms spanning the 5 samples.
    assert counter.get() == pytest.approx(4 / 0.2)


def test_fps_is_zero_when_no_time_has_passed(clock):
    counter = FPSCounter()
    counter.tick()
    counter.tick()  # frozen clock → zero elapsed

    assert counter.get() == 0.0


def test_window_bounds_the_history(clock):
    counter = FPSCounter(window=3)
    for _ in range(10):
        counter.tick()
        clock.advance(0.1)

    # Only the 3 newest samples are retained → 2 intervals of 100 ms.
    assert counter.get() == pytest.approx(2 / 0.2)


def test_reset_clears_the_counter(clock):
    counter = FPSCounter()
    counter.tick()
    clock.advance(0.1)
    counter.tick()

    counter.reset()

    assert counter.get() == 0.0


# ── StageTimer ────────────────────────────────────────────────────────────────


def test_context_manager_records_elapsed_milliseconds(clock):
    timer = StageTimer()

    with timer as entered:
        clock.advance(0.02)

    assert entered is timer
    assert timer.last_ms == pytest.approx(20.0)


def test_manual_start_stop_records_elapsed_milliseconds(clock):
    timer = StageTimer()

    timer.start()
    clock.advance(0.005)
    elapsed = timer.stop()

    assert elapsed == pytest.approx(5.0)
    assert timer.last_ms == pytest.approx(5.0)


def test_statistics_default_to_zero_without_samples():
    timer = StageTimer()

    assert timer.mean_ms == 0.0
    assert timer.max_ms == 0.0


def test_statistics_summarise_the_history(clock):
    timer = StageTimer()
    for duration in (0.01, 0.03):
        timer.start()
        clock.advance(duration)
        timer.stop()

    assert timer.mean_ms == pytest.approx(20.0)
    assert timer.max_ms == pytest.approx(30.0)
    assert timer.last_ms == pytest.approx(30.0)


# ── LatencyProfiler ───────────────────────────────────────────────────────────


def test_marks_measure_time_since_the_previous_mark(clock):
    profiler = LatencyProfiler()
    profiler.start_frame()
    clock.advance(0.002)

    camera_ms = profiler.mark("camera")
    clock.advance(0.003)
    detection_ms = profiler.mark("detection")

    assert camera_ms == pytest.approx(2.0)
    assert detection_ms == pytest.approx(3.0)


def test_end_frame_reports_stages_plus_total(clock):
    profiler = LatencyProfiler()
    profiler.start_frame()
    clock.advance(0.002)
    profiler.mark("camera")
    clock.advance(0.001)

    report = profiler.end_frame()

    assert report == {"camera": pytest.approx(2.0), "total": pytest.approx(3.0)}


def test_stage_timers_are_created_and_reused(clock):
    profiler = LatencyProfiler()
    assert profiler.stage("camera") is None

    for expected in (2.0, 4.0):
        profiler.start_frame()
        clock.advance(expected / 1000.0)
        profiler.mark("camera")

    timer = profiler.stage("camera")
    assert isinstance(timer, StageTimer)
    assert timer is profiler.stage("camera")  # reused, not recreated
    assert timer.last_ms == pytest.approx(4.0)
    assert timer.mean_ms == pytest.approx(3.0)


def test_averages_are_empty_before_any_frame():
    assert LatencyProfiler().averages() == {}


def test_averages_summarise_recorded_frames(clock):
    profiler = LatencyProfiler()
    for _ in range(3):
        profiler.start_frame()
        clock.advance(0.002)
        profiler.mark("camera")
        clock.advance(0.004)
        profiler.mark("mesh")
        profiler.end_frame()

    assert profiler.averages() == {
        "camera": pytest.approx(2.0),
        "mesh": pytest.approx(4.0),
        "total": pytest.approx(6.0),
    }


def test_averages_are_keyed_by_the_first_frame(clock):
    """Documented narrow spot: stages absent from frame 1 never appear."""
    profiler = LatencyProfiler()
    profiler.start_frame()
    clock.advance(0.002)
    profiler.mark("camera")
    profiler.end_frame()

    profiler.start_frame()
    clock.advance(0.009)
    profiler.mark("detection")
    profiler.end_frame()

    averages = profiler.averages()
    assert "detection" not in averages
    assert averages["camera"] == pytest.approx(2.0)


def test_history_is_bounded(clock):
    profiler = LatencyProfiler(history_size=2)
    for i in range(5):
        profiler.start_frame()
        clock.advance((i + 1) / 1000.0)
        profiler.mark("camera")
        profiler.end_frame()

    assert len(profiler._history) == 2
    assert profiler.averages()["camera"] == pytest.approx(4.5)  # mean of 4 and 5 ms


def test_report_str_joins_the_averages(clock):
    profiler = LatencyProfiler()
    assert profiler.report_str() == ""

    profiler.start_frame()
    clock.advance(0.002)
    profiler.mark("camera")
    profiler.end_frame()

    assert profiler.report_str() == "camera=2.0ms | total=2.0ms"
