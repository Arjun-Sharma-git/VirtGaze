"""FPS counter and per-stage latency profiler."""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Optional


class FPSCounter:
    """Rolling-window FPS counter.

    Example::

        fps = FPSCounter(window=60)
        while True:
            fps.tick()
            print(fps.get())
    """

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        """Record the current timestamp as a processed frame."""
        self._timestamps.append(time.monotonic())

    def get(self) -> float:
        """Return the current FPS estimate (0.0 if fewer than 2 samples)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


class StageTimer:
    """Simple context-manager / manual timer for a single pipeline stage.

    Usage::

        timer = StageTimer()
        with timer:
            do_work()
        print(timer.last_ms)  # ms for last call
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.last_ms: float = 0.0
        self._history: deque = deque(maxlen=100)

    def __enter__(self) -> StageTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_) -> None:
        self.last_ms = (time.monotonic() - self._start) * 1_000.0
        self._history.append(self.last_ms)

    def start(self) -> None:
        """Manual start (use when context manager is inconvenient)."""
        self._start = time.monotonic()

    def stop(self) -> float:
        """Manual stop; returns elapsed ms."""
        self.last_ms = (time.monotonic() - self._start) * 1_000.0
        self._history.append(self.last_ms)
        return self.last_ms

    @property
    def mean_ms(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def max_ms(self) -> float:
        return max(self._history) if self._history else 0.0


class LatencyProfiler:
    """Track per-stage latency and end-to-end latency for each frame.

    Usage::

        profiler = LatencyProfiler()
        profiler.start_frame()
        profiler.mark("camera")
        profiler.mark("detection")
        profiler.mark("mesh")
        report = profiler.end_frame()
        # report = {"camera": 2.1, "detection": 5.3, "total": 7.4, ...}
    """

    def __init__(self, history_size: int = 100) -> None:
        self._stage_timers: Dict[str, StageTimer] = {}
        self._history: deque = deque(maxlen=history_size)
        self._frame_start: float = 0.0
        self._last_mark: float = 0.0
        self._current_frame: Dict[str, float] = {}

    def start_frame(self) -> None:
        """Mark the start of a new frame."""
        self._frame_start = time.monotonic()
        self._last_mark = self._frame_start
        self._current_frame = {}

    def mark(self, stage: str) -> float:
        """Record elapsed time since the previous mark (or frame start).

        Returns the ms elapsed for this stage.
        """
        now = time.monotonic()
        elapsed_ms = (now - self._last_mark) * 1_000.0
        self._last_mark = now
        self._current_frame[stage] = elapsed_ms
        if stage not in self._stage_timers:
            self._stage_timers[stage] = StageTimer()
        self._stage_timers[stage]._history.append(elapsed_ms)
        self._stage_timers[stage].last_ms = elapsed_ms
        return elapsed_ms

    def end_frame(self) -> Dict[str, float]:
        """Finalise the frame.  Returns per-stage latency dict + 'total'."""
        total_ms = (time.monotonic() - self._frame_start) * 1_000.0
        self._current_frame["total"] = total_ms
        self._history.append(dict(self._current_frame))
        return dict(self._current_frame)

    def averages(self) -> Dict[str, float]:
        """Return average latency (ms) per stage over all recorded frames."""
        if not self._history:
            return {}
        keys = self._history[0].keys()
        result: Dict[str, float] = {}
        for k in keys:
            vals = [f[k] for f in self._history if k in f]
            result[k] = sum(vals) / len(vals) if vals else 0.0
        return result

    def stage(self, name: str) -> Optional[StageTimer]:
        """Return the StageTimer for *name*, or None if not yet seen."""
        return self._stage_timers.get(name)

    def report_str(self) -> str:
        """Return a one-line summary of average latency per stage."""
        avgs = self.averages()
        parts = [f"{k}={v:.1f}ms" for k, v in avgs.items()]
        return " | ".join(parts)
