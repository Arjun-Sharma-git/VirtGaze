"""Tests for StageThread tap queues and put_or_drop backpressure."""

from __future__ import annotations

import queue
import threading

from gaze_estimation.pipeline.thread_base import StageThread, put_or_drop


class _Emitter(StageThread):
    """Source stage that emits an incrementing counter."""

    def __init__(self, output_queue, tap_queue, stop_event):
        super().__init__(
            input_queue=None,
            output_queue=output_queue,
            stop_event=stop_event,
            tap_queue=tap_queue,
            name="emitter",
        )
        self.count = 0

    def process(self, item) -> None:
        self.emit(self.count)
        self.count += 1


def test_put_or_drop_drops_oldest_when_full():
    q: queue.Queue = queue.Queue(maxsize=2)
    for i in range(5):
        put_or_drop(q, i)
    assert q.qsize() == 2
    assert q.get_nowait() == 3  # 0..2 were dropped
    assert q.get_nowait() == 4


def test_emit_copies_to_output_and_tap():
    out: queue.Queue = queue.Queue()
    tap: queue.Queue = queue.Queue()
    stage = _Emitter(out, tap, threading.Event())

    stage.process(None)

    assert out.qsize() == 1
    assert tap.qsize() == 1
    assert out.get_nowait() == tap.get_nowait() == 0


def test_emit_without_tap_is_safe():
    out: queue.Queue = queue.Queue()
    stage = _Emitter(out, None, threading.Event())

    stage.process(None)

    assert out.qsize() == 1


def test_tap_uses_drop_oldest_semantics():
    tap: queue.Queue = queue.Queue(maxsize=2)
    stage = _Emitter(queue.Queue(), tap, threading.Event())

    for _ in range(5):
        stage.process(None)

    assert tap.qsize() == 2
    assert tap.get_nowait() == 3  # only the two newest survive
    assert tap.get_nowait() == 4
