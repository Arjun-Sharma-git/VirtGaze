"""Tests for the video-file capture source.

A tiny MJPG clip is generated with cv2 so the whole read/emit path runs without
a camera; if the installed OpenCV build cannot write MJPG the tests skip rather
than fail.
"""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import pytest

import gaze_estimation.capture.video_source as video_source_module
from gaze_estimation.capture.video_source import VideoFileSource

FRAME_W, FRAME_H = 64, 48


def _write_clip(path, frames: int = 5, fps: float = 30.0) -> bool:
    """Write a tiny MJPG .avi.  Returns False if the codec is unavailable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        return False
    for i in range(frames):
        writer.write(np.full((FRAME_H, FRAME_W, 3), i * 20, dtype=np.uint8))
    writer.release()
    return True


@pytest.fixture
def clip_path(tmp_path):
    path = tmp_path / "clip.avi"
    if not _write_clip(path):
        pytest.skip("this OpenCV build cannot write MJPG video")
    return str(path)


def _source(path, *, loop: bool = False, realtime: bool = False):
    out: queue.Queue = queue.Queue()
    src = VideoFileSource(path, out, threading.Event(), loop=loop, realtime=realtime)
    return src, out


class _FakeTime:
    """Stand-in for the stdlib ``time`` module used by VideoFileSource."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def perf_counter(self) -> float:
        return 1234.5


def test_missing_file_raises_when_opened(tmp_path):
    src, _ = _source(str(tmp_path / "nope.mp4"))
    with pytest.raises(FileNotFoundError):
        src._open()


def test_thread_exits_without_raising_when_the_file_is_missing(tmp_path):
    """A bad source must not take the process down or spin forever."""
    src, out = _source(str(tmp_path / "nope.mp4"))

    src.start()
    src.join(timeout=10)

    assert not src.is_alive()
    assert out.qsize() == 0


def test_open_reads_the_frame_rate_from_the_file(clip_path):
    src, _ = _source(clip_path)
    src._open()
    assert src._frame_interval == pytest.approx(1.0 / 30.0, rel=0.2)


def test_process_emits_a_frame_packet(clip_path):
    src, out = _source(clip_path)
    src._open()

    src.process(None)

    packet = out.get_nowait()
    assert packet.frame_id == 0
    assert packet.frame.shape == (FRAME_H, FRAME_W, 3)
    assert packet.timestamp > 0.0


def test_frame_ids_increment(clip_path):
    src, out = _source(clip_path)
    src._open()

    for _ in range(3):
        src.process(None)

    assert [out.get_nowait().frame_id for _ in range(3)] == [0, 1, 2]


def test_realtime_mode_paces_frames(clip_path, monkeypatch):
    fake_time = _FakeTime()
    monkeypatch.setattr(video_source_module, "time", fake_time)
    src, _ = _source(clip_path, realtime=True)
    src._open()

    src.process(None)

    assert fake_time.sleeps == [pytest.approx(1.0 / 30.0, rel=0.2)]


def test_benchmark_mode_does_not_sleep(clip_path, monkeypatch):
    fake_time = _FakeTime()
    monkeypatch.setattr(video_source_module, "time", fake_time)
    src, _ = _source(clip_path, realtime=False)
    src._open()

    src.process(None)

    assert fake_time.sleeps == []


def test_end_of_file_stops_the_source(clip_path):
    src, out = _source(clip_path, loop=False)
    src._open()

    emitted = 0
    for _ in range(200):
        before = out.qsize()
        src.process(None)
        if out.qsize() == before:  # nothing emitted → EOF branch ran
            break
        emitted += 1

    assert emitted >= 1
    assert src.stop_event.is_set()
    assert [out.get_nowait().frame_id for _ in range(emitted)] == list(range(emitted))


def test_loop_rewinds_instead_of_stopping(clip_path):
    src, out = _source(clip_path, loop=True)
    src._open()

    emitted = 0
    for _ in range(200):
        before = out.qsize()
        src.process(None)
        if out.qsize() == before:
            break
        emitted += 1

    assert emitted >= 1
    assert not src.stop_event.is_set()
    assert src._frame_id == 0  # rewound


def test_process_before_open_stops_the_source(tmp_path):
    src, _ = _source(str(tmp_path / "nope.mp4"))

    src.process(None)

    assert src.stop_event.is_set()


def test_teardown_releases_the_capture(clip_path):
    src, _ = _source(clip_path)
    src._open()
    assert src._cap is not None

    src._teardown()

    assert src._cap is None


def test_teardown_before_open_is_safe(tmp_path):
    src, _ = _source(str(tmp_path / "nope.mp4"))
    src._teardown()  # must not raise
    assert src._cap is None
