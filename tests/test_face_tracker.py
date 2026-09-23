"""Tests for face ROI tracking (FaceTracker + FaceDetector integration)."""
from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from gaze_estimation.detection.face_detector import FaceDetector
from gaze_estimation.detection.face_tracker import FaceTracker
from gaze_estimation.pipeline.schemas import FramePacket


def _frame() -> np.ndarray:
    """A dark frame with a bright rectangular 'face' blob."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 200:400] = 200
    return img


def test_face_tracker_requires_init():
    tracker = FaceTracker()
    bbox, confidence = tracker.update(_frame())
    assert bbox is None
    assert confidence == 0.0


def test_face_tracker_tracks_initialised_bbox():
    tracker = FaceTracker()
    frame = _frame()
    tracker.init(frame, (200, 100, 200, 200))

    bbox, confidence = tracker.update(frame)

    assert bbox is not None
    assert len(bbox) == 4
    assert confidence > 0.0


def test_face_tracker_reset_clears_state():
    tracker = FaceTracker()
    tracker.init(_frame(), (200, 100, 200, 200))
    tracker.reset()
    bbox, _ = tracker.update(_frame())
    assert bbox is None


def test_detector_tracks_between_full_detections(monkeypatch):
    detector = FaceDetector(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
        detection_interval=10,
    )
    calls = {"full": 0}

    def _fake_full(frame):
        calls["full"] += 1
        return (200, 100, 200, 200), 0.9, None

    emitted: list = []
    monkeypatch.setattr(detector, "_detect_full", _fake_full)
    monkeypatch.setattr(detector, "emit", emitted.append)

    for i in range(5):
        detector.process(FramePacket(timestamp=float(i), frame=_frame(), frame_id=i))

    assert calls["full"] == 1                     # only the first frame detects
    assert len(emitted) == 5
    assert all(p.face_bbox is not None for p in emitted)
    assert emitted[0].detection_confidence == pytest.approx(0.9)
    # Tracked frames report the tracker's own (lower) confidence
    assert emitted[1].detection_confidence < 0.9


def test_detector_resets_tracker_when_face_is_lost(monkeypatch):
    detector = FaceDetector(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
        detection_interval=1,
    )
    emitted: list = []
    monkeypatch.setattr(detector, "_detect_full", lambda frame: None)
    monkeypatch.setattr(detector, "emit", emitted.append)

    detector.process(FramePacket(timestamp=0.0, frame=_frame(), frame_id=0))

    assert emitted[0].face_bbox is None
    assert emitted[0].detection_confidence == 0.0
    assert detector._tracker._hist is None        # tracker state dropped


def test_detector_ignores_non_frame_packets():
    detector = FaceDetector(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
    )
    detector.process("not a frame packet")        # must not raise
