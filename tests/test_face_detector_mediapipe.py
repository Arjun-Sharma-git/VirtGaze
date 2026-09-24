"""Tests for FaceDetector's MediaPipe path, via a faithful fake.

MediaPipe publishes no wheel for every interpreter this project supports, so the
detector is driven through a stand-in injected into ``sys.modules``.  The fake
mirrors the real result object's shape, which makes the geometry (relative
bounding box → pixels, normalised keypoints → pixels) testable deterministically.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import types

import numpy as np
import pytest

from gaze_estimation.detection.face_detector import FaceDetector, resolve_detection_model
from gaze_estimation.pipeline.schemas import FacePacket, FramePacket

W, H = 640, 480
FRAME = np.zeros((H, W, 3), dtype=np.uint8)


class _RelativeBBox:
    def __init__(self, xmin, ymin, width, height):
        self.xmin, self.ymin, self.width, self.height = xmin, ymin, width, height


class _Keypoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _LocationData:
    def __init__(self, bbox, keypoints):
        self.relative_bounding_box = bbox
        self.relative_keypoints = keypoints


class _Detection:
    def __init__(self, score, bbox, keypoints=()):
        self.score = score
        self.location_data = _LocationData(bbox, list(keypoints))


class _Results:
    def __init__(self, detections):
        self.detections = list(detections)


class FakeFaceDetection:
    """Stands in for ``mp.solutions.face_detection.FaceDetection``."""

    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.closed = False
        self.processed: list[np.ndarray] = []
        self.next_results = _Results([])

    def process(self, rgb):
        self.processed.append(rgb)
        return self.next_results

    def close(self):
        self.closed = True


@pytest.fixture
def mediapipe(monkeypatch):
    """Install a fake mediapipe; returns the factory class actually constructed."""

    constructed: list[FakeFaceDetection] = []

    class _Factory(FakeFaceDetection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            constructed.append(self)

    mp = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    face_detection = types.ModuleType("mediapipe.solutions.face_detection")
    face_detection.FaceDetection = _Factory
    solutions.face_detection = face_detection
    mp.solutions = solutions

    for name, module in (
        ("mediapipe", mp),
        ("mediapipe.solutions", solutions),
        ("mediapipe.solutions.face_detection", face_detection),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return constructed


def _detector(**kwargs) -> FaceDetector:
    return FaceDetector(queue.Queue(), queue.Queue(), threading.Event(), **kwargs)


def _packet(frame=FRAME) -> FramePacket:
    return FramePacket(timestamp=1.0, frame=frame, frame_id=0)


def _one_detection(score=(0.9,), bbox=(0.25, 0.5, 0.25, 0.5), keypoints=()) -> _Results:
    return _Results([_Detection(list(score), _RelativeBBox(*bbox), keypoints)])


# ── _setup / _teardown ────────────────────────────────────────────────────────


def test_setup_constructs_the_mediapipe_detector(mediapipe):
    detector = _detector()

    detector._setup()

    assert len(mediapipe) == 1
    assert detector._detector is mediapipe[0]


def test_setup_passes_the_configured_model_and_confidence(mediapipe):
    detector = _detector(model="mediapipe_full", min_confidence=0.8)

    detector._setup()

    assert mediapipe[0].model_selection == resolve_detection_model("mediapipe_full")[1]
    assert mediapipe[0].model_selection == 1
    assert mediapipe[0].min_detection_confidence == 0.8


def test_setup_logs_that_mediapipe_is_ready(mediapipe, caplog):
    detector = _detector()

    # _setup logs through the stage's own logger (gaze_estimation.detection_thread),
    # so the root level must be lowered for an INFO record to propagate.
    with caplog.at_level(logging.INFO):
        detector._setup()

    assert "initialised" in caplog.text


def test_setup_failure_is_logged_and_not_raised(monkeypatch, caplog):
    class _Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("no mediapipe backend")

    mp = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    face_detection = types.ModuleType("mediapipe.solutions.face_detection")
    face_detection.FaceDetection = _Broken
    solutions.face_detection = face_detection
    mp.solutions = solutions
    monkeypatch.setitem(sys.modules, "mediapipe", mp)

    detector = _detector()
    with caplog.at_level(logging.ERROR, logger="gaze_estimation.detection.face_detector"):
        detector._setup()

    assert detector._detector is None
    assert "Could not initialise MediaPipe" in caplog.text


def test_teardown_closes_the_detector(mediapipe):
    detector = _detector()
    detector._setup()

    detector._teardown()

    assert mediapipe[0].closed is True


def test_teardown_without_a_detector_is_safe():
    _detector()._teardown()  # must not raise


# ── _detect_full with MediaPipe ───────────────────────────────────────────────


def test_detect_full_returns_the_bbox_and_confidence(mediapipe):
    detector = _detector()
    detector._setup()
    detector._detector.next_results = _one_detection(score=(0.9,))

    bbox, confidence, landmarks = detector._detect_full(FRAME)

    # 0.25/0.5 of a 640x480 frame
    assert bbox == (160, 240, 160, 240)
    assert confidence == pytest.approx(0.9)
    assert landmarks is None


def test_detect_full_converts_the_frame_to_rgb(mediapipe):
    """MediaPipe expects RGB; a channel-order slip would silently hurt accuracy."""
    channelised = np.zeros((2, 2, 3), dtype=np.uint8)
    channelised[:, :, 0] = 10  # B
    channelised[:, :, 1] = 20  # G
    channelised[:, :, 2] = 30  # R
    detector = _detector()
    detector._setup()

    detector._detect_full(channelised)

    passed = detector._detector.processed[0]
    assert passed[0, 0].tolist() == [30, 20, 10]


def test_keypoints_become_pixel_coordinates(mediapipe):
    keypoints = [_Keypoint(0.1, 0.2), _Keypoint(0.75, 0.5)]
    detector = _detector()
    detector._setup()
    detector._detector.next_results = _one_detection(score=(0.9,), keypoints=keypoints)

    _, _, landmarks = detector._detect_full(FRAME)

    assert landmarks is not None
    assert landmarks.shape == (2, 2)
    assert landmarks.dtype == np.float32
    np.testing.assert_allclose(landmarks[0], [0.1 * W, 0.2 * H])
    np.testing.assert_allclose(landmarks[1], [0.75 * W, 0.5 * H])


def test_no_detections_returns_none(mediapipe):
    detector = _detector()
    detector._setup()

    assert detector._detect_full(FRAME) is None


def test_detection_below_the_confidence_threshold_is_rejected(mediapipe):
    detector = _detector(min_confidence=0.8)
    detector._setup()
    detector._detector.next_results = _one_detection(score=(0.4,))

    assert detector._detect_full(FRAME) is None


def test_a_missing_score_counts_as_zero_confidence(mediapipe):
    """MediaPipe leaves score unset on some builds; that must not crash."""
    detector = _detector(min_confidence=0.8)
    detector._setup()
    detector._detector.next_results = _one_detection(score=())

    assert detector._detect_full(FRAME) is None


def test_bbox_is_clamped_to_the_frame(mediapipe):
    detector = _detector()
    detector._setup()

    detector._detector.next_results = _one_detection(bbox=(0.95, 0.5, 0.5, 0.5))
    assert detector._detect_full(FRAME)[0] == (608, 240, 32, 240)

    detector._detector.next_results = _one_detection(bbox=(-0.1, -0.1, 0.5, 0.5))
    bbox = detector._detect_full(FRAME)[0]
    assert bbox[0] == 0 and bbox[1] == 0
    assert bbox[2] == int(0.5 * W) and bbox[3] == int(0.5 * H)


def test_the_first_detection_is_used(mediapipe):
    detector = _detector()
    detector._setup()
    detector._detector.next_results = _Results(
        [
            _Detection([0.95], _RelativeBBox(0.0, 0.0, 0.1, 0.1)),
            _Detection([0.60], _RelativeBBox(0.5, 0.5, 0.1, 0.1)),
        ]
    )

    bbox, confidence, _ = detector._detect_full(FRAME)

    assert bbox == (0, 0, 64, 48)
    assert confidence == pytest.approx(0.95)


def test_missing_detector_falls_back_to_haar(mediapipe, monkeypatch):
    detector = _detector()
    monkeypatch.setattr(detector, "_fallback_detect", lambda frame: "haar-result")

    assert detector._detect_full(FRAME) == "haar-result"


# ── process() with MediaPipe ──────────────────────────────────────────────────


def test_process_emits_a_face_packet(mediapipe):
    detector = _detector()
    detector._setup()
    detector._detector.next_results = _one_detection(score=(0.9,))

    detector.process(_packet())

    packet = detector.output_queue.get_nowait()
    assert isinstance(packet, FacePacket)
    assert packet.face_bbox == (160, 240, 160, 240)
    assert packet.detection_confidence == pytest.approx(0.9)


def test_process_reports_a_miss_with_an_empty_box(mediapipe):
    detector = _detector()
    detector._setup()

    detector.process(_packet())

    packet = detector.output_queue.get_nowait()
    assert packet.face_bbox is None
    assert packet.detection_confidence == 0.0
