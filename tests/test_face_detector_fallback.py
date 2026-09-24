"""Tests for the Haar-cascade fallback's OpenCV-version handling.

OpenCV 5 removed both ``cv2.CascadeClassifier`` and the bundled cascade XML
files, so the fallback must detect that and report "no face" rather than raising
``AttributeError`` on every frame.  cv2's attributes are patched explicitly, so
both the available and unavailable cases run on any OpenCV version.
"""

from __future__ import annotations

import logging
import queue
import threading

import cv2
import numpy as np
import pytest

import gaze_estimation.detection.face_detector as detector_module
from gaze_estimation.detection.face_detector import _HAAR_CASCADE_XML, FaceDetector

FRAME = np.full((480, 640, 3), 100, dtype=np.uint8)


def _detector() -> FaceDetector:
    return FaceDetector(queue.Queue(), queue.Queue(), threading.Event())


class FakeClassifier:
    """Stands in for cv2.CascadeClassifier."""

    def __init__(self, path, faces=((10, 20, 30, 40),)):
        self.path = path
        self.faces = np.array(faces)
        self.detected_on = []

    def detectMultiScale(self, gray, **kwargs):  # noqa: N802 - mirrors OpenCV
        self.detected_on.append(gray.shape)
        return self.faces


@pytest.fixture
def xml_dir(tmp_path):
    """A directory that looks like cv2.data.haarcascades with the XML present."""
    (tmp_path / _HAAR_CASCADE_XML).write_text("<opencv_storage/>")
    return tmp_path


def _patch_cv2(monkeypatch, classifier, xml_root):
    """Point the module's cv2 at a given classifier class and cascade directory."""
    monkeypatch.setattr(detector_module.cv2, "CascadeClassifier", classifier, raising=False)
    monkeypatch.setattr(
        detector_module.cv2, "data", type("D", (), {"haarcascades": str(xml_root)})(), raising=False
    )


# ── Unavailable ───────────────────────────────────────────────────────────────


def test_missing_classifier_degrades_instead_of_raising(monkeypatch, xml_dir, caplog):
    _patch_cv2(monkeypatch, None, xml_dir)
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        assert detector._load_cascade() is None

    assert "Haar cascade fallback unavailable" in caplog.text


def test_missing_cascade_xml_degrades(monkeypatch, tmp_path, caplog):
    _patch_cv2(monkeypatch, FakeClassifier, tmp_path)  # dir has no XML
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        assert detector._load_cascade() is None

    assert "Haar cascade fallback unavailable" in caplog.text


def test_unavailable_cascade_reports_no_face(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, None, xml_dir)
    detector = _detector()

    assert detector._fallback_detect(FRAME) is None


def test_warning_is_emitted_once(monkeypatch, xml_dir, caplog):
    _patch_cv2(monkeypatch, None, xml_dir)
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        detector._load_cascade()
        detector._load_cascade()
        detector._fallback_detect(FRAME)

    assert caplog.text.count("Haar cascade fallback unavailable") == 1


def test_classifier_construction_failure_degrades(monkeypatch, xml_dir, caplog):
    def _boom(path):
        raise RuntimeError("bad xml")

    _patch_cv2(monkeypatch, _boom, xml_dir)
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        assert detector._load_cascade() is None

    assert "Could not load Haar cascade" in caplog.text


def test_missing_data_module_degrades(monkeypatch, caplog):
    monkeypatch.setattr(detector_module.cv2, "CascadeClassifier", FakeClassifier, raising=False)
    monkeypatch.setattr(detector_module.cv2, "data", None, raising=False)
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        assert detector._load_cascade() is None

    assert "Haar cascade fallback unavailable" in caplog.text


# ── Available ─────────────────────────────────────────────────────────────────


def test_available_cascade_is_loaded_from_the_xml(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, FakeClassifier, xml_dir)
    detector = _detector()

    cascade = detector._load_cascade()

    assert isinstance(cascade, FakeClassifier)
    assert cascade.path == str(xml_dir / _HAAR_CASCADE_XML)


def test_available_cascade_detects_a_face(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, FakeClassifier, xml_dir)
    detector = _detector()

    result = detector._fallback_detect(FRAME)

    assert result is not None
    bbox, confidence, landmarks = result
    assert bbox == (10, 20, 30, 40)
    assert confidence == 0.5  # Haar detections carry no real score
    assert landmarks is None


def test_available_cascade_runs_on_a_grayscale_frame(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, FakeClassifier, xml_dir)
    detector = _detector()
    detector._load_cascade()

    detector._fallback_detect(FRAME)

    assert detector._cascade.detected_on == [(480, 640)]


def test_no_detection_returns_none(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, lambda path: FakeClassifier(path, faces=[]), xml_dir)

    assert _detector()._fallback_detect(FRAME) is None


def test_classifier_is_cached_across_calls(monkeypatch, xml_dir):
    _patch_cv2(monkeypatch, FakeClassifier, xml_dir)
    detector = _detector()

    assert detector._load_cascade() is detector._load_cascade()


# ── Real OpenCV smoke tests ───────────────────────────────────────────────────


def test_resolving_the_cascade_never_raises_on_the_installed_opencv(caplog):
    """Whichever OpenCV is installed, resolving must either work or degrade."""
    detector = _detector()

    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        cascade = detector._load_cascade()

    assert cascade is not None or "Haar cascade fallback unavailable" in caplog.text


def test_fallback_detect_returns_a_well_formed_result_or_none():
    result = _detector()._fallback_detect(FRAME)

    assert result is None or (isinstance(result, tuple) and len(result) == 3)


def test_real_cv2_version_is_reported_when_unavailable(caplog):
    if getattr(cv2, "CascadeClassifier", None) is not None:
        pytest.skip("this OpenCV still ships the cascade classifier")

    detector = _detector()
    with caplog.at_level(logging.WARNING, logger="gaze_estimation.detection.face_detector"):
        detector._load_cascade()

    assert cv2.__version__ in caplog.text
