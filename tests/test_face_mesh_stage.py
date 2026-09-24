"""Tests for FaceMeshExtractor, driven through a fake MediaPipe Face Mesh.

The fake mirrors the real result shape (``multi_face_landmarks[0].landmark`` with
normalised ``.x``/``.y``), which makes the mesh slicing and the iris-circle fits
checkable without the real package.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import types

import numpy as np
import pytest

from gaze_estimation.mesh.face_mesh import FaceMeshExtractor
from gaze_estimation.pipeline.schemas import FacePacket

W, H = 640, 480
FRAME = np.zeros((H, W, 3), dtype=np.uint8)
N_LMS = 478


class _Landmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FaceLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _Results:
    def __init__(self, faces):
        self.multi_face_landmarks = list(faces) if faces else None


class FakeFaceMesh:
    """Stands in for ``mp.solutions.face_mesh.FaceMesh``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.processed: list[np.ndarray] = []
        self.next_results = _Results(None)

    def process(self, rgb):
        self.processed.append(rgb)
        return self.next_results

    def close(self):
        self.closed = True


@pytest.fixture
def mediapipe(monkeypatch):
    constructed: list[FakeFaceMesh] = []

    class _Factory(FakeFaceMesh):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            constructed.append(self)

    mp = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    face_mesh = types.ModuleType("mediapipe.solutions.face_mesh")
    face_mesh.FaceMesh = _Factory
    solutions.face_mesh = face_mesh
    mp.solutions = solutions
    for name, module in (
        ("mediapipe", mp),
        ("mediapipe.solutions", solutions),
        ("mediapipe.solutions.face_mesh", face_mesh),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return constructed


def _extractor(**kwargs) -> FaceMeshExtractor:
    kwargs.setdefault("input_queue", queue.Queue())
    kwargs.setdefault("output_queue", queue.Queue())
    kwargs.setdefault("stop_event", threading.Event())
    return FaceMeshExtractor(**kwargs)


def _packet(frame=FRAME) -> FacePacket:
    return FacePacket(
        timestamp=3.0,
        frame=frame,
        face_bbox=(10, 20, 30, 40),
        detection_confidence=0.9,
        face_landmarks_2d=None,
    )


def _landmarks(n=N_LMS, centre=(0.3, 0.4), radius=0.02) -> list[_Landmark]:
    """Landmarks spread over the frame; indices 468+ form the two iris rings."""
    rng = np.random.default_rng(0)
    points = [
        _Landmark(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))) for _ in range(n)
    ]
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    for base, (cx, cy) in ((468, centre), (473, (centre[0] + 0.1, centre[1]))):
        for i, angle in enumerate(angles):
            index = base + i
            if index < n:
                points[index] = _Landmark(cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    return points


# ── _setup / _teardown ────────────────────────────────────────────────────────


def test_setup_constructs_the_face_mesh(mediapipe):
    extractor = _extractor()

    extractor._setup()

    assert len(mediapipe) == 1
    assert extractor._face_mesh is mediapipe[0]


def test_setup_passes_every_configured_option(mediapipe):
    extractor = _extractor(
        refine_iris=False,
        max_num_faces=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        static_image_mode=True,
    )

    extractor._setup()

    assert mediapipe[0].kwargs == {
        "max_num_faces": 2,
        "refine_landmarks": False,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.6,
        "static_image_mode": True,
    }


def test_setup_logs_the_configuration(mediapipe, caplog):
    extractor = _extractor(static_image_mode=True)

    with caplog.at_level(logging.INFO):
        extractor._setup()

    assert "FaceMesh initialised" in caplog.text
    assert "static_image_mode=True" in caplog.text


def test_setup_failure_is_logged_and_not_raised(monkeypatch, caplog):
    class _Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("no face mesh backend")

    mp = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    face_mesh = types.ModuleType("mediapipe.solutions.face_mesh")
    face_mesh.FaceMesh = _Broken
    solutions.face_mesh = face_mesh
    mp.solutions = solutions
    monkeypatch.setitem(sys.modules, "mediapipe", mp)

    extractor = _extractor()
    with caplog.at_level(logging.ERROR):
        extractor._setup()

    assert extractor._face_mesh is None
    assert "Could not initialise MediaPipe FaceMesh" in caplog.text


def test_teardown_closes_the_face_mesh(mediapipe):
    extractor = _extractor()
    extractor._setup()

    extractor._teardown()

    assert mediapipe[0].closed is True


def test_teardown_without_a_face_mesh_is_safe():
    _extractor()._teardown()  # must not raise


# ── process() ─────────────────────────────────────────────────────────────────


def test_process_ignores_a_packet_of_the_wrong_type():
    extractor = _extractor()

    extractor.process("not a face packet")

    assert extractor.output_queue.qsize() == 0


def test_process_without_a_face_mesh_emits_an_empty_packet():
    extractor = _extractor()

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.mesh_468.shape == (468, 2)
    assert not packet.mesh_468.any()
    assert packet.confidence == 0.0
    assert packet.face_bbox == (10, 20, 30, 40)  # still relayed
    assert packet.timestamp == 3.0


def test_process_without_landmarks_emits_an_empty_packet(mediapipe):
    extractor = _extractor()
    extractor._setup()

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.confidence == 0.0
    assert packet.iris_478 is None
    assert packet.left_iris_center is None


def test_process_converts_the_frame_to_rgb(mediapipe):
    channelised = np.zeros((2, 2, 3), dtype=np.uint8)
    channelised[:, :, 0] = 10  # B
    channelised[:, :, 1] = 20  # G
    channelised[:, :, 2] = 30  # R
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks())])

    extractor.process(_packet(frame=channelised))

    assert extractor._face_mesh.processed[0][0, 0].tolist() == [30, 20, 10]


def test_landmarks_become_pixel_coordinates(mediapipe):
    extractor = _extractor()
    extractor._setup()
    lms = _landmarks()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(lms)])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.mesh_468.shape == (468, 2)
    assert packet.mesh_468.dtype == np.float32
    np.testing.assert_allclose(packet.mesh_468[0], [lms[0].x * W, lms[0].y * H])


def test_iris_refinement_keeps_the_478_point_array(mediapipe):
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks())])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.iris_478 is not None
    assert packet.iris_478.shape == (478, 2)
    assert packet.mesh_468.shape == (468, 2)


def test_iris_centres_are_fitted_from_the_ring_landmarks(mediapipe):
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks())])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.left_iris_center is not None
    assert packet.right_iris_center is not None
    # The left ring was planted at (0.30, 0.40) of a 640x480 frame.
    assert abs(packet.left_iris_center[0] - 0.30 * W) < 2.0
    assert abs(packet.left_iris_center[1] - 0.40 * H) < 2.0
    assert packet.left_iris_radius > 0.0
    assert packet.right_iris_radius > 0.0
    # The right ring sits 0.10 to the right of the left one.
    assert packet.right_iris_center[0] > packet.left_iris_center[0]
    assert packet.confidence == 1.0  # MediaPipe exposes no per-frame score


def test_a_468_point_mesh_yields_no_iris(mediapipe):
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks(n=468))])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.mesh_468.shape == (468, 2)
    assert packet.iris_478 is None
    assert packet.left_iris_center is None
    assert packet.right_iris_center is None


def test_a_partial_iris_ring_fits_the_left_eye_only(mediapipe):
    """473-477 landmarks exist but the right ring is incomplete.

    Documenting reachability: the "else" branch in ``process`` is entered only
    when fewer than 478 landmarks are returned, so the left ring (indices
    468-472) can be fitted while the right ring (473-477) cannot.
    """
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks(n=475))])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.iris_478 is None
    assert packet.left_iris_center is not None
    assert packet.left_iris_radius > 0.0
    assert packet.right_iris_center is None
    assert packet.right_iris_radius is None


def test_only_the_first_face_is_used(mediapipe):
    extractor = _extractor()
    extractor._setup()
    first = _landmarks(centre=(0.3, 0.4))
    second = _landmarks(centre=(0.7, 0.6))
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(first), _FaceLandmarks(second)])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert abs(packet.left_iris_center[0] - 0.30 * W) < 2.0


def test_frame_and_bbox_are_relayed_untouched(mediapipe):
    extractor = _extractor()
    extractor._setup()
    extractor._face_mesh.next_results = _Results([_FaceLandmarks(_landmarks())])

    extractor.process(_packet())

    packet = extractor.output_queue.get_nowait()
    assert packet.frame is FRAME
    assert packet.face_bbox == (10, 20, 30, 40)
