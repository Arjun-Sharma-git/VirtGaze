"""Tests for the webcam capture stage.

``cv2.VideoCapture`` is replaced with a recording stand-in injected into the
camera module's namespace, so opening, auto-exposure, read failures, reconnects,
and in-place undistortion are all exercised without a physical camera.  Time is
also faked, so the reconnect delay never slows the suite down.
"""

from __future__ import annotations

import queue
import threading
import types

import numpy as np
import pytest

import gaze_estimation.capture.camera as camera_module
from gaze_estimation.capture.camera import CameraCapture
from gaze_estimation.utils.camera_calibration import estimate_camera_matrix

W, H = 1280, 720
FRAME = np.full((H, W, 3), 64, dtype=np.uint8)


class FakeCapture:
    """Records driver calls and returns queued frames."""

    def __init__(self, opened=True, frames=None, fail_reads=0, size=(W, H), fps=30.0):
        self._opened = opened
        self._frames = list(frames) if frames else []
        self._fail_reads = fail_reads
        self._size = size
        self._fps = fps
        self.sets: list[tuple[int, float]] = []
        self.gets: list[int] = []
        self.read_calls = 0
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop, value) -> bool:
        self.sets.append((prop, value))
        return True

    def get(self, prop) -> float:
        self.gets.append(prop)
        if prop == FakeCV2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == FakeCV2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        if prop == FakeCV2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == FakeCV2.CAP_PROP_AUTO_EXPOSURE:
            return 0.75
        return 0.0

    def read(self):
        self.read_calls += 1
        if self._fail_reads > 0:
            self._fail_reads -= 1
            return False, None
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1


class FakeCV2(types.ModuleType):
    """Just the pieces of cv2 that CameraCapture touches."""

    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_AUTO_EXPOSURE = 21

    def __init__(self, **capture_kwargs):
        super().__init__("fake_cv2")
        self.capture_kwargs = capture_kwargs
        self.created: list[FakeCapture] = []
        self.requested_indices: list[int] = []

    def VideoCapture(self, index):  # noqa: N802 - mirrors the OpenCV name
        self.requested_indices.append(index)
        capture = FakeCapture(**self.capture_kwargs)
        self.created.append(capture)
        return capture


class FakeTime(types.ModuleType):
    def __init__(self):
        super().__init__("fake_time")
        self.sleeps: list[float] = []
        self.now = 100.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def perf_counter(self) -> float:
        self.now += 0.01
        return self.now


@pytest.fixture
def env(monkeypatch):
    """Install fake cv2/time into the camera module and build a stage."""

    def make(capture_kwargs=None, **kwargs):
        fake_cv2 = FakeCV2(**(capture_kwargs or {}))
        fake_time = FakeTime()
        monkeypatch.setattr(camera_module, "cv2", fake_cv2)
        monkeypatch.setattr(camera_module, "time", fake_time)
        q_out: queue.Queue = queue.Queue()
        kwargs.setdefault("camera_index", 0)
        kwargs.setdefault("width", W)
        kwargs.setdefault("height", H)
        stage = CameraCapture(q_out, threading.Event(), **kwargs)
        return stage, fake_cv2, fake_time, q_out

    return make


# ── _open_camera ──────────────────────────────────────────────────────────────


def test_open_requests_the_configured_stream_settings(env):
    stage, fake_cv2, _, _ = env()

    stage._open_camera()

    assert fake_cv2.requested_indices == [0]
    assert fake_cv2.created[0].sets[:3] == [
        (FakeCV2.CAP_PROP_FRAME_WIDTH, float(W)),
        (FakeCV2.CAP_PROP_FRAME_HEIGHT, float(H)),
        (FakeCV2.CAP_PROP_FPS, 60.0),
    ]
    assert stage._cap is fake_cv2.created[0]


def test_open_uses_the_requested_camera_index(env):
    stage, fake_cv2, _, _ = env(camera_index=2)

    stage._open_camera()

    assert fake_cv2.requested_indices == [2]


def test_auto_exposure_enabled_sets_both_driver_conventions(env):
    """V4L2 uses 3 = auto; DirectShow uses 0.75.  Both must be sent."""
    stage, fake_cv2, _, _ = env(auto_exposure=True)

    stage._open_camera()

    exposure_sets = [
        value for prop, value in fake_cv2.created[0].sets if prop == FakeCV2.CAP_PROP_AUTO_EXPOSURE
    ]
    assert exposure_sets == [3.0, 0.75]
    # The driver's actual reading is queried for the log line.
    assert FakeCV2.CAP_PROP_AUTO_EXPOSURE in fake_cv2.created[0].gets


def test_auto_exposure_disabled_sends_the_manual_values(env):
    stage, fake_cv2, _, _ = env(auto_exposure=False)

    stage._open_camera()

    exposure_sets = [
        value for prop, value in fake_cv2.created[0].sets if prop == FakeCV2.CAP_PROP_AUTO_EXPOSURE
    ]
    assert exposure_sets == [1.0, 0.25]


def test_failed_open_leaves_no_capture(env):
    stage, fake_cv2, _, _ = env(capture_kwargs={"opened": False})

    stage._open_camera()

    assert stage._cap is None
    assert fake_cv2.created[0].release_calls == 0


def test_intrinsics_follow_a_resolution_the_driver_changed(env):
    stage, _, _, _ = env(capture_kwargs={"size": (640, 480)})

    stage._open_camera()

    assert stage._camera_matrix[0, 2] == pytest.approx(320.0)
    assert stage._camera_matrix[0, 0] == pytest.approx(640.0)


def test_intrinsics_are_kept_when_the_resolution_matches(env):
    custom = estimate_camera_matrix(320, 240)
    stage, _, _, _ = env(camera_matrix=custom)

    stage._open_camera()

    np.testing.assert_allclose(stage._camera_matrix, custom)


def test_get_camera_intrinsics_returns_the_constructed_values(env):
    custom = estimate_camera_matrix(320, 240)
    stage, _, _, _ = env(camera_matrix=custom, dist_coeffs=np.zeros(5))

    matrix, coeffs = stage.get_camera_intrinsics()

    np.testing.assert_allclose(matrix, custom)
    assert coeffs.shape == (5,)


def test_teardown_releases_the_capture(env):
    stage, fake_cv2, _, _ = env()
    stage._open_camera()

    stage._teardown()

    assert fake_cv2.created[0].release_calls == 1
    assert stage._cap is None


def test_teardown_before_open_is_safe(env):
    stage, _, _, _ = env()
    stage._teardown()  # must not raise
    assert stage._cap is None


def test_setup_opens_the_camera(env):
    stage, fake_cv2, _, _ = env()

    stage._setup()

    assert stage._cap is not None
    assert len(fake_cv2.created) == 1


# ── process() ─────────────────────────────────────────────────────────────────


def test_process_emits_a_frame_packet(env):
    stage, _, fake_time, q_out = env(capture_kwargs={"frames": [FRAME]})
    stage._open_camera()

    stage.process(None)

    packet = q_out.get_nowait()
    assert packet.frame is FRAME
    assert packet.frame_id == 0
    assert packet.timestamp == pytest.approx(fake_time.now)


def test_frame_ids_increment(env):
    stage, _, _, q_out = env(capture_kwargs={"frames": [FRAME] * 3})
    stage._open_camera()

    for _ in range(3):
        stage.process(None)

    assert [q_out.get_nowait().frame_id for _ in range(3)] == [0, 1, 2]


def test_process_emits_nothing_when_the_device_is_closed(env):
    stage, fake_cv2, _, q_out = env(capture_kwargs={"frames": [FRAME]})

    stage.process(None)  # no _open_camera() first

    assert q_out.qsize() == 0
    assert len(fake_cv2.created) == 1  # reconnect attempted → one open call


def test_a_successful_read_clears_the_failure_count(env):
    stage, _, _, _ = env(capture_kwargs={"frames": [FRAME]})
    stage._open_camera()
    stage._failure_count = 3

    stage.process(None)

    assert stage._failure_count == 0


def test_a_failed_read_is_counted_and_emits_nothing(env):
    stage, _, _, q_out = env(capture_kwargs={"frames": [FRAME], "fail_reads": 1})
    stage._open_camera()

    stage.process(None)

    assert stage._failure_count == 1
    assert q_out.qsize() == 0


def test_five_consecutive_failures_trigger_a_reconnect(env):
    stage, fake_cv2, fake_time, q_out = env(capture_kwargs={"fail_reads": 100})
    stage._open_camera()
    assert len(fake_cv2.created) == 1

    for _ in range(5):
        stage.process(None)

    assert len(fake_cv2.created) == 2  # reopened
    assert fake_time.sleeps == [2.0]
    assert q_out.qsize() == 0


def test_reconnect_resets_the_failure_count(env):
    stage, _, _, _ = env(capture_kwargs={"fail_reads": 100})
    stage._open_camera()

    for _ in range(5):
        stage.process(None)

    assert stage._failure_count == 0


def test_fewer_than_five_failures_does_not_reconnect(env):
    stage, fake_cv2, fake_time, _ = env(capture_kwargs={"fail_reads": 4})
    stage._open_camera()

    for _ in range(4):
        stage.process(None)

    assert len(fake_cv2.created) == 1
    assert fake_time.sleeps == []


# ── Undistortion ──────────────────────────────────────────────────────────────


def test_undistortion_is_off_by_default(env):
    stage, _, _, _ = env()

    stage._open_camera()

    assert stage._map1 is None and stage._map2 is None


def test_enabling_undistortion_builds_maps_on_open(env):
    stage, _, _, _ = env(undistort=True)

    stage._open_camera()

    assert stage._map1 is not None
    assert stage._map2 is not None


def test_enabled_undistortion_replaces_the_frame(env):
    stage, _, _, q_out = env(undistort=True, capture_kwargs={"frames": [FRAME.copy()]})
    stage._open_camera()

    stage.process(None)

    packet = q_out.get_nowait()
    assert packet.frame.shape == FRAME.shape
    assert packet.frame is not FRAME


def test_disabled_undistortion_passes_the_frame_through(env):
    stage, _, _, q_out = env(undistort=False, capture_kwargs={"frames": [FRAME]})
    stage._open_camera()

    stage.process(None)

    assert q_out.get_nowait().frame is FRAME


def test_set_camera_intrinsics_rebuilds_the_maps(env):
    stage, _, _, _ = env(undistort=True)
    stage._open_camera()
    before = stage._map1

    stage.set_camera_intrinsics(estimate_camera_matrix(640, 480), np.zeros(5))

    assert stage._map1 is not None
    assert stage._map1 is not before  # rebuilt, not reused
    np.testing.assert_allclose(stage._camera_matrix, estimate_camera_matrix(640, 480))


def test_rebuilt_maps_follow_the_driver_reported_size(env):
    """The maps are sized from the stream resolution, not the intrinsics."""
    stage, _, _, _ = env(undistort=True, capture_kwargs={"size": (640, 480)})
    stage._open_camera()

    stage.set_camera_intrinsics(estimate_camera_matrix(1280, 720), np.zeros(5))

    assert stage._map1.shape == (480, 640)


def test_set_camera_intrinsics_without_undistortion_keeps_maps_empty(env):
    stage, _, _, _ = env(undistort=False)
    stage._open_camera()

    stage.set_camera_intrinsics(estimate_camera_matrix(640, 480), np.zeros(5))

    assert stage._map1 is None and stage._map2 is None
