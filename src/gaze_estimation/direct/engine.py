"""Direct gaze engine: one loop from camera frame to filtered screen point.

The staged pipeline (camera, detection, mesh, pose, gaze, inference and
filter threads joined by bounded, frame-dropping queues) exists for the 3D
eye model.  The landmark ridge model needs none of the middle stages, and a
side-by-side test showed the staged path losing a lot of accuracy: the same
features and model scored ~110 px in a plain capture loop against ~230-280 px
through the pipeline.  This engine is that plain loop:

    frame -> FaceLandmarker -> face-aligned landmark features -> ridge model
          -> FilterStage (UKF, One Euro, stabiliser, blink hold)

It mirrors the parts of :class:`GazeEstimationPipeline` the scripts use
(``start``, ``stop``, ``set_ridge_model``, ``get_latest_estimate``,
``tap_predictions``), so check_accuracy.py and tune_filters.py can drive it
with ``--direct``.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gaze_estimation.config.config import Config
from gaze_estimation.model.landmark_features import extract_landmark_features
from gaze_estimation.pipeline.schemas import GazeEstimate, PredictionPacket
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("direct.engine")


def apply_camera_mode(config: Config, mode: Optional[str]) -> None:
    """Set camera.width/height/fourcc from a "WxH" string ("" or None: keep).

    1280x720 also requests MJPG: uncompressed 720p runs at ~10 FPS on most
    UVC webcams.
    """
    if not mode:
        return
    w, h = (int(v) for v in str(mode).lower().split("x"))
    config.set("camera.width", w)
    config.set("camera.height", h)
    config.set("camera.fourcc", "MJPG" if w * h > 640 * 480 else "")


def camera_mode(config: Config) -> tuple:
    cam = config.section("camera")
    return (int(cam.get("width", 0) or 0), int(cam.get("height", 0) or 0),
            str(cam.get("fourcc", "") or ""))


def eye_aspect_ratios(points: np.ndarray) -> tuple:
    """(left, right) eye openness: lid gap / eye width, from normalised landmarks."""

    def ratio(outer: int, inner: int, top: int, bottom: int) -> float:
        width = np.linalg.norm(points[outer, :2] - points[inner, :2])
        return float(np.linalg.norm(points[top, :2] - points[bottom, :2]) / (width + 1e-9))

    return ratio(33, 133, 159, 145), ratio(263, 362, 386, 374)


@dataclass
class DirectSample:
    """One processed camera frame."""

    timestamp: float
    frame: np.ndarray
    landmark_features: Optional[np.ndarray]  # None: no face
    left_ear: float
    right_ear: float
    confidence: float  # 1.0 with a face, 0.0 without


class DirectGazeEngine:
    """Single-thread camera loop producing filtered gaze estimates.

    Args:
        config:        Configuration (``camera.*``, ``filtering.*``).
        screen_width:  Display width, pixels.
        screen_height: Display height, pixels.
    """

    def __init__(self, config: Config, screen_width: int, screen_height: int) -> None:
        self._config = config
        self._sw = int(screen_width)
        self._sh = int(screen_height)
        self._ridge = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_estimate: Optional[GazeEstimate] = None
        self._latest_sample: Optional[DirectSample] = None
        self._sample_subscribers: list = []
        self._prediction_tap: Optional[queue.Queue] = None
        self._fps = 0.0
        self._filter = None
        self._filter_out: queue.Queue = queue.Queue()

    # ── Pipeline-compatible API ─────────────────────────────────────────────

    def set_ridge_model(self, ridge) -> None:
        self._ridge = ridge
        camera = getattr(ridge, "camera", None)
        if camera is not None and self._thread is None:
            # Before start(): run the camera in the mode the model was calibrated in
            w, h, fcc = camera
            self._config.set("camera.width", w)
            self._config.set("camera.height", h)
            self._config.set("camera.fourcc", fcc)

    def set_model(self, *args, **kwargs) -> None:  # MLP: not used in direct mode
        _logger.warning("Direct mode uses the landmark ridge model only; MLP ignored")

    def set_camera_intrinsics(self, *args, **kwargs) -> None:  # not needed
        pass

    def update_kappa(self, *args, **kwargs) -> None:  # not needed
        pass

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("DirectGazeEngine is already running")
        from gaze_estimation.pipeline.pipeline import FilterStage

        self._filter = FilterStage(
            queue.Queue(), self._filter_out, threading.Event(), config=self._config
        )
        self._stop.clear()
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True,
                                        name="direct_gaze")
        self._thread.start()
        ready.wait(timeout=15.0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def get_latest_estimate(self) -> Optional[GazeEstimate]:
        with self._lock:
            return self._latest_estimate

    def get_latest_sample(self) -> Optional[DirectSample]:
        with self._lock:
            return self._latest_sample

    def tap_predictions(self, maxsize: int = 10000) -> queue.Queue:
        """Queue receiving every raw (unfiltered) prediction, for filter tuning."""
        self._prediction_tap = queue.Queue(maxsize=maxsize)
        return self._prediction_tap

    def subscribe_samples(self, maxsize: int = 1000) -> queue.Queue:
        """Queue receiving every processed frame (:class:`DirectSample`)."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._sample_subscribers.append(q)
        return q

    def unsubscribe_samples(self, q: queue.Queue) -> None:
        if q in self._sample_subscribers:
            self._sample_subscribers.remove(q)

    @property
    def fps(self) -> float:
        return self._fps

    # ── Loop ──────────────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        cam = self._config.section("camera")
        cap = cv2.VideoCapture(int(cam.get("index", 0)))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {cam.get('index', 0)}")
        # Only change what was asked for; the driver defaults gave the best
        # landmarks in testing (no forced exposure, FPS or compression).
        fourcc = str(cam.get("fourcc", "") or "")
        make_fourcc = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc and make_fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, make_fourcc(*fourcc[:4]))
        w, h = int(cam.get("width", 0) or 0), int(cam.get("height", 0) or 0)
        if w and h:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Auto-exposure is a mode the driver remembers between programs, so set
        # it explicitly (an earlier bug left this webcam stuck in manual).
        from gaze_estimation.capture.camera import _auto_exposure_value

        value = _auto_exposure_value(cap, bool(cam.get("auto_exposure", True)))
        if value is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
        return cap

    def _run(self, ready: threading.Event) -> None:
        from gaze_estimation.utils.mediapipe_compat import create_face_mesh

        try:
            cap = self._open_camera()
            mesh = create_face_mesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
            )
        except Exception as exc:
            _logger.error("Direct engine could not start: %s", exc)
            ready.set()
            return
        ready.set()
        frames, t_fps = 0, time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                ts = time.perf_counter()
                sample = self._process_frame(mesh, frame, ts)
                self._publish(sample)
                frames += 1
                if ts - t_fps >= 1.0:
                    self._fps = frames / (ts - t_fps)
                    frames, t_fps = 0, ts
        finally:
            mesh.close()
            cap.release()

    def _process_frame(self, mesh, frame: np.ndarray, ts: float) -> DirectSample:
        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return DirectSample(ts, frame, None, 0.0, 0.0, 0.0)
        lms = result.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms], dtype=np.float64)
        left, right = eye_aspect_ratios(pts)
        feats = extract_landmark_features(pts)
        return DirectSample(ts, frame, feats, left, right, 1.0 if feats is not None else 0.0)

    def _publish(self, sample: DirectSample) -> None:
        for q in list(self._sample_subscribers):
            try:
                q.put_nowait(sample)
            except queue.Full:
                pass
        estimate = None
        ridge = self._ridge
        if ridge is not None and sample.landmark_features is not None:
            out = ridge.predict(sample.landmark_features)[0]
            x = float(min(max(out[0] * self._sw, 0.0), self._sw - 1))
            y = float(min(max(out[1] * self._sh, 0.0), self._sh - 1))
            packet = PredictionPacket(
                timestamp=sample.timestamp,
                screen_x=x,
                screen_y=y,
                confidence=1.0,
                raw_features={"left_ear": sample.left_ear, "right_ear": sample.right_ear},
                source="ridge",
            )
            if self._prediction_tap is not None:
                try:
                    self._prediction_tap.put_nowait(packet)
                except queue.Full:
                    pass
            self._filter.process(packet)
            try:
                estimate = self._filter_out.get_nowait()
            except queue.Empty:
                estimate = None
        with self._lock:
            self._latest_sample = sample
            if estimate is not None:
                self._latest_estimate = estimate
