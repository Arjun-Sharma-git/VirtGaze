"""GazeEstimationPipeline: full end-to-end pipeline orchestrator."""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import Optional, Protocol

import numpy as np

from gaze_estimation.capture.camera import CameraCapture
from gaze_estimation.config.config import Config
from gaze_estimation.detection.face_detector import FaceDetector
from gaze_estimation.filtering.fixation_detector import FixationDetector
from gaze_estimation.filtering.one_euro import OneEuroFilter2D
from gaze_estimation.filtering.unscented_kalman import UnscentedKalmanFilter
from gaze_estimation.gaze.gaze_geometry import GazeGeometryEstimator
from gaze_estimation.mesh.face_mesh import FaceMeshExtractor
from gaze_estimation.pipeline.schemas import (
    GazeEstimate,
    GazePacket,
    MeshPacket,
    PredictionPacket,
)
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.pose.head_pose import HeadPoseEstimator
from gaze_estimation.utils.camera_calibration import estimate_camera_matrix, zero_dist_coeffs
from gaze_estimation.utils.device import describe_device
from gaze_estimation.utils.logging import get_logger
from gaze_estimation.utils.timing import FPSCounter, LatencyProfiler

_logger = get_logger("pipeline")


# The inference collaborators are duck-typed on purpose: importing torch (for
# GazeMLP), onnxruntime or tensorrt at module level would make the whole package
# depend on every optional backend.  These protocols give mypy the interfaces
# without the imports.
class _ModelLike(Protocol):
    """In-process torch model."""

    def predict_numpy(self, X: np.ndarray) -> np.ndarray: ...


class _TrainerLike(Protocol):
    """Trainer holding the feature-normalisation statistics."""

    def features_dict_to_vector(self, features: dict) -> np.ndarray: ...


class _PredictorLike(Protocol):
    """External backend (ONNX Runtime / TensorRT)."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _drain_latest(q: queue.Queue) -> Optional[object]:
    """Non-blocking drain of *q*; returns the newest item (or None)."""
    latest: Optional[object] = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


class InferenceStage(StageThread):
    """Screen coordinate inference from GazePackets using MLP or geometric fallback.

    Backends (selected by ``inference.backend``):

    - ``"torch"``     — in-process :class:`GazeMLP` (``predict_numpy``)
    - ``"onnx"``      — ONNX Runtime (:class:`ONNXInference`)
    - ``"tensorrt"``  — TensorRT engine (:class:`TensorRTInference`)

    Every backend receives the *normalised* feature vector produced by the
    trainer and returns normalised ``(1, 2)`` screen coordinates.  An optional
    :class:`BiasMap` is applied to model predictions to remove residual
    spatial error.

    If no model is available (or confidence is too low) the geometric
    gaze-to-screen projection is used, unless
    ``inference.fallback_to_geometric`` is false — then (and below
    ``inference.geometric_min_confidence``) the estimate holds at the screen
    centre.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        config: Config,
        screen_width: int = 1920,
        screen_height: int = 1080,
        tap_queue: Optional[queue.Queue] = None,
        name: str = "inference_thread",
    ) -> None:
        super().__init__(
            input_queue=input_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            tap_queue=tap_queue,
            name=name,
        )
        self._config = config
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._model: Optional[_ModelLike] = None        # Torch GazeMLP
        self._trainer: Optional[_TrainerLike] = None    # Feature normalisation stats
        self._predictor: Optional[_PredictorLike] = None  # ONNX / TensorRT backend
        self._source_name = "mlp"     # Reported in GazeEstimate.source
        self._bias_map = None         # Optional BiasMap spatial correction
        self._conf_threshold = float(config.get("inference.confidence_threshold", 0.7))
        self._geometric_min_conf = float(
            config.get("inference.geometric_min_confidence", 0.4)
        )
        self._fallback_distance_mm = float(
            config.get("inference.fallback_distance_mm", 600.0)
        )
        # When False the geometric projection is never used: an unusable model
        # or a low-confidence sample holds the estimate at the screen centre.
        self._fallback_enabled = bool(
            config.get("inference.fallback_to_geometric", True)
        )
        if not self._fallback_enabled:
            self._logger.info(
                "Geometric fallback disabled (inference.fallback_to_geometric=false)"
            )

    def set_model(self, model, trainer, predictor=None, backend: str = "torch") -> None:
        """Attach a trained model.

        Args:
            model:     Torch ``GazeMLP`` (used when *predictor* is ``None``).
            trainer:   ``MLPTrainer`` holding the normalisation statistics.
            predictor: Optional external predictor exposing
                       ``predict(normalised_features) -> (N, 2)`` — an
                       ``ONNXInference`` or ``TensorRTInference`` instance.
            backend:   Backend name, reported in ``GazeEstimate.source``.
        """
        self._model = model
        self._trainer = trainer
        self._predictor = predictor
        self._source_name = "mlp" if backend == "torch" else str(backend)

    def set_bias_map(self, bias_map) -> None:
        """Attach a :class:`~gaze_estimation.correction.bias_map.BiasMap`.

        The map is applied to model predictions (``source == "mlp"``) only.
        """
        self._bias_map = bias_map

    def process(self, item: object) -> None:
        if not isinstance(item, GazePacket):
            return

        source = "hold"
        screen_x, screen_y = self._screen_width / 2, self._screen_height / 2
        confidence = item.confidence

        # Bind the collaborators to locals: mypy narrows locals reliably, while
        # attribute narrowing does not always survive intervening calls.
        trainer = self._trainer
        model = self._model
        predictor = self._predictor
        has_model = trainer is not None and (predictor is not None or model is not None)
        # Set when this packet should be projected geometrically.  The model
        # path may also set it, on failure, so that the fallback still applies
        # when the model is confident but throws.
        use_geometric = False

        if confidence >= self._conf_threshold and has_model:
            assert trainer is not None          # implied by has_model
            try:
                x_norm = trainer.features_dict_to_vector(item.features)
                if predictor is not None:
                    out = predictor.predict(x_norm)
                else:
                    assert model is not None    # implied by has_model
                    out = model.predict_numpy(x_norm)
                screen_x = float(out[0, 0]) * self._screen_width
                screen_y = float(out[0, 1]) * self._screen_height
                source = self._source_name
                if self._bias_map is not None:
                    screen_x, screen_y = self._bias_map.apply(
                        screen_x, screen_y, self._screen_width, self._screen_height
                    )
                    screen_x, screen_y = self._clamp(screen_x, screen_y)
            except Exception as exc:
                self._logger.debug("%s inference failed: %s", self._source_name, exc)
                use_geometric = True

        if (
            not use_geometric
            and source == "hold"
            and confidence >= self._geometric_min_conf
        ):
            use_geometric = True

        if use_geometric and self._fallback_enabled:
            # Geometric fallback: project the gaze angles onto the screen plane
            # at the configured viewing distance.  Angles are right/up-positive
            # (see geometry.ray_to_angles), so the screen Y axis is inverted.
            dist = self._fallback_distance_mm
            screen_x = (
                self._screen_width / 2 + math.tan(math.radians(item.gaze_yaw)) * dist
            )
            screen_y = (
                self._screen_height / 2 - math.tan(math.radians(item.gaze_pitch)) * dist
            )
            screen_x, screen_y = self._clamp(screen_x, screen_y)
            source = "geometric"

        self.emit(
            PredictionPacket(
                timestamp=item.timestamp,
                screen_x=screen_x,
                screen_y=screen_y,
                confidence=confidence,
                raw_features=dict(item.features),
                source=source,
            )
        )

    def _clamp(self, x: float, y: float) -> tuple:
        """Clamp a screen position to the visible area."""
        return (
            float(min(max(x, 0.0), self._screen_width - 1)),
            float(min(max(y, 0.0), self._screen_height - 1)),
        )


class FilterStage(StageThread):
    """UKF + One Euro filter + fixation detection on prediction packets."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        config: Config,
        name: str = "filter_thread",
    ) -> None:
        super().__init__(input_queue, output_queue, stop_event, name=name)
        ukf_cfg = config.section("filtering").get("ukf", {})
        oef_cfg = config.section("filtering").get("one_euro", {})
        fix_cfg = config.section("filtering").get("fixation", {})

        self._ukf = UnscentedKalmanFilter(
            process_noise=float(ukf_cfg.get("process_noise", 1.0)),
            measurement_noise=float(ukf_cfg.get("measurement_noise", 5.0)),
            alpha=float(ukf_cfg.get("alpha", 1e-3)),
            beta=float(ukf_cfg.get("beta", 2.0)),
            kappa=float(ukf_cfg.get("kappa", 0.0)),
        )
        self._oef = OneEuroFilter2D(
            min_cutoff=float(oef_cfg.get("min_cutoff", 1.0)),
            beta=float(oef_cfg.get("beta", 0.015)),
            d_cutoff=float(oef_cfg.get("d_cutoff", 1.0)),
        )
        self._fixation = FixationDetector(
            fixation_velocity_threshold=float(fix_cfg.get("fixation_velocity_threshold", 100.0)),
            saccade_velocity_threshold=float(fix_cfg.get("saccade_velocity_threshold", 500.0)),
            blink_ear_threshold=float(fix_cfg.get("blink_ear_threshold", 0.15)),
            min_fixation_duration=float(fix_cfg.get("min_fixation_duration", 0.1)),
        )
        self._prev_t: float = 0.0
        # Fixation state drives the UKF measurement noise: a larger multiplier
        # means "trust the measurement less" (more smoothing).  Disabled until
        # the detector has seen a first sample, otherwise the initial LOST
        # state would freeze the filter.
        self._fixation_ready = False

    def process(self, item: object) -> None:
        if not isinstance(item, PredictionPacket):
            return

        dt = item.timestamp - self._prev_t if self._prev_t > 0 else 0.016
        self._prev_t = item.timestamp
        dt = max(0.001, min(dt, 0.5))

        # Adapt measurement noise to the previous fixation state (FIXATION →
        # more smoothing, SACCADE → more responsive, BLINK/LOST → hold).
        if self._fixation_ready:
            multiplier = self._fixation.get_smoothing_multiplier()
            self._ukf.set_measurement_scale(multiplier)

        # UKF
        self._ukf.predict(dt)
        pos = self._ukf.update(np.array([item.screen_x, item.screen_y]))

        # One Euro
        fx, fy = self._oef.filter(pos[0], pos[1], item.timestamp)

        # Fixation
        left_ear = float(item.raw_features.get("left_ear", 1.0))
        right_ear = float(item.raw_features.get("right_ear", 1.0))
        fix_info = self._fixation.update(fx, fy, item.timestamp, left_ear, right_ear)
        self._fixation_ready = True

        latency_ms = (time.perf_counter() - item.timestamp) * 1000.0

        self.emit(
            GazeEstimate(
                timestamp=item.timestamp,
                screen_x=fx,
                screen_y=fy,
                raw_x=item.screen_x,
                raw_y=item.screen_y,
                velocity=fix_info.velocity,
                confidence=item.confidence,
                fixation_state=fix_info.state,
                fixation_duration=fix_info.duration,
                source=item.source,
                latency_ms=latency_ms,
            )
        )


class GazeEstimationPipeline:
    """Full real-time gaze estimation pipeline.

    Creates and manages all stage threads with bounded queues.

    Usage::

        pipeline = GazeEstimationPipeline(config)
        pipeline.start()
        try:
            while True:
                est = pipeline.get_latest_estimate()
                if est:
                    print(est.screen_x, est.screen_y)
        finally:
            pipeline.stop()
    """

    QUEUE_MAXSIZE = 2
    # Calibration taps need to buffer a burst of gaze packets (the consumer is
    # slower than the 60 fps producer), so they get a deeper queue.
    CALIBRATION_QUEUE_MAXSIZE = 256
    # The preview tap only ever needs the newest MeshPacket.
    PREVIEW_QUEUE_MAXSIZE = 2

    def __init__(
        self,
        config: Optional[Config] = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self._config = config or Config.default()
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._stop_event = threading.Event()

        # Camera intrinsics (can be updated before start())
        cam_w = int(self._config.get("camera.width", 1280))
        cam_h = int(self._config.get("camera.height", 720))
        self._camera_matrix = estimate_camera_matrix(cam_w, cam_h)
        self._dist_coeffs = zero_dist_coeffs()

        # Queues
        self._queues: dict = {
            name: queue.Queue(maxsize=self.QUEUE_MAXSIZE)
            for name in ["frame", "face", "mesh", "pose", "gaze", "prediction", "estimate"]
        }
        # Dedicated calibration tap.  ``GazeGeometryEstimator`` copies every
        # GazePacket it produces onto this queue, so calibration reads an
        # independent stream instead of competing with inference for
        # ``_queues["gaze"]``.
        self._queues["calibration"] = queue.Queue(
            maxsize=self.CALIBRATION_QUEUE_MAXSIZE
        )
        # Preview tap: the newest MeshPacket (frame + mesh + iris), used by the
        # tracker's live overlay window.
        self._queues["preview"] = queue.Queue(maxsize=self.PREVIEW_QUEUE_MAXSIZE)

        self._stages: list = []
        self._fps_counter = FPSCounter()
        self._profiler = LatencyProfiler()
        self._latest_estimate: Optional[GazeEstimate] = None
        self._estimate_lock = threading.Lock()

        # MLP model reference (set after calibration)
        self._inference_stage: Optional[InferenceStage] = None

        # Calibration/model state applied to stages when they are constructed,
        # so that set_model()/update_kappa()/set_bias_map() also work *before*
        # start() (which is how the entry-point scripts use them).
        gaze_cfg = self._config.section("gaze")
        self._kappa_yaw = float(gaze_cfg.get("kappa_yaw", 0.0))
        self._kappa_pitch = float(gaze_cfg.get("kappa_pitch", 0.0))
        self._pending_model = None
        self._pending_trainer = None
        self._pending_predictor = None
        self._pending_backend = "torch"
        self._pending_bias_map = None

        # Optional custom frame source (video file / synthetic) set before start()
        self._source: Optional[StageThread] = None
        self._collector: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, frame_source: Optional[StageThread] = None) -> None:
        """Create and start all pipeline threads.

        Args:
            frame_source: Optional replacement for the webcam capture stage
                          (e.g. :class:`~gaze_estimation.capture.video_source.VideoFileSource`).
                          It must emit ``FramePacket`` objects onto
                          :meth:`frame_queue`.  When ``None`` a
                          :class:`CameraCapture` is created.
        """
        if self._stages:
            raise RuntimeError("Pipeline is already running")
        # Allow a restart after stop() by using a fresh stop event.  Custom
        # sources created before start() keep the old (never-set) event, which
        # is correct for the first run and is why we only reset when set.
        if self._stop_event.is_set():
            self._stop_event = threading.Event()
        if frame_source is not None and self._source is None:
            self._source = frame_source

        cfg = self._config
        c = cfg.section("camera")
        d = cfg.section("detection")
        m = cfg.section("mesh")
        p = cfg.section("pose")

        if self._source is not None:
            cam_thread: StageThread = self._source
        else:
            cam_thread = CameraCapture(
                output_queue=self._queues["frame"],
                stop_event=self._stop_event,
                camera_index=int(c.get("index", 0)),
                width=int(c.get("width", 1280)),
                height=int(c.get("height", 720)),
                fps=int(c.get("fps", 60)),
                camera_matrix=self._camera_matrix,
                dist_coeffs=self._dist_coeffs,
                undistort=bool(c.get("undistort", False)),
                auto_exposure=bool(c.get("auto_exposure", True)),
            )
            self._source = cam_thread

        det_thread = FaceDetector(
            input_queue=self._queues["frame"],
            output_queue=self._queues["face"],
            stop_event=self._stop_event,
            detection_interval=int(d.get("detection_interval", 5)),
            min_confidence=float(d.get("min_confidence", 0.5)),
            model=str(d.get("model", "mediapipe_short")),
        )

        mesh_thread = FaceMeshExtractor(
            input_queue=self._queues["face"],
            output_queue=self._queues["mesh"],
            stop_event=self._stop_event,
            refine_iris=bool(m.get("refine_iris", True)),
            max_num_faces=int(m.get("max_num_faces", 1)),
            min_detection_confidence=float(m.get("min_detection_confidence", 0.5)),
            static_image_mode=bool(m.get("static_image_mode", False)),
            tap_queue=self._queues["preview"],
        )

        pose_thread = HeadPoseEstimator(
            input_queue=self._queues["mesh"],
            output_queue=self._queues["pose"],
            stop_event=self._stop_event,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            solvepnp_method=str(p.get("solvepnp_method", "SOLVEPNP_ITERATIVE")),
            use_ransac=bool(p.get("use_ransac", False)),
        )

        gaze_cfg = cfg.section("gaze")
        gaze_thread = GazeGeometryEstimator(
            input_queue=self._queues["pose"],
            output_queue=self._queues["gaze"],
            stop_event=self._stop_event,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            kappa_yaw=float(gaze_cfg.get("kappa_yaw", 0.0)),
            kappa_pitch=float(gaze_cfg.get("kappa_pitch", 0.0)),
            eyeball_radius=float(gaze_cfg.get("eyeball_radius_mm", 12.0)),
            tap_queue=self._queues["calibration"],
        )
        # Honour kappa updated before start()
        gaze_thread.update_kappa(self._kappa_yaw, self._kappa_pitch)

        inf_thread = InferenceStage(
            input_queue=self._queues["gaze"],
            output_queue=self._queues["prediction"],
            stop_event=self._stop_event,
            config=cfg,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            tap_queue=self._queues["calibration"],
        )
        self._inference_stage = inf_thread
        # Apply anything that was configured before start()
        if self._pending_trainer is not None:
            inf_thread.set_model(
                self._pending_model,
                self._pending_trainer,
                predictor=self._pending_predictor,
                backend=self._pending_backend,
            )
        if self._pending_bias_map is not None:
            inf_thread.set_bias_map(self._pending_bias_map)

        filter_thread = FilterStage(
            input_queue=self._queues["prediction"],
            output_queue=self._queues["estimate"],
            stop_event=self._stop_event,
            config=cfg,
        )

        # Collector thread that puts estimates into _latest_estimate
        def _collect() -> None:
            while not self._stop_event.is_set():
                try:
                    est: GazeEstimate = self._queues["estimate"].get(timeout=0.1)
                    with self._estimate_lock:
                        self._latest_estimate = est
                    self._fps_counter.tick()
                except queue.Empty:
                    pass

        collector = threading.Thread(target=_collect, name="collector", daemon=True)
        self._collector = collector

        self._stages = [
            cam_thread, det_thread, mesh_thread,
            pose_thread, gaze_thread, inf_thread, filter_thread,
        ]
        for s in self._stages:
            s.start()
        collector.start()
        _logger.info(
            "Pipeline started (%d stages) | Device: %s",
            len(self._stages),
            describe_device(),
        )

    def stop(self) -> None:
        """Signal and join all threads (including the estimate collector)."""
        self._stop_event.set()
        for s in self._stages:
            s.join(timeout=3.0)
        still_running = [s.name for s in self._stages if s.is_alive()]
        if still_running:
            _logger.warning("Stages still alive after stop timeout: %s", still_running)
        if self._collector is not None:
            self._collector.join(timeout=1.0)
            self._collector = None
        self._stages = []
        # Drop the cached source so a subsequent start() builds a fresh capture
        # stage instead of reusing a dead thread.
        self._source = None
        _logger.info("Pipeline stopped")

    # ── Public API ────────────────────────────────────────────────────────

    def get_latest_estimate(self) -> Optional[GazeEstimate]:
        """Non-blocking read of the most recent gaze estimate."""
        with self._estimate_lock:
            return self._latest_estimate

    @property
    def frame_queue(self) -> queue.Queue:
        """Queue of raw :class:`FramePacket`s produced by the capture stage.

        Pass this to a custom frame source when calling :meth:`start` with
        ``frame_source=...``.
        """
        return self._queues["frame"]

    @property
    def stop_event(self) -> threading.Event:
        """Shared shutdown event (needed by custom frame sources)."""
        return self._stop_event

    def get_gaze_queue(self) -> queue.Queue:
        """Return the dedicated calibration tap queue of ``GazePacket``s.

        This is an independent copy of the gaze stream produced by the
        gaze-geometry stage, so calibration never steals packets from the live
        pipeline.
        """
        return self._queues["calibration"]

    def get_latest_gaze_packet(self) -> Optional[GazePacket]:
        """Drain the gaze tap and return the newest :class:`GazePacket`.

        Used by implicit (click-based) calibration, which needs the current
        feature vector.  Note that this *consumes* the tap, so it should not be
        combined with calibration in the same process.
        """
        return _drain_latest(self._queues["calibration"])

    def get_latest_mesh_packet(self) -> Optional[MeshPacket]:
        """Drain the preview tap and return the newest :class:`MeshPacket`.

        Provides the frame + face-mesh + iris data used to render the live
        overlay window.
        """
        return _drain_latest(self._queues["preview"])

    @property
    def fps(self) -> float:
        """Output FPS of the full pipeline."""
        return self._fps_counter.get()

    def set_model(self, model, trainer, predictor=None, backend: str = "torch") -> None:
        """Attach a trained model to the inference stage.

        Works both before :meth:`start` (stored and applied when the stage is
        created) and at runtime.

        Args:
            model:     Torch ``GazeMLP`` (used when *predictor* is ``None``).
            trainer:   ``MLPTrainer`` holding the normalisation statistics.
            predictor: Optional external backend (``ONNXInference`` /
                       ``TensorRTInference``) with a ``predict`` method.
            backend:   Backend name, reported in ``GazeEstimate.source``.
        """
        self._pending_model = model
        self._pending_trainer = trainer
        self._pending_predictor = predictor
        self._pending_backend = backend
        if self._inference_stage is not None:
            self._inference_stage.set_model(
                model, trainer, predictor=predictor, backend=backend
            )

    def set_bias_map(self, bias_map) -> None:
        """Attach a :class:`BiasMap` applied to model predictions.

        Works both before :meth:`start` and at runtime.
        """
        self._pending_bias_map = bias_map
        if self._inference_stage is not None:
            self._inference_stage.set_bias_map(bias_map)

    def update_kappa(self, kappa_yaw: float, kappa_pitch: float) -> None:
        """Update kappa angles (works before :meth:`start` and at runtime)."""
        self._kappa_yaw = float(kappa_yaw)
        self._kappa_pitch = float(kappa_pitch)
        for s in self._stages:
            if isinstance(s, GazeGeometryEstimator):
                s.update_kappa(self._kappa_yaw, self._kappa_pitch)
                break

    def set_camera_intrinsics(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """Set camera intrinsics.

        Safe to call before :meth:`start` (the values are used when the stages
        are constructed) *and* at runtime (each affected stage is updated in
        place — the matrices are read once per frame).
        """
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)

        if not self._stages:
            return
        for stage in self._stages:
            update = getattr(stage, "set_camera_intrinsics", None)
            if callable(update) and isinstance(
                stage, (HeadPoseEstimator, GazeGeometryEstimator, CameraCapture)
            ):
                update(self._camera_matrix, self._dist_coeffs)
