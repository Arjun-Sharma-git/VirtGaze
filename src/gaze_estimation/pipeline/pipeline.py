"""GazeEstimationPipeline: full end-to-end pipeline orchestrator."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

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
    GazeEstimate, GazePacket, GazeState,
    PredictionPacket,
)
from gaze_estimation.pipeline.thread_base import StageThread, put_or_drop
from gaze_estimation.pose.head_pose import HeadPoseEstimator
from gaze_estimation.utils.camera_calibration import estimate_camera_matrix, zero_dist_coeffs
from gaze_estimation.utils.device import describe_device, get_onnx_providers
from gaze_estimation.utils.logging import get_logger
from gaze_estimation.utils.timing import FPSCounter, LatencyProfiler

_logger = get_logger("pipeline")


class InferenceStage(StageThread):
    """Screen coordinate inference from GazePackets using MLP or geometric fallback.

    If an MLP model is available and confidence is high, use MLP; otherwise
    fall back the geometric gaze-to-screen projection.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        config: Config,
        screen_width: int = 1920,
        screen_height: int = 1080,
        name: str = "inference_thread",
    ) -> None:
        super().__init__(input_queue, output_queue, stop_event, name=name)
        self._config = config
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._model = None           # Set via set_model()
        self._trainer = None         # Set via set_trainer()
        self._conf_threshold = float(config.get("inference.confidence_threshold", 0.7))

    def set_model(self, model, trainer) -> None:
        """Attach a trained GazeMLP + MLPTrainer (thread-safe write)."""
        self._model = model
        self._trainer = trainer

    def process(self, item: object) -> None:
        if not isinstance(item, GazePacket):
            return

        source = "hold"
        screen_x, screen_y = self._screen_width / 2, self._screen_height / 2
        confidence = item.confidence

        if confidence >= self._conf_threshold and self._model is not None and self._trainer is not None:
            try:
                x_norm = self._trainer.features_dict_to_vector(item.features)
                out = self._model.predict_numpy(x_norm)
                screen_x = float(out[0, 0]) * self._screen_width
                screen_y = float(out[0, 1]) * self._screen_height
                source = "mlp"
            except Exception as exc:
                self._logger.debug("MLP inference failed: %s", exc)
                source = "geometric"
        elif confidence >= 0.4:
            # Geometric fallback: forward gaze angles → screen
            yaw_deg = item.gaze_yaw
            pitch_deg = item.gaze_pitch
            import math
            screen_x = (math.tan(math.radians(yaw_deg)) * 600 + 0) + self._screen_width / 2
            screen_y = (math.tan(math.radians(pitch_deg)) * 600 + 0) + self._screen_height / 2
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

    def process(self, item: object) -> None:
        if not isinstance(item, PredictionPacket):
            return

        dt = item.timestamp - self._prev_t if self._prev_t > 0 else 0.016
        self._prev_t = item.timestamp
        dt = max(0.001, min(dt, 0.5))

        # UKF
        self._ukf.predict(dt)
        pos = self._ukf.update(np.array([item.screen_x, item.screen_y]))

        # One Euro
        fx, fy = self._oef.filter(pos[0], pos[1], item.timestamp)

        # Fixation
        left_ear = float(item.raw_features.get("left_ear", 1.0))
        right_ear = float(item.raw_features.get("right_ear", 1.0))
        fix_info = self._fixation.update(fx, fy, item.timestamp, left_ear, right_ear)

        latency_ms = (time.time() - item.timestamp) * 1000.0

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

        self._stages: list = []
        self._fps_counter = FPSCounter()
        self._profiler = LatencyProfiler()
        self._latest_estimate: Optional[GazeEstimate] = None
        self._estimate_lock = threading.Lock()

        # MLP model reference (set after calibration)
        self._inference_stage: Optional[InferenceStage] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create and start all pipeline threads."""
        cfg = self._config
        c = cfg.section("camera")
        d = cfg.section("detection")
        m = cfg.section("mesh")

        cam_thread = CameraCapture(
            output_queue=self._queues["frame"],
            stop_event=self._stop_event,
            camera_index=int(c.get("index", 0)),
            width=int(c.get("width", 1280)),
            height=int(c.get("height", 720)),
            fps=int(c.get("fps", 60)),
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
        )

        det_thread = FaceDetector(
            input_queue=self._queues["frame"],
            output_queue=self._queues["face"],
            stop_event=self._stop_event,
            detection_interval=int(d.get("detection_interval", 5)),
            min_confidence=float(d.get("min_confidence", 0.5)),
        )

        mesh_thread = FaceMeshExtractor(
            input_queue=self._queues["face"],
            output_queue=self._queues["mesh"],
            stop_event=self._stop_event,
            refine_iris=bool(m.get("refine_iris", True)),
            max_num_faces=int(m.get("max_num_faces", 1)),
            min_detection_confidence=float(m.get("min_detection_confidence", 0.5)),
        )

        pose_thread = HeadPoseEstimator(
            input_queue=self._queues["mesh"],
            output_queue=self._queues["pose"],
            stop_event=self._stop_event,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
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
        )

        inf_thread = InferenceStage(
            input_queue=self._queues["gaze"],
            output_queue=self._queues["prediction"],
            stop_event=self._stop_event,
            config=cfg,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
        )
        self._inference_stage = inf_thread

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
        """Signal and join all threads."""
        self._stop_event.set()
        for s in self._stages:
            s.join(timeout=3.0)
        _logger.info("Pipeline stopped")

    # ── Public API ────────────────────────────────────────────────────────

    def get_latest_estimate(self) -> Optional[GazeEstimate]:
        """Non-blocking read of the most recent gaze estimate."""
        with self._estimate_lock:
            return self._latest_estimate

    def get_gaze_queue(self) -> queue.Queue:
        """Return the GazePacket queue (used by calibration engine)."""
        return self._queues["gaze"]

    @property
    def fps(self) -> float:
        """Output FPS of the full pipeline."""
        return self._fps_counter.get()

    def set_model(self, model, trainer) -> None:
        """Attach a trained GazeMLP to the inference stage."""
        if self._inference_stage is not None:
            self._inference_stage.set_model(model, trainer)

    def update_kappa(self, kappa_yaw: float, kappa_pitch: float) -> None:
        """Update kappa angles in the gaze geometry stage."""
        for s in self._stages:
            if isinstance(s, GazeGeometryEstimator):
                s.update_kappa(kappa_yaw, kappa_pitch)
                break

    def set_camera_intrinsics(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """Update camera intrinsics before starting (or at runtime)."""
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)
