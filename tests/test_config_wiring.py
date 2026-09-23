"""Config-to-object wiring tests.

The drift guard (:mod:`tests.test_config_drift`) proves every key in
``default_config.yaml`` is *read*.  These tests prove the value that is read
actually reaches the object that uses it — the other half of the bug class
where a documented setting had no effect.

Where a component needs hardware (the camera), the pipeline's call site is
verified by substituting a recording stub for the capture stage, so the wiring
is covered without a webcam.
"""
from __future__ import annotations

import math
import queue
import threading
import time

import cv2
import numpy as np
import pytest

from gaze_estimation.calibration.quick_calibration import QuickCalibration
from gaze_estimation.capture.camera import CameraCapture
from gaze_estimation.config.config import Config
from gaze_estimation.detection.face_detector import FaceDetector, resolve_detection_model
from gaze_estimation.mesh.face_mesh import FaceMeshExtractor
from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline, InferenceStage
from gaze_estimation.pipeline.schemas import FEATURE_DIM, FEATURE_KEYS, GazePacket
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.pose.head_pose import HeadPoseEstimator, resolve_solvepnp_flags
from gaze_estimation.visualization.overlay import OverlayRenderer

SCREEN_W, SCREEN_H = 1920, 1080


def _gaze_packet(yaw: float = 10.0, pitch: float = 5.0, confidence: float = 0.9):
    return GazePacket(
        timestamp=time.perf_counter(),
        gaze_ray_left=None,
        gaze_ray_right=None,
        gaze_yaw=yaw,
        gaze_pitch=pitch,
        features={k: 0.0 for k in FEATURE_KEYS},
        head_pose=None,
        confidence=confidence,
    )


def _inference_stage(config: Config):
    """Build a standalone InferenceStage (no thread is started)."""
    out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=out,
        stop_event=threading.Event(),
        config=config,
        screen_width=SCREEN_W,
        screen_height=SCREEN_H,
    )
    return stage, out


# ── inference.fallback_to_geometric ──────────────────────────────────────────

class _ExplodingPredictor:
    """Stands in for an unavailable ONNX/TensorRT backend."""

    def predict(self, features):
        raise RuntimeError("backend unavailable")


def test_fallback_disabled_holds_at_screen_centre():
    cfg = Config.default()
    cfg.set("inference.fallback_to_geometric", False)
    stage, out = _inference_stage(cfg)

    stage.process(_gaze_packet())

    packet = out.get_nowait()
    assert packet.source == "hold"
    assert packet.screen_x == pytest.approx(SCREEN_W / 2)
    assert packet.screen_y == pytest.approx(SCREEN_H / 2)


def test_fallback_enabled_projects_geometrically():
    stage, out = _inference_stage(Config.default())

    stage.process(_gaze_packet(yaw=10.0, pitch=5.0))

    packet = out.get_nowait()
    assert packet.source == "geometric"
    distance = 600.0                                   # default fallback distance
    assert packet.screen_x == pytest.approx(
        SCREEN_W / 2 + math.tan(math.radians(10.0)) * distance
    )
    # pitch is up-positive, so a positive pitch moves *up* the screen
    assert packet.screen_y == pytest.approx(
        SCREEN_H / 2 - math.tan(math.radians(5.0)) * distance
    )


def test_fallback_distance_comes_from_config():
    cfg = Config.default()
    cfg.set("inference.fallback_distance_mm", 300.0)
    stage, out = _inference_stage(cfg)

    stage.process(_gaze_packet(yaw=10.0, pitch=0.0))

    assert out.get_nowait().screen_x == pytest.approx(
        SCREEN_W / 2 + math.tan(math.radians(10.0)) * 300.0
    )


def test_model_failure_falls_back_when_enabled():
    stage, out = _inference_stage(Config.default())
    stage.set_model(None, MLPTrainer(), predictor=_ExplodingPredictor(), backend="onnx")

    stage.process(_gaze_packet())

    assert out.get_nowait().source == "geometric"


def test_model_failure_holds_when_fallback_disabled():
    cfg = Config.default()
    cfg.set("inference.fallback_to_geometric", False)
    stage, out = _inference_stage(cfg)
    stage.set_model(None, MLPTrainer(), predictor=_ExplodingPredictor(), backend="onnx")

    stage.process(_gaze_packet())

    packet = out.get_nowait()
    assert packet.source == "hold"
    assert packet.screen_x == pytest.approx(SCREEN_W / 2)


def test_low_confidence_holds_below_geometric_threshold():
    cfg = Config.default()
    cfg.set("inference.geometric_min_confidence", 0.6)
    stage, out = _inference_stage(cfg)

    stage.process(_gaze_packet(confidence=0.3))

    assert out.get_nowait().source == "hold"


# ── mlp.* and inference.device ───────────────────────────────────────────────

def test_trainer_from_config_reads_mlp_section():
    cfg = Config.default()
    cfg.set("mlp.hidden_dims", [8, 4])
    cfg.set("mlp.learning_rate", 5e-4)
    cfg.set("mlp.weight_decay", 2e-4)
    cfg.set("mlp.epochs", 7)
    cfg.set("mlp.batch_size", 8)
    cfg.set("mlp.early_stopping_patience", 3)
    cfg.set("mlp.input_dim", FEATURE_DIM)

    trainer = MLPTrainer.from_config(cfg)

    assert trainer.input_dim == FEATURE_DIM
    assert trainer.hidden_dims == [8, 4]
    assert trainer.lr == pytest.approx(5e-4)
    assert trainer.weight_decay == pytest.approx(2e-4)
    assert trainer.epochs == 7
    assert trainer.batch_size == 8
    assert trainer.patience == 3


def test_trainer_load_keeps_architecture_and_configures_optimiser(tmp_path):
    save_path = str(tmp_path / "weights.pt")
    trainer = MLPTrainer(epochs=1)
    model = GazeMLP(input_dim=FEATURE_DIM, hidden_dims=[8, 4])
    trainer.save(model, save_path)

    cfg = Config.default()
    cfg.set("mlp.learning_rate", 3e-4)
    cfg.set("mlp.epochs", 11)

    loaded, restored = MLPTrainer().load(save_path, config=cfg)

    assert loaded.input_dim == FEATURE_DIM
    assert loaded.hidden_dims == [8, 4]        # architecture from the checkpoint
    assert restored.hidden_dims == [8, 4]
    assert restored.input_dim == FEATURE_DIM
    assert restored.lr == pytest.approx(3e-4)  # optimisation from the config
    assert restored.epochs == 11


# ── detection.model / mesh.static_image_mode ─────────────────────────────────

def test_resolve_detection_model_maps_config_names():
    assert resolve_detection_model("mediapipe_short") == ("mediapipe_short", 0)
    assert resolve_detection_model("MediaPipe_Full") == ("mediapipe_full", 1)
    assert resolve_detection_model("nonsense") == ("mediapipe_short", 0)


def test_face_detector_uses_configured_model():
    detector = FaceDetector(
        queue.Queue(), queue.Queue(), threading.Event(), model="mediapipe_full"
    )
    assert detector._model_selection == 1
    assert detector._model_name == "mediapipe_full"


def test_face_mesh_static_image_mode_flag():
    mesh = FaceMeshExtractor(
        queue.Queue(), queue.Queue(), threading.Event(), static_image_mode=True
    )
    assert mesh._static_image_mode is True


# ── pose.solvepnp_method / pose.use_ransac ───────────────────────────────────

def test_resolve_solvepnp_flags_accepts_names_and_falls_back():
    assert resolve_solvepnp_flags("SOLVEPNP_ITERATIVE") == cv2.SOLVEPNP_ITERATIVE
    assert resolve_solvepnp_flags("epnp") == cv2.SOLVEPNP_EPNP
    assert resolve_solvepnp_flags("not_a_solver") == cv2.SOLVEPNP_ITERATIVE


def test_head_pose_estimator_honours_solver_options(
    camera_matrix_640x480, dist_coeffs_zero
):
    estimator = HeadPoseEstimator(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
        camera_matrix=camera_matrix_640x480,
        dist_coeffs=dist_coeffs_zero,
        solvepnp_method="SOLVEPNP_EPNP",
        use_ransac=True,
    )
    assert estimator._flags == cv2.SOLVEPNP_EPNP
    assert estimator._use_ransac is True


# ── quick_calibration.points ─────────────────────────────────────────────────

def test_quick_calibration_supports_five_and_nine_point_layouts():
    five = QuickCalibration(SCREEN_W, SCREEN_H, points=5)
    assert len(five.generate_targets()) == 5

    nine = QuickCalibration(SCREEN_W, SCREEN_H, points=9)
    targets = nine.generate_targets()
    assert len(targets) == 9
    assert (SCREEN_W * 0.5, SCREEN_H * 0.5) in targets


def test_quick_calibration_falls_back_to_five_points():
    odd = QuickCalibration(SCREEN_W, SCREEN_H, points=7)

    assert len(odd.generate_targets()) == 5
    assert odd.points == 5


# ── logging.log_fps / logging.log_latency ────────────────────────────────────

def test_overlay_hud_flags_toggle_the_text():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)

    quiet = OverlayRenderer(draw_fps=False, draw_latency=False)
    assert not quiet.render(frame, None, None, fps=60.0, latency_ms=5.0).any()

    loud = OverlayRenderer(draw_fps=True, draw_latency=True)
    assert loud.render(frame, None, None, fps=60.0, latency_ms=5.0).any()


# ── camera.auto_exposure ─────────────────────────────────────────────────────

def test_camera_capture_stores_auto_exposure_flag():
    disabled = CameraCapture(
        output_queue=queue.Queue(), stop_event=threading.Event(), auto_exposure=False
    )
    assert disabled._auto_exposure is False

    default = CameraCapture(output_queue=queue.Queue(), stop_event=threading.Event())
    assert default._auto_exposure is True


# ── pipeline call sites ──────────────────────────────────────────────────────

class _NullSource(StageThread):
    """A frame source that emits nothing (avoids needing a camera)."""

    def __init__(self, output_queue, stop_event):
        super().__init__(
            input_queue=None,
            output_queue=output_queue,
            stop_event=stop_event,
            name="null_source",
        )

    def process(self, item) -> None:
        time.sleep(0.01)


def test_pipeline_threads_camera_options_into_capture_stage(monkeypatch):
    """camera.undistort / camera.auto_exposure reach the capture stage."""
    import gaze_estimation.pipeline.pipeline as pipeline_module

    captured: dict = {}

    class _RecordingCamera(StageThread):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(
                input_queue=None,
                output_queue=kwargs["output_queue"],
                stop_event=kwargs["stop_event"],
                name="recording_camera",
            )

        def process(self, item) -> None:
            time.sleep(0.01)

    monkeypatch.setattr(pipeline_module, "CameraCapture", _RecordingCamera)

    cfg = Config.default()
    cfg.set("camera.auto_exposure", False)
    cfg.set("camera.undistort", True)

    pipeline = GazeEstimationPipeline(config=cfg)
    try:
        pipeline.start()
    finally:
        pipeline.stop()

    assert captured["auto_exposure"] is False
    assert captured["undistort"] is True


def test_pipeline_threads_config_into_detection_mesh_and_pose_stages():
    cfg = Config.default()
    cfg.set("detection.model", "mediapipe_full")
    cfg.set("mesh.static_image_mode", True)
    cfg.set("pose.solvepnp_method", "SOLVEPNP_EPNP")
    cfg.set("pose.use_ransac", True)
    cfg.set("inference.fallback_to_geometric", False)

    pipeline = GazeEstimationPipeline(config=cfg)
    source = _NullSource(pipeline.frame_queue, pipeline.stop_event)
    try:
        pipeline.start(frame_source=source)

        detectors = [s for s in pipeline._stages if isinstance(s, FaceDetector)]
        meshes = [s for s in pipeline._stages if isinstance(s, FaceMeshExtractor)]
        poses = [s for s in pipeline._stages if isinstance(s, HeadPoseEstimator)]

        assert detectors[0]._model_selection == 1
        assert meshes[0]._static_image_mode is True
        assert poses[0]._flags == cv2.SOLVEPNP_EPNP
        assert poses[0]._use_ransac is True
        assert pipeline._inference_stage._fallback_enabled is False
    finally:
        pipeline.stop()
