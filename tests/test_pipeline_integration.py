"""Integration tests: config loading, schema validation, adaptation buffer."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gaze_estimation.config.config import Config
from gaze_estimation.pipeline.schemas import (
    FEATURE_DIM,
    FEATURE_KEYS,
    FramePacket,
    GazeState,
)

# ── Config tests ─────────────────────────────────────────────────────────────

def test_default_config_loads():
    cfg = Config.default()
    assert cfg.get("camera.fps") == 60
    assert cfg.get("mlp.input_dim") == 34


def test_config_dot_access():
    cfg = Config.default()
    assert cfg.get("filtering.one_euro.min_cutoff") == 1.0
    assert cfg.get("missing.key", "fallback") == "fallback"


def test_config_set():
    cfg = Config.default()
    cfg.set("camera.fps", 120)
    assert cfg.get("camera.fps") == 120


def test_config_section():
    cfg = Config.default()
    section = cfg.section("calibration")
    assert "grid_cols" in section


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_frame_packet():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pkt = FramePacket(timestamp=time.time(), frame=frame, frame_id=0)
    assert pkt.frame.shape == (480, 640, 3)
    assert pkt.frame_id == 0


def test_gaze_state_enum():
    assert GazeState.FIXATION.value == "fixation"
    assert GazeState.SACCADE.value == "saccade"


def test_feature_keys_unique():
    assert len(FEATURE_KEYS) == len(set(FEATURE_KEYS)), "Duplicate FEATURE_KEYS!"


# ── Adaptation buffer integration ─────────────────────────────────────────────

def test_adaptation_buffer_add_get():
    from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
    buf = AdaptationBuffer(max_size=100)
    feats = {k: 0.5 for k in FEATURE_KEYS}
    for i in range(10):
        buf.add(feats, float(100 + i), float(200 + i))
    X, Y = buf.get_batch()
    assert X.shape == (10, FEATURE_DIM)
    assert Y.shape == (10, 2)
    assert len(buf) == 10


def test_adaptation_buffer_max_size():
    from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
    buf = AdaptationBuffer(max_size=5)
    feats = {k: 0.0 for k in FEATURE_KEYS}
    for i in range(20):
        buf.add(feats, float(i), float(i))
    assert len(buf) == 5  # Ring buffer capped at 5


def test_adaptation_buffer_new_counter():
    from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
    buf = AdaptationBuffer()
    feats = {k: 0.0 for k in FEATURE_KEYS}
    buf.add(feats, 0, 0)
    buf.add(feats, 0, 0)
    assert buf.count_new_since_last_train() == 2
    buf.mark_trained()
    assert buf.count_new_since_last_train() == 0


# ── Pipeline wiring: calibration tap + frame source ────────────────────────────

def test_pipeline_exposes_distinct_calibration_tap():
    """Calibration must read its own queue, not the live inference queue."""
    import threading

    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline

    pipeline = GazeEstimationPipeline(config=Config.default())

    tap = pipeline.get_gaze_queue()
    assert tap is not pipeline._queues["gaze"], "calibration steals inference packets"
    assert tap.maxsize == GazeEstimationPipeline.CALIBRATION_QUEUE_MAXSIZE
    assert pipeline.frame_queue is pipeline._queues["frame"]
    assert isinstance(pipeline.stop_event, threading.Event)


def test_inference_stage_tap_receives_emitted_prediction():
    """A stage's tap queue receives a copy of what the stage *emits*."""
    import queue
    import threading

    from gaze_estimation.pipeline.pipeline import InferenceStage
    from gaze_estimation.pipeline.schemas import GazePacket, PredictionPacket, empty_features

    tap: queue.Queue = queue.Queue()
    out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=out,
        stop_event=threading.Event(),
        config=Config.default(),
        tap_queue=tap,
    )
    packet = GazePacket(
        timestamp=1.0,
        gaze_ray_left=None,
        gaze_ray_right=None,
        gaze_yaw=0.0,
        gaze_pitch=0.0,
        features=empty_features(),
        head_pose=None,
        confidence=0.9,
    )

    stage.process(packet)

    assert tap.qsize() == 1
    tapped = tap.get_nowait()
    assert isinstance(tapped, PredictionPacket)
    assert tapped is out.get_nowait()


def test_inference_stage_geometric_fallback_clamps_and_inverts_pitch():
    """Fallback projects onto the screen plane, inverted in Y and clamped."""
    import queue
    import threading

    from gaze_estimation.pipeline.pipeline import InferenceStage
    from gaze_estimation.pipeline.schemas import GazePacket, empty_features

    out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=out,
        stop_event=threading.Event(),
        config=Config.default(),
        screen_width=1920,
        screen_height=1080,
    )

    def _packet(yaw, pitch):
        return GazePacket(
            timestamp=1.0,
            gaze_ray_left=None,
            gaze_ray_right=None,
            gaze_yaw=yaw,
            gaze_pitch=pitch,
            features=empty_features(),
            head_pose=None,
            confidence=0.5,          # below MLP threshold, above geometric minimum
        )

    # Looking up (positive pitch) must move the estimate *up* the screen
    stage.process(_packet(0.0, 30.0))
    up = out.get_nowait()
    assert up.source == "geometric"
    assert up.screen_y < 540

    # Extreme angles must be clamped into the visible area
    stage.process(_packet(80.0, -80.0))
    extreme = out.get_nowait()
    assert 0.0 <= extreme.screen_x <= 1919.0
    assert 0.0 <= extreme.screen_y <= 1079.0


def test_inference_stage_uses_external_predictor_and_reports_backend():
    import queue
    import threading

    from gaze_estimation.pipeline.pipeline import InferenceStage
    from gaze_estimation.pipeline.schemas import GazePacket, empty_features

    class _Predictor:
        def predict(self, features):
            return np.array([[0.25, 0.75]], dtype=np.float32)

    class _Trainer:
        def features_dict_to_vector(self, features):
            return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=out,
        stop_event=threading.Event(),
        config=Config.default(),
        screen_width=1920,
        screen_height=1080,
    )
    stage.set_model(None, _Trainer(), predictor=_Predictor(), backend="onnx")

    stage.process(
        GazePacket(
            timestamp=1.0,
            gaze_ray_left=None,
            gaze_ray_right=None,
            gaze_yaw=0.0,
            gaze_pitch=0.0,
            features=empty_features(),
            head_pose=None,
            confidence=0.9,
        )
    )

    prediction = out.get_nowait()
    assert prediction.source == "onnx"
    assert prediction.screen_x == pytest.approx(0.25 * 1920)
    assert prediction.screen_y == pytest.approx(0.75 * 1080)


def test_inference_stage_applies_bias_map():
    import queue
    import threading

    from gaze_estimation.correction.bias_map import BiasMap
    from gaze_estimation.pipeline.pipeline import InferenceStage
    from gaze_estimation.pipeline.schemas import GazePacket, empty_features

    class _Predictor:
        def predict(self, features):
            return np.array([[0.5, 0.5]], dtype=np.float32)

    class _Trainer:
        def features_dict_to_vector(self, features):
            return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    # A uniform +50 px x-correction across the grid
    bias_map = BiasMap(cols=4, rows=4)
    for x in (0, 480, 960, 1440, 1919):
        for y in (0, 540, 1079):
            bias_map.add_sample(float(x), float(y), 50.0, 0.0, 1920, 1080)

    out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=out,
        stop_event=threading.Event(),
        config=Config.default(),
        screen_width=1920,
        screen_height=1080,
    )
    stage.set_model(None, _Trainer(), predictor=_Predictor(), backend="onnx")
    stage.set_bias_map(bias_map)

    stage.process(
        GazePacket(
            timestamp=1.0,
            gaze_ray_left=None,
            gaze_ray_right=None,
            gaze_yaw=0.0,
            gaze_pitch=0.0,
            features=empty_features(),
            head_pose=None,
            confidence=0.9,
        )
    )

    prediction = out.get_nowait()
    assert prediction.screen_x == pytest.approx(960.0 + 50.0, abs=2.0)


def test_inference_stage_without_tap_queue_is_safe():
    import queue
    import threading

    from gaze_estimation.pipeline.pipeline import InferenceStage
    from gaze_estimation.pipeline.schemas import GazePacket, empty_features

    stage = InferenceStage(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
        config=Config.default(),
    )
    stage.process(
        GazePacket(
            timestamp=1.0,
            gaze_ray_left=None,
            gaze_ray_right=None,
            gaze_yaw=0.0,
            gaze_pitch=0.0,
            features=empty_features(),
            head_pose=None,
            confidence=0.9,
        )
    )  # must not raise


def test_latest_packet_getters_drain_taps():
    """The calibration/preview taps can be polled for the newest packet."""
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.pipeline.schemas import GazePacket, empty_features

    pipeline = GazeEstimationPipeline(config=Config.default())
    for i in range(3):
        pipeline._queues["calibration"].put(
            GazePacket(
                timestamp=float(i),
                gaze_ray_left=None,
                gaze_ray_right=None,
                gaze_yaw=0.0,
                gaze_pitch=0.0,
                features=empty_features(),
                head_pose=None,
                confidence=0.5,
            )
        )

    latest = pipeline.get_latest_gaze_packet()
    assert latest is not None
    assert latest.timestamp == 2.0                 # newest of the three
    assert pipeline._queues["calibration"].empty()  # drained
    assert pipeline.get_latest_gaze_packet() is None
    assert pipeline.get_latest_mesh_packet() is None


def test_pipeline_applies_model_and_kappa_set_before_start_integration():
    """set_model()/update_kappa() before start() must not be silently dropped."""
    import time

    from gaze_estimation.gaze.gaze_geometry import GazeGeometryEstimator
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.pipeline.thread_base import StageThread

    class _NullSource(StageThread):
        def __init__(self, output_queue, stop_event):
            super().__init__(
                input_queue=None,
                output_queue=output_queue,
                stop_event=stop_event,
                name="null_source",
            )

        def process(self, item) -> None:
            time.sleep(0.01)

    class _Trainer:
        feature_mean = None
        feature_std = None

        def features_dict_to_vector(self, features):
            return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    pipeline = GazeEstimationPipeline(config=Config.default())
    source = _NullSource(pipeline.frame_queue, pipeline.stop_event)
    pipeline.set_model(object(), _Trainer())
    pipeline.update_kappa(1.5, -2.5)
    try:
        pipeline.start(frame_source=source)
        assert pipeline._inference_stage is not None
        assert pipeline._inference_stage._trainer is not None

        gaze_stages = [
            s for s in pipeline._stages if isinstance(s, GazeGeometryEstimator)
        ]
        assert len(gaze_stages) == 1
        assert gaze_stages[0]._kappa_yaw == pytest.approx(1.5)
        assert gaze_stages[0]._kappa_pitch == pytest.approx(-2.5)

        # set_camera_intrinsics is safe at runtime and reaches the stages
        cm = np.eye(3, dtype=np.float64) * 500.0
        cm[2, 2] = 1.0
        pipeline.set_camera_intrinsics(cm, np.zeros(5))
        assert gaze_stages[0]._camera_matrix[0, 0] == pytest.approx(500.0)
    finally:
        pipeline.stop()

    # Double start after stop is allowed (fresh stop event)
    pipeline.start(frame_source=_NullSource(pipeline.frame_queue, pipeline.stop_event))
    pipeline.stop()


def test_pipeline_runs_with_custom_frame_source_integration():
    """A synthetic (non-camera) source can drive the full pipeline."""
    import time

    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.pipeline.thread_base import StageThread

    class _NullSource(StageThread):
        def __init__(self, output_queue, stop_event):
            super().__init__(
                input_queue=None,
                output_queue=output_queue,
                stop_event=stop_event,
                name="null_source",
            )

        def process(self, item) -> None:
            time.sleep(0.01)

    pipeline = GazeEstimationPipeline(config=Config.default())
    source = _NullSource(pipeline.frame_queue, pipeline.stop_event)
    try:
        pipeline.start(frame_source=source)
        assert pipeline._stages, "pipeline did not start"
        time.sleep(0.3)
    finally:
        pipeline.stop()

    assert pipeline._stages == []