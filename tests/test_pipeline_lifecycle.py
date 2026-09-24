"""Tests for InferenceStage's default path and the pipeline's runtime API.

The plain torch model path — ``predictor is None`` with a model attached, which
is what ``inference.backend: torch`` produces — was untested, as were the
pipeline's runtime setters and the collector thread that publishes
``get_latest_estimate()``.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from gaze_estimation.config.config import Config
from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline, InferenceStage
from gaze_estimation.pipeline.schemas import (
    FEATURE_DIM,
    GazeEstimate,
    GazePacket,
    GazeState,
    empty_features,
)
from gaze_estimation.pipeline.thread_base import StageThread


class _Trainer:
    """Stands in for MLPTrainer's normalisation helper."""

    feature_mean = None
    feature_std = None

    def __init__(self) -> None:
        self.calls = 0

    def features_dict_to_vector(self, features):
        self.calls += 1
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)


class _Model:
    """Minimal stand-in for GazeMLP exposing predict_numpy."""

    def __init__(self, output=(0.5, 0.25)) -> None:
        self.output = np.array([output], dtype=np.float32)
        self.calls = 0

    def predict_numpy(self, features):
        self.calls += 1
        return self.output


class _Predictor:
    def __init__(self, output=(0.1, 0.2)) -> None:
        self.output = np.array([output], dtype=np.float32)
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return self.output


def _stage(**config_overrides):
    config = Config.default()
    for dotted, value in config_overrides.items():
        section, _, key = dotted.partition(".")
        config.raw.setdefault(section, {})[key] = value
    q_out: queue.Queue = queue.Queue()
    stage = InferenceStage(
        queue.Queue(), q_out, threading.Event(), config, screen_width=1920, screen_height=1080
    )
    return stage, q_out


def _gaze_packet(confidence=0.9, yaw=0.0, pitch=0.0):
    return GazePacket(
        timestamp=1.0,
        gaze_ray_left=None,
        gaze_ray_right=None,
        gaze_yaw=yaw,
        gaze_pitch=pitch,
        features=empty_features(),
        head_pose=None,
        confidence=confidence,
    )


# ── Default (torch) inference path ────────────────────────────────────────────


def test_process_ignores_a_packet_of_the_wrong_type():
    stage, q_out = _stage()

    stage.process("not a gaze packet")

    assert q_out.qsize() == 0


def test_plain_model_path_uses_predict_numpy():
    """`backend: torch` attaches a model and no predictor — the default path."""
    stage, q_out = _stage()
    trainer, model = _Trainer(), _Model()
    stage.set_model(model, trainer)

    stage.process(_gaze_packet())

    assert model.calls == 1
    assert trainer.calls == 1
    assert q_out.get_nowait().source == "mlp"


def test_plain_model_output_is_scaled_to_screen_pixels():
    stage, q_out = _stage()
    stage.set_model(_Model(output=(0.5, 0.25)), _Trainer())

    stage.process(_gaze_packet())

    prediction = q_out.get_nowait()
    assert prediction.screen_x == pytest.approx(0.5 * 1920)
    assert prediction.screen_y == pytest.approx(0.25 * 1080)


def test_non_torch_backend_is_reported_as_the_source():
    stage, q_out = _stage()
    stage.set_model(_Model(), _Trainer(), predictor=_Predictor(), backend="onnx")

    stage.process(_gaze_packet())

    assert q_out.get_nowait().source == "onnx"


def test_predictor_takes_precedence_over_the_model():
    stage, _ = _stage()
    model, predictor = _Model(), _Predictor()
    stage.set_model(model, trainer=_Trainer(), predictor=predictor, backend="onnx")

    stage.process(_gaze_packet())

    assert predictor.calls == 1
    assert model.calls == 0


def test_model_failure_falls_back_to_geometry():
    class _Broken(_Model):
        def predict_numpy(self, features):
            raise RuntimeError("model exploded")

    stage, q_out = _stage()
    stage.set_model(_Broken(), _Trainer())

    stage.process(_gaze_packet(yaw=20.0, pitch=10.0))

    assert q_out.get_nowait().source == "geometric"


def test_confidence_below_threshold_skips_the_model():
    stage, q_out = _stage()
    trainer, model = _Trainer(), _Model()
    stage.set_model(model, trainer)

    stage.process(_gaze_packet(confidence=0.1, yaw=20.0))

    assert model.calls == 0
    assert q_out.get_nowait().source in ("geometric", "hold")


def test_no_model_holds_at_the_screen_centre_when_the_fallback_is_off():
    stage, q_out = _stage(**{"inference.fallback_to_geometric": False})

    stage.process(_gaze_packet(yaw=25.0, pitch=25.0))

    prediction = q_out.get_nowait()
    assert prediction.source == "hold"
    assert prediction.screen_x == pytest.approx(960.0)
    assert prediction.screen_y == pytest.approx(540.0)


# ── Pipeline lifecycle ────────────────────────────────────────────────────────


class _NullSource(StageThread):
    """A frame source that emits nothing, for lifecycle tests."""

    def __init__(self, output_queue, stop_event):
        super().__init__(
            input_queue=None, output_queue=output_queue, stop_event=stop_event, name="null_source"
        )

    def process(self, item) -> None:
        time.sleep(0.01)


def _pipeline():
    pipeline = GazeEstimationPipeline(config=Config.default())
    return pipeline, _NullSource(pipeline.frame_queue, pipeline.stop_event)


def test_double_start_is_rejected():
    pipeline, source = _pipeline()
    try:
        pipeline.start(frame_source=source)
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.start(frame_source=source)
    finally:
        pipeline.stop()


def test_latest_estimate_is_none_before_anything_is_published():
    pipeline, _ = _pipeline()

    assert pipeline.get_latest_estimate() is None


def test_collector_publishes_estimates_for_get_latest_estimate():
    """The documented usage loop must actually receive estimates."""
    pipeline, source = _pipeline()
    estimate = GazeEstimate(
        timestamp=1.0,
        screen_x=123.0,
        screen_y=456.0,
        raw_x=123.0,
        raw_y=456.0,
        velocity=0.0,
        confidence=0.9,
        fixation_state=GazeState.FIXATION,
        fixation_duration=0.2,
        source="mlp",
        latency_ms=5.0,
    )
    try:
        pipeline.start(frame_source=source)
        pipeline._queues["estimate"].put_nowait(estimate)

        deadline = time.time() + 5.0
        while time.time() < deadline and pipeline.get_latest_estimate() is None:
            time.sleep(0.01)

        assert pipeline.get_latest_estimate() is estimate
    finally:
        pipeline.stop()


def test_set_model_at_runtime_reaches_the_live_stage():
    pipeline, source = _pipeline()
    try:
        pipeline.start(frame_source=source)
        trainer, model = _Trainer(), _Model()

        pipeline.set_model(model, trainer, backend="onnx")

        stage = pipeline._inference_stage
        assert stage is not None
        assert stage._model is model
        assert stage._trainer is trainer
        assert stage._source_name == "onnx"
    finally:
        pipeline.stop()


def test_set_bias_map_before_start_is_applied_to_the_stage():
    class _BiasMap:
        def apply(self, x, y, width, height):
            return x, y

    pipeline, source = _pipeline()
    bias_map = _BiasMap()
    pipeline.set_bias_map(bias_map)
    try:
        pipeline.start(frame_source=source)
        assert pipeline._inference_stage is not None
        assert pipeline._inference_stage._bias_map is bias_map
    finally:
        pipeline.stop()


def test_set_bias_map_at_runtime_reaches_the_live_stage():
    class _BiasMap:
        def apply(self, x, y, width, height):
            return x, y

    pipeline, source = _pipeline()
    try:
        pipeline.start(frame_source=source)
        bias_map = _BiasMap()

        pipeline.set_bias_map(bias_map)

        assert pipeline._inference_stage is not None
        assert pipeline._inference_stage._bias_map is bias_map
    finally:
        pipeline.stop()


def test_set_camera_intrinsics_before_start_is_remembered():
    pipeline, source = _pipeline()
    matrix = np.eye(3, dtype=np.float64) * 700.0
    matrix[2, 2] = 1.0

    pipeline.set_camera_intrinsics(matrix, np.zeros(5))  # no stages yet

    assert pipeline._camera_matrix[0, 0] == pytest.approx(700.0)
    try:
        pipeline.start(frame_source=source)
        assert pipeline._stages
    finally:
        pipeline.stop()


def test_update_kappa_at_runtime_reaches_the_gaze_stage():
    from gaze_estimation.gaze.gaze_geometry import GazeGeometryEstimator

    pipeline, source = _pipeline()
    try:
        pipeline.start(frame_source=source)
        gaze_stages = [s for s in pipeline._stages if isinstance(s, GazeGeometryEstimator)]
        assert len(gaze_stages) == 1

        pipeline.update_kappa(3.0, -1.0)

        assert gaze_stages[0]._kappa_yaw == pytest.approx(3.0)
        assert gaze_stages[0]._kappa_pitch == pytest.approx(-1.0)
    finally:
        pipeline.stop()


def test_stop_clears_stages_and_allows_a_restart():
    pipeline, source = _pipeline()
    pipeline.start(frame_source=source)
    pipeline.stop()

    assert pipeline._stages == []
    assert pipeline._collector is None

    pipeline.start(frame_source=_NullSource(pipeline.frame_queue, pipeline.stop_event))
    assert pipeline._stages
    pipeline.stop()


def test_fps_is_reported():
    pipeline, _ = _pipeline()

    assert pipeline.fps >= 0.0


def test_stop_warns_when_a_stage_outlives_the_join_timeout(caplog):
    """A wedged stage is reported rather than silently abandoned.

    This test costs ~3 s on purpose: the point is that stop() genuinely waits
    out its join timeout before deciding a stage is stuck.
    """
    import logging

    class _StuckSource(_NullSource):
        def process(self, item) -> None:
            time.sleep(4.0)  # outlives stop()'s 3 s join window

    pipeline = GazeEstimationPipeline(config=Config.default())
    source = _StuckSource(pipeline.frame_queue, pipeline.stop_event)
    pipeline.start(frame_source=source)
    try:
        with caplog.at_level(logging.WARNING, logger="gaze_estimation.pipeline"):
            pipeline.stop()

        assert "still alive after stop timeout" in caplog.text
        assert "null_source" in caplog.text
    finally:
        pipeline.stop()
