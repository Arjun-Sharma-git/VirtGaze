"""Tests for FilterStage: the stage that produces the final GazeEstimate.

This stage had no test coverage at all, despite being the only component that
runs the UKF, the One Euro filter, fixation detection, and the latency
measurement together.  The tests below drive real packets through it (no mocks of
the filters) and assert on the emitted estimates.
"""

from __future__ import annotations

import math
import queue
import threading
import types

import numpy as np
import pytest

import gaze_estimation.pipeline.pipeline as pipeline_module
from gaze_estimation.config.config import Config
from gaze_estimation.pipeline.pipeline import FilterStage
from gaze_estimation.pipeline.schemas import GazeState, PredictionPacket

FRAME_DT = 1.0 / 60.0


def _stage(config: Config | None = None) -> tuple[FilterStage, queue.Queue]:
    q_in: queue.Queue = queue.Queue()
    q_out: queue.Queue = queue.Queue()
    stage = FilterStage(q_in, q_out, threading.Event(), config or Config.default())
    return stage, q_out


def _prediction(
    timestamp: float = 1.0,
    x: float = 960.0,
    y: float = 540.0,
    confidence: float = 0.9,
    source: str = "mlp",
    features: dict | None = None,
) -> PredictionPacket:
    return PredictionPacket(
        timestamp=timestamp,
        screen_x=x,
        screen_y=y,
        confidence=confidence,
        raw_features={"left_ear": 0.3, "right_ear": 0.3} if features is None else features,
        source=source,
    )


def _feed(stage: FilterStage, q_out: queue.Queue, count: int, **kwargs):
    """Push *count* identical-ish packets spaced one frame apart; return the last."""
    estimate = None
    for i in range(count):
        stage.process(_prediction(timestamp=1.0 + i * FRAME_DT, **kwargs))
        estimate = q_out.get_nowait()
    return estimate


class _FakeClock(types.ModuleType):
    def __init__(self, now: float = 0.0):
        super().__init__("fake_time")
        self.now = now

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(pipeline_module, "time", fake)
    return fake


# ── Packet handling ───────────────────────────────────────────────────────────


def test_ignores_a_packet_of_the_wrong_type():
    stage, q_out = _stage()

    stage.process("not a prediction")

    assert q_out.qsize() == 0


def test_emits_one_estimate_per_prediction():
    stage, q_out = _stage()

    stage.process(_prediction())

    assert q_out.qsize() == 1


def test_estimate_relays_raw_values_and_confidence():
    stage, q_out = _stage()

    stage.process(_prediction(x=1234.0, y=567.0, confidence=0.77))

    estimate = q_out.get_nowait()
    assert estimate.raw_x == 1234.0
    assert estimate.raw_y == 567.0
    assert estimate.confidence == 0.77
    assert estimate.timestamp == 1.0


def test_source_is_propagated_unchanged():
    stage, q_out = _stage()

    stage.process(_prediction(source="geometric"))

    assert q_out.get_nowait().source == "geometric"


def test_estimate_is_finite_for_extreme_inputs():
    """A wild packet must not produce NaN/inf downstream."""
    stage, q_out = _stage()

    stage.process(_prediction(x=1e9, y=-1e9))

    estimate = q_out.get_nowait()
    assert math.isfinite(estimate.screen_x)
    assert math.isfinite(estimate.screen_y)
    assert math.isfinite(estimate.velocity)


# ── Smoothing behaviour ───────────────────────────────────────────────────────


def test_output_converges_on_a_stationary_target():
    stage, q_out = _stage()

    estimate = _feed(stage, q_out, 120, x=960.0, y=540.0)

    assert abs(estimate.screen_x - 960.0) < 5.0
    assert abs(estimate.screen_y - 540.0) < 5.0


def test_first_estimate_initialises_on_the_first_measurement():
    """No startup transient: the filter begins at the first sample, not at 0,0."""
    stage, q_out = _stage()

    stage.process(_prediction(x=400.0, y=300.0))

    estimate = q_out.get_nowait()
    assert abs(estimate.screen_x - 400.0) < 1.0
    assert abs(estimate.screen_y - 300.0) < 1.0


def test_jitter_is_attenuated():
    """Alternating small offsets must come out flatter than they went in.

    Measured on the default config: a ratio of ~0.03 (97 % of the ±20 px jitter
    removed).  The 0.75 bound leaves headroom for platform float differences
    while still failing a pass-through filter, which would score 1.0.
    """
    stage, q_out = _stage()
    inputs, outputs = [], []
    for i in range(90):
        x = 960.0 + (20.0 if i % 2 else -20.0)
        stage.process(_prediction(timestamp=1.0 + i * FRAME_DT, x=x, y=540.0))
        estimate = q_out.get_nowait()
        if i >= 30:  # skip the settling period
            inputs.append(x)
            outputs.append(estimate.screen_x)

    assert np.std(outputs) < np.std(inputs) * 0.75


def test_smoothing_suppresses_a_single_sample_spike():
    stage, q_out = _stage()
    _feed(stage, q_out, 60, x=960.0, y=540.0)

    stage.process(_prediction(timestamp=1.0 + 60 * FRAME_DT, x=1900.0, y=1000.0))
    estimate = q_out.get_nowait()

    # The output moves far less than the full input jump (measured: ~21 % of it),
    # while the raw value is still reported untouched downstream.
    assert abs(estimate.screen_x - 960.0) < abs(1900.0 - 960.0)
    assert estimate.raw_x == 1900.0


def test_timestep_history_is_tracked():
    stage, _ = _stage()

    stage.process(_prediction(timestamp=5.0))

    assert stage._prev_t == 5.0


def test_a_large_time_gap_is_clamped():
    """dt is clamped to 0.5 s so a stall cannot destabilise the filter."""
    stage, q_out = _stage()
    _feed(stage, q_out, 5)

    stage.process(_prediction(timestamp=1000.0))

    assert math.isfinite(q_out.get_nowait().screen_x)


def test_time_running_backwards_is_clamped():
    stage, q_out = _stage()
    _feed(stage, q_out, 5)

    stage.process(_prediction(timestamp=0.0))

    assert math.isfinite(q_out.get_nowait().screen_x)


# ── Latency ───────────────────────────────────────────────────────────────────


def test_latency_is_measured_from_the_packet_timestamp(clock):
    """latency_ms is the age of the packet, not the time spent filtering."""
    stage, q_out = _stage()
    clock.now = 1.02  # 20 ms after the packet was captured

    stage.process(_prediction(timestamp=1.0))

    assert q_out.get_nowait().latency_ms == pytest.approx(20.0, abs=1e-6)


def test_latency_is_zero_when_the_clock_matches_the_packet(clock):
    stage, q_out = _stage()
    clock.now = 1.0

    stage.process(_prediction(timestamp=1.0))

    assert q_out.get_nowait().latency_ms == pytest.approx(0.0, abs=1e-6)


# ── Fixation-adaptive smoothing ───────────────────────────────────────────────


def test_first_packet_does_not_adapt_measurement_noise(monkeypatch):
    """The initial LOST state would otherwise freeze the filter."""
    stage, _ = _stage()
    calls: list[float] = []
    monkeypatch.setattr(stage._ukf, "set_measurement_scale", calls.append)

    stage.process(_prediction())

    assert calls == []


def test_measurement_noise_is_adapted_after_the_first_packet(monkeypatch):
    stage, _ = _stage()
    calls: list[float] = []
    monkeypatch.setattr(stage._ukf, "set_measurement_scale", calls.append)

    stage.process(_prediction())
    stage.process(_prediction(timestamp=1.0 + FRAME_DT))

    assert len(calls) == 1


def test_measurement_scale_uses_the_fixation_multiplier(monkeypatch):
    stage, _ = _stage()
    calls: list[float] = []
    monkeypatch.setattr(stage._ukf, "set_measurement_scale", calls.append)
    monkeypatch.setattr(stage._fixation, "get_smoothing_multiplier", lambda: 7.5)

    stage.process(_prediction())
    stage.process(_prediction(timestamp=1.0 + FRAME_DT))

    assert calls == [7.5]


def test_fixation_ready_flag_latches_after_the_first_packet():
    stage, _ = _stage()
    assert stage._fixation_ready is False

    stage.process(_prediction())

    assert stage._fixation_ready is True


# ── Fixation / blink state ────────────────────────────────────────────────────


def test_blink_is_detected_from_a_low_eye_aspect_ratio():
    stage, q_out = _stage()

    estimate = _feed(stage, q_out, 3, features={"left_ear": 0.05, "right_ear": 0.05})

    assert estimate.fixation_state is GazeState.BLINK


def test_a_normal_eye_aspect_ratio_is_not_a_blink():
    stage, q_out = _stage()

    estimate = _feed(stage, q_out, 3, features={"left_ear": 0.35, "right_ear": 0.35})

    assert estimate.fixation_state is not GazeState.BLINK


def test_missing_ear_features_default_to_open_eyes():
    """A packet without EAR keys must not be mistaken for a blink."""
    stage, q_out = _stage()

    estimate = _feed(stage, q_out, 3, features={})

    assert estimate.fixation_state is not GazeState.BLINK


def test_the_blink_threshold_is_taken_from_config():
    config = Config.default()
    config.raw.setdefault("filtering", {}).setdefault("fixation", {})["blink_ear_threshold"] = 0.5
    stage, q_out = _stage(config)

    estimate = _feed(stage, q_out, 3, features={"left_ear": 0.35, "right_ear": 0.35})

    # 0.35 is below the configured 0.5 threshold, so this counts as a blink.
    assert estimate.fixation_state is GazeState.BLINK


def test_a_stationary_gaze_becomes_a_fixation():
    stage, q_out = _stage()

    estimate = _feed(stage, q_out, 90, x=960.0, y=540.0)

    assert estimate.fixation_state is GazeState.FIXATION
    assert estimate.fixation_duration > 0.0


def test_velocity_is_reported_for_a_moving_gaze():
    stage, q_out = _stage()
    estimate = None
    for i in range(10):
        stage.process(_prediction(timestamp=1.0 + i * FRAME_DT, x=100.0 + i * 120.0, y=540.0))
        estimate = q_out.get_nowait()

    assert estimate.velocity > 0.0


def test_fixation_duration_grows_while_stationary():
    stage, q_out = _stage()
    durations = []
    for i in range(40):
        stage.process(_prediction(timestamp=1.0 + i * FRAME_DT, x=960.0, y=540.0))
        durations.append(q_out.get_nowait().fixation_duration)

    assert durations[-1] > durations[0]


# ── Configuration ─────────────────────────────────────────────────────────────


def test_filters_are_built_from_the_config_section():
    config = Config.default()
    stage, _ = _stage(config)

    assert stage._ukf is not None
    assert stage._oef is not None
    assert stage._fixation is not None
