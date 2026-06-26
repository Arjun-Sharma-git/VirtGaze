"""Integration tests: config loading, schema validation, adaptation buffer."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gaze_estimation.config.config import Config
from gaze_estimation.pipeline.schemas import (
    FEATURE_KEYS, FEATURE_DIM, CalibrationSample, GazeState, FramePacket,
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
