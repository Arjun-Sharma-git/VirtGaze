"""Tests for CalibrationEngine: target generation, outlier removal."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gaze_estimation.calibration.calibration_engine import CalibrationEngine
from gaze_estimation.pipeline.schemas import FEATURE_KEYS, CalibrationSample


def _make_sample(screen_x, screen_y, gaze_yaw=0.0, gaze_pitch=0.0, conf=0.9):
    feats = {k: 0.0 for k in FEATURE_KEYS}
    feats["gaze_yaw_avg"] = gaze_yaw
    feats["gaze_pitch_avg"] = gaze_pitch
    feats["landmark_confidence"] = conf
    return CalibrationSample(
        features=feats,
        screen_x=screen_x,
        screen_y=screen_y,
        timestamp=time.time(),
    )


def test_generate_targets_count():
    engine = CalibrationEngine(1920, 1080, grid_cols=5, grid_rows=5)
    targets = engine.generate_targets()
    assert len(targets) == 25


def test_generate_targets_range():
    engine = CalibrationEngine(1920, 1080, grid_cols=3, grid_rows=3)
    targets = engine.generate_targets()
    for x, y in targets:
        assert 0 < x < 1920
        assert 0 < y < 1080


def test_outlier_removal_keeps_good():
    engine = CalibrationEngine(1920, 1080, outlier_sigma=2.0)
    samples = [_make_sample(960, 540, gaze_yaw=float(i) * 0.01) for i in range(50)]
    cleaned = engine._remove_outliers(samples)
    assert len(cleaned) >= 40  # Most should be kept


def test_outlier_removal_removes_bad():
    engine = CalibrationEngine(1920, 1080, outlier_sigma=1.5)
    samples = [_make_sample(960, 540, gaze_yaw=0.0) for _ in range(28)]
    # Add 2 extreme outliers
    samples.append(_make_sample(960, 540, gaze_yaw=100.0))
    samples.append(_make_sample(960, 540, gaze_yaw=-100.0))
    cleaned = engine._remove_outliers(samples)
    yaws = [s.features["gaze_yaw_avg"] for s in cleaned]
    assert max(yaws) < 50.0


def test_targets_are_randomised():
    import random

    engine = CalibrationEngine(1920, 1080)
    random.seed(1234)
    t1 = engine.generate_targets()
    random.seed(4321)
    t2 = engine.generate_targets()
    # Same grid points, different visit order.
    assert set(t1) == set(t2)
    assert t1 != t2


# ── Residual bias map construction ───────────────────────────────────────────

def _identity_model():
    """Stand-in model mapping the first two features straight to screen coords."""
    class _Model:
        def predict_numpy(self, X):
            return X[:, :2]

    return _Model()


class _PassThroughTrainer:
    def normalise(self, X):
        return X


def test_build_residual_bias_map_learns_prediction_error():
    from gaze_estimation.calibration.calibration_engine import build_residual_bias_map

    X = np.array([[0.5, 0.5]], dtype=np.float32)
    Y = np.array([[1000.0, 560.0]], dtype=np.float32)

    bias_map = build_residual_bias_map(
        _identity_model(), _PassThroughTrainer(), X, Y, 1920, 1080, smoothing=0.0
    )

    assert bias_map is not None
    # prediction = (0.5, 0.5) * (1920, 1080) = (960, 540); target = (1000, 560)
    dx, dy = bias_map.get_correction(960.0, 540.0, 1920, 1080)
    assert dx == pytest.approx(40.0, abs=1.0)
    assert dy == pytest.approx(20.0, abs=1.0)


def test_build_residual_bias_map_returns_none_on_model_failure():
    from gaze_estimation.calibration.calibration_engine import build_residual_bias_map

    class _Broken:
        def predict_numpy(self, X):
            raise RuntimeError("boom")

    bias_map = build_residual_bias_map(
        _Broken(),
        _PassThroughTrainer(),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        1920,
        1080,
    )
    assert bias_map is None


def test_calibration_engine_accepts_bias_map_grid_config():
    engine = CalibrationEngine(
        1920, 1080, bias_map_cols=8, bias_map_rows=4, bias_map_smoothing=2.0
    )
    assert (engine._bias_map_cols, engine._bias_map_rows) == (8, 4)
    assert engine._bias_map_smoothing == pytest.approx(2.0)
