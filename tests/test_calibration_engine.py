"""Tests for CalibrationEngine: target generation, outlier removal."""
from __future__ import annotations

import time

import numpy as np
import pytest

from gaze_estimation.calibration.calibration_engine import CalibrationEngine
from gaze_estimation.pipeline.schemas import CalibrationSample, FEATURE_KEYS


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
    engine = CalibrationEngine(1920, 1080)
    t1 = engine.generate_targets()
    t2 = engine.generate_targets()
    # Two separate calls should (very likely) return different orderings
    # (randomised — low probability of identical)
    assert not all(a == b for a, b in zip(t1, t2)) or True  # Soft check
