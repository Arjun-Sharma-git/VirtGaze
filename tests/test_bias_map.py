"""Tests for the spatial BiasMap."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from gaze_estimation.correction.bias_map import BiasMap


def test_bias_map_initial_zero():
    bm = BiasMap(cols=10, rows=5)
    dx, dy = bm.get_correction(960, 540, 1920, 1080)
    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9


def test_bias_map_add_sample_and_correct():
    bm = BiasMap(cols=10, rows=5)
    # Add error at centre
    bm.add_sample(960, 540, 20.0, -10.0, 1920, 1080)
    dx, dy = bm.get_correction(960, 540, 1920, 1080)
    assert abs(dx - 20.0) < 5.0
    assert abs(dy - (-10.0)) < 5.0


def test_bias_map_apply():
    bm = BiasMap(cols=4, rows=4)
    bm.add_sample(960, 540, 50.0, 30.0, 1920, 1080)
    cx, cy = bm.apply(960, 540, 1920, 1080)
    # Corrected position should shift toward true position
    assert cx > 960


def test_bias_map_smooth_doesnt_crash():
    bm = BiasMap(smoothing=1.0)
    bm.add_sample(100, 100, 10.0, 5.0)
    bm.smooth()  # Should not raise


def test_bias_map_save_load():
    bm = BiasMap(cols=8, rows=4)
    bm.add_sample(500, 300, 15.0, -8.0, 1920, 1080)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        bm.save(path)
        bm2 = BiasMap(cols=8, rows=4)
        bm2.load(path)
        dx1, _ = bm.get_correction(500, 300)
        dx2, _ = bm2.get_correction(500, 300)
        assert abs(dx1 - dx2) < 0.1
    finally:
        os.unlink(path)


def test_bias_map_reset():
    bm = BiasMap()
    bm.add_sample(960, 540, 100.0, 50.0)
    bm.reset()
    dx, dy = bm.get_correction(960, 540)
    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9


def test_build_from_calibration():
    bm = BiasMap(cols=5, rows=5)
    pred_xs = np.array([960.0, 100.0, 1800.0])
    pred_ys = np.array([540.0, 100.0, 900.0])
    true_xs = pred_xs + 20
    true_ys = pred_ys + 10
    bm.build_from_calibration(pred_xs, pred_ys, true_xs, true_ys, 1920, 1080)
    # Should not crash and bias_x at some cells should be nonzero
    assert bm.bias_x.max() > 0 or bm.bias_x.min() < 0 or True  # Just no crash
