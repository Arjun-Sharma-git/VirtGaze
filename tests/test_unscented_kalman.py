"""Tests for the Unscented Kalman Filter."""
from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.filtering.unscented_kalman import UnscentedKalmanFilter


def test_ukf_first_update():
    """After the first update, position should match measurement."""
    ukf = UnscentedKalmanFilter()
    meas = np.array([100.0, 200.0])
    pos = ukf.update(meas)
    assert np.allclose(pos, meas, atol=1e-6)


def test_ukf_predict_coast():
    """Coast (predict-only) should return a position, not crash."""
    ukf = UnscentedKalmanFilter()
    ukf.update(np.array([500.0, 300.0]))
    pos = ukf.coast(dt=0.016)
    assert pos.shape == (2,)
    assert np.isfinite(pos).all()


def test_ukf_convergence():
    """UKF should converge to a stationary point after many updates."""
    ukf = UnscentedKalmanFilter(measurement_noise=2.0)
    target = np.array([640.0, 360.0])
    for _ in range(100):
        ukf.predict(0.016)
        pos = ukf.update(target + np.random.randn(2) * 2.0)
    assert np.linalg.norm(pos - target) < 30.0


def test_ukf_reset():
    ukf = UnscentedKalmanFilter()
    ukf.update(np.array([100.0, 200.0]))
    ukf.reset()
    assert not ukf._initialised


def test_ukf_state_dimension():
    ukf = UnscentedKalmanFilter()
    ukf.update(np.array([0.0, 0.0]))
    state = ukf.get_state()
    assert state.shape == (6,)
