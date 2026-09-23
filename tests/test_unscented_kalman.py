"""Tests for the Unscented Kalman Filter."""
from __future__ import annotations

import numpy as np

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


# ── Fixation-driven measurement-noise scaling ────────────────────────────────

def test_set_measurement_scale_updates_r():
    ukf = UnscentedKalmanFilter(measurement_noise=5.0)
    assert ukf.measurement_scale == 1.0

    ukf.set_measurement_scale(2.0)

    assert ukf.measurement_scale == 2.0
    assert np.allclose(ukf._R, np.eye(2) * 10.0)


def test_set_measurement_scale_clamps_to_sane_range():
    ukf = UnscentedKalmanFilter()
    ukf.set_measurement_scale(1e9)
    assert ukf.measurement_scale <= 10.0
    ukf.set_measurement_scale(-5.0)
    assert ukf.measurement_scale >= 0.05


def test_reset_restores_unit_measurement_scale():
    ukf = UnscentedKalmanFilter()
    ukf.set_measurement_scale(3.0)
    ukf.reset()
    assert ukf.measurement_scale == 1.0


def test_larger_measurement_scale_smooths_more():
    """A bigger scale means trusting the measurement less (more smoothing)."""

    def _response_after_step(scale: float) -> float:
        ukf = UnscentedKalmanFilter(process_noise=0.01, measurement_noise=5.0)
        ukf.update(np.array([500.0, 500.0]))       # start at 500
        ukf.set_measurement_scale(scale)
        for _ in range(5):
            ukf.predict(0.016)
            ukf.update(np.array([900.0, 500.0]))   # step input
        return float(ukf.get_position()[0])

    responsive = _response_after_step(0.05)
    smoothed = _response_after_step(10.0)
    assert responsive > smoothed
