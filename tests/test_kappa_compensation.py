"""Tests for kappa-angle estimation and eyeball-radius fitting."""
from __future__ import annotations

import pytest

from gaze_estimation.gaze.kappa_compensation import estimate_kappa, fit_eyeball_radius
from gaze_estimation.pipeline.schemas import FEATURE_KEYS, CalibrationSample


def _sample(screen_x, screen_y, yaw, pitch, l_iris=0.0, r_iris=0.0, tz=600.0):
    feats = {k: 0.0 for k in FEATURE_KEYS}
    feats["gaze_yaw_avg"] = yaw
    feats["gaze_pitch_avg"] = pitch
    feats["left_iris_radius"] = l_iris
    feats["right_iris_radius"] = r_iris
    feats["head_tz"] = tz
    return CalibrationSample(
        features=feats, screen_x=screen_x, screen_y=screen_y, timestamp=0.0
    )


# ── estimate_kappa ────────────────────────────────────────────────────────────

def test_estimate_kappa_target_centre_matches_zero_gaze():
    samples = [_sample(960, 540, 0.0, 0.0) for _ in range(10)]
    kappa_yaw, kappa_pitch = estimate_kappa(samples, 1920, 1080)
    assert abs(kappa_yaw) < 1e-6
    assert abs(kappa_pitch) < 1e-6


def test_estimate_kappa_positive_for_right_target():
    samples = [_sample(1800, 540, 0.0, 0.0) for _ in range(5)]
    kappa_yaw, _ = estimate_kappa(samples, 1920, 1080)
    assert kappa_yaw > 0.0


def test_estimate_kappa_empty_samples():
    assert estimate_kappa([], 1920, 1080) == (0.0, 0.0)


# ── fit_eyeball_radius ────────────────────────────────────────────────────────

def test_fit_eyeball_radius_without_intrinsics_returns_default():
    samples = [_sample(960, 540, 0, 0, l_iris=0.02, r_iris=0.02)]
    assert fit_eyeball_radius(samples) == pytest.approx(12.0)


def test_fit_eyeball_radius_is_metric():
    # r_norm=0.008, frame 1280 px, f=1280 px, z=600 mm
    #   L_iris = 0.008 * 1280 * 600 / 1280 = 4.8 mm
    #   R_eyeball = 4.8 / 0.48 = 10.0 mm  (inside the plausible range)
    samples = [
        _sample(960, 540, 0, 0, l_iris=0.008, r_iris=0.008) for _ in range(5)
    ]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(10.0, abs=0.1)


def test_fit_eyeball_radius_rejects_implausible_estimates():
    # r_norm=0.05 -> L_iris = 30 mm -> R_eyeball = 62.5 mm, not human
    samples = [
        _sample(960, 540, 0, 0, l_iris=0.05, r_iris=0.05) for _ in range(5)
    ]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(12.0)


def test_fit_eyeball_radius_no_valid_samples_returns_default():
    samples = [_sample(960, 540, 0, 0, l_iris=0.0, r_iris=0.0, tz=0.0)]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(12.0)
