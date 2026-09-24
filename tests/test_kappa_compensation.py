"""Tests for kappa-angle estimation and eyeball-radius fitting."""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import pytest

from gaze_estimation.gaze.gaze_geometry import GazeGeometryEstimator
from gaze_estimation.gaze.kappa_compensation import estimate_kappa, fit_eyeball_radius
from gaze_estimation.pipeline.schemas import (
    FEATURE_KEYS,
    CalibrationSample,
    HeadPose,
    PosePacket,
)


def _sample(screen_x, screen_y, yaw, pitch, l_iris=0.0, r_iris=0.0, tz=600.0):
    feats = {k: 0.0 for k in FEATURE_KEYS}
    feats["gaze_yaw_avg"] = yaw
    feats["gaze_pitch_avg"] = pitch
    feats["left_iris_radius"] = l_iris
    feats["right_iris_radius"] = r_iris
    feats["head_tz"] = tz
    return CalibrationSample(features=feats, screen_x=screen_x, screen_y=screen_y, timestamp=0.0)


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
    samples = [_sample(960, 540, 0, 0, l_iris=0.008, r_iris=0.008) for _ in range(5)]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(10.0, abs=0.1)


def test_fit_eyeball_radius_rejects_implausible_estimates():
    # r_norm=0.05 -> L_iris = 30 mm -> R_eyeball = 62.5 mm, not human
    samples = [_sample(960, 540, 0, 0, l_iris=0.05, r_iris=0.05) for _ in range(5)]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(12.0)


def test_fit_eyeball_radius_no_valid_samples_returns_default():
    samples = [_sample(960, 540, 0, 0, l_iris=0.0, r_iris=0.0, tz=0.0)]
    radius = fit_eyeball_radius(samples, focal_length_px=1280.0, frame_width_px=1280)
    assert radius == pytest.approx(12.0)


# ── Coordinate frames ─────────────────────────────────────────────────────────


def test_world_angles_take_precedence_over_the_head_frame_features():
    """The world field is the one comparable with a screen-referenced target."""
    sample = _sample(960, 540, yaw=10.0, pitch=5.0)
    sample.gaze_yaw_world = 2.0
    sample.gaze_pitch_world = -1.0

    kappa_yaw, kappa_pitch = estimate_kappa([sample] * 3, 1920, 1080)

    assert kappa_yaw == pytest.approx(-2.0, abs=0.01)  # 0 - 2, not 0 - 10
    assert kappa_pitch == pytest.approx(1.0, abs=0.01)


def test_kappa_ignores_head_rotation_when_world_angles_are_present():
    """Regression: kappa used to absorb the head rotation.

    Every sample looks straight at the screen centre, but the head is turned by
    ~20°, so the head-frame gaze angle is ~-20°.  Comparing that against the
    screen-referenced target would report kappa ≈ +20° instead of ≈ 0.
    """
    width, height = 640, 480
    camera_matrix = np.array([[640.0, 0.0, width / 2], [0.0, 640.0, height / 2], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.35, 0.0]))  # ~20° head yaw
    head_pose = HeadPose(
        rvec=np.array([0.0, 0.35, 0.0]),
        tvec=np.array([0.0, 0.0, 600.0]),
        euler_angles=np.zeros(3),
        rotation_matrix=rotation,
    )
    out: queue.Queue = queue.Queue()
    estimator = GazeGeometryEstimator(
        queue.Queue(),
        out,
        threading.Event(),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    samples = []
    for _ in range(5):
        estimator.process(
            PosePacket(
                timestamp=0.0,
                frame=np.zeros((height, width, 3), dtype=np.uint8),
                mesh_468=np.full((468, 2), 100.0, dtype=np.float32),
                iris_478=None,
                left_iris_center=(width / 2, height / 2),  # principal point
                right_iris_center=(width / 2, height / 2),
                left_iris_radius=8.0,
                right_iris_radius=8.0,
                head_pose=head_pose,
                confidence=0.9,
            )
        )
        packet = out.get_nowait()
        samples.append(
            CalibrationSample(
                features=dict(packet.features),
                screen_x=960,
                screen_y=540,  # screen centre -> world angles (0, 0)
                timestamp=0.0,
                gaze_yaw_world=packet.gaze_yaw,
                gaze_pitch_world=packet.gaze_pitch,
            )
        )

    # The head-frame angle really is ~-20°, so the old code could not have
    # returned zero here.
    assert samples[0].features["gaze_yaw_avg"] < -15.0

    kappa_yaw, kappa_pitch = estimate_kappa(samples, 1920, 1080)

    assert kappa_yaw == pytest.approx(0.0, abs=0.5)
    assert kappa_pitch == pytest.approx(0.0, abs=0.5)
