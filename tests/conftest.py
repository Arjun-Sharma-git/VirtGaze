"""Pytest fixtures shared across all test modules."""
from __future__ import annotations

import time

import numpy as np
import pytest

# ── Synthetic data ────────────────────────────────────────────────────────────

@pytest.fixture
def frame_480p() -> np.ndarray:
    """A 480x640 BGR image filled with noise (simulates a camera frame)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def camera_matrix_640x480() -> np.ndarray:
    """A reasonable pinhole camera matrix for 640x480."""
    return np.array([
        [640.0,   0.0, 320.0],
        [  0.0, 640.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)


@pytest.fixture
def dist_coeffs_zero() -> np.ndarray:
    return np.zeros(5, dtype=np.float64)


@pytest.fixture
def mock_mesh_468() -> np.ndarray:
    """Synthetic 468 face-mesh landmarks for a roughly frontal face (640×480)."""
    rng = np.random.default_rng(0)
    # Place landmarks roughly in the centre of a 640x480 frame
    lms = rng.uniform(200, 440, (468, 2)).astype(np.float32)
    return lms


@pytest.fixture
def mock_head_pose():
    """HeadPose with identity rotation (yaw=0, pitch=0, roll=0)."""
    from gaze_estimation.pipeline.schemas import HeadPose
    return HeadPose(
        rvec=np.zeros(3, dtype=np.float64),
        tvec=np.array([0.0, 0.0, 600.0], dtype=np.float64),
        euler_angles=np.zeros(3, dtype=np.float64),
        rotation_matrix=np.eye(3, dtype=np.float64),
    )


@pytest.fixture
def calibration_samples():
    """200 synthetic CalibrationSamples for MLP training tests."""
    from gaze_estimation.pipeline.schemas import FEATURE_KEYS, CalibrationSample
    rng = np.random.default_rng(99)
    samples = []
    for _ in range(200):
        feats = {k: float(rng.uniform(-1, 1)) for k in FEATURE_KEYS}
        feats["landmark_confidence"] = 0.9
        sx = float(rng.uniform(100, 1820))
        sy = float(rng.uniform(100, 980))
        samples.append(CalibrationSample(
            features=feats,
            screen_x=sx,
            screen_y=sy,
            timestamp=time.time(),
        ))
    return samples
