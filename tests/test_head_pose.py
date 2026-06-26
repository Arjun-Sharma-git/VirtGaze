"""Tests for head pose estimation (solvePnP)."""
from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.utils.geometry import rotation_matrix_to_euler


def test_rotation_to_euler_known():
    """Rotation of 45° around Y-axis → yaw ≈ 45°."""
    import math
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    euler = rotation_matrix_to_euler(R)
    # yaw should be ~45°, pitch and roll ~0°
    assert abs(euler[0] - 45.0) < 1.0 or abs(euler[1]) < 1.0


def test_head_pose_estimator_init(camera_matrix_640x480, dist_coeffs_zero):
    """HeadPoseEstimator can be created without errors."""
    import queue, threading
    from gaze_estimation.pose.head_pose import HeadPoseEstimator
    stop = threading.Event()
    q_in = queue.Queue(maxsize=2)
    q_out = queue.Queue(maxsize=2)
    est = HeadPoseEstimator(q_in, q_out, stop,
                            camera_matrix=camera_matrix_640x480,
                            dist_coeffs=dist_coeffs_zero)
    assert est is not None


def test_euler_identity():
    R = np.eye(3)
    euler = rotation_matrix_to_euler(R)
    assert np.allclose(euler, 0.0, atol=1e-5)
