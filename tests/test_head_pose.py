"""Tests for head pose estimation (solvePnP)."""

from __future__ import annotations

import numpy as np

from gaze_estimation.utils.geometry import rotation_matrix_to_euler


def test_rotation_to_euler_known():
    """A 45° rotation about the Y axis maps to *pitch* in the ZYX convention.

    ``rotation_matrix_to_euler`` returns [yaw, pitch, roll] following
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (verified against
    ``scipy.spatial.transform.Rotation.as_euler('zyx')``), so a pure Ry
    rotation appears in the pitch slot while yaw and roll stay at 0.
    """
    import math

    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    euler = rotation_matrix_to_euler(R)
    assert abs(euler[0]) < 1e-6  # yaw   (about Z)
    assert abs(euler[1] - 45.0) < 1e-6  # pitch (about Y)
    assert abs(euler[2]) < 1e-6  # roll  (about X)


def test_rotation_to_euler_yaw_about_z():
    """A pure Rz rotation must land in the yaw slot."""
    import math

    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    euler = rotation_matrix_to_euler(R)
    assert abs(euler[0] - 30.0) < 1e-6
    assert abs(euler[1]) < 1e-6
    assert abs(euler[2]) < 1e-6


def test_head_pose_estimator_init(camera_matrix_640x480, dist_coeffs_zero):
    """HeadPoseEstimator can be created without errors."""
    import queue
    import threading

    from gaze_estimation.pose.head_pose import HeadPoseEstimator

    stop = threading.Event()
    q_in = queue.Queue(maxsize=2)
    q_out = queue.Queue(maxsize=2)
    est = HeadPoseEstimator(
        q_in, q_out, stop, camera_matrix=camera_matrix_640x480, dist_coeffs=dist_coeffs_zero
    )
    assert est is not None


def test_euler_identity():
    R = np.eye(3)
    euler = rotation_matrix_to_euler(R)
    assert np.allclose(euler, 0.0, atol=1e-5)
