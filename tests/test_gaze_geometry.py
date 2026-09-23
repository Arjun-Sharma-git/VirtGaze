"""Tests for 3D gaze ray computation and angle utilities."""
from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.utils.geometry import (
    angles_to_ray,
    angular_error_deg,
    normalize,
    ray_plane_intersection,
    ray_to_angles,
    rotation_matrix_to_euler,
    screen_to_angles,
)


def test_ray_to_angles_forward():
    """A ray pointing straight ahead should produce yaw=0, pitch=0."""
    direction = np.array([0.0, 0.0, 1.0])
    yaw, pitch = ray_to_angles(direction)
    assert abs(yaw) < 0.1
    assert abs(pitch) < 0.1


def test_ray_to_angles_right():
    """A ray pointing right should produce positive yaw."""
    direction = np.array([1.0, 0.0, 1.0])
    yaw, _pitch = ray_to_angles(normalize(direction))
    assert yaw > 0


def test_angles_roundtrip():
    """Converting yaw/pitch → ray → yaw/pitch should be stable."""
    for yaw in [-30, 0, 15]:
        for pitch in [-20, 0, 10]:
            ray = angles_to_ray(yaw, pitch)
            y2, p2 = ray_to_angles(ray)
            assert abs(y2 - yaw) < 0.5, f"yaw mismatch: {y2} vs {yaw}"
            assert abs(p2 - pitch) < 0.5, f"pitch mismatch: {p2} vs {pitch}"


def test_normalize():
    v = np.array([3.0, 4.0, 0.0])
    n = normalize(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9


def test_normalize_zero():
    v = np.zeros(3)
    n = normalize(v)
    assert np.allclose(n, 0.0)


def test_angular_error_same():
    err = angular_error_deg(10, 5, 10, 5)
    assert abs(err) < 0.01


def test_angular_error_orthogonal():
    err = angular_error_deg(0, 0, 90, 0)
    assert abs(err - 90.0) < 1.0


def test_ray_plane_intersection():
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_point = np.array([0.0, 0.0, 5.0])
    pt = ray_plane_intersection(origin, direction, plane_normal, plane_point)
    assert pt is not None
    assert abs(pt[2] - 5.0) < 1e-9


def test_ray_plane_parallel():
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_point = np.array([0.0, 0.0, 5.0])
    pt = ray_plane_intersection(origin, direction, plane_normal, plane_point)
    assert pt is None


def test_rotation_matrix_to_euler_identity():
    R = np.eye(3)
    euler = rotation_matrix_to_euler(R)
    assert np.allclose(euler, 0.0, atol=1e-6)


def test_screen_to_angles_center():
    """Screen centre should give 0 yaw and 0 pitch."""
    yaw, pitch = screen_to_angles(960, 540, 1920, 1080, 600)
    assert abs(yaw) < 0.01
    assert abs(pitch) < 0.01


def test_screen_to_angles_matches_ray_convention():
    """screen_to_angles and ray_to_angles share the up/right-positive sign.

    Below-centre targets and downward gaze rays must both be negative in
    pitch — otherwise kappa estimation compares mismatched conventions.
    """
    below_centre = screen_to_angles(960, 1000, 1920, 1080, 600)
    assert below_centre[0] == pytest.approx(0.0, abs=1e-6)
    assert below_centre[1] < 0.0

    down_ray = np.array([0.0, 1.0, 2.0])          # Y is down in camera coords
    _, ray_pitch = ray_to_angles(normalize(down_ray))
    assert ray_pitch < 0.0

    right_of_centre = screen_to_angles(1800, 540, 1920, 1080, 600)
    assert right_of_centre[0] > 0.0
    right_ray = np.array([1.0, 0.0, 2.0])
    ray_yaw, _ = ray_to_angles(normalize(right_ray))
    assert ray_yaw > 0.0
