"""Tests for the 3D gaze-geometry pipeline stage.

Ray directions are checked against the pinhole model directly (a centre pixel must
back-project to a forward ray, a right-of-centre pixel to positive yaw), and the
kappa offset is checked to land on the emitted angles.
"""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import pytest

from gaze_estimation.gaze.gaze_geometry import GazeGeometryEstimator
from gaze_estimation.pipeline.schemas import (
    FEATURE_KEYS,
    HeadPose,
    PosePacket,
)
from gaze_estimation.utils.geometry import ray_to_angles, rotation_matrix_to_euler

W, H = 640, 480
CAM = np.array([[640.0, 0.0, W / 2], [0.0, 640.0, H / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)
FRAME = np.zeros((H, W, 3), dtype=np.uint8)


def _head_pose(rot_vector=(0.0, 0.0, 0.0), tvec=(0.0, 0.0, 600.0)) -> HeadPose:
    rot = np.asarray(rot_vector, dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rot)
    return HeadPose(
        rvec=rot,
        tvec=np.asarray(tvec, dtype=np.float64),
        euler_angles=rotation_matrix_to_euler(rotation),
        rotation_matrix=rotation,
    )


def _estimator(**kwargs):
    q_in: queue.Queue = queue.Queue()
    q_out: queue.Queue = queue.Queue()
    kwargs.setdefault("camera_matrix", CAM)
    kwargs.setdefault("dist_coeffs", DIST)
    return GazeGeometryEstimator(q_in, q_out, threading.Event(), **kwargs), q_in, q_out


def _pose_packet(
    head_pose=_head_pose(),
    left=(320.0, 240.0),
    right=(340.0, 240.0),
    confidence=0.9,
    frame=FRAME,
    mesh=None,
):
    return PosePacket(
        timestamp=2.0,
        frame=frame,
        mesh_468=np.full((468, 2), 200.0, dtype=np.float32) if mesh is None else mesh,
        iris_478=None,
        left_iris_center=left,
        right_iris_center=right,
        left_iris_radius=8.0,
        right_iris_radius=8.0,
        head_pose=head_pose,
        confidence=confidence,
    )


# ── process() ─────────────────────────────────────────────────────────────────


def test_process_ignores_a_packet_of_the_wrong_type():
    estimator, _, q_out = _estimator()

    estimator.process(object())

    assert q_out.qsize() == 0


def test_process_emits_one_gaze_packet_per_pose_packet():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet())

    assert q_out.qsize() == 1
    assert q_out.get_nowait().timestamp == 2.0


def test_without_head_pose_there_are_no_rays_or_angles():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(head_pose=None))

    packet = q_out.get_nowait()
    assert packet.gaze_ray_left is None
    assert packet.gaze_ray_right is None
    assert packet.gaze_yaw == 0.0
    assert packet.gaze_pitch == 0.0
    assert packet.confidence == pytest.approx(0.9)  # still relayed


def test_zero_confidence_suppresses_the_rays():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(confidence=0.0))

    packet = q_out.get_nowait()
    assert packet.gaze_ray_left is None
    assert packet.gaze_ray_right is None


def test_both_iris_centres_produce_both_rays():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet())

    packet = q_out.get_nowait()
    assert packet.gaze_ray_left is not None
    assert packet.gaze_ray_right is not None


def test_a_missing_iris_centre_only_drops_that_ray():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(left=None))

    packet = q_out.get_nowait()
    assert packet.gaze_ray_left is None
    assert packet.gaze_ray_right is not None


def test_features_cover_every_declared_key():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet())

    features = q_out.get_nowait().features
    assert set(features) == set(FEATURE_KEYS)
    assert all(isinstance(v, float) for v in features.values())


def test_features_use_the_default_frame_shape_when_the_frame_is_missing():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(frame=None))

    features = q_out.get_nowait().features
    assert features["left_iris_x"] == pytest.approx(320.0 / 1280.0)


def test_head_pose_is_forwarded_for_downstream_stages():
    estimator, _, q_out = _estimator()
    head_pose = _head_pose()

    estimator.process(_pose_packet(head_pose=head_pose))

    assert q_out.get_nowait().head_pose is head_pose


# ── gaze angles ───────────────────────────────────────────────────────────────


def test_a_centred_iris_gives_a_forward_ray():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(left=(W / 2, H / 2), right=(W / 2, H / 2)))

    packet = q_out.get_nowait()
    np.testing.assert_allclose(packet.gaze_ray_left.direction, [0.0, 0.0, 1.0], atol=1e-6)
    assert packet.gaze_yaw == pytest.approx(0.0, abs=1e-6)
    assert packet.gaze_pitch == pytest.approx(0.0, abs=1e-6)


def test_right_of_centre_iris_gives_positive_yaw():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(left=(520.0, 240.0), right=(520.0, 240.0)))

    assert q_out.get_nowait().gaze_yaw > 0.0


def test_below_centre_iris_gives_negative_pitch():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(left=(320.0, 400.0), right=(320.0, 400.0)))

    assert q_out.get_nowait().gaze_pitch < 0.0


def test_gaze_angles_are_the_mean_of_the_two_rays():
    estimator, _, q_out = _estimator()

    estimator.process(_pose_packet(left=(300.0, 240.0), right=(340.0, 240.0)))
    packet = q_out.get_nowait()

    left_angles = ray_to_angles(packet.gaze_ray_left.direction)
    right_angles = ray_to_angles(packet.gaze_ray_right.direction)
    assert packet.gaze_yaw == pytest.approx((left_angles[0] + right_angles[0]) / 2)
    assert packet.gaze_pitch == pytest.approx((left_angles[1] + right_angles[1]) / 2)


def test_kappa_is_added_to_both_angles():
    estimator, _, q_out = _estimator(kappa_yaw=5.0, kappa_pitch=-3.0)

    estimator.process(_pose_packet(left=(W / 2, H / 2), right=(W / 2, H / 2)))

    packet = q_out.get_nowait()
    assert packet.gaze_yaw == pytest.approx(5.0, abs=1e-6)
    assert packet.gaze_pitch == pytest.approx(-3.0, abs=1e-6)


def test_update_kappa_takes_effect_immediately():
    estimator, _, q_out = _estimator()

    estimator.update_kappa(2.5, 1.5)
    estimator.process(_pose_packet(left=(W / 2, H / 2), right=(W / 2, H / 2)))

    packet = q_out.get_nowait()
    assert packet.gaze_yaw == pytest.approx(2.5)
    assert packet.gaze_pitch == pytest.approx(1.5)


# ── _compute_gaze_ray ─────────────────────────────────────────────────────────


def test_ray_origin_is_the_head_translation():
    estimator, _, _ = _estimator()
    head_pose = _head_pose(tvec=(11.0, -22.0, 550.0))

    ray = estimator._compute_gaze_ray(np.array([320.0, 240.0]), head_pose)

    np.testing.assert_allclose(ray.origin, [11.0, -22.0, 550.0])


def test_ray_direction_is_a_unit_vector():
    estimator, _, _ = _estimator()

    ray = estimator._compute_gaze_ray(np.array([420.0, 300.0]), _head_pose())

    assert np.linalg.norm(ray.direction) == pytest.approx(1.0)


def test_an_unrotated_head_leaves_the_ray_in_camera_coordinates():
    estimator, _, _ = _estimator()

    ray = estimator._compute_gaze_ray(np.array([500.0, 200.0]), _head_pose())

    expected = np.array([(500.0 - 320.0) / 640.0, (200.0 - 240.0) / 640.0, 1.0])
    np.testing.assert_allclose(ray.direction, expected / np.linalg.norm(expected), atol=1e-9)


def test_a_rotated_head_transforms_the_ray_by_the_inverse_rotation():
    estimator, _, _ = _estimator()
    head_pose = _head_pose(rot_vector=(0.0, 0.4, 0.0))
    pixel = np.array([500.0, 200.0])

    ray = estimator._compute_gaze_ray(pixel, head_pose)

    cam = np.array([(500.0 - 320.0) / 640.0, (200.0 - 240.0) / 640.0, 1.0])
    cam /= np.linalg.norm(cam)
    expected = head_pose.rotation_matrix.T @ cam
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(ray.direction, expected, atol=1e-9)
    assert not np.allclose(ray.direction, cam)  # the rotation actually did something


def test_set_camera_intrinsics_is_read_per_frame():
    estimator, _, _ = _estimator()

    estimator.set_camera_intrinsics(CAM, DIST)
    ray = estimator._compute_gaze_ray(np.array([W / 2, H / 2]), _head_pose())

    np.testing.assert_allclose(ray.direction, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(estimator._camera_matrix, CAM)


def test_a_shifted_principal_point_changes_the_ray():
    estimator, _, _ = _estimator()
    shifted = CAM.copy()
    shifted[0, 2] = 100.0
    estimator.set_camera_intrinsics(shifted, DIST)

    ray = estimator._compute_gaze_ray(np.array([W / 2, H / 2]), _head_pose())

    assert ray.direction[0] > 0.0  # the pixel is now right of the new centre


def test_distortion_coefficients_are_applied():
    estimator, _, _ = _estimator()
    distorting = np.array([0.2, -0.1, 0.0, 0.0, 0.0])

    estimator.set_camera_intrinsics(CAM, distorting)
    distorted_ray = estimator._compute_gaze_ray(np.array([500.0, 200.0]), _head_pose())

    estimator.set_camera_intrinsics(CAM, DIST)
    clean_ray = estimator._compute_gaze_ray(np.array([500.0, 200.0]), _head_pose())

    assert not np.allclose(distorted_ray.direction, clean_ray.direction)
