"""3D gaze ray computation from iris centres and head pose."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np

from gaze_estimation.gaze.gaze_features import FeatureExtractor
from gaze_estimation.pipeline.schemas import (
    GazePacket,
    GazeRay,
    HeadPose,
    PosePacket,
)
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.utils.geometry import angles_to_ray, normalize, ray_to_angles


class GazeGeometryEstimator(StageThread):
    """Compute 3D gaze rays and the full feature vector from head-pose data.

    Pipeline stage: PosePacket -> GazePacket
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        kappa_yaw: float = 0.0,
        kappa_pitch: float = 0.0,
        eyeball_radius: float = 12.0,
        tap_queue: Optional[queue.Queue] = None,
        name: str = "gaze_thread",
    ) -> None:
        super().__init__(
            input_queue=input_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            tap_queue=tap_queue,
            name=name,
        )
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)
        self._kappa_yaw = kappa_yaw
        self._kappa_pitch = kappa_pitch
        self._eyeball_radius = eyeball_radius
        self._feature_extractor = FeatureExtractor()

    # ── Public API ─────────────────────────────────────────────────────────

    def update_kappa(self, kappa_yaw: float, kappa_pitch: float) -> None:
        """Update kappa angles (called after calibration)."""
        self._kappa_yaw = kappa_yaw
        self._kappa_pitch = kappa_pitch

    def set_camera_intrinsics(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """Update camera intrinsics at runtime (read per-frame, so this is safe)."""
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)

    # ── StageThread ────────────────────────────────────────────────────────

    def process(self, item: object) -> None:
        if not isinstance(item, PosePacket):
            return

        gaze_ray_left: Optional[GazeRay] = None
        gaze_ray_right: Optional[GazeRay] = None
        gaze_yaw = 0.0
        gaze_pitch = 0.0
        confidence = item.confidence

        if item.head_pose is not None and confidence > 0.0:
            if item.left_iris_center is not None:
                gaze_ray_left = self._compute_gaze_ray(
                    np.array(item.left_iris_center, dtype=np.float64),
                    item.head_pose,
                )
            if item.right_iris_center is not None:
                gaze_ray_right = self._compute_gaze_ray(
                    np.array(item.right_iris_center, dtype=np.float64),
                    item.head_pose,
                )

            # Angles from the average of the available rays (+ kappa).  The rays
            # themselves are eye-in-head (see _compute_gaze_ray), so these are
            # head-frame angles.
            angles = []
            for ray in [gaze_ray_left, gaze_ray_right]:
                if ray is not None:
                    y, p = ray_to_angles(ray.direction)
                    angles.append((y, p))

            if angles:
                avg_yaw = float(np.mean([a[0] for a in angles]))
                avg_pitch = float(np.mean([a[1] for a in angles]))
                # Kappa is the offset between the optical and visual axes, which
                # is fixed within the eye — so it is applied here, in the head
                # frame, before the direction is rotated into the camera frame.
                head_yaw = avg_yaw + self._kappa_yaw
                head_pitch = avg_pitch + self._kappa_pitch
                # The packet angles are consumed as *camera-frame* angles: the
                # geometric screen fallback projects them onto the screen plane,
                # and kappa estimation compares them with screen-referenced
                # target angles.  Rotating the head-frame direction by the head
                # rotation is what makes those two comparisons valid — without it
                # a turned head is mistaken for a turned gaze.  The rays and the
                # MLP feature vector stay head-frame on purpose, so the learned
                # mapping is invariant to head pose.
                world_ray = item.head_pose.rotation_matrix @ angles_to_ray(head_yaw, head_pitch)
                gaze_yaw, gaze_pitch = ray_to_angles(world_ray)

        features = self._feature_extractor.extract(
            mesh_468=item.mesh_468,
            left_iris_center=item.left_iris_center,
            right_iris_center=item.right_iris_center,
            left_iris_radius=item.left_iris_radius,
            right_iris_radius=item.right_iris_radius,
            head_pose=item.head_pose,
            gaze_ray_left=gaze_ray_left,
            gaze_ray_right=gaze_ray_right,
            frame_shape=item.frame.shape if item.frame is not None else (720, 1280, 3),
            face_bbox=item.face_bbox,
            confidence=confidence,
        )

        self.emit(
            GazePacket(
                timestamp=item.timestamp,
                gaze_ray_left=gaze_ray_left,
                gaze_ray_right=gaze_ray_right,
                gaze_yaw=gaze_yaw,
                gaze_pitch=gaze_pitch,
                features=features,
                head_pose=item.head_pose,
                confidence=confidence,
            )
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _compute_gaze_ray(
        self,
        iris_center_2d: np.ndarray,
        head_pose: HeadPose,
    ) -> Optional[GazeRay]:
        """Undistort iris centre, back-project to 3D, transform to head frame."""
        # Undistort the iris 2D point
        pts = iris_center_2d.reshape(1, 1, 2)
        undistorted = cv2.undistortPoints(
            pts.astype(np.float64),
            self._camera_matrix,
            self._dist_coeffs,
            P=self._camera_matrix,
        )
        ux, uy = undistorted.reshape(2)

        # Normalised camera-space ray
        fx = self._camera_matrix[0, 0]
        fy = self._camera_matrix[1, 1]
        cx = self._camera_matrix[0, 2]
        cy = self._camera_matrix[1, 2]

        ray_cam = np.array([(ux - cx) / fx, (uy - cy) / fy, 1.0], dtype=np.float64)
        ray_cam = normalize(ray_cam)

        # Transform ray from camera frame to head frame
        R_inv = head_pose.rotation_matrix.T  # R is orthogonal
        ray_head = R_inv @ ray_cam
        ray_head = normalize(ray_head)

        # Gaze origin: approximate eye position in camera frame
        # tvec is the head origin; we use it as the ray origin
        origin = head_pose.tvec.copy()

        return GazeRay(origin=origin, direction=ray_head)
