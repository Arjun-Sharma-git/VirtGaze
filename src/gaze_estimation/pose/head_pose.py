"""Head pose estimation thread using solvePnP on face mesh landmarks."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np

from gaze_estimation.pipeline.schemas import HeadPose, MeshPacket, PosePacket
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.pose.canonical_face import CANONICAL_3D_POINTS, LANDMARK_INDICES
from gaze_estimation.utils.geometry import rotation_matrix_to_euler


class HeadPoseEstimator(StageThread):
    """Estimate 3-DOF head rotation + translation using OpenCV solvePnP.

    Subscribes to :class:`~gaze_estimation.pipeline.schemas.MeshPacket` and
    emits :class:`~gaze_estimation.pipeline.schemas.PosePacket`.

    The head pose is expressed as:
    - ``rvec``  — Rodrigues rotation vector  (3,)
    - ``tvec``  — translation vector in camera frame (mm)  (3,)
    - ``euler_angles`` — [yaw, pitch, roll] in degrees
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tap_queue: Optional[queue.Queue] = None,
        name: str = "pose_thread",
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
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def set_camera_intrinsics(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """Update camera intrinsics at runtime (read per-frame, so this is safe)."""
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)
        # Stale extrinsic guesses are worse than none after an intrinsics change
        self._prev_rvec = None
        self._prev_tvec = None

    # ── StageThread ────────────────────────────────────────────────────────

    def process(self, item: object) -> None:
        if not isinstance(item, MeshPacket):
            return

        head_pose: Optional[HeadPose] = None
        confidence = item.confidence

        if item.confidence > 0.0 and item.mesh_468 is not None and len(item.mesh_468) >= 468:
            head_pose = self._solve_pnp(item.mesh_468)

        self.emit(
            PosePacket(
                timestamp=item.timestamp,
                frame=item.frame,
                mesh_468=item.mesh_468,
                iris_478=item.iris_478,
                left_iris_center=item.left_iris_center,
                right_iris_center=item.right_iris_center,
                left_iris_radius=item.left_iris_radius,
                right_iris_radius=item.right_iris_radius,
                head_pose=head_pose,
                confidence=confidence if head_pose is not None else 0.0,
            )
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _solve_pnp(self, mesh_468: np.ndarray) -> Optional[HeadPose]:
        """Run solvePnP on the stable subset of face landmarks."""
        image_points = mesh_468[LANDMARK_INDICES].astype(np.float64)

        # Use previous solution as initial guess for speed + stability
        if self._prev_rvec is not None and self._prev_tvec is not None:
            use_extrinsic = True
            init_rvec: Optional[np.ndarray] = self._prev_rvec.copy()
            init_tvec: Optional[np.ndarray] = self._prev_tvec.copy()
        else:
            use_extrinsic = False
            init_rvec = None
            init_tvec = None

        try:
            success, rvec, tvec = cv2.solvePnP(
                CANONICAL_3D_POINTS,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                rvec=init_rvec,
                tvec=init_tvec,
                useExtrinsicGuess=use_extrinsic,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:
            self._logger.debug("solvePnP failed: %s", exc)
            return None

        if not success:
            return None

        rvec = rvec.flatten()
        tvec = tvec.flatten()
        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()

        R, _ = cv2.Rodrigues(rvec)
        euler = rotation_matrix_to_euler(R)

        return HeadPose(
            rvec=rvec,
            tvec=tvec,
            euler_angles=euler,
            rotation_matrix=R,
        )

    @staticmethod
    def rotation_to_euler(rvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector to [yaw, pitch, roll] degrees (public helper)."""
        R, _ = cv2.Rodrigues(rvec)
        return rotation_matrix_to_euler(R)
