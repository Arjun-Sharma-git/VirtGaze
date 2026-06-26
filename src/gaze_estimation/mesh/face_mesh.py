"""Face mesh + iris landmark extraction using MediaPipe Face Mesh."""
from __future__ import annotations

import queue
import threading
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from gaze_estimation.pipeline.schemas import FacePacket, MeshPacket
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.utils.geometry import fit_circle

# MediaPipe iris landmark indices in the 478-point model
LEFT_IRIS_INDICES = list(range(468, 473))   # 5 points
RIGHT_IRIS_INDICES = list(range(473, 478))  # 5 points


class FaceMeshExtractor(StageThread):
    """Extract 468-point face mesh and iris landmarks.

    Uses MediaPipe Face Mesh with iris refinement enabled.  Emits
    :class:`~gaze_estimation.pipeline.schemas.MeshPacket` objects.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        refine_iris: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        name: str = "mesh_thread",
    ) -> None:
        super().__init__(
            input_queue=input_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            name=name,
        )
        self._refine_iris = refine_iris
        self._max_num_faces = max_num_faces
        self._min_det_conf = min_detection_confidence
        self._min_trk_conf = min_tracking_confidence
        self._face_mesh = None

    # ── StageThread ────────────────────────────────────────────────────────

    def _setup(self) -> None:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self._max_num_faces,
                refine_landmarks=self._refine_iris,
                min_detection_confidence=self._min_det_conf,
                min_tracking_confidence=self._min_trk_conf,
            )
            self._logger.info(
                "MediaPipe FaceMesh initialised (refine_iris=%s)", self._refine_iris
            )
        except Exception as exc:
            self._logger.error("Could not initialise MediaPipe FaceMesh: %s", exc)

    def _teardown(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def process(self, item: object) -> None:
        if not isinstance(item, FacePacket):
            return

        frame = item.frame
        h, w = frame.shape[:2]

        if self._face_mesh is None:
            self._emit_empty(item)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._emit_empty(item)
            return

        face_lms = results.multi_face_landmarks[0].landmark

        # Extract mesh_468 (or 478 if iris refinement is on)
        n_lms = len(face_lms)
        all_lms = np.array(
            [[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32
        )

        mesh_468 = all_lms[:468]
        iris_478: Optional[np.ndarray] = all_lms if n_lms >= 478 else None

        # Extract iris centres and radii
        left_center = left_radius = None
        right_center = right_radius = None

        if iris_478 is not None and n_lms >= 478:
            left_pts = iris_478[LEFT_IRIS_INDICES]
            right_pts = iris_478[RIGHT_IRIS_INDICES]
            lx, ly, lr = fit_circle(left_pts)
            rx, ry, rr = fit_circle(right_pts)
            left_center = (float(lx), float(ly))
            left_radius = float(lr)
            right_center = (float(rx), float(ry))
            right_radius = float(rr)
        else:
            # Fallback: use landmark 468-477 if available
            if n_lms > 472:
                left_pts = all_lms[LEFT_IRIS_INDICES]
                lx, ly, lr = fit_circle(left_pts)
                left_center = (float(lx), float(ly))
                left_radius = float(lr)
            if n_lms > 477:
                right_pts = all_lms[RIGHT_IRIS_INDICES]
                rx, ry, rr = fit_circle(right_pts)
                right_center = (float(rx), float(ry))
                right_radius = float(rr)

        confidence = 1.0  # MediaPipe doesn't expose a per-frame confidence score

        self.emit(
            MeshPacket(
                timestamp=item.timestamp,
                frame=frame,
                face_bbox=item.face_bbox,
                mesh_468=mesh_468,
                iris_478=iris_478,
                left_iris_center=left_center,
                right_iris_center=right_center,
                left_iris_radius=left_radius,
                right_iris_radius=right_radius,
                confidence=confidence,
            )
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _emit_empty(self, item: FacePacket) -> None:
        """Emit a MeshPacket with empty landmarks when detection fails."""
        h, w = item.frame.shape[:2]
        self.emit(
            MeshPacket(
                timestamp=item.timestamp,
                frame=item.frame,
                face_bbox=item.face_bbox,
                mesh_468=np.zeros((468, 2), dtype=np.float32),
                iris_478=None,
                left_iris_center=None,
                right_iris_center=None,
                left_iris_radius=None,
                right_iris_radius=None,
                confidence=0.0,
            )
        )
