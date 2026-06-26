"""Webcam capture thread: reads frames and pushes FramePackets to output queue."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from gaze_estimation.pipeline.schemas import FramePacket
from gaze_estimation.pipeline.thread_base import StageThread, put_or_drop
from gaze_estimation.utils.camera_calibration import estimate_camera_matrix, zero_dist_coeffs
from gaze_estimation.utils.logging import get_logger

_MAX_CONSECUTIVE_FAILURES = 5
_RECONNECT_DELAY_SEC = 2.0


class CameraCapture(StageThread):
    """Webcam capture stage.

    Runs on its own daemon thread.  On each iteration it reads a raw frame
    from the specified camera device, wraps it in a :class:`FramePacket`,
    and pushes it onto ``output_queue``.

    On >5 consecutive read failures the thread attempts to re-open the device
    after a short delay.
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        name: str = "camera_thread",
    ) -> None:
        super().__init__(
            input_queue=None,          # Source stage — no input queue
            output_queue=output_queue,
            stop_event=stop_event,
            name=name,
        )
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._failure_count: int = 0

        # Camera intrinsics (used downstream)
        self._camera_matrix = (
            camera_matrix if camera_matrix is not None
            else estimate_camera_matrix(width, height)
        )
        self._dist_coeffs = dist_coeffs if dist_coeffs is not None else zero_dist_coeffs()

    # ── StageThread interface ─────────────────────────────────────────────

    def _setup(self) -> None:
        self._open_camera()

    def _teardown(self) -> None:
        self._release()

    def process(self, _item: object) -> None:
        """Read one frame from the camera and emit a FramePacket."""
        if self._cap is None or not self._cap.isOpened():
            self._attempt_reconnect()
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failure_count += 1
            self._logger.warning(
                "Camera read failed (count=%d)", self._failure_count
            )
            if self._failure_count >= _MAX_CONSECUTIVE_FAILURES:
                self._logger.error(
                    "Too many consecutive failures — attempting reconnect"
                )
                self._attempt_reconnect()
            return

        self._failure_count = 0
        packet = FramePacket(
            timestamp=time.time(),
            frame=frame,
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        self.emit(packet)

    # ── Public helpers ────────────────────────────────────────────────────

    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (camera_matrix, dist_coeffs)."""
        return self._camera_matrix, self._dist_coeffs

    def set_camera_intrinsics(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """Update camera intrinsics (e.g. after chessboard calibration)."""
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = dist_coeffs.astype(np.float64)

    # ── Private ───────────────────────────────────────────────────────────

    def _open_camera(self) -> None:
        self._release()
        self._logger.info("Opening camera index=%d", self._camera_index)
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self._logger.error("Failed to open camera index=%d", self._camera_index)
            self._cap = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self._logger.info(
            "Camera opened: %dx%d @ %.0f FPS (requested %dx%d @ %d)",
            actual_w, actual_h, actual_fps,
            self._width, self._height, self._target_fps,
        )

        # Re-estimate intrinsics if resolution changed
        if actual_w != self._width or actual_h != self._height:
            self._camera_matrix = estimate_camera_matrix(actual_w, actual_h)

        self._cap = cap
        self._failure_count = 0

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _attempt_reconnect(self) -> None:
        self._logger.info(
            "Reconnecting to camera in %.1fs …", _RECONNECT_DELAY_SEC
        )
        self._release()
        time.sleep(_RECONNECT_DELAY_SEC)
        self._open_camera()
