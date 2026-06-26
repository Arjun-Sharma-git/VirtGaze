"""Face detection thread using MediaPipe Face Detection."""
from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from gaze_estimation.pipeline.schemas import FacePacket, FramePacket
from gaze_estimation.pipeline.thread_base import StageThread

# MediaPipe landmark indices used as face_landmarks_2d (for solvePnP seed)
_MP_FACE_KEY_INDICES = [0, 1, 2, 3, 4, 5]  # 6 keypoints from mediapipe face detection


class FaceDetector(StageThread):
    """Detect the face on each frame.

    Runs MediaPipe Face Detection every *detection_interval* frames; between
    detections, the bounding-box from the previous frame is tracked using a
    lightweight mean-shift / optical-flow tracker.

    Emits :class:`~gaze_estimation.pipeline.schemas.FacePacket` objects.
    If no face is detected, emits a FacePacket with ``face_bbox=None`` so
    downstream stages know to enter coasting mode.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        detection_interval: int = 5,
        min_confidence: float = 0.5,
        name: str = "detection_thread",
    ) -> None:
        super().__init__(
            input_queue=input_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            name=name,
        )
        self._detection_interval = detection_interval
        self._min_confidence = min_confidence

        self._detector = None           # MediaPipe detector (lazy init in thread)
        self._frame_counter: int = 0
        self._prev_bbox: Optional[tuple] = None
        self._prev_gray: Optional[np.ndarray] = None

    # ── StageThread ────────────────────────────────────────────────────────

    def _setup(self) -> None:
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self._min_confidence,
            )
            self._logger.info("MediaPipe FaceDetection initialised")
        except Exception as exc:
            self._logger.error("Could not initialise MediaPipe: %s", exc)

    def _teardown(self) -> None:
        if self._detector is not None:
            self._detector.close()

    def process(self, item: object) -> None:
        if not isinstance(item, FramePacket):
            return

        frame = item.frame
        self._frame_counter += 1

        # Full detection every N frames, else ROI track
        if self._frame_counter % self._detection_interval == 0 or self._prev_bbox is None:
            result = self._detect_full(frame)
        else:
            result = self._track_roi(frame, self._prev_bbox)

        if result is not None:
            bbox, confidence, landmarks = result
            self._prev_bbox = bbox
        else:
            bbox, confidence, landmarks = None, 0.0, None
            self._prev_bbox = None

        self.emit(
            FacePacket(
                timestamp=item.timestamp,
                frame=frame,
                face_bbox=bbox,
                detection_confidence=float(confidence),
                face_landmarks_2d=landmarks,
            )
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _detect_full(
        self, frame: np.ndarray
    ) -> Optional[Tuple[tuple, float, Optional[np.ndarray]]]:
        """Run MediaPipe face detection on the full frame."""
        if self._detector is None:
            return self._fallback_detect(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return None

        det = results.detections[0]  # Take highest-confidence detection
        confidence = det.score[0] if det.score else 0.0
        if confidence < self._min_confidence:
            return None

        h, w = frame.shape[:2]
        bbox_rel = det.location_data.relative_bounding_box
        x = int(bbox_rel.xmin * w)
        y = int(bbox_rel.ymin * h)
        bw = int(bbox_rel.width * w)
        bh = int(bbox_rel.height * h)
        x, y = max(0, x), max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        bbox = (x, y, bw, bh)

        # Extract key landmarks as (N, 2) pixel array
        kps = det.location_data.relative_keypoints
        landmarks = np.array(
            [[kp.x * w, kp.y * h] for kp in kps], dtype=np.float32
        ) if kps else None

        return bbox, confidence, landmarks

    def _track_roi(
        self, frame: np.ndarray, prev_bbox: Optional[tuple]
    ) -> Optional[Tuple[tuple, float, Optional[np.ndarray]]]:
        """Lightweight mean-shift tracking within the previous ROI."""
        if prev_bbox is None:
            return None

        x, y, bw, bh = prev_bbox
        # Clamp ROI to frame
        h_f, w_f = frame.shape[:2]
        x = max(0, min(x, w_f - 1))
        y = max(0, min(y, h_f - 1))
        bw = max(1, min(bw, w_f - x))
        bh = max(1, min(bh, h_f - y))

        try:
            roi = frame[y: y + bh, x: x + bw]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            back_proj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
            back_proj &= mask
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            window = (x, y, bw, bh)
            _ret, track_window = cv2.meanShift(back_proj, window, term_crit)
            tx, ty, tw, th = track_window
            return (tx, ty, tw, th), 0.6, None   # Medium confidence for tracked ROI
        except Exception:
            return prev_bbox, 0.5, None

    def _fallback_detect(
        self, frame: np.ndarray
    ) -> Optional[Tuple[tuple, float, Optional[np.ndarray]]]:
        """Haar-cascade fallback if MediaPipe is not available."""
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, bw, bh = faces[0]
        return (int(x), int(y), int(bw), int(bh)), 0.5, None
