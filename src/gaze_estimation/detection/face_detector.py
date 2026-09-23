"""Face detection thread using MediaPipe Face Detection."""
from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from gaze_estimation.detection.face_tracker import FaceTracker
from gaze_estimation.pipeline.schemas import FacePacket, FramePacket
from gaze_estimation.pipeline.thread_base import StageThread
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("detection.face_detector")

# MediaPipe FaceDetection ``model_selection`` values:
#   0 — short-range model (within ~2 m; faster, the webcam default)
#   1 — full-range model (within ~5 m)
_DETECTION_MODELS = {
    "mediapipe_short": 0,
    "mediapipe_full": 1,
}
_DEFAULT_DETECTION_MODEL = "mediapipe_short"

# MediaPipe landmark indices used as face_landmarks_2d (for solvePnP seed)
_MP_FACE_KEY_INDICES = [0, 1, 2, 3, 4, 5]  # 6 keypoints from mediapipe face detection


def resolve_detection_model(model: str) -> Tuple[str, int]:
    """Map a ``detection.model`` config string to ``(name, model_selection)``.

    Unknown names fall back to the default short-range model with a warning.
    """
    key = str(model).strip().lower()
    if key not in _DETECTION_MODELS:
        _logger.warning(
            "Unknown detection.model %r — using %s", model, _DEFAULT_DETECTION_MODEL
        )
        key = _DEFAULT_DETECTION_MODEL
    return key, _DETECTION_MODELS[key]


class FaceDetector(StageThread):
    """Detect the face on each frame.

    Runs MediaPipe Face Detection every *detection_interval* frames; between
    detections, the bounding-box from the previous frame is tracked using a
    lightweight mean-shift / optical-flow tracker.

    Emits :class:`~gaze_estimation.pipeline.schemas.FacePacket` objects.
    If no face is detected, emits a FacePacket with ``face_bbox=None`` so
    downstream stages know to enter coasting mode.

    Args:
        detection_interval: Run full detection every N frames.
        min_confidence:     MediaPipe detection confidence threshold.
        model:              ``"mediapipe_short"`` (≤2 m) or
                            ``"mediapipe_full"`` (≤5 m).
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        detection_interval: int = 5,
        min_confidence: float = 0.5,
        model: str = _DEFAULT_DETECTION_MODEL,
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
        self._model_name, self._model_selection = resolve_detection_model(model)

        self._detector = None           # MediaPipe detector (lazy init in thread)
        self._frame_counter: int = 0
        self._prev_bbox: Optional[tuple] = None
        self._tracker = FaceTracker()   # Between-detection ROI tracking

    # ── StageThread ────────────────────────────────────────────────────────

    def _setup(self) -> None:
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self._model_selection,
                min_detection_confidence=self._min_confidence,
            )
            self._logger.info(
                "MediaPipe FaceDetection initialised (model=%s, selection=%d)",
                self._model_name,
                self._model_selection,
            )
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

        # Full MediaPipe detection every N frames; mean-shift tracking between
        bbox: Optional[tuple] = None
        landmarks: Optional[np.ndarray] = None
        confidence = 0.0

        if self._frame_counter % self._detection_interval == 0 or self._prev_bbox is None:
            result = self._detect_full(frame)
            if result is not None:
                bbox, confidence, landmarks = result
                # (Re)initialise the ROI tracker from the fresh detection
                self._tracker.init(frame, bbox)
        else:
            bbox, confidence = self._tracker.update(frame)

        self._prev_bbox = bbox
        if bbox is None:
            # Lost the face — drop tracker state so the next frame re-detects
            self._tracker.reset()
            confidence = 0.0

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
