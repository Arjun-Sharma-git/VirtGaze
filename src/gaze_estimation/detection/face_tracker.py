"""Face ROI tracker — wraps OpenCV mean-shift for run-time use."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FaceTracker:
    """Lightweight face-region tracker using mean-shift / CamShift.

    This class is a helper used by :class:`FaceDetector` to persist tracking
    between full MediaPipe detection runs.  It is not a StageThread on its own.

    Usage::

        tracker = FaceTracker()
        # When a full detection is available:
        tracker.init(frame, bbox)
        # On subsequent frames:
        new_bbox, confidence = tracker.update(frame)
    """

    def __init__(self) -> None:
        self._hist: Optional[np.ndarray] = None
        self._window: Optional[Tuple[int, int, int, int]] = None
        self._term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Initialise the tracker from a detection bounding box."""
        x, y, bw, bh = bbox
        h_f, w_f = frame.shape[:2]
        x = max(0, min(x, w_f - 1))
        y = max(0, min(y, h_f - 1))
        bw = max(1, min(bw, w_f - x))
        bh = max(1, min(bh, h_f - y))

        roi = frame[y: y + bh, x: x + bw]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self._hist = hist
        self._window = (x, y, bw, bh)

    def update(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Track the face in the new frame.

        Returns:
            (bbox, confidence) where bbox is (x, y, w, h) and confidence is
            in [0, 1].  Returns (None, 0.0) if tracking is not initialised.
        """
        if self._hist is None or self._window is None:
            return None, 0.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0.0, 60.0, 32.0]), np.array([180.0, 255.0, 255.0]))
        back_proj = cv2.calcBackProject([hsv], [0], self._hist, [0, 180], 1)
        back_proj &= mask

        try:
            _ret, track_window = cv2.meanShift(back_proj, self._window, self._term_crit)
            self._window = track_window
            return tuple(int(v) for v in track_window), 0.6  # type: ignore[return-value]
        except cv2.error:
            return self._window, 0.4

    def reset(self) -> None:
        """Reset the tracker state."""
        self._hist = None
        self._window = None
