"""Online adaptation buffer — stores (features, cursor) pairs for implicit calibration."""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

from gaze_estimation.pipeline.schemas import FEATURE_KEYS


class AdaptationBuffer:
    """Ring-buffer of (feature_vector, cursor_x, cursor_y) triples.

    Filled by click events.  Consumed by :class:`OnlineTrainer`.

    Args:
        max_size: Maximum number of samples to keep (ring-buffer).
    """

    def __init__(self, max_size: int = 5000) -> None:
        self._max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._new_since_train: int = 0

    def add(
        self, features: dict, cursor_x: float, cursor_y: float
    ) -> None:
        """Store one implicit calibration sample.

        Args:
            features:  Feature dict (FEATURE_KEYS keys).
            cursor_x:  Mouse cursor X at click time (pixels).
            cursor_y:  Mouse cursor Y at click time (pixels).
        """
        vec = np.array(
            [features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32
        )
        self._buffer.append((vec, float(cursor_x), float(cursor_y)))
        self._new_since_train += 1

    def get_batch(
        self, n: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) arrays.

        Args:
            n: If given, return only the most recent *n* samples.

        Returns:
            X: (N, 34) float32 feature matrix.
            Y: (N, 2)  float32 screen coordinates in pixels.
        """
        samples = list(self._buffer) if n is None else list(self._buffer)[-n:]
        if not samples:
            return np.zeros((0, len(FEATURE_KEYS)), np.float32), np.zeros((0, 2), np.float32)
        X = np.stack([s[0] for s in samples])
        Y = np.array([[s[1], s[2]] for s in samples], dtype=np.float32)
        return X, Y

    def count_new_since_last_train(self) -> int:
        """Number of samples added since the last :meth:`mark_trained` call."""
        return self._new_since_train

    def mark_trained(self) -> None:
        """Reset the new-sample counter after a training run."""
        self._new_since_train = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._new_since_train = 0
