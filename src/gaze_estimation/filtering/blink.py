"""Relative blink detection from the eye aspect ratio (EAR).

A fixed EAR threshold fits some eyes and not others, so a blink is an EAR
below a fraction of this person's recent open-eye level (the approach
EyeTrax uses).  Until enough history exists, a fixed fallback threshold
applies.
"""

from __future__ import annotations

from collections import deque


class BlinkDetector:
    """Flag blinks as an EAR well below the recent open-eye EAR.

    Args:
        ratio:        Blink when EAR < ratio * mean of recent open-eye EARs.
        history:      Open-eye samples kept for the running mean.
        min_history:  Samples needed before the relative test is used.
        fallback:     Absolute EAR threshold used until then (0 = none).
    """

    def __init__(
        self,
        ratio: float = 0.8,
        history: int = 50,
        min_history: int = 15,
        fallback: float = 0.0,
    ) -> None:
        self.ratio = float(ratio)
        self.min_history = int(min_history)
        self.fallback = float(fallback)
        self._ears: deque = deque(maxlen=int(history))

    def is_blink(self, left_ear: float, right_ear: float) -> bool:
        """Update with one frame; True if it is a blink (not added to history)."""
        if self.ratio <= 0.0:
            return False
        ear = (float(left_ear) + float(right_ear)) / 2.0
        if ear <= 0.0:  # no landmarks: not a blink, just no data
            return False
        if len(self._ears) >= self.min_history:
            threshold = self.ratio * (sum(self._ears) / len(self._ears))
        else:
            threshold = self.fallback
        if ear < threshold:
            return True
        self._ears.append(ear)
        return False
