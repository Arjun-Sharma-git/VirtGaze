"""Fixation / saccade / blink state machine."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from gaze_estimation.pipeline.schemas import FixationInfo, GazeState


class FixationDetector:
    """Classify gaze samples into fixation, saccade, blink, or lost states.

    State machine transitions:
    - velocity < fixation_threshold  → FIXATION
    - velocity > saccade_threshold   → SACCADE
    - EAR < blink_threshold          → BLINK
    - no data                        → LOST

    Args:
        fixation_velocity_threshold: Max velocity (px/s) to be considered fixating.
        saccade_velocity_threshold:  Min velocity (px/s) to be considered saccading.
        blink_ear_threshold:         EAR below this triggers blink state.
        min_fixation_duration:       Minimum seconds before FIXATION is confirmed.
    """

    def __init__(
        self,
        fixation_velocity_threshold: float = 100.0,
        saccade_velocity_threshold: float = 500.0,
        blink_ear_threshold: float = 0.15,
        min_fixation_duration: float = 0.1,
    ) -> None:
        self._fix_thresh = fixation_velocity_threshold
        self._sacc_thresh = saccade_velocity_threshold
        self._blink_ear = blink_ear_threshold
        self._min_fix_dur = min_fixation_duration

        self._state = GazeState.LOST
        self._state_start: float = time.perf_counter()
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._fix_points: list = []   # Accumulate positions for centroid

    def update(
        self,
        x: float,
        y: float,
        timestamp: float,
        left_ear: float = 1.0,
        right_ear: float = 1.0,
    ) -> FixationInfo:
        """Process a new gaze sample and return current state.

        Args:
            x:          Filtered screen X position.
            y:          Filtered screen Y position.
            timestamp:  Monotonic timestamp (seconds).
            left_ear:   Left eye aspect ratio (1.0 = wide open).
            right_ear:  Right eye aspect ratio.

        Returns:
            :class:`~gaze_estimation.pipeline.schemas.FixationInfo` describing
            current state, velocity, duration, and centroid.
        """
        velocity = 0.0
        avg_ear = (left_ear + right_ear) / 2.0

        # Compute velocity if we have a previous sample
        if self._prev_x is not None and self._prev_t is not None:
            dt = timestamp - self._prev_t
            if dt > 0:
                dx = x - self._prev_x
                dy = y - (self._prev_y or 0.0)
                velocity = float(np.hypot(dx, dy) / dt)

        self._prev_x = x
        self._prev_y = y
        self._prev_t = timestamp

        # Determine new state
        if avg_ear < self._blink_ear:
            new_state = GazeState.BLINK
        elif velocity > self._sacc_thresh:
            new_state = GazeState.SACCADE
        elif velocity < self._fix_thresh:
            new_state = GazeState.FIXATION
        else:
            # Between thresholds — keep current state
            new_state = self._state if self._state != GazeState.LOST else GazeState.FIXATION

        # State transition
        if new_state != self._state:
            self._state = new_state
            self._state_start = timestamp
            self._fix_points = []

        duration = timestamp - self._state_start

        # Accumulate fixation centroid
        if self._state == GazeState.FIXATION:
            self._fix_points.append((x, y))

        fix_point: Optional[Tuple[float, float]] = None
        if self._state == GazeState.FIXATION and len(self._fix_points) > 0:
            pts = np.array(self._fix_points)
            fix_point = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        return FixationInfo(
            state=self._state,
            velocity=velocity,
            duration=duration,
            fixation_point=fix_point,
        )

    def get_smoothing_multiplier(self) -> float:
        """Return a smoothing multiplier appropriate for the current state.

        - FIXATION → 2.0  (more smoothing — stable)
        - SACCADE  → 0.3  (less smoothing — be responsive)
        - BLINK    → 5.0  (heavy smoothing — hold last position)
        - LOST     → 5.0  (heavy smoothing)
        """
        if self._state == GazeState.FIXATION:
            return 2.0
        if self._state == GazeState.SACCADE:
            return 0.3
        return 5.0

    def reset(self) -> None:
        """Reset detector state."""
        self._state = GazeState.LOST
        self._state_start = time.perf_counter()
        self._prev_x = None
        self._prev_y = None
        self._prev_t = None
        self._fix_points = []
