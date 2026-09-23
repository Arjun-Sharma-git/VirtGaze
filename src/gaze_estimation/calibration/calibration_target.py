"""Animated calibration target — pulse + circular motion."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class TargetState:
    """Current visual state of the animated calibration target."""
    x: float = 0.0
    y: float = 0.0
    radius: float = 20.0
    alpha: float = 1.0    # Opacity 0-1
    phase: float = 0.0    # Animation phase 0-1


class CalibrationTarget:
    """Animate a pulsing, circularly-drifting calibration dot.

    Call :meth:`set_position` when the target moves to a new calibration
    point, then call :meth:`update` each frame to get the current
    :class:`TargetState` for rendering.

    Args:
        base_radius:       Base dot radius in pixels (default 20).
        pulse_amplitude:   Pulse amplitude as fraction of base_radius (default 0.3).
        pulse_freq_hz:     Pulse frequency in Hz (default 2.5).
        circular_radius:   Radius of circular motion in pixels (default 15).
        circular_freq_hz:  Circular motion frequency in Hz (default 1.0).
        fade_in_sec:       Fade-in duration when target appears (default 0.2).
    """

    def __init__(
        self,
        base_radius: float = 20.0,
        pulse_amplitude: float = 0.3,
        pulse_freq_hz: float = 2.5,
        circular_radius: float = 15.0,
        circular_freq_hz: float = 1.0,
        fade_in_sec: float = 0.2,
    ) -> None:
        self._base_radius = base_radius
        self._pulse_amp = pulse_amplitude * base_radius
        self._pulse_freq = pulse_freq_hz
        self._circ_radius = circular_radius
        self._circ_freq = circular_freq_hz
        self._fade_in_sec = fade_in_sec

        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._appear_time: float = 0.0

    def set_position(self, x: float, y: float) -> None:
        """Move the target to a new position and restart animations."""
        self._center_x = x
        self._center_y = y
        self._appear_time = time.monotonic()

    def update(self, t: float | None = None) -> TargetState:
        """Compute the current animated target state.

        Args:
            t: Monotonic timestamp (seconds).  Defaults to time.monotonic().

        Returns:
            :class:`TargetState` with current x, y, radius, alpha.
        """
        if t is None:
            t = time.monotonic()

        elapsed = t - self._appear_time

        # Fade in
        alpha = min(1.0, elapsed / max(self._fade_in_sec, 1e-6))

        # Pulse (scale radius)
        pulse = self._pulse_amp * math.sin(2 * math.pi * self._pulse_freq * elapsed)
        radius = self._base_radius + pulse

        # Circular drift
        angle = 2 * math.pi * self._circ_freq * elapsed
        cx = self._center_x + self._circ_radius * math.cos(angle)
        cy = self._center_y + self._circ_radius * math.sin(angle)

        return TargetState(x=cx, y=cy, radius=radius, alpha=alpha)
