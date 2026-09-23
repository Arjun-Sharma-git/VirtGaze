"""One Euro Filter — adaptive low-pass filter for gaze smoothing."""

from __future__ import annotations

import math
from typing import Optional


class _LowPassFilter:
    """Single exponential low-pass filter."""

    def __init__(self) -> None:
        self._y: Optional[float] = None
        self._s: Optional[float] = None

    def filter(self, value: float, alpha: float) -> float:
        if self._y is None:
            self._y = value
            self._s = value
            return value
        self._y = alpha * value + (1.0 - alpha) * self._y
        return self._y

    @property
    def last_value(self) -> Optional[float]:
        return self._y

    def reset(self) -> None:
        self._y = None
        self._s = None


class OneEuroFilter:
    """One Euro Filter for scalar real-time signal smoothing.

    Behaviour:
    - **Stable when stationary** (low velocity -> high smoothing).
    - **Responsive during movement** (high velocity -> low smoothing).

    Reference: Casiez et al. (2012), "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems".

    Args:
        min_cutoff: Minimum cutoff frequency in Hz.
                    Lower = more smoothing at low velocity.
        beta:       Speed coefficient.
                    Higher = less smoothing at high velocity (more responsive).
        d_cutoff:   Cutoff for the derivative (speed) estimate.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._prev_t: Optional[float] = None

    def filter(self, value: float, timestamp: float) -> float:
        """Apply the filter to a scalar value at the given timestamp.

        Args:
            value:     Raw (noisy) input value.
            timestamp: Monotonic timestamp in *seconds*.

        Returns:
            Filtered value.
        """
        if self._prev_t is None:
            self._prev_t = timestamp
            return self._x_filter.filter(value, 1.0)

        t_e = timestamp - self._prev_t
        self._prev_t = timestamp
        if t_e <= 0.0:
            t_e = 1e-6

        # Estimate derivative
        d_alpha = self._smoothing_factor(t_e, self._d_cutoff)
        x_prev = self._x_filter.last_value
        d_x = 0.0 if x_prev is None else (value - x_prev) / t_e
        d_x_hat = self._dx_filter.filter(d_x, d_alpha)

        # Adaptive cutoff
        cutoff = self._min_cutoff + self._beta * abs(d_x_hat)
        alpha = self._smoothing_factor(t_e, cutoff)
        return self._x_filter.filter(value, alpha)

    def reset(self) -> None:
        """Reset filter state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._prev_t = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)


class OneEuroFilter2D:
    """One Euro Filter applied independently to x and y coordinates."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        """Filter a 2D (x, y) point."""
        return self._fx.filter(x, timestamp), self._fy.filter(y, timestamp)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
