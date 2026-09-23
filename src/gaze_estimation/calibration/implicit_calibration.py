"""ImplicitCalibration: accumulate click-based gaze-cursor pairs for online adaptation."""
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from gaze_estimation.adaptation.online_trainer import OnlineTrainer
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("calibration.implicit")


class ImplicitCalibration:
    """Hook into mouse click events to do passive, ongoing calibration.

    At each deliberate click event the system assumes gaze ≈ cursor position
    (±50 px) and stores the pair for later fine-tuning.

    Usage — call :meth:`on_click` from wherever mouse events are captured::

        implicit = ImplicitCalibration(online_trainer)
        # ... on each click event:
        implicit.on_click(gaze_features, cursor_x, cursor_y, mouse_velocity)

    Args:
        online_trainer:          The :class:`OnlineTrainer` that fine-tunes the MLP.
        click_velocity_threshold: Ignore clicks when mouse speed > this (px/s).
    """

    def __init__(
        self,
        online_trainer: OnlineTrainer,
        click_velocity_threshold: float = 500.0,
        on_retrain: Optional[Callable[[], None]] = None,
    ) -> None:
        self._trainer = online_trainer
        self._click_vel_threshold = click_velocity_threshold
        # Called after a successful fine-tune so the caller can publish the
        # updated model to the live pipeline.
        self._on_retrain = on_retrain
        self._total_clicks = 0
        self._skipped_clicks = 0
        self._retrain_count = 0
        self._prev_cursor: Optional[Tuple[float, float]] = None
        self._prev_cursor_time: Optional[float] = None

    def on_click(
        self,
        gaze_features: dict,
        cursor_x: float,
        cursor_y: float,
        mouse_velocity: Optional[float] = None,
    ) -> bool:
        """Process one click event.

        Args:
            gaze_features:  Feature dict from the latest GazePacket.
            cursor_x, cursor_y: Mouse cursor position at click time (pixels).
            mouse_velocity: Mouse speed at click time (px/s). Computed
                            internally from position delta if not provided.

        Returns:
            True if the sample was accepted (added to buffer).
        """
        now = time.monotonic()

        # Compute velocity from history if not provided
        if mouse_velocity is None and self._prev_cursor is not None and self._prev_cursor_time is not None:
            dt = now - self._prev_cursor_time
            dx = cursor_x - self._prev_cursor[0]
            dy = cursor_y - self._prev_cursor[1]
            import math
            mouse_velocity = math.hypot(dx, dy) / max(dt, 1e-6)

        self._prev_cursor = (cursor_x, cursor_y)
        self._prev_cursor_time = now

        # Filter out fast-motion clicks
        if mouse_velocity is not None and mouse_velocity > self._click_vel_threshold:
            self._skipped_clicks += 1
            return False

        self._total_clicks += 1
        self._trainer.feed_click(
            gaze_features, cursor_x, cursor_y,
            mouse_velocity=mouse_velocity or 0.0,
            click_velocity_threshold=self._click_vel_threshold,
        )

        # Trigger fine-tune if enough new samples
        retrained = self._trainer.maybe_retrain()
        if retrained:
            self._retrain_count += 1
            _logger.info(
                "Implicit calibration triggered fine-tune "
                "(total clicks=%d, skipped=%d, retrains=%d)",
                self._total_clicks, self._skipped_clicks, self._retrain_count,
            )
            if self._on_retrain is not None:
                try:
                    self._on_retrain()
                except Exception as exc:
                    _logger.warning("on_retrain callback failed: %s", exc)
        return True

    @property
    def total_clicks(self) -> int:
        return self._total_clicks

    @property
    def skipped_clicks(self) -> int:
        return self._skipped_clicks

    @property
    def retrain_count(self) -> int:
        """Number of fine-tunes triggered by implicit calibration."""
        return self._retrain_count
