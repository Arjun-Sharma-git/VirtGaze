"""QuickCalibration: 5-point recalibration (fine-tune existing MLP)."""

from __future__ import annotations

import queue
from typing import Callable, List, Optional, Tuple

import numpy as np

from gaze_estimation.calibration.calibration_engine import (
    _drain,
    build_residual_bias_map,
)
from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.schemas import (
    CalibrationResult,
    CalibrationSample,
    GazePacket,
)
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("calibration.quick")

# 5-point layout: center + 4 corners at 15% / 85% margins
_QUICK_MARGINS = 0.15


class QuickCalibration:
    """Run a fast 5-point recalibration to fine-tune an existing MLP.

    Points: centre + 4 corners.  Each target collects ~60 samples for
    1 second, giving 300 total.  The existing MLP is fine-tuned rather
    than retrained from scratch.

    Args:
        screen_width:       Display width in pixels.
        screen_height:      Display height in pixels.
        samples_per_target: Gaze samples per target (default 60).
        target_duration_sec: Seconds per target (default 1.0).
        points:             Number of calibration targets: 5 (centre + 4
                            corners) or 9 (3×3 grid).  Anything else falls back
                            to 5 with a warning.
        trainer:            Trainer with stored normalisation stats.
    """

    # Layouts understood by :meth:`generate_targets`.
    SUPPORTED_POINTS = (5, 9)

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        samples_per_target: int = 60,
        target_duration_sec: float = 1.0,
        points: int = 5,
        trainer: Optional[MLPTrainer] = None,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.samples_per_target = samples_per_target
        self.target_duration_sec = target_duration_sec
        self.points = int(points)
        self._trainer = trainer or MLPTrainer()

    def generate_targets(self) -> List[Tuple[float, float]]:
        """Return the target positions for :attr:`points`.

        The 5-point layout is centre + 4 corners at 15 % / 85 % margins (a
        3×3 grid with the edge midpoints omitted); the 9-point layout is the
        full 3×3 grid.
        """
        m = _QUICK_MARGINS
        w, h = self.screen_width, self.screen_height

        if self.points not in self.SUPPORTED_POINTS:
            _logger.warning("Unsupported quick_calibration.points=%s — using 5", self.points)
            self.points = 5

        if self.points == 9:
            xs = [w * m, w * 0.5, w * (1 - m)]
            ys = [h * m, h * 0.5, h * (1 - m)]
            return [(x, y) for y in ys for x in xs]

        return [
            (w * 0.5, h * 0.5),  # Centre
            (w * m, h * m),  # Top-left
            (w * (1 - m), h * m),  # Top-right
            (w * m, h * (1 - m)),  # Bottom-left
            (w * (1 - m), h * (1 - m)),  # Bottom-right
        ]

    def run(
        self,
        gaze_queue: queue.Queue,
        existing_model: GazeMLP,
        on_target_change: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        mlp_save_path: Optional[str] = None,
    ) -> CalibrationResult:
        """Run quick calibration and fine-tune *existing_model*.

        Returns:
            :class:`CalibrationResult` with updated kappa and model path.
        """
        import time

        from gaze_estimation.pipeline.schemas import FEATURE_KEYS

        targets = self.generate_targets()
        all_samples: List[CalibrationSample] = []

        for i, (tx, ty) in enumerate(targets):
            if on_target_change:
                on_target_change(tx, ty)
            _drain(gaze_queue)
            time.sleep(0.3)

            collected: List[CalibrationSample] = []
            deadline = time.monotonic() + self.target_duration_sec
            while time.monotonic() < deadline and len(collected) < self.samples_per_target:
                try:
                    pkt: GazePacket = gaze_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if pkt.confidence < 0.3:
                    continue
                collected.append(
                    CalibrationSample(
                        features=dict(pkt.features),
                        screen_x=tx,
                        screen_y=ty,
                        timestamp=pkt.timestamp,
                    )
                )
            all_samples.extend(collected)
            if on_progress:
                on_progress((i + 1) / len(targets))

        if not all_samples:
            _logger.warning("Quick calibration: no samples collected")
            return CalibrationResult(
                samples=[],
                mlp_weights_path=None,
                kappa_yaw=0.0,
                kappa_pitch=0.0,
                eyeball_radius=12.0,
                bias_map=None,
                timestamp=time.time(),
                screen_resolution=(self.screen_width, self.screen_height),
            )

        X = np.array(
            [[s.features.get(k, 0.0) for k in FEATURE_KEYS] for s in all_samples],
            dtype=np.float32,
        )
        Y = np.array([[s.screen_x, s.screen_y] for s in all_samples], dtype=np.float32)

        _logger.info("Fine-tuning MLP on %d quick-calib samples …", len(X))
        model = self._trainer.fine_tune(
            existing_model,
            X,
            Y,
            epochs=3,
        )
        if mlp_save_path:
            self._trainer.save(model, mlp_save_path)

        # Refresh the residual bias map for the fine-tuned model
        bias_map = build_residual_bias_map(
            model, self._trainer, X, Y, self.screen_width, self.screen_height
        )

        from gaze_estimation.gaze.kappa_compensation import estimate_kappa

        kappa_yaw, kappa_pitch = estimate_kappa(all_samples, self.screen_width, self.screen_height)

        return CalibrationResult(
            samples=all_samples,
            mlp_weights_path=mlp_save_path,
            kappa_yaw=kappa_yaw,
            kappa_pitch=kappa_pitch,
            eyeball_radius=12.0,
            bias_map=bias_map,
            timestamp=time.time(),
            screen_resolution=(self.screen_width, self.screen_height),
        )
