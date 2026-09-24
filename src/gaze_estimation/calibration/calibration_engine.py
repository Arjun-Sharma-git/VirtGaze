"""CalibrationEngine: orchestrates a 25-point gaze calibration session."""

from __future__ import annotations

import queue
import random
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from gaze_estimation.correction.bias_map import BiasMap
from gaze_estimation.gaze.kappa_compensation import estimate_kappa, fit_eyeball_radius
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.schemas import (
    CalibrationResult,
    CalibrationSample,
    GazePacket,
)
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("calibration.engine")


def build_residual_bias_map(
    model,
    trainer,
    X: np.ndarray,
    Y: np.ndarray,
    screen_width: int,
    screen_height: int,
    cols: int = 40,
    rows: int = 20,
    smoothing: float = 1.0,
):
    """Build a residual :class:`BiasMap` from a model's own calibration errors.

    The map stores the mean (target − prediction) error per screen region as
    measured on the calibration targets, so that the model's residual spatial
    bias can be corrected at inference time without retraining.

    Returns ``None`` when the model cannot be evaluated.
    """
    try:
        preds = model.predict_numpy(trainer.normalise(X))
    except Exception as exc:
        _logger.warning("Could not build bias map: %s", exc)
        return None

    bias_map = BiasMap(cols=cols, rows=rows, smoothing=smoothing)
    bias_map.build_from_calibration(
        preds[:, 0] * screen_width,
        preds[:, 1] * screen_height,
        Y[:, 0],
        Y[:, 1],
        screen_width,
        screen_height,
    )
    return bias_map


# Default screen margins as fraction of screen dimensions
_MARGIN = 0.1  # 10% margin on each side → targets from 10% to 90%


class CalibrationEngine:
    """Orchestrate a full 25-point calibration session.

    Generates a 5×5 grid of calibration targets, collects gaze samples,
    removes outliers, estimates kappa, fits the eyeball radius, trains the
    personalised MLP, and returns a :class:`CalibrationResult`.

    Args:
        screen_width:         Display resolution in pixels.
        screen_height:        Display resolution in pixels.
        grid_cols:            Calibration grid columns (default 5).
        grid_rows:            Calibration grid rows (default 5).
        samples_per_target:   Gaze samples to collect per target (default 120).
        target_duration_sec:  Seconds the user looks at each target (default 2).
        outlier_sigma:        Reject samples > N σ from per-target mean (default 2).
        mlp_trainer:          MLPTrainer instance; a default one is created if None.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        grid_cols: int = 5,
        grid_rows: int = 5,
        samples_per_target: int = 120,
        target_duration_sec: float = 2.0,
        outlier_sigma: float = 2.0,
        mlp_trainer: Optional[MLPTrainer] = None,
        camera_matrix: Optional[np.ndarray] = None,
        frame_width: Optional[int] = None,
        bias_map_cols: int = 40,
        bias_map_rows: int = 20,
        bias_map_smoothing: float = 1.0,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.samples_per_target = samples_per_target
        self.target_duration_sec = target_duration_sec
        self.outlier_sigma = outlier_sigma
        self._trainer = mlp_trainer or MLPTrainer()
        self._camera_matrix = camera_matrix
        self._frame_width = frame_width
        self._bias_map_cols = bias_map_cols
        self._bias_map_rows = bias_map_rows
        self._bias_map_smoothing = bias_map_smoothing

    # ── Main API ───────────────────────────────────────────────────────────

    def generate_targets(self) -> List[Tuple[float, float]]:
        """Generate calibration target positions (randomised order)."""
        xs = np.linspace(_MARGIN, 1.0 - _MARGIN, self.grid_cols) * self.screen_width
        ys = np.linspace(_MARGIN, 1.0 - _MARGIN, self.grid_rows) * self.screen_height
        targets = [(float(x), float(y)) for y in ys for x in xs]
        random.shuffle(targets)
        return targets

    def run(
        self,
        gaze_queue: queue.Queue,
        on_target_change: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        mlp_save_path: Optional[str] = None,
    ) -> CalibrationResult:
        """Run the calibration session and return results.

        Args:
            gaze_queue:       Live queue of GazePackets from the pipeline.
            on_target_change: Callback(target_x, target_y) when target moves.
            on_progress:      Callback(fraction_complete: float).
            mlp_save_path:    If given, save the trained MLP weights here.

        Returns:
            A :class:`CalibrationResult` with trained MLP path, kappa, bias map.
        """
        targets = self.generate_targets()
        all_samples: List[CalibrationSample] = []

        n_targets = len(targets)
        for i, (tx, ty) in enumerate(targets):
            _logger.info("Target %d/%d  (%.0f, %.0f)", i + 1, n_targets, tx, ty)
            if on_target_change:
                on_target_change(tx, ty)

            # Drain stale packets
            _drain(gaze_queue)

            # Wait briefly for eyes to fixate on new target
            time.sleep(0.3)

            # Collect samples
            samples = self._collect_samples(gaze_queue, tx, ty)
            all_samples.extend(samples)

            if on_progress:
                on_progress((i + 1) / n_targets)

        _logger.info("Collected %d total samples from %d targets", len(all_samples), n_targets)

        # Estimate kappa
        kappa_yaw, kappa_pitch = estimate_kappa(all_samples, self.screen_width, self.screen_height)
        _logger.info("Kappa: yaw=%.2f°  pitch=%.2f°", kappa_yaw, kappa_pitch)

        # Fit eyeball radius (needs camera intrinsics for a metric estimate;
        # falls back to the anthropometric default when unavailable)
        focal_px = float(self._camera_matrix[0, 0]) if self._camera_matrix is not None else None
        radius = fit_eyeball_radius(
            all_samples,
            focal_length_px=focal_px,
            frame_width_px=self._frame_width,
        )

        # Train MLP + build the residual bias map
        mlp_path = None
        bias_map = None
        if all_samples:
            from gaze_estimation.pipeline.schemas import FEATURE_KEYS

            X = np.array(
                [[s.features.get(k, 0.0) for k in FEATURE_KEYS] for s in all_samples],
                dtype=np.float32,
            )
            Y = np.array([[s.screen_x, s.screen_y] for s in all_samples], dtype=np.float32)

            _logger.info("Training MLP on %d samples …", len(X))
            model = self._trainer.train(X, Y, self.screen_width, self.screen_height)

            if mlp_save_path:
                self._trainer.save(model, mlp_save_path)
                # Only report the path once the file actually exists, otherwise
                # callers would try to load a model that was never written.
                mlp_path = mlp_save_path
                _logger.info("MLP saved to %s", mlp_path)

            bias_map = self._build_bias_map(model, X, Y)

        return CalibrationResult(
            samples=all_samples,
            mlp_weights_path=mlp_path,
            kappa_yaw=kappa_yaw,
            kappa_pitch=kappa_pitch,
            eyeball_radius=radius,
            bias_map=bias_map,
            timestamp=time.time(),
            screen_resolution=(self.screen_width, self.screen_height),
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _build_bias_map(self, model, X: np.ndarray, Y: np.ndarray):
        """Build a residual :class:`BiasMap` from the trained model's errors."""
        bias_map = build_residual_bias_map(
            model,
            self._trainer,
            X,
            Y,
            self.screen_width,
            self.screen_height,
            cols=self._bias_map_cols,
            rows=self._bias_map_rows,
            smoothing=self._bias_map_smoothing,
        )
        if bias_map is not None:
            _logger.info(
                "Bias map built (%dx%d grid, %d samples)",
                self._bias_map_rows,
                self._bias_map_cols,
                len(X),
            )
        return bias_map

    def _collect_samples(
        self,
        gaze_queue: queue.Queue,
        target_x: float,
        target_y: float,
    ) -> List[CalibrationSample]:
        """Collect *samples_per_target* samples and remove outliers."""
        collected: List[CalibrationSample] = []
        deadline = time.monotonic() + self.target_duration_sec

        while time.monotonic() < deadline and len(collected) < self.samples_per_target:
            try:
                packet: GazePacket = gaze_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if packet.confidence < 0.3:
                continue

            collected.append(
                CalibrationSample(
                    features=dict(packet.features),
                    screen_x=target_x,
                    screen_y=target_y,
                    timestamp=packet.timestamp,
                )
            )

        return self._remove_outliers(collected)

    def _remove_outliers(self, samples: List[CalibrationSample]) -> List[CalibrationSample]:
        """Reject samples more than *outlier_sigma* σ from mean gaze angles."""
        if len(samples) < 4:
            return samples

        yaws = np.array([s.features.get("gaze_yaw_avg", 0.0) for s in samples])
        pitches = np.array([s.features.get("gaze_pitch_avg", 0.0) for s in samples])

        mean_y, std_y = yaws.mean(), yaws.std()
        mean_p, std_p = pitches.mean(), pitches.std()

        cleaned = [
            s
            for s, y, p in zip(samples, yaws, pitches)
            if (
                abs(y - mean_y) <= self.outlier_sigma * (std_y + 1e-6)
                and abs(p - mean_p) <= self.outlier_sigma * (std_p + 1e-6)
            )
        ]

        removed = len(samples) - len(cleaned)
        if removed:
            _logger.debug("Outlier removal: %d/%d samples kept", len(cleaned), len(samples))

        return cleaned


def _drain(q: queue.Queue, max_drain: int = 100) -> None:
    """Drain stale items from a queue (non-blocking)."""
    for _ in range(max_drain):
        try:
            q.get_nowait()
        except queue.Empty:
            break
