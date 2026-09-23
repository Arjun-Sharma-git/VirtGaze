"""Online incremental MLP fine-tuner using click-based adaptation samples."""
from __future__ import annotations

from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("adaptation.online_trainer")


class OnlineTrainer:
    """Monitor the :class:`AdaptationBuffer` and periodically fine-tune the MLP.

    Args:
        mlp:                 The current GazeMLP to fine-tune.
        buffer:              The adaptation buffer.
        trainer:             The MLPTrainer with stored normalisation stats.
        retrain_threshold:   Number of new samples to trigger a retrain.
        retrain_epochs:      Epochs to use for fine-tuning.
        learning_rate:       Fine-tune learning rate (lower than initial training).
        batch_size:          Fine-tune batch size.
        screen_width:        Display width for label normalisation.
        screen_height:       Display height for label normalisation.
    """

    def __init__(
        self,
        mlp: GazeMLP,
        buffer: AdaptationBuffer,
        trainer: MLPTrainer,
        retrain_threshold: int = 50,
        retrain_epochs: int = 2,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self._mlp = mlp
        self._buffer = buffer
        self._trainer = trainer
        self._threshold = retrain_threshold
        self._retrain_epochs = retrain_epochs
        self._lr = learning_rate
        self._batch_size = batch_size
        self._screen_width = screen_width
        self._screen_height = screen_height

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def model(self) -> GazeMLP:
        """The current (possibly fine-tuned) model."""
        return self._mlp

    def maybe_retrain(self) -> bool:
        """Fine-tune the model if enough new samples have accumulated.

        Returns:
            True if a retrain was performed.
        """
        if self._buffer.count_new_since_last_train() < self._threshold:
            return False

        X, Y = self._buffer.get_batch()
        if len(X) < self._threshold:
            return False

        _logger.info(
            "Fine-tuning MLP on %d adaptation samples "
            "(%d new) for %d epochs …",
            len(X), self._buffer.count_new_since_last_train(), self._retrain_epochs,
        )

        # Store original settings, override for fine-tune
        orig_epochs = self._trainer.epochs
        orig_lr = self._trainer.lr
        orig_bs = self._trainer.batch_size

        self._trainer.epochs = self._retrain_epochs
        self._trainer.lr = self._lr
        self._trainer.batch_size = self._batch_size

        try:
            self._mlp = self._trainer.train(
                X, Y,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
                existing_model=self._mlp,
                refit_normalisation=False,
            )
        except Exception as exc:
            _logger.error("Fine-tuning failed: %s", exc)
            return False
        finally:
            # Restore original settings
            self._trainer.epochs = orig_epochs
            self._trainer.lr = orig_lr
            self._trainer.batch_size = orig_bs

        self._buffer.mark_trained()
        _logger.info("Fine-tuning complete")
        return True

    def feed_click(
        self,
        features: dict,
        cursor_x: float,
        cursor_y: float,
        mouse_velocity: float = 0.0,
        click_velocity_threshold: float = 500.0,
    ) -> None:
        """Add a click event to the adaptation buffer.

        Ignores clicks during fast mouse movement (not likely gaze-directed).

        Args:
            features:                Feature dict from the current GazePacket.
            cursor_x:                Mouse X at click time (pixels).
            cursor_y:                Mouse Y at click time (pixels).
            mouse_velocity:          Mouse movement speed at click time (px/s).
            click_velocity_threshold: Ignore clicks above this speed.
        """
        if mouse_velocity > click_velocity_threshold:
            return
        self._buffer.add(features, cursor_x, cursor_y)
