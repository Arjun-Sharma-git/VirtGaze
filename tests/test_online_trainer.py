"""Tests for OnlineTrainer, the real incremental fine-tuner.

The implicit-calibration tests drive a *stub* trainer, so the real class was never
instantiated.  These tests exercise it against a real `MLPTrainer` and
`AdaptationBuffer`, including the setting override/restore contract and the
promise that fine-tuning does not disturb the stored normalisation statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
from gaze_estimation.adaptation.online_trainer import OnlineTrainer
from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.schemas import FEATURE_KEYS

W, H = 1920, 1080


def _trainer(**kwargs) -> MLPTrainer:
    kwargs.setdefault("epochs", 5)
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("hidden_dims", [8, 8])
    kwargs.setdefault("learning_rate", 0.01)
    kwargs.setdefault("device", "cpu")
    return MLPTrainer(**kwargs)


class _RecordingTrainer(MLPTrainer):
    """Records the settings in force at call time, then trains for real."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: list[dict] = []
        self.fail = False
        self.returned: list[GazeMLP] = []

    def train(
        self,
        X,
        Y,
        screen_width,
        screen_height,
        existing_model=None,
        refit_normalisation=True,
    ):
        self.seen.append(
            {
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "refit_normalisation": refit_normalisation,
                "n_samples": len(X),
                "existing_model": existing_model,
            }
        )
        if self.fail:
            raise RuntimeError("training blew up")
        model = super().train(
            X,
            Y,
            screen_width,
            screen_height,
            existing_model=existing_model,
            refit_normalisation=refit_normalisation,
        )
        self.returned.append(model)
        return model


def _features(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {key: float(rng.normal()) for key in FEATURE_KEYS}


def _filled_buffer(n: int, max_size: int = 5000) -> AdaptationBuffer:
    buffer = AdaptationBuffer(max_size=max_size)
    for i in range(n):
        buffer.add(_features(i), float((i * 37) % W), float((i * 53) % H))
    return buffer


def _click_pairs(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(features, screen coords) shaped like a real adaptation batch."""
    X = np.random.default_rng(7).normal(size=(n, len(FEATURE_KEYS))).astype(np.float32)
    Y = np.random.default_rng(8).uniform(0, 1000, size=(n, 2)).astype(np.float32)
    return X, Y


# ── Construction ──────────────────────────────────────────────────────────────


def test_model_property_returns_the_current_model():
    mlp = GazeMLP(hidden_dims=[8, 8])
    trainer = OnlineTrainer(mlp, AdaptationBuffer(), _trainer())

    assert trainer.model is mlp


# ── maybe_retrain gating ──────────────────────────────────────────────────────


def test_no_retrain_below_the_threshold():
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    buffer = _filled_buffer(5)
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, recorder, retrain_threshold=50)

    assert trainer.maybe_retrain() is False
    assert recorder.seen == []


def test_no_retrain_when_the_batch_is_too_small():
    """Enough *new* samples, but the ring buffer dropped most of them."""
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    buffer = _filled_buffer(20, max_size=5)
    assert buffer.count_new_since_last_train() == 20  # above the threshold
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, recorder, retrain_threshold=10)

    assert trainer.maybe_retrain() is False
    assert recorder.seen == []


def test_retrain_happens_once_the_threshold_is_reached():
    X, Y = _click_pairs(60)
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    mlp = GazeMLP(hidden_dims=[8, 8])
    buffer = AdaptationBuffer()
    for features, coords in zip(X, Y):
        buffer.add(dict(zip(FEATURE_KEYS, features.tolist())), float(coords[0]), float(coords[1]))
    trainer = OnlineTrainer(mlp, buffer, recorder, retrain_threshold=50)

    assert trainer.maybe_retrain() is True
    assert recorder.seen[0]["n_samples"] == 60
    # Fine-tuning is in place: the live model instance is updated, not replaced.
    assert recorder.seen[0]["existing_model"] is mlp
    assert trainer.model is mlp
    assert isinstance(trainer.model, GazeMLP)


def test_a_second_call_does_not_retrain_immediately():
    buffer = _filled_buffer(60)
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, recorder, retrain_threshold=50)

    assert trainer.maybe_retrain() is True
    assert buffer.count_new_since_last_train() == 0  # marked trained
    assert trainer.maybe_retrain() is False
    assert len(recorder.seen) == 1


# ── Setting override / restore ────────────────────────────────────────────────


def _tuned_recorder() -> _RecordingTrainer:
    """A trainer whose defaults differ from the fine-tune settings."""
    return _RecordingTrainer(
        hidden_dims=[8, 8], epochs=9, batch_size=4, learning_rate=0.05, device="cpu"
    )


def test_fine_tune_settings_apply_during_training():
    recorder = _tuned_recorder()
    buffer = _filled_buffer(60)
    trainer = OnlineTrainer(
        GazeMLP(hidden_dims=[8, 8]),
        buffer,
        recorder,
        retrain_threshold=50,
        retrain_epochs=2,
        learning_rate=1e-4,
        batch_size=16,
    )

    assert trainer.maybe_retrain() is True

    assert recorder.seen[0]["epochs"] == 2
    assert recorder.seen[0]["lr"] == pytest.approx(1e-4)
    assert recorder.seen[0]["batch_size"] == 16


def test_original_settings_are_restored_afterwards():
    recorder = _tuned_recorder()
    buffer = _filled_buffer(60)
    trainer = OnlineTrainer(
        GazeMLP(hidden_dims=[8, 8]),
        buffer,
        recorder,
        retrain_threshold=50,
        retrain_epochs=2,
        learning_rate=1e-4,
        batch_size=16,
    )

    trainer.maybe_retrain()

    assert recorder.epochs == 9
    assert recorder.lr == pytest.approx(0.05)
    assert recorder.batch_size == 4


def test_settings_are_restored_even_when_training_fails():
    """The restore lives in a finally block, so a crash must not leak the override."""
    recorder = _tuned_recorder()
    recorder.fail = True
    buffer = _filled_buffer(60)
    trainer = OnlineTrainer(
        GazeMLP(hidden_dims=[8, 8]),
        buffer,
        recorder,
        retrain_threshold=50,
        retrain_epochs=2,
        learning_rate=1e-4,
        batch_size=16,
    )

    assert trainer.maybe_retrain() is False

    assert recorder.epochs == 9
    assert recorder.lr == pytest.approx(0.05)
    assert recorder.batch_size == 4


def test_a_failed_retrain_does_not_mark_the_buffer_trained():
    recorder = _tuned_recorder()
    recorder.fail = True
    buffer = _filled_buffer(60)
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, recorder, retrain_threshold=50)

    trainer.maybe_retrain()

    assert buffer.count_new_since_last_train() == 60  # still pending
    assert trainer.maybe_retrain() is False


def test_a_failed_retrain_keeps_the_previous_model():
    recorder = _tuned_recorder()
    recorder.fail = True
    mlp = GazeMLP(hidden_dims=[8, 8])
    trainer = OnlineTrainer(mlp, _filled_buffer(60), recorder, retrain_threshold=50)

    trainer.maybe_retrain()

    assert trainer.model is mlp


def test_fine_tuning_does_not_refit_normalisation_statistics():
    """Refitting on a small click batch would shift the live model's inputs."""
    X_seed, Y_seed = _click_pairs(64)
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    recorder.train(X_seed, Y_seed, W, H)  # establishes feature_mean/std
    mean_before = recorder.feature_mean.copy()
    std_before = recorder.feature_std.copy()

    trainer = OnlineTrainer(
        GazeMLP(hidden_dims=[8, 8]), _filled_buffer(60), recorder, retrain_threshold=50
    )
    assert trainer.maybe_retrain() is True

    assert recorder.seen[-1]["refit_normalisation"] is False
    np.testing.assert_allclose(recorder.feature_mean, mean_before)
    np.testing.assert_allclose(recorder.feature_std, std_before)


# ── feed_click ────────────────────────────────────────────────────────────────


def test_feed_click_stores_the_sample():
    buffer = AdaptationBuffer()
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, _trainer())

    trainer.feed_click(_features(), 640.0, 360.0)

    assert len(buffer) == 1
    assert buffer.count_new_since_last_train() == 1
    X, Y = buffer.get_batch()
    np.testing.assert_allclose(Y[0], [640.0, 360.0])


def test_feed_click_ignores_a_fast_mouse():
    """A click during a flick is not a gaze-directed target."""
    buffer = AdaptationBuffer()
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, _trainer())

    trainer.feed_click(_features(), 640.0, 360.0, mouse_velocity=900.0)

    assert len(buffer) == 0


def test_feed_click_accepts_a_slow_mouse():
    buffer = AdaptationBuffer()
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, _trainer())

    trainer.feed_click(_features(), 640.0, 360.0, mouse_velocity=10.0)

    assert len(buffer) == 1


def test_feed_click_threshold_is_configurable():
    buffer = AdaptationBuffer()
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, _trainer())

    trainer.feed_click(
        _features(), 640.0, 360.0, mouse_velocity=200.0, click_velocity_threshold=100.0
    )

    assert len(buffer) == 0


def test_clicks_feed_a_retrain():
    """The end-to-end adaptation loop: clicks accumulate, then trigger training."""
    recorder = _RecordingTrainer(hidden_dims=[8, 8], batch_size=8, epochs=1)
    buffer = AdaptationBuffer()
    trainer = OnlineTrainer(GazeMLP(hidden_dims=[8, 8]), buffer, recorder, retrain_threshold=20)

    for i in range(25):
        trainer.feed_click(_features(i), float(i * 10), float(i * 5))

    assert trainer.maybe_retrain() is True
    assert len(recorder.seen) == 1
