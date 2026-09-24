"""Tests for MLPTrainer: training convergence, save/load, normalisation."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.schemas import FEATURE_DIM


def _make_data(n: int = 300):
    """Synthetic linear mapping: y = f(x[:2]) for testing convergence."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32)
    # Simple linear target (should be learnable)
    sw, sh = 1920, 1080
    Y = np.column_stack(
        [
            sw * 0.5 + X[:, 0] * 50,
            sh * 0.5 + X[:, 1] * 50,
        ]
    ).astype(np.float32)
    return X, Y


def test_trainer_train_returns_model():
    X, Y = _make_data()
    trainer = MLPTrainer(epochs=10, batch_size=64, hidden_dims=[32])
    model = trainer.train(X, Y, screen_width=1920, screen_height=1080)
    assert isinstance(model, GazeMLP)


# ── One-sample tail batches ───────────────────────────────────────────────────

# Sample counts whose 90 % training split leaves a remainder of exactly 1 for at
# least one of the batch sizes below, e.g. 55 → 49 train → 16+16+16+1.
_TAIL_BATCH_COUNTS = [10, 11, 19, 28, 37, 46, 55, 64, 73, 82, 90, 91, 108]


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("n", _TAIL_BATCH_COUNTS)
def test_training_survives_a_one_sample_tail_batch(n, batch_size):
    """BatchNorm1d cannot train on a batch of one.

    A 90 % split can strand exactly one sample in the last batch, which used to
    raise ``ValueError: Expected more than 1 value per channel``.  OnlineTrainer
    catches that and logs it, so click adaptation silently never ran for these
    counts — hence the explicit regression list.
    """
    X, Y = _make_data(n)
    trainer = MLPTrainer(epochs=1, batch_size=batch_size, hidden_dims=[8, 8], device="cpu")

    model = trainer.train(X, Y, screen_width=1920, screen_height=1080)

    assert isinstance(model, GazeMLP)
    assert model.predict_numpy(trainer.normalise(X[:1])).shape == (1, 2)


def test_training_still_uses_every_sample_when_no_tail_exists():
    """The tail is only dropped in the pathological case."""
    X, Y = _make_data(90)  # 81 train samples, 81 % 32 = 17
    trainer = MLPTrainer(epochs=1, batch_size=32, hidden_dims=[8, 8], device="cpu")

    model = trainer.train(X, Y)

    assert isinstance(model, GazeMLP)


def test_trainer_output_in_range():
    X, Y = _make_data()
    trainer = MLPTrainer(epochs=5, batch_size=64, hidden_dims=[32])
    model = trainer.train(X, Y)
    import torch

    x = torch.from_numpy(trainer.normalise(X[:4]))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_trainer_save_load():
    X, Y = _make_data(100)
    trainer = MLPTrainer(epochs=5, batch_size=32, hidden_dims=[16])
    model = trainer.train(X, Y)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.save(model, path)
        model2, trainer2 = trainer.load(path)
        assert isinstance(model2, GazeMLP)
        assert trainer2.feature_mean is not None
    finally:
        os.unlink(path)


def test_trainer_normalise_consistent():
    X, _ = _make_data()
    trainer = MLPTrainer(epochs=5, batch_size=32, hidden_dims=[16])
    trainer._normalise_features(X, fit=True)
    X2 = trainer.normalise(X[:10])
    assert X2.shape == (10, FEATURE_DIM)
    assert np.isfinite(X2).all()


def test_fine_tune_preserves_normalisation_statistics():
    """Fine-tuning must not refit the z-score stats on the small new batch."""
    X, Y = _make_data(300)
    trainer = MLPTrainer(epochs=10, batch_size=64, hidden_dims=[32])
    model = trainer.train(X, Y, screen_width=1920, screen_height=1080)
    mean_before = trainer.feature_mean.copy()
    std_before = trainer.feature_std.copy()

    # Deliberately shifted, tiny batch: refitting here would corrupt the stats.
    trainer.fine_tune(model, (X[:20] + 5.0).astype(np.float32), Y[:20], epochs=2)

    assert np.allclose(trainer.feature_mean, mean_before)
    assert np.allclose(trainer.feature_std, std_before)


def test_fine_tune_restores_learning_rate_and_epochs():
    X, Y = _make_data(100)
    trainer = MLPTrainer(epochs=7, batch_size=32, hidden_dims=[16], learning_rate=1e-3)
    model = trainer.train(X, Y)

    trainer.fine_tune(model, X[:20], Y[:20], epochs=2, lr=1e-5)

    assert trainer.epochs == 7
    assert trainer.lr == 1e-3


def test_features_dict_to_vector():
    from gaze_estimation.pipeline.schemas import FEATURE_KEYS

    X, _ = _make_data()
    trainer = MLPTrainer(epochs=3, batch_size=32, hidden_dims=[16])
    trainer._normalise_features(X, fit=True)
    feats = {k: 0.0 for k in FEATURE_KEYS}
    vec = trainer.features_dict_to_vector(feats)
    assert vec.shape == (1, FEATURE_DIM)
