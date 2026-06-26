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
    Y = np.column_stack([
        sw * 0.5 + X[:, 0] * 50,
        sh * 0.5 + X[:, 1] * 50,
    ]).astype(np.float32)
    return X, Y


def test_trainer_train_returns_model():
    X, Y = _make_data()
    trainer = MLPTrainer(epochs=10, batch_size=64, hidden_dims=[32])
    model = trainer.train(X, Y, screen_width=1920, screen_height=1080)
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


def test_features_dict_to_vector():
    from gaze_estimation.pipeline.schemas import FEATURE_KEYS
    X, _ = _make_data()
    trainer = MLPTrainer(epochs=3, batch_size=32, hidden_dims=[16])
    trainer._normalise_features(X, fit=True)
    feats = {k: 0.0 for k in FEATURE_KEYS}
    vec = trainer.features_dict_to_vector(feats)
    assert vec.shape == (1, FEATURE_DIM)
