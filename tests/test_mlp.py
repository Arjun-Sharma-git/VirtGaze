"""Tests for GazeMLP model."""

from __future__ import annotations

import numpy as np
import torch

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.pipeline.schemas import FEATURE_DIM


def test_mlp_output_shape():
    model = GazeMLP(input_dim=FEATURE_DIM, hidden_dims=[32, 32])
    model.eval()
    x = torch.randn(4, FEATURE_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)


def test_mlp_output_range():
    """Sigmoid output should be in [0, 1]."""
    model = GazeMLP()
    model.eval()
    x = torch.randn(16, FEATURE_DIM) * 10  # Large inputs
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_mlp_no_nan():
    model = GazeMLP()
    model.eval()
    x = torch.zeros(1, FEATURE_DIM)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_mlp_predict_numpy():
    model = GazeMLP()
    x = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    out = model.predict_numpy(x)
    assert out.shape == (1, 2)
    assert np.isfinite(out).all()


def test_mlp_default_hidden_dims():
    model = GazeMLP()
    assert model.hidden_dims == [64, 128, 64]


def test_mlp_custom_hidden_dims():
    model = GazeMLP(hidden_dims=[16, 32])
    x = torch.randn(2, FEATURE_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
