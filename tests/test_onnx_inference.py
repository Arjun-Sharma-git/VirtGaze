"""Tests for the ONNX Runtime inference wrapper.

These run against a real exported model, so the wrapper's contract with the
exported graph (input name, float32 input, batch handling) is verified rather
than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.onnx_export import export_to_onnx
from gaze_estimation.model.onnx_inference import ONNXInference
from gaze_estimation.pipeline.schemas import FEATURE_KEYS

FEATURE_DIM = 34


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory):
    """(path, torch model) for one exported GazeMLP, shared by this module."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")  # the exporter needs it
    torch.manual_seed(0)
    model = GazeMLP()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_to_onnx(model, str(path))
    return str(path), model


@pytest.fixture
def inference(exported_model):
    return ONNXInference(exported_model[0], device="CPU")


def _features(batch: int = 1, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).random((batch, FEATURE_DIM)).astype(np.float32)


def test_predict_matches_the_torch_model(inference, exported_model):
    _, model = exported_model
    features = _features()

    got = inference.predict(features)

    with torch.no_grad():
        expected = model(torch.from_numpy(features.astype(np.float32))).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_predict_returns_float32(inference):
    assert inference.predict(_features()).dtype == np.float32


def test_predict_promotes_a_single_unbatched_sample(inference):
    got = inference.predict(_features()[0])  # shape (34,)

    assert got.shape == (1, 2)


def test_predict_handles_a_batch(inference):
    got = inference.predict(_features(batch=4))

    assert got.shape == (4, 2)
    assert np.all((got >= 0.0) & (got <= 1.0))  # sigmoid output


def test_predict_dict_without_statistics_matches_predict(inference):
    features = {key: float(value) for key, value in zip(FEATURE_KEYS, _features()[0])}

    got = inference.predict_dict(features, list(FEATURE_KEYS))

    expected = inference.predict(
        np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_predict_dict_with_missing_keys_defaults_to_zero(inference):
    got = inference.predict_dict({}, list(FEATURE_KEYS))

    assert got.shape == (1, 2)


def test_predict_dict_keeps_float32_with_float64_statistics(inference, exported_model):
    """Regression: float64 z-score stats used to promote the ORT input type."""
    _, model = exported_model
    raw = _features()[0]
    features = {key: float(value) for key, value in zip(FEATURE_KEYS, raw)}
    mean = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    std = np.full(len(FEATURE_KEYS), 2.0, dtype=np.float64)

    got = inference.predict_dict(features, list(FEATURE_KEYS), mean=mean, std=std)

    assert got.dtype == np.float32
    expected = model.predict_numpy((raw / 2.0).astype(np.float32))  # (raw - 0) / 2
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_cpu_provider_is_active(inference):
    assert "CPUExecutionProvider" in inference.active_providers


def test_repr_lists_the_active_providers(inference):
    assert repr(inference) == f"ONNXInference(providers={inference.active_providers})"


def test_gpu_request_degrades_to_a_working_session(exported_model):
    """A CUDA/ROCm request on a CPU-only build must still produce a session."""
    inference = ONNXInference(exported_model[0], device="CUDA")

    assert "CPUExecutionProvider" in inference.active_providers
    assert inference.predict(_features()).shape == (1, 2)


def test_unknown_device_falls_back_to_cpu(exported_model):
    inference = ONNXInference(exported_model[0], device="banana")

    assert inference.active_providers == ["CPUExecutionProvider"]


def test_unreadable_model_path_raises(tmp_path):
    """create_predictor relies on this raising so it can fall back to torch."""
    with pytest.raises(Exception):
        ONNXInference(str(tmp_path / "does-not-exist.onnx"))
