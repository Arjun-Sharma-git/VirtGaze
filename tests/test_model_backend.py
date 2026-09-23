"""Tests for the inference backend factory."""

from __future__ import annotations

from gaze_estimation.model.backend import VALID_BACKENDS, create_predictor


def test_torch_backend_returns_no_external_predictor():
    predictor, name = create_predictor("torch")
    assert predictor is None
    assert name == "torch"


def test_backend_name_is_case_insensitive():
    _, name = create_predictor("TORCH")
    assert name == "torch"


def test_empty_backend_defaults_to_torch():
    _, name = create_predictor("")
    assert name == "torch"


def test_missing_model_path_falls_back_to_torch():
    predictor, name = create_predictor("onnx", model_path=None)
    assert predictor is None
    assert name == "torch"


def test_unknown_backend_falls_back_to_torch():
    predictor, name = create_predictor("banana", model_path="whatever.onnx")
    assert predictor is None
    assert name == "torch"


def test_unloadable_model_falls_back_to_torch(tmp_path):
    """A missing/invalid model file must degrade gracefully, not raise."""
    for backend in ("onnx", "tensorrt"):
        predictor, name = create_predictor(backend, model_path=str(tmp_path / "missing-model"))
        assert predictor is None
        assert name == "torch"


def test_valid_backends_are_declared():
    assert set(VALID_BACKENDS) == {"torch", "onnx", "tensorrt"}
