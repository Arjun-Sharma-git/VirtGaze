"""Tests for the TensorRT inference wrapper's optional-dependency contract.

TensorRT and PyCUDA are not installable from PyPI in this project's dependency
set, so the important behaviour is that the *module* imports cleanly and the
*class* fails loudly and helpfully when the runtime is missing.
"""

from __future__ import annotations

import importlib.util

import pytest

from gaze_estimation.model.tensorrt_inference import TensorRTInference


def _runtime_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("tensorrt", "pycuda.driver"))


def test_module_imports_without_tensorrt():
    """The module must not import tensorrt at module scope."""
    import gaze_estimation.model.tensorrt_inference as module

    assert module.TensorRTInference is TensorRTInference


@pytest.mark.skipif(
    _runtime_available(), reason="tensorrt installed — the missing-runtime path cannot run"
)
def test_missing_runtime_raises_with_install_hint():
    with pytest.raises(RuntimeError, match="TensorRT or PyCUDA not installed"):
        TensorRTInference("model.trt")


@pytest.mark.skipif(
    _runtime_available(), reason="tensorrt installed — the missing-runtime path cannot run"
)
def test_error_chains_the_import_error():
    with pytest.raises(RuntimeError) as excinfo:
        TensorRTInference("model.trt")

    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.skipif(
    _runtime_available(), reason="tensorrt installed — the missing-runtime path cannot run"
)
def test_failure_happens_before_the_engine_file_is_read(tmp_path):
    """A missing runtime must be reported, not a confusing FileNotFoundError."""
    missing_engine = tmp_path / "does-not-exist.trt"

    with pytest.raises(RuntimeError):
        TensorRTInference(str(missing_engine))
