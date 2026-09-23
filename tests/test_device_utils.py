"""Tests for device selection and ONNX execution-provider mapping.

Availability is forced with monkeypatch rather than detected, so the assertions
hold on a CPU-only machine, a CUDA box, and CI alike.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from gaze_estimation.utils.device import (
    Backend,
    best_device_string,
    describe_device,
    get_onnx_providers,
    get_torch_device,
    is_cuda,
    is_rocm,
)


def _force_cuda_available(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def test_backend_enum_values():
    assert [b.value for b in Backend] == ["CPU", "CUDA", "ROCM", "TENSORRT"]


def test_cpu_request_returns_cpu():
    assert get_torch_device("CPU") == torch.device("cpu")


def test_unrecognised_request_returns_cpu():
    assert get_torch_device("banana") == torch.device("cpu")


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    _force_cuda_available(monkeypatch, False)
    assert get_torch_device("auto") == torch.device("cpu")


def test_auto_prefers_cuda_when_available(monkeypatch):
    _force_cuda_available(monkeypatch, True)
    assert get_torch_device("auto") == torch.device("cuda")


def test_cuda_request_warns_and_falls_back_when_unavailable(monkeypatch):
    _force_cuda_available(monkeypatch, False)
    with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
        assert get_torch_device("CUDA") == torch.device("cpu")


def test_rocm_request_uses_the_same_cuda_device(monkeypatch):
    _force_cuda_available(monkeypatch, True)
    assert get_torch_device("ROCM") == torch.device("cuda")


def test_is_cuda_matches_torch():
    assert is_cuda() is bool(torch.cuda.is_available())


def test_is_rocm_matches_the_torch_build():
    assert is_rocm() is (getattr(torch.version, "hip", None) is not None)


def test_device_probes_degrade_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail

    assert is_cuda() is False
    assert is_rocm() is False
    assert describe_device() == "CPU (torch not available or error)"
    assert best_device_string() == "CPU"


def test_describe_device_reports_cpu_without_cuda(monkeypatch):
    _force_cuda_available(monkeypatch, False)
    assert describe_device() == "CPU"


def test_best_device_string_reports_cpu_without_cuda(monkeypatch):
    _force_cuda_available(monkeypatch, False)
    assert best_device_string() == "CPU"


def test_best_device_string_distinguishes_rocm_from_cuda(monkeypatch):
    _force_cuda_available(monkeypatch, True)
    expected = "ROCM" if getattr(torch.version, "hip", None) else "CUDA"
    assert best_device_string() == expected


def test_cuda_providers_are_requested_before_cpu():
    assert get_onnx_providers("CUDA") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_rocm_providers_are_requested_before_cpu():
    assert get_onnx_providers("ROCM") == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_unrecognised_device_maps_to_cpu():
    assert get_onnx_providers("banana") == ["CPUExecutionProvider"]


def test_auto_prefers_rocm_over_cuda(monkeypatch):
    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: ["ROCMExecutionProvider", "CUDAExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    assert get_onnx_providers("auto")[0] == "ROCMExecutionProvider"

    fake.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert get_onnx_providers("auto")[0] == "CUDAExecutionProvider"


def test_auto_without_onnxruntime_uses_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert get_onnx_providers("auto") == ["CPUExecutionProvider"]


def test_auto_matches_the_installed_providers():
    ort = pytest.importorskip("onnxruntime")
    available = ort.get_available_providers()

    providers = get_onnx_providers("auto")

    if "ROCMExecutionProvider" in available:
        assert providers[0] == "ROCMExecutionProvider"
    elif "CUDAExecutionProvider" in available:
        assert providers[0] == "CUDAExecutionProvider"
    else:
        assert providers == ["CPUExecutionProvider"]
