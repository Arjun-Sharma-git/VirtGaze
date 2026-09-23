"""Device detection and selection utilities for CPU / CUDA / ROCm."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Backend(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    ROCM = "ROCM"
    TENSORRT = "TENSORRT"


def get_torch_device(requested: str = "CPU") -> torch.device:
    """Return the best available torch device for the requested backend.

    Args:
        requested: One of "CPU", "CUDA", "ROCM", "auto".
                   "auto" picks ROCM → CUDA → CPU in priority order.

    Returns:
        A ``torch.device`` object.

    Notes:
        - On AMD GPUs with ROCm, PyTorch exposes HIP through the same
          ``torch.cuda.*`` API.  ``torch.version.hip`` is not None when
          running with ROCm.
        - Pass "ROCM" and "CUDA" interchangeably — the runtime decides.
    """
    import torch

    req = requested.upper()

    if req == "AUTO":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if req in ("CUDA", "ROCM"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        import warnings

        warnings.warn(
            f"Requested device '{requested}' but CUDA/ROCm is not available. "
            "Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device("cpu")


def is_rocm() -> bool:
    """Return True if the installed PyTorch was built with ROCm / HIP."""
    try:
        import torch

        return torch.version.hip is not None  # type: ignore[attr-defined]
    except Exception:
        return False


def is_cuda() -> bool:
    """Return True if CUDA is available (also True on ROCm systems)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def describe_device() -> str:
    """Return a human-readable description of the best available device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU"
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if n > 0 else "unknown"
        hip = torch.version.hip  # type: ignore[attr-defined]
        if hip:
            return f"ROCm ({hip}) — AMD GPU: {name} ({n} device(s))"
        cuda = torch.version.cuda
        return f"CUDA ({cuda}) — GPU: {name} ({n} device(s))"
    except Exception:
        return "CPU (torch not available or error)"


def get_onnx_providers(device: str = "CPU") -> list[str]:
    """Return the ordered list of ONNX Runtime execution providers.

    Args:
        device: "CPU", "CUDA", "ROCM", or "auto".

    Returns:
        List of provider strings for ``onnxruntime.InferenceSession``.

    Notes:
        - ROCm requires ``onnxruntime-rocm`` package.
        - CUDA requires ``onnxruntime-gpu`` package.
        - Falls back to ``CPUExecutionProvider`` if GPU providers are
          unavailable at runtime.
    """
    dev = device.upper()

    if dev in ("CUDA",):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if dev == "ROCM":
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]

    if dev == "AUTO":
        # Probe available providers
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if "ROCMExecutionProvider" in available:
                return ["ROCMExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            pass
        return ["CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def best_device_string() -> str:
    """Return "ROCM", "CUDA", or "CPU" based on available hardware."""
    try:
        import torch

        if torch.cuda.is_available():
            if is_rocm():
                return "ROCM"
            return "CUDA"
    except ImportError:
        pass
    return "CPU"
