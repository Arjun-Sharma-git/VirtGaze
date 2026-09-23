"""Inference backend factory — selects the torch / ONNX / TensorRT predictor."""
from __future__ import annotations

from typing import Optional, Tuple

from gaze_estimation.utils.logging import get_logger

_logger = get_logger("model.backend")

VALID_BACKENDS = ("torch", "onnx", "tensorrt")


def create_predictor(
    backend: str,
    *,
    model=None,
    model_path: Optional[str] = None,
    device: str = "CPU",
    input_dim: int = 34,
) -> Tuple[Optional[object], str]:
    """Create an inference predictor for the requested backend.

    Returns ``(predictor, backend_name)``.  ``predictor`` is ``None`` when the
    in-process torch model should be used — either because that was requested,
    or because the requested backend is unavailable — in which case
    ``backend_name`` is ``"torch"``.

    Every returned predictor accepts a *normalised* ``(1, input_dim)`` feature
    array (see :meth:`MLPTrainer.features_dict_to_vector`) and returns
    ``(1, 2)`` normalised screen coordinates in ``[0, 1]``.

    Args:
        backend:    ``"torch"``, ``"onnx"`` or ``"tensorrt"`` (case-insensitive).
        model:      Trained ``GazeMLP`` (torch backend).
        model_path: Path to an exported ``.onnx`` / ``.trt`` file.
        device:     ONNX Runtime device: ``"CPU"``, ``"CUDA"``, ``"ROCM"``, ``"auto"``.
        input_dim:  Feature dimension (TensorRT only).
    """
    name = (backend or "torch").strip().lower()
    if name == "torch":
        return None, "torch"

    if not model_path:
        _logger.warning(
            "Backend '%s' requested but no model path was given — "
            "using the in-process torch model instead.",
            name,
        )
        return None, "torch"

    try:
        if name == "onnx":
            from gaze_estimation.model.onnx_inference import ONNXInference

            return ONNXInference(model_path, device=device), "onnx"
        if name == "tensorrt":
            from gaze_estimation.model.tensorrt_inference import TensorRTInference

            return TensorRTInference(model_path, input_dim=input_dim), "tensorrt"
    except Exception as exc:
        _logger.warning(
            "Could not initialise the '%s' backend (%s) — "
            "using the in-process torch model instead.",
            name,
            exc,
        )
        return None, "torch"

    _logger.warning("Unknown inference backend '%s' — using torch.", name)
    return None, "torch"
