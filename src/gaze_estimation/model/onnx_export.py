"""Export a trained GazeMLP to ONNX format."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaze_estimation.model.mlp import GazeMLP


def _supports_dynamo_kwarg(torch_module: Any) -> bool:
    """Whether ``torch.onnx.export`` accepts the ``dynamo`` switch.

    Introduced with the dynamo exporter (torch 2.6); older versions raise
    ``TypeError`` on an unknown keyword, so callers must ask first.
    """
    try:
        return "dynamo" in inspect.signature(torch_module.onnx.export).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic/decorated builds
        return False


def export_to_onnx(model: GazeMLP, path: str, input_dim: int = 34) -> None:
    """Export *model* to ONNX at *path*.

    The exported graph expects the **normalised** feature vector produced by
    :meth:`MLPTrainer.features_dict_to_vector` — normalisation is deliberately
    not part of the graph, so callers must apply it before inference.

    The TorchScript exporter is selected explicitly.  torch ≥ 2.6 defaults to
    the dynamo exporter, which imports ``onnxscript`` — a package this project
    does not depend on — so the default path fails outright with
    ``ModuleNotFoundError: No module named 'onnxscript'``.  The TorchScript
    exporter emits the same small graph without that dependency.

    Args:
        model:     Trained GazeMLP in eval mode.
        path:      Destination .onnx file path.
        input_dim: Number of input features (default 34).
    """
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, input_dim)

    export_kwargs: dict[str, Any] = {
        "input_names": ["features"],
        "output_names": ["screen_coords"],
        "dynamic_axes": {
            "features": {0: "batch"},
            "screen_coords": {0: "batch"},
        },
        "opset_version": 17,
        "do_constant_folding": True,
    }
    if _supports_dynamo_kwarg(torch):
        export_kwargs["dynamo"] = False

    # torch.onnx.export accepts a Tensor as arg 2; its stubs mistype it as a
    # tuple of positional args.
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore[arg-type]
        path,
        **export_kwargs,
    )
    print(f"[onnx_export] Exported to {path}")
