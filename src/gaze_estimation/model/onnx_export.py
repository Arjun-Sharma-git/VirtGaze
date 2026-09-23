"""Export a trained GazeMLP to ONNX format."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaze_estimation.model.mlp import GazeMLP


def export_to_onnx(model: GazeMLP, path: str, input_dim: int = 34) -> None:
    """Export *model* to ONNX at *path*.

    The exported graph expects the **normalised** feature vector produced by
    :meth:`MLPTrainer.features_dict_to_vector` — normalisation is deliberately
    not part of the graph, so callers must apply it before inference.

    Args:
        model:     Trained GazeMLP in eval mode.
        path:      Destination .onnx file path.
        input_dim: Number of input features (default 34).
    """
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["features"],
        output_names=["screen_coords"],
        dynamic_axes={
            "features": {0: "batch"},
            "screen_coords": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[onnx_export] Exported to {path}")
