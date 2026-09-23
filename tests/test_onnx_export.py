"""Tests for exporting a GazeMLP to ONNX.

The export is exercised for real whenever ``onnx`` / ``onnxruntime`` are
importable, including a numerical comparison against the torch model — the point
of the file is that the exported graph agrees with the trainer it was built from.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np
import pytest
import torch

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.model.onnx_export import export_to_onnx

FEATURE_DIM = 34


@pytest.fixture
def tiny_mlp():
    torch.manual_seed(0)
    return GazeMLP()


@pytest.fixture
def exported(tmp_path, tiny_mlp):
    path = tmp_path / "model.onnx"
    export_to_onnx(tiny_mlp, str(path))
    return str(path), tiny_mlp


def test_creates_missing_parent_directories(tmp_path, tiny_mlp):
    dest = tmp_path / "runs" / "2026-01" / "model.onnx"

    export_to_onnx(tiny_mlp, str(dest))

    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_puts_the_model_in_eval_mode(tmp_path, tiny_mlp):
    tiny_mlp.train()

    export_to_onnx(tiny_mlp, str(tmp_path / "model.onnx"))

    assert tiny_mlp.training is False


def test_reports_the_destination(tmp_path, tiny_mlp, capsys):
    dest = tmp_path / "model.onnx"

    export_to_onnx(tiny_mlp, str(dest))

    assert f"[onnx_export] Exported to {dest}" in capsys.readouterr().out


def test_selects_the_torchscript_exporter(tmp_path, tiny_mlp, monkeypatch):
    """torch >= 2.6 defaults to the dynamo exporter, which needs onnxscript."""
    real_export = torch.onnx.export
    captured = {}

    @functools.wraps(real_export)  # keeps inspect.signature() following __wrapped__
    def spy(model, args, f, **kwargs):
        captured.update(kwargs)
        return real_export(model, args, f, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", spy)

    export_to_onnx(tiny_mlp, str(tmp_path / "model.onnx"))

    assert captured["opset_version"] == 17
    assert captured["input_names"] == ["features"]
    assert captured["output_names"] == ["screen_coords"]
    assert captured["dynamic_axes"] == {
        "features": {0: "batch"},
        "screen_coords": {0: "batch"},
    }
    assert captured["do_constant_folding"] is True
    if "dynamo" in inspect.signature(real_export).parameters:
        assert captured["dynamo"] is False


def test_graph_has_the_expected_io_names_and_opset(exported):
    onnx = pytest.importorskip("onnx")
    path, _ = exported

    proto = onnx.load(path)

    assert [i.name for i in proto.graph.input] == ["features"]
    assert [o.name for o in proto.graph.output] == ["screen_coords"]
    assert {"ai.onnx": 17} == {o.domain or "ai.onnx": o.version for o in proto.opset_import}
    onnx.checker.check_model(proto)


def test_exported_graph_matches_the_torch_model(exported):
    ort = pytest.importorskip("onnxruntime")
    path, model = exported
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    features = np.random.default_rng(7).random((1, FEATURE_DIM)).astype(np.float32)

    got = session.run(None, {"features": features})[0]

    with torch.no_grad():
        expected = model(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_exported_graph_accepts_a_variable_batch(exported):
    ort = pytest.importorskip("onnxruntime")
    path, _ = exported
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    for batch in (1, 4):
        features = np.random.default_rng(batch).random((batch, FEATURE_DIM)).astype(np.float32)
        assert session.run(None, {"features": features})[0].shape == (batch, 2)


def test_custom_input_dimension_is_honoured(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = GazeMLP(input_dim=8)
    dest = tmp_path / "small.onnx"

    export_to_onnx(model, str(dest), input_dim=8)

    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[-1] == 8
    features = np.random.default_rng(1).random((1, 8)).astype(np.float32)
    assert session.run(None, {"features": features})[0].shape == (1, 2)
