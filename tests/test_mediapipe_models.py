"""Pinned MediaPipe model downloads are checksum-verified."""

from __future__ import annotations

import hashlib

import pytest

from gaze_estimation.utils import mediapipe_compat as mc


def _model(content: bytes):
    return ("m.task", "https://example.invalid/m.task", hashlib.sha256(content).hexdigest())


def _fake_download(content: bytes, calls: list):
    def fake(url, dst):
        calls.append(url)
        with open(dst, "wb") as fh:
            fh.write(content)

    return fake


def test_good_download_is_kept(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setenv("GAZE_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mc.urllib.request, "urlretrieve", _fake_download(b"ok", calls))

    path = mc.ensure_model(_model(b"ok"))
    again = mc.ensure_model(_model(b"ok"))

    assert path == again and open(path, "rb").read() == b"ok"
    assert len(calls) == 1  # cached once verified


def test_tampered_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZE_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mc.urllib.request, "urlretrieve", _fake_download(b"evil", []))

    with pytest.raises(RuntimeError, match="checksum"):
        mc.ensure_model(_model(b"ok"))
    assert list(tmp_path.iterdir()) == []


def test_stale_local_file_is_replaced(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setenv("GAZE_MODELS_DIR", str(tmp_path))
    (tmp_path / "m.task").write_bytes(b"old")
    monkeypatch.setattr(mc.urllib.request, "urlretrieve", _fake_download(b"ok", calls))

    path = mc.ensure_model(_model(b"ok"))

    assert open(path, "rb").read() == b"ok" and len(calls) == 1


def test_pinned_models_are_versioned():
    for _, url, sha in (mc.FACE_DETECTOR_MODEL, mc.FACE_LANDMARKER_MODEL):
        assert "/latest/" not in url and len(sha) == 64
