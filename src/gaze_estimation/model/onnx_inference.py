"""ONNX Runtime inference wrapper — supports CPU, CUDA, and ROCm."""
from __future__ import annotations

import os

import numpy as np


class ONNXInference:
    """Run the exported GazeMLP ONNX model via ONNX Runtime.

    Supports three backends via the *device* argument:

    ============  ========================  ==============================
    device        ORT Provider              Required package
    ============  ========================  ==============================
    ``"CPU"``     CPUExecutionProvider      ``onnxruntime``
    ``"CUDA"``    CUDAExecutionProvider     ``onnxruntime-gpu``
    ``"ROCM"``    ROCMExecutionProvider     ``onnxruntime-rocm``
    ``"auto"``    Best available            any of the above
    ============  ========================  ==============================

    Args:
        model_path: Path to the exported .onnx file.
        device:     Backend selector — "CPU", "CUDA", "ROCM", or "auto".
    """

    def __init__(self, model_path: str, device: str = "CPU") -> None:
        import onnxruntime as ort

        from gaze_estimation.utils.device import get_onnx_providers

        providers = get_onnx_providers(device)

        # Filter to providers actually installed / available
        available = ort.get_available_providers()
        filtered_providers = [p for p in providers if p in available]
        if not filtered_providers:
            filtered_providers = ["CPUExecutionProvider"]

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        n_cores = os.cpu_count() or 1
        session_opts.intra_op_num_threads = n_cores

        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=filtered_providers,
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._active_providers = self._session.get_providers()

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            features: (1, 34) or (N, 34) float32 feature array.

        Returns:
            (1, 2) or (N, 2) float32 — (screen_x_norm, screen_y_norm) in [0, 1].
        """
        x = features.astype(np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self._session.run(None, {self._input_name: x})[0]

    def predict_dict(
        self, features_dict: dict, feature_keys: list, mean=None, std=None
    ) -> np.ndarray:
        """Predict from a feature dict (keyed by ``FEATURE_KEYS``).

        The exported ONNX graph contains **no** feature normalisation, so the
        z-score statistics stored in the ``MLPTrainer`` must be supplied for
        the prediction to be correct.  Prefer
        ``trainer.features_dict_to_vector(features)`` followed by
        :meth:`predict`, which applies them automatically.
        """
        raw = np.array(
            [features_dict.get(k, 0.0) for k in feature_keys], dtype=np.float32
        ).reshape(1, -1)
        if mean is not None and std is not None:
            # Keep the result float32.  The ONNX graph's input is float32, and
            # dividing by a float64 std would silently promote the array, which
            # ORT then rejects as an input-type mismatch.
            mean_arr = np.asarray(mean, dtype=np.float32)
            std_arr = np.asarray(std, dtype=np.float32)
            raw = ((raw - mean_arr) / std_arr).astype(np.float32)
        return self.predict(raw)

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def active_providers(self) -> list:
        """ORT execution providers actually used by this session."""
        return list(self._active_providers)

    def __repr__(self) -> str:
        return f"ONNXInference(providers={self.active_providers})"
