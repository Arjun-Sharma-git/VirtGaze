"""TensorRT inference wrapper (optional GPU acceleration).

This module provides a drop-in replacement for :class:`ONNXInference` when
TensorRT is available.  The ``trtexec`` tool (bundled with TensorRT) must be
used to convert the ONNX model to a .trt engine first::

    trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

If TensorRT is not installed this module still imports successfully; the
class will raise :class:`RuntimeError` when instantiated.
"""
from __future__ import annotations

import numpy as np


class TensorRTInference:
    """Run the GazeMLP via TensorRT for maximum GPU throughput.

    Requires:
        - NVIDIA GPU with CUDA
        - TensorRT 8.6+ installed (``pip install tensorrt``)
        - A pre-built .trt engine file

    Args:
        engine_path: Path to the .trt engine file.
        input_dim:   Feature dimension (default 34).
    """

    def __init__(self, engine_path: str, input_dim: int = 34) -> None:
        try:
            import pycuda.autoinit  # type: ignore[import]  # noqa: F401
            import pycuda.driver as cuda  # type: ignore[import]
            import tensorrt as trt  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT or PyCUDA not installed. "
                "Install with: pip install tensorrt pycuda"
            ) from exc

        self._input_dim = input_dim
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()

        # Allocate host + device buffers
        self._h_input = cuda.pagelocked_empty((1, input_dim), dtype=np.float32)
        self._h_output = cuda.pagelocked_empty((1, 2), dtype=np.float32)
        self._d_input = cuda.mem_alloc(self._h_input.nbytes)
        self._d_output = cuda.mem_alloc(self._h_output.nbytes)
        self._stream = cuda.Stream()
        self._cuda = cuda

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run FP16 TensorRT inference.

        Args:
            features: (1, 34) float32 array.

        Returns:
            (1, 2) float32 — (screen_x_norm, screen_y_norm) in [0, 1].
        """
        np.copyto(self._h_input, features.astype(np.float32))
        self._cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)
        self._context.execute_async_v2(
            bindings=[int(self._d_input), int(self._d_output)],
            stream_handle=self._stream.handle,
        )
        self._cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
        self._stream.synchronize()
        return self._h_output.copy()
