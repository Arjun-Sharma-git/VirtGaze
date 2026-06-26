# 3D Gaze Estimation System

A production-grade real-time 3D gaze estimation system from a standard webcam, targeting:

| Metric | Target |
|--------|--------|
| Angular error (post-calibration) | < 3° |
| Screen-space error (1080p) | < 40 px |
| End-to-end latency | < 20 ms |
| Inference FPS (CPU) | 60 |
| Inference FPS (GPU) | 120 |

---

## Architecture

```
Camera → Face Detection → Face Mesh + Iris → Head Pose (solvePnP)
      → 3D Gaze Geometry → Personalized MLP → Temporal Filtering → Screen Coords
```

All pipeline stages run on separate threads with bounded queues for natural backpressure and low latency.

---

## Quick Start

```bash
# 1. Install
pip install -e .
# or for development:
pip install -e ".[dev]"

# 2. Run the tracker (first run will prompt for calibration)
python scripts/run_tracker.py --user alice

# 3. Run calibration standalone
python scripts/run_calibration.py --user alice

# 4. Quick 5-point recalibration
python scripts/run_calibration.py --user alice --quick
```

---

## Project Structure

```
3dEyeEstim/
├── src/gaze_estimation/       # Main package
│   ├── pipeline/              # Pipeline orchestrator + base thread
│   ├── capture/               # Webcam + video file input
│   ├── detection/             # Face detection + tracking
│   ├── mesh/                  # Face mesh + iris landmarks (MediaPipe)
│   ├── pose/                  # Head pose (solvePnP)
│   ├── gaze/                  # 3D gaze ray + feature extraction
│   ├── calibration/           # 25-point / 5-point / implicit calibration
│   ├── model/                 # GazeMLP + ONNX inference
│   ├── filtering/             # UKF + One Euro + fixation detection
│   ├── adaptation/            # Online adaptation (click-based)
│   ├── correction/            # Bias map (spatial error correction)
│   ├── profile/               # User profile persistence
│   ├── config/                # YAML config loader
│   ├── visualization/         # Overlay renderer + calibration UI
│   └── utils/                 # Geometry, logging, timing helpers
├── tests/                     # Unit + integration tests
├── scripts/                   # Entry points
├── configs/                   # User-facing config overrides
├── profiles/                  # Runtime user profiles
├── models/                    # Exported MLP models
└── data/                      # Canonical face model, test fixtures
```

---

## Configuration

Copy and edit the example config:

```bash
cp configs/example_config.yaml configs/my_config.yaml
python scripts/run_tracker.py --config configs/my_config.yaml
```

Key configuration sections:

```yaml
camera:
  index: 0
  width: 1280
  height: 720
  fps: 60

inference:
  backend: "onnx"    # "onnx" | "torch" | "tensorrt"
  device: "CPU"      # "CPU" | "CUDA"

mlp:
  hidden_dims: [64, 128, 64]
  epochs: 200
```

---

## Calibration

### Full 25-Point Calibration (~90 seconds)
```bash
python scripts/run_calibration.py --user alice
```
Collects 3000 samples across a 5×5 grid. Trains a personalized MLP (~14K parameters, <200ms training).

### Quick 5-Point Recalibration (~15 seconds)
```bash
python scripts/run_calibration.py --user alice --quick
```
Fine-tunes existing MLP weights. Recommended every 7 days or after camera/screen changes.

---

## Testing

```bash
# All tests
make test

# Unit tests only (no camera required)
make test-unit

# With coverage report
make test-cov
```

---

## Benchmarking

```bash
# Accuracy benchmark (requires ground truth video + CSV)
python scripts/benchmark_accuracy.py --video data/test.mp4 --gt data/ground_truth.csv

# Latency benchmark
python scripts/benchmark_latency.py
```

---

## Model Export

```bash
# Export MLP to ONNX (for deployment)
python scripts/export_model.py --user alice --output models/gaze_alice.onnx

# TensorRT (requires CUDA + TensorRT)
python scripts/export_model.py --user alice --format tensorrt --output models/gaze_alice.trt
```

---

## Dependencies

- Python ≥ 3.9
- OpenCV ≥ 4.8
- MediaPipe ≥ 0.10
- PyTorch ≥ 2.0
- ONNX Runtime ≥ 1.15
- PyYAML, Pydantic ≥ 2.0, SciPy, Pygame

Optional NVIDIA GPU: `pip install tensorrt` (requires CUDA 11+)

Optional AMD GPU: see ROCm section below.

---

## ROCm Support (AMD GPU)

The system fully supports AMD GPUs via the ROCm / HIP backend:

| Component | ROCm backend |
|-----------|-------------|
| PyTorch MLP training | `torch` ROCm wheel (HIP) |
| ONNX Runtime inference | `onnxruntime-rocm` + `ROCMExecutionProvider` |
| MediaPipe face mesh | CPU only (MediaPipe has no ROCm GPU delegate) |

### Install (AMD GPU)

```bash
# 1. Install ROCm system packages (follow AMD docs for your distro + ROCm version)
#    https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

# 2. Install PyTorch with ROCm wheel + onnxruntime-rocm
pip install -r requirements-rocm.txt

# 3. Install the package itself (no-deps, wheels already installed above)
pip install -e . --no-deps

# Or use the Makefile shortcut:
make install-rocm
```

### Configure for ROCm

Edit your config YAML or pass via CLI:

```yaml
# configs/my_config_rocm.yaml
inference:
  backend: "onnx"
  device: "ROCM"    # ← enables ROCMExecutionProvider in ONNX Runtime

mlp:
  device: "rocm"    # ← trains GazeMLP on your AMD GPU
```

Then run:
```bash
python scripts/run_tracker.py --config configs/my_config_rocm.yaml --user alice
```

### Auto-detection

Set `device: "auto"` to let the system pick the best available backend at startup:

```yaml
inference:
  device: "auto"   # ROCm → CUDA → CPU, in priority order
```

### Verify

```python
from gaze_estimation.utils.device import describe_device, is_rocm
print(describe_device())
# → "ROCm (6.0.0) — AMD GPU: Radeon RX 7900 XTX (1 device(s))"
print(is_rocm())   # True on ROCm build
```

### Notes

- `torch.cuda.*` APIs work transparently on ROCm (`torch.version.hip` is non-None).
- MediaPipe always runs on CPU.  The GPU stages are MLP training and ONNX inference.
- TensorRT is NVIDIA-only and is skipped on ROCm.

---

## Implementation Plan

See [`docs/GAZE_ESTIMATION_PLAN.md`](docs/GAZE_ESTIMATION_PLAN.md) for the full specification.
