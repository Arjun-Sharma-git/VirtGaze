# 3D Gaze Estimation System

![CI](https://github.com/Arjun-Sharma-git/VirtGaze/actions/workflows/ci.yml/badge.svg)

A production-grade real-time 3D gaze estimation system from a standard webcam, targeting:

| Metric | Target |
|--------|--------|
| Angular error (post-calibration) | < 3° |
| Screen-space error (1080p) | < 40 px |
| End-to-end latency | < 20 ms |
| Inference FPS (CPU) | 60 |
| Inference FPS (GPU) | 120 |

These are **design targets, not measured results** — accuracy depends on your camera,
lighting and calibration quality. Measure them on your hardware with
`scripts/benchmark_accuracy.py` and `scripts/benchmark_latency.py`.

---

## Architecture

```
Camera → Face Detection → Face Mesh + Iris → Head Pose (solvePnP)
      → 3D Gaze Geometry → Personalized MLP → Temporal Filtering → Screen Coords
```

All pipeline stages run on separate threads with bounded queues for natural backpressure and low latency.

### Coordinate frames

Gaze directions are expressed in two different frames, and confusing them silently
breaks both calibration and the uncalibrated cursor:

| Value | Frame | Why |
|---|---|---|
| `GazeRay.direction` (`gaze_ray_left/right`) | **head** | Eye-in-head direction. The MLP features (`gaze_yaw_avg`, `gaze_pitch_avg`, …) are derived from it, which is what makes the learned mapping invariant to head pose |
| `GazePacket.gaze_yaw` / `gaze_pitch` | **camera** | Kappa is applied in the head frame, then the direction is rotated by the head rotation. The geometric screen fallback projects these onto the screen plane |
| `CalibrationSample.gaze_yaw_world` / `gaze_pitch_world` | **camera** | Recorded next to the target so `estimate_kappa` compares screen-referenced angles with screen-referenced angles, instead of absorbing head rotation into kappa |
| `HeadPose.euler_angles` | — | `[yaw, pitch, roll]` following `R = Rz·Ry·Rx`. Note the camera frame has **Y down and Z forward**, so a physical head *turn* about the vertical axis is a rotation about Y and lands in `euler_angles[1]`, the slot labelled "pitch". `rotation_matrix_to_euler` is verified against `scipy`'s `as_euler("zyx")` |

`inference.fallback_to_geometric` therefore consumes camera-frame angles, while a
person model is trained on head-frame features — that split is deliberate.

---

## Quick Start

```bash
# 1. Install (creates .venv, picks the right OpenCV/PyTorch, then verifies it)
python scripts/bootstrap.py
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Confirm the environment is ready (versions, device, screen, camera)
python scripts/bootstrap.py --check

# 3. Run the tracker (first run will prompt for calibration)
python scripts/run_tracker.py --user alice

# 4. Run calibration standalone
python scripts/run_calibration.py --user alice

# 5. Quick 5-point recalibration
python scripts/run_calibration.py --user alice --quick

# 6. Run with click-based online adaptation
python scripts/run_tracker.py --user alice --implicit-calibration
```

`make setup` is equivalent to step 1, `make doctor` to step 2.

---

## Installation

One command sets up a working environment on Linux, macOS, or Windows:

```bash
python scripts/bootstrap.py
```

It exists because a bare `pip install -e .` fails on a fresh machine in four
common and confusing ways, and it handles each of them:

| Situation | What `pip install -e .` does | What the bootstrap does |
|---|---|---|
| Debian/Ubuntu 24.04+ | Refuses: *externally managed environment* (PEP 668) | Creates a virtualenv, so PEP 668 never applies |
| Python 3.13+ | Fails on a dependency with no wheel | Asks PyPI which wheels exist, names the blocking package, and points at 3.10–3.12 |
| Headless server | Installs, then `import cv2` dies on `libGL.so.1` | Detects the missing display and installs `opencv-python-headless` |
| NVIDIA machine | Silently installs CPU-only PyTorch | Uses the CUDA-enabled wheel by default; `--torch cpu/rocm` to override |

Then it verifies the result rather than assuming:

```
Environment check
=================

  Python        : 3.12.4
  PyTorch device: CUDA (12.4) — GPU: NVIDIA GeForce RTX 4070 (1 device(s))
  Screen        : 2560x1440
  OpenCV build  : opencv-python 4.10.0.84

  Packages
    cv2          4.10.0
    mediapipe    0.10.14
    torch        2.4.1+cu124
    ...

  Everything needed is importable.
```

### Options

| Flag | Purpose |
|---|---|
| `--dev` | Also install the test/lint tooling (`pytest`, `ruff`, `mypy`, `pre-commit`) |
| `--torch cpu` / `rocm` / `none` | Choose the PyTorch build, or leave an existing one alone |
| `--opencv display` / `headless` | Override the automatic display detection |
| `--venv PATH` | Put the virtualenv somewhere else (default `.venv`) |
| `--python PATH` | Build with a specific interpreter, e.g. `--python python3.12` |
| `--check` | Verify an existing environment and change nothing |
| `--probe-camera` | Also try to open camera index 0 |
| `--dry-run` | Print the exact commands without running them |
| `--no-deps` | Install the package without re-resolving dependencies (used by the ROCm path) |

Common combinations:

```bash
python scripts/bootstrap.py --dev                    # development setup
python scripts/bootstrap.py --torch cpu              # CPU-only PyTorch
python scripts/bootstrap.py --opencv headless        # server, no display
python scripts/bootstrap.py --torch rocm             # AMD GPU
python scripts/bootstrap.py --dry-run                # see the plan first
```

Exit codes: `0` success, `1` the environment check found missing required
packages, `2` unsupported interpreter or a failed install. Re-running is safe —
an existing virtualenv is reused, and a failed run can simply be repeated.

### Manual installation

If you would rather not use the script, the equivalent steps are:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python-headless        # or opencv-python when you have a display
pip install torch                          # add --index-url https://download.pytorch.org/whl/cpu for CPU-only
pip install -e .
python scripts/bootstrap.py --check        # verify
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `error: externally-managed-environment` | Use a virtualenv — `python scripts/bootstrap.py` does it for you |
| `No matching distribution found for pygame` (or mediapipe) | That Python is too new. `--python python3.12`; on Ubuntu `sudo apt install python3.12 python3.12-venv` |
| `ImportError: libGL.so.1` | Install `opencv-python-headless` instead of `opencv-python` (or `apt install libgl1`) |
| `ensurepip is not available` | `sudo apt install python3-venv`, or let the script fall back to `get-pip.py` |
| PyTorch reports CPU on a CUDA machine | Reinstall with the CUDA index, or check `nvidia-smi` / the driver |

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

Key configuration sections (every key below is honoured by the code — see
`tests/test_config_drift.py`):

```yaml
camera:
  index: 0
  width: 1280
  height: 720
  fps: 60
  auto_exposure: true        # false locks exposure — avoids iris drift mid-session
  undistort: false           # Apply lens-distortion correction
  intrinsics_path: ""        # .npz produced by scripts/calibrate_camera.py

detection:
  model: "mediapipe_short"   # "mediapipe_short" (≤2 m) | "mediapipe_full" (≤5 m)
  detection_interval: 5      # full detection every N frames; tracked in between
  min_confidence: 0.5

mesh:
  refine_iris: true
  static_image_mode: false   # true for still images / recordings (slower, live video)

pose:
  solvepnp_method: "SOLVEPNP_ITERATIVE"   # any cv2.SOLVEPNP_* constant name
  use_ransac: false                       # solvePnPRansac outlier rejection

inference:
  backend: "onnx"            # "onnx" | "torch" | "tensorrt"
  device: "CPU"              # "CPU" | "CUDA" | "ROCm" | "auto"
                             # also selects the PyTorch training device
  fallback_to_geometric: true
  fallback_distance_mm: 600.0
  geometric_min_confidence: 0.4

mlp:
  hidden_dims: [64, 128, 64]
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 200
  batch_size: 32
  early_stopping_patience: 20

calibration:
  grid_cols: 5
  grid_rows: 5
  samples_per_target: 120
  target_duration_sec: 2.0
  pulse_animation: true      # animated target; false = static dot
  circular_motion: true
  circular_radius_px: 15
  outlier_sigma_threshold: 2.0

quick_calibration:
  points: 5                  # 5 (centre + 4 corners) | 9 (full 3×3 grid)

profile:
  auto_save: true            # false = run without writing any calibration output
  quick_calib_interval_days: 7   # staleness hint printed by run_tracker

logging:
  log_fps: true              # console FPS line + overlay HUD
  log_latency: true
  save_debug_frames: false   # dump annotated frames to debug_frames/ (throttled)
```

---

## Runtime Features

### Inference backends
`inference.backend` selects the real backend at runtime:

| backend | implementation | requirement |
|---------|----------------|-------------|
| `torch` | in-process `GazeMLP` | none |
| `onnx`  | ONNX Runtime (`ONNXInference`) | exported `.onnx` per profile |
| `tensorrt` | TensorRT (`TensorRTInference`) | NVIDIA GPU + prebuilt `.trt` |

The exported ONNX/TensorRT graph expects the **normalised** feature vector, so
feature z-scoring is always applied by the trainer before the backend runs. If a
requested backend cannot be initialised, the tracker logs a warning and falls
back to `torch` rather than failing.

### Geometric fallback
When no personal model is available, or confidence is below
`inference.geometric_min_confidence`, the gaze angles are projected onto the
screen plane at `inference.fallback_distance_mm`. Set

```yaml
inference:
  fallback_to_geometric: false
```

to disable that entirely: an unusable model then holds the estimate at the
screen centre instead of reporting a plausible-looking but uncalibrated point.
Use this when a wrong-but-confident cursor is worse than no cursor.

### Spatial bias correction
During calibration a residual `BiasMap` is built from the model's own
prediction errors on the calibration targets (`bias_map.npz` per profile) and is
applied to model predictions at inference time, removing systematic
location-dependent error.

### Fixation-adaptive smoothing
The UKF measurement noise is scaled by the current gaze state: heavier
smoothing while fixating or blinking, more responsive during saccades.

### Live preview & implicit calibration
`scripts/run_tracker.py` renders a live overlay (face mesh, iris, gaze dot,
FPS/latency HUD) from the pipeline's preview tap. With
`--implicit-calibration` the window is fullscreen and each deliberate
left-click is treated as a gaze-directed click: the model is fine-tuned
in the background and the updated weights are saved automatically.

```bash
python scripts/run_tracker.py --user alice --implicit-calibration
```

### Camera calibration pipeline
```bash
# 1. Capture chessboard intrinsics
python scripts/calibrate_camera.py --output data/camera_intrinsics.npz
# 2. Point camera.intrinsics_path at it and enable undistortion
#    camera: {undistort: true, intrinsics_path: "data/camera_intrinsics.npz"}
```

### Debug frames
With `logging.save_debug_frames: true` the tracker writes annotated frames to
`debug_frames/` — at most one every 0.5 s, 200 files total, so a long session
cannot fill the disk.

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

## Code Quality

```bash
make lint          # ruff (rule set pinned in pyproject.toml)
make format        # black + ruff --fix
make typecheck     # mypy src/gaze_estimation
make check         # lint + typecheck + tests — exactly what CI enforces
```

Git hooks (ruff, black, mypy plus whitespace/YAML/large-file checks):

```bash
make precommit-install   # once per clone
make precommit           # run against the whole tree
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request:

| job | what it does |
|-----|--------------|
| `lint` | `ruff check` and `black --check`, with exact pinned tool versions |
| `typecheck` | `mypy src/gaze_estimation` on Python 3.12 |
| `test` | `pytest tests/` on Python 3.10, 3.11 and 3.12 |

Python 3.9 is **not** supported: `pyproject.toml` requires `>=3.10`, which the
bootstrap script reads directly rather than duplicating.

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

- Python 3.10 – 3.12 (the tested range; check with `python scripts/bootstrap.py --check`)
- OpenCV ≥ 4.8 — `opencv-python` for a display, `opencv-python-headless` on a server
- MediaPipe ≥ 0.10
- PyTorch ≥ 2.0 (the PyPI wheel is CUDA-enabled on Linux/Windows; use the CPU index for CPU-only)
- ONNX Runtime ≥ 1.15
- PyYAML, Pydantic ≥ 2.0, SciPy, Pygame, screeninfo

Newer interpreters may work: `scripts/bootstrap.py` asks PyPI whether every
required package still publishes a wheel for the interpreter and reports exactly
which ones block the install (pygame is usually the first to lag).

OpenCV 5 is supported, with one caveat: it removed `cv2.CascadeClassifier` and
the bundled cascade XML files, so the Haar detection fallback — used only when
MediaPipe is unavailable — cannot work there. The detector logs a warning naming
the version and reports no face rather than failing. Install `opencv-python<5`
if you need that fallback.

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

# 2. One command: creates .venv, installs the ROCm PyTorch wheel,
#    onnxruntime-rocm and the package, then verifies the result
python scripts/bootstrap.py --torch rocm

# Manual equivalent:
pip install -r requirements-rocm.txt
pip install -e . --no-deps
```

`--torch rocm` uses the `rocm6.2` wheel index by default; pass
`--rocm-version 6.4` (or whichever series matches your system ROCm) to change it.

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

See [`docs/GAZE_ESTIMATION_PLAN.md`](docs/GAZE_ESTIMATION_PLAN.md) for the original
design specification (theory, module rationale, calibration protocol). It was
written before the code and is kept as design rationale — it is **not** a
description of the current system, and it opens with a list of divergences. This
README and the sources are authoritative for current behaviour.
