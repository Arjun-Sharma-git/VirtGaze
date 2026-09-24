# Production-Grade 3D Gaze Estimation System — Comprehensive Implementation Plan

> **Document Purpose**: This document is a complete, self-contained specification for building a production-grade 3D gaze estimation system from a standard webcam. Any AI model or developer can pick up this document and implement the full system without additional context.

---

> ## Status: historical design specification — the code is authoritative
>
> This document was written **before** the implementation and is kept for its design
> rationale. It is **not** a description of the current system: the sources under
> `src/gaze_estimation/` are the source of truth, and [`README.md`](../README.md)
> documents current behaviour. Read this for *why* things are shaped as they are,
> not for *what* the APIs look like.
>
> ### Known divergences
>
> - **The phase checkboxes in §13 are the original roadmap, not a status tracker.**
>   They all remain unchecked although every phase is implemented; the test suite
>   (612 tests) is the live signal. `setup.py` was also dropped in favour of
>   `pyproject.toml` alone.
> - **Coordinate frames.** §5.5 gives `GazeRay.direction` as "camera frame", which
>   contradicts §3 and §5.5's own `_compute_gaze_ray` docstring. The **head frame**
>   is correct: rays and MLP features are head-frame (head-pose invariant) while
>   `GazePacket.gaze_yaw/gaze_pitch` are camera-frame with kappa applied. §7's
>   "apply kappa compensation to optical axis, then transform the visual axis to
>   the camera frame using head pose" describes exactly what the code does.
> - **Schemas gained fields** not listed in §6: `PosePacket.face_bbox`,
>   `CalibrationSample.gaze_yaw_world`/`gaze_pitch_world`, and `GazePacket`'s frame
>   annotations.
> - **Code snippets are sketches, not shipped signatures.** Angle conversion moved to
>   `utils.geometry.ray_to_angles`, feature extraction to
>   `gaze_features.FeatureExtractor`, and `GazeGeometryEstimator` gained a `tap_queue`
>   so calibration can observe packets without stealing them from the pipeline.
> - **Features added beyond this plan:** runtime inference-backend selection
>   (`model/backend.py`, `inference.backend`), residual spatial bias correction
>   (`correction/bias_map.py`), click-based online adaptation, fixation-adaptive
>   smoothing, in-place lens-distortion removal, and a live preview overlay.
> - **Configuration is read from `config/default_config.yaml`**; every key there is
>   consumed (guarded by `tests/test_config_drift.py`), and all are wired through to
>   the objects that use them (guarded by `tests/test_config_wiring.py`).

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [System Architecture](#2-system-architecture)
3. [3D Gaze Estimation Theory](#3-3d-gaze-estimation-theory)
4. [Complete Folder Structure](#4-complete-folder-structure)
5. [Module Specifications](#5-module-specifications)
6. [Data Schemas](#6-data-schemas)
7. [Calibration Protocol](#7-calibration-protocol)
8. [Model Architecture](#8-model-architecture)
9. [Configuration System](#9-configuration-system)
10. [Dependencies & Environment](#10-dependencies--environment)
11. [Testing & Evaluation](#11-testing--evaluation)
12. [Performance Optimization](#12-performance-optimization)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Project Overview & Goals

### 1.1 Objective

Build a real-time 3D gaze estimation system that runs on a standard webcam (60–120 FPS) with sub-3° angular accuracy after personal calibration, and <20ms end-to-end latency on commodity hardware.

### 1.2 Target Metrics

| Metric | Target | Stretch |
|--------|--------|---------|
| Angular error (post-calibration) | < 3° | < 1° |
| Screen-space error (1080p) | < 40 px | < 15 px |
| End-to-end latency | < 20 ms | < 10 ms |
| Inference FPS (CPU) | 60 | 90 |
| Inference FPS (GPU) | 120 | 240 |
| Calibration time (25-point) | < 90 s | < 60 s |
| Quick recalibration (5-point) | < 15 s | < 10 s |

### 1.3 Design Principles

1. **Dual-path gaze estimation**: Geometric 3D model (interpretable, calibration-free baseline) + learned personalization layer (accuracy boost).
2. **Modular pipeline**: Each stage is independently testable, replaceable, and has a well-defined interface.
3. **Graceful degradation**: If a module fails or confidence drops, the system falls back to a simpler model rather than crashing.
4. **Thread-safe concurrency**: Pipeline stages run on separate threads with bounded queues.
5. **Profile persistence**: User calibration data survives restarts.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
┌──────────────┐
│  Camera      │  Thread: camera_thread
│  Capture     │  Queue: frame_queue (maxsize=2)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Face        │  Thread: detection_thread
│  Detection   │  Queue: face_queue (maxsize=2)
│  + Tracking  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Face Mesh   │  Thread: mesh_thread
│  + Iris      │  Queue: mesh_queue (maxsize=2)
│  Landmarks   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Head Pose   │  Thread: pose_thread
│  (solvePnP)  │  Queue: pose_queue (maxsize=2)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  3D Gaze     │  Thread: gaze_thread
│  Geometry    │  Queue: gaze_queue (maxsize=2)
│  + Features  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Personalized│  Thread: inference_thread
│  MLP Mapping │  Queue: prediction_queue (maxsize=2)
│  + Bias Map  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Temporal    │  Thread: filter_thread
│  Filtering   │  (UKF → One Euro)
│  + Fixation  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Screen      │  Output: GazeEstimate
│  Coordinates │
└──────────────┘
```

### 2.2 Threading Model

```python
# Each stage is a Thread subclass with:
# - input_queue: queue.Queue (bounded, maxsize=2)
# - output_queue: queue.Queue (bounded, maxsize=2)
# - stop_event: threading.Event
# - name: str (for logging)

# Bounded queues provide natural backpressure.
# If a downstream stage is slow, the upstream stage blocks
# instead of accumulating unbounded memory.

# The main thread reads from prediction_queue at display rate.
```

### 2.3 Data Flow

```
Frame (np.ndarray, BGR)
  → FramePacket(timestamp, frame)
  → FacePacket(timestamp, frame, face_bbox, face_landmarks_2d)
  → MeshPacket(timestamp, frame, face_bbox, mesh_468, iris_2d, left_iris_center, right_iris_center)
  → PosePacket(timestamp, frame, mesh_468, iris_2d, head_pose_rvec, head_pose_tvec, head_pose_euler)
  → GazePacket(timestamp, gaze_ray_left, gaze_ray_right, gaze_yaw, gaze_pitch, features_dict)
  → PredictionPacket(timestamp, screen_x, screen_y, confidence, raw_features)
  → GazeEstimate(timestamp, screen_x, screen_y, filtered_x, filtered_y, confidence, fixation_state, velocity)
```

### 2.4 Fallback Strategy

```
Confidence > 0.7  → Full pipeline (MLP + bias map + UKF + One Euro)
Confidence 0.4-0.7 → Geometric model only + One Euro filter
Confidence < 0.4  → Hold last prediction + increase smoothing
Face lost          → coast (predict via UKF velocity) for 10 frames, then LOST state
```

---

## 3. 3D Gaze Estimation Theory

### 3.1 Eye Model

Each eye is modeled as a sphere (eyeball) with:
- **Center**: 3D point in head coordinate frame
- **Radius**: ~12 mm (average), fitted per-user during calibration
- **Iris**: circle on the sphere surface, whose 3D center and normal define the optical axis

```
Eye coordinate frame (per-eye, head-relative):

       y (up)
       │
       │
       ○───── x (right, toward temple)
      ╱
    ╱
  z (forward, toward camera)

Eyeball center: C_eye = (x_c, y_c, z_c) in head frame
Iris center:    P_iris = C_eye + r * n  where n is the optical axis unit vector
```

### 3.2 Optical Axis vs Visual Axis

- **Optical axis**: 3D vector from eyeball center through iris center. This is what we can measure geometrically.
- **Visual axis**: 3D vector from fovea through pupil center. This is where the user is *actually* looking.
- **Kappa angle (∠κ)**: Offset between optical and visual axes, typically 4°–8° horizontally, 1°–2° vertically. This is **user-specific** and must be calibrated.

```
         Visual axis (where user looks)
        ╱
       ╱  ∠κ (kappa)
      ╱─────────
     ╱
    Optical axis (measured)

Kappa compensation:
  visual_yaw   = optical_yaw   + kappa_yaw
  visual_pitch = optical_pitch + kappa_pitch
```

### 3.3 Head Pose Estimation (solvePnP)

Using the 468 MediaPipe face mesh landmarks, select a subset of stable 3D reference points (nose tip, eye corners, mouth corners, etc.) with known canonical 3D coordinates. Solve for head pose:

```python
import cv2
import numpy as np

# Canonical 3D face model (head coordinate frame, mm)
# These are average human face landmark positions.
CANONICAL_3D = np.array([
    # nose tip
    [0.0, 0.0, 0.0],
    # chin
    [0.0, -63.6, -12.5],
    # left eye outer corner
    [-43.3, 20.0, -5.0],
    # right eye outer corner
    [43.3, 20.0, -5.0],
    # left mouth corner
    [-28.0, -28.0, -10.0],
    # right mouth corner
    [28.0, -28.0, -10.0],
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

def estimate_head_pose(mesh_landmarks_2d: np.ndarray, camera_matrix, dist_coeffs):
    """Returns (rvec, tvec) where rvec encodes rotation and tvec encodes translation."""
    image_points = mesh_landmarks_2d[LANDMARK_INDICES].astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        CANONICAL_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rvec, tvec
```

### 3.4 Gaze Ray Computation

From the iris 2D center and head pose, compute the 3D gaze ray:

```python
def compute_gaze_ray(iris_center_2d, head_rvec, head_tvec, camera_matrix, dist_coeffs):
    """
    1. Undistort the iris 2D center.
    2. Compute a ray from camera through the undistorted iris point.
    3. Transform the ray into the head coordinate frame using the inverse of head pose.
    4. The ray direction in head frame is the optical axis direction.
    """
    # Undistort
    pts = np.array([[iris_center_2d]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    undistorted = undistorted.reshape(-1, 2)

    # Ray in camera frame
    ray_cam = np.array([undistorted[0][0], undistorted[0][1], 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # Transform to head frame
    R, _ = cv2.Rodrigues(head_rvec)
    R_inv = R.T  # Rotation matrix is orthogonal
    ray_head = R_inv @ ray_cam

    return ray_head  # Optical axis direction in head frame
```

### 3.5 Gaze-to-Screen Projection

After computing the visual axis (optical + kappa) and head pose, project the gaze ray to the screen plane:

```python
def gaze_ray_to_screen(gaze_origin_3d, gaze_direction_3d, screen_plane_normal, screen_plane_point):
    """
    Intersect gaze ray with screen plane.
    screen_plane_normal: normal of screen (typically [0, 0, 1] in world)
    screen_plane_point: center of screen in world coordinates
    """
    denom = np.dot(gaze_direction_3d, screen_plane_normal)
    if abs(denom) < 1e-6:
        return None  # Ray parallel to screen
    t = np.dot(screen_plane_point - gaze_origin_3d, screen_plane_normal) / denom
    intersection = gaze_origin_3d + t * gaze_direction_3d
    return intersection[:2]  # (x, y) in screen plane
```

### 3.6 Feature Vector

The geometric model produces gaze yaw/pitch, but the learned model uses a richer feature vector:

```python
FEATURE_KEYS = [
    # 3D gaze (geometric)
    "gaze_yaw_left", "gaze_pitch_left",
    "gaze_yaw_right", "gaze_pitch_right",
    "gaze_yaw_avg", "gaze_pitch_avg",

    # Head pose
    "head_yaw", "head_pitch", "head_roll",
    "head_tx", "head_ty", "head_tz",

    # Iris 2D positions (normalized)
    "left_iris_x", "left_iris_y",
    "right_iris_x", "right_iris_y",
    "left_iris_radius", "right_iris_radius",

    # Eye aspect ratios (eyelid openness)
    "left_ear", "right_ear",

    # Pupil-to-eye-corner vectors
    "left_pupil_to_inner_x", "left_pupil_to_inner_y",
    "left_pupil_to_outer_x", "left_pupil_to_outer_y",
    "right_pupil_to_inner_x", "right_pupil_to_inner_y",
    "right_pupil_to_outer_x", "right_pupil_to_outer_y",

    # Face position (normalized)
    "face_x", "face_y", "face_width", "face_height",

    # Interpupil distance (pixels)
    "interpupil_distance",

    # Confidence
    "landmark_confidence",
]
```

**Total**: 34 features.

---

## 4. Complete Folder Structure

```
3dEyeEstim/
│
├── docs/
│   └── GAZE_ESTIMATION_PLAN.md       # This document
│
├── src/
│   └── gaze_estimation/
│       ├── __init__.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── pipeline.py            # Pipeline orchestrator: threads + queues
│       │   ├── frame_packet.py        # FramePacket dataclass
│       │   └── thread_base.py         # Base StageThread class
│       │
│       ├── capture/
│       │   ├── __init__.py
│       │   ├── camera.py              # CameraCapture: webcam input, thread
│       │   └── video_source.py        # VideoFileSource: for testing with recorded video
│       │
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── face_detector.py       # FaceDetector: MediaPipe Face Detection
│       │   └── face_tracker.py        # FaceTracker: ROI tracking between detections
│       │
│       ├── mesh/
│       │   ├── __init__.py
│       │   ├── face_mesh.py           # FaceMeshExtractor: MediaPipe 468 landmarks
│       │   └── iris_tracker.py        # IrisTracker: MediaPipe Iris (478 landmarks)
│       │
│       ├── pose/
│       │   ├── __init__.py
│       │   ├── head_pose.py           # HeadPoseEstimator: solvePnP
│       │   └── canonical_face.py      # 3D canonical face model constants
│       │
│       ├── gaze/
│       │   ├── __init__.py
│       │   ├── gaze_geometry.py       # 3D gaze ray computation, eyeball model
│       │   ├── kappa_compensation.py  # Kappa angle estimation during calibration
│       │   └── gaze_features.py       # FeatureExtraction: builds FEATURE_KEYS vector
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   ├── calibration_engine.py   # CalibrationEngine: orchestrates 25-point session
│       │   ├── calibration_target.py   # Animated target: pulse + circular motion
│       │   ├── quick_calibration.py    # QuickCalibration: 5-point recalibration
│       │   └── implicit_calibration.py # ImplicitCalibration: click-based adaptation
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── mlp.py                  # GazeMLP: PyTorch model definition
│       │   ├── trainer.py             # MLPTrainer: trains on calibration data
│       │   ├── onnx_export.py          # Export PyTorch → ONNX
│       │   ├── onnx_inference.py       # ONNX Runtime inference wrapper
│       │   └── tensorrt_inference.py   # TensorRT inference wrapper (optional)
│       │
│       ├── filtering/
│       │   ├── __init__.py
│       │   ├── unscented_kalman.py     # UnscentedKalmanFilter: 6-state UKF
│       │   ├── one_euro.py            # OneEuroFilter: adaptive low-pass
│       │   └── fixation_detector.py    # FixationDetector: state machine
│       │
│       ├── adaptation/
│       │   ├── __init__.py
│       │   ├── adaptation_buffer.py    # AdaptationBuffer: stores (features, cursor) pairs
│       │   └── online_trainer.py       # OnlineTrainer: incremental MLP weight updates
│       │
│       ├── correction/
│       │   ├── __init__.py
│       │   └── bias_map.py            # BiasMap: 2D spatial error correction grid
│       │
│       ├── profile/
│       │   ├── __init__.py
│       │   ├── profile_manager.py     # ProfileManager: load/save user profiles
│       │   └── schema.py              # Profile JSON schema (Pydantic model)
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── config.py             # Config loader: YAML → dataclass
│       │   └── default_config.yaml   # Default configuration
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── overlay.py            # OverlayRenderer: draw gaze, mesh, debug info
│       │   └── calibration_ui.py     # CalibrationUI: fullscreen target display
│       │
│       └── utils/
│           ├── __init__.py
│           ├── geometry.py           # vec math, euler conversions, ray-plane
│           ├── camera_calibration.py # Camera intrinsics estimation
│           ├── logging.py            # Structured logging setup
│           └── timing.py             # FPS counter, latency profiler
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures (synthetic frames, etc.)
│   ├── test_camera.py
│   ├── test_face_detector.py
│   ├── test_face_mesh.py
│   ├── test_head_pose.py
│   ├── test_gaze_geometry.py
│   ├── test_gaze_features.py
│   ├── test_calibration_engine.py
│   ├── test_mlp.py
│   ├── test_mlp_trainer.py
│   ├── test_unscented_kalman.py
│   ├── test_one_euro.py
│   ├── test_fixation_detector.py
│   ├── test_bias_map.py
│   ├── test_profile_manager.py
│   ├── test_pipeline_integration.py
│   └── test_accuracy_benchmark.py
│
├── scripts/
│   ├── run_tracker.py                # Main entry point: real-time tracking
│   ├── run_calibration.py            # Standalone calibration session
│   ├── benchmark_accuracy.py         # Evaluation against ground truth
│   ├── benchmark_latency.py          # Measure per-stage latency
│   ├── export_model.py               # Export MLP to ONNX/TensorRT
│   └── calibrate_camera.py           # Chessboard camera intrinsics calibration
│
├── data/
│   └── canonical_face_model.json     # 3D canonical face landmarks
│
├── profiles/                         # User profiles (runtime-created)
│   └── .gitkeep
│
├── models/                           # Pre-trained and exported models
│   └── .gitkeep
│
├── configs/                          # User-facing config overrides
│   └── example_config.yaml
│
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── .gitignore
├── Makefile
└── README.md
```

---

## 5. Module Specifications

### 5.1 Camera Capture

**File**: `src/gaze_estimation/capture/camera.py`

**Purpose**: Capture frames from webcam at maximum FPS, assign timestamps, push to `frame_queue`.

```python
import time
import threading
import queue
import cv2
from dataclasses import dataclass
import numpy as np

@dataclass
class FramePacket:
    timestamp: float           # Unix timestamp
    frame: np.ndarray          # BGR image, shape (H, W, 3)
    frame_id: int              # Monotonic counter

class CameraCapture(threading.Thread):
    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        name: str = "camera_thread"
    ):
        ...

    def run(self) -> None:
        """Main loop: read frames, wrap in FramePacket, put on queue."""

    def get_camera_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (camera_matrix, dist_coeffs). Loads from profile or estimates."""

    def release(self) -> None:
        """Release camera resource."""
```

**Error handling**: If camera read fails for >5 consecutive frames, emit a `CameraError` event and attempt reconnection.

---

### 5.2 Face Detection + Tracking

**File**: `src/gaze_estimation/detection/face_detector.py`

**Purpose**: Detect face bounding box using MediaPipe Face Detection. Run every N frames; in between, track via ROI (cheaper).

```python
@dataclass
class FacePacket:
    timestamp: float
    frame: np.ndarray
    face_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    detection_confidence: float
    face_landmarks_2d: np.ndarray | None   # 6 key points for solvePnP

class FaceDetector(threading.Thread):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        detection_interval: int = 5,  # Run full detection every 5 frames
        min_confidence: float = 0.5,
        name: str = "detection_thread"
    ):
        ...

    def run(self) -> None:
        """Alternate between full detection and ROI tracking."""

    def _detect_full(self, frame: np.ndarray) -> tuple | None:
        """MediaPipe Face Detection. Returns (bbox, confidence, landmarks)."""

    def _track_roi(self, frame: np.ndarray, prev_bbox) -> tuple:
        """Mean-shift or optical-flow tracking within previous ROI."""
```

**Fallback**: If face not detected, pass `FacePacket(face_bbox=None)` downstream so the system knows to enter coasting mode.

---

### 5.3 Face Mesh + Iris Landmark Extraction

**File**: `src/gaze_estimation/mesh/face_mesh.py`

**Purpose**: Extract 468-point face mesh and refine iris landmarks using MediaPipe Face Mesh + Iris.

```python
@dataclass
class MeshPacket:
    timestamp: float
    frame: np.ndarray
    face_bbox: tuple[int, int, int, int] | None
    mesh_468: np.ndarray          # Shape (468, 2), pixel coords
    iris_478: np.ndarray | None  # Shape (478, 2) when iris refinement enabled
    left_iris_center: tuple[float, float] | None
    right_iris_center: tuple[float, float] | None
    left_iris_radius: float | None
    right_iris_radius: float | None
    confidence: float

class FaceMeshExtractor(threading.Thread):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        refine_iris: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        name: str = "mesh_thread"
    ):
        ...

    def run(self) -> None:
        """Extract face mesh, extract iris centers and radii."""

    def _extract_iris(self, iris_landmarks: np.ndarray) -> tuple:
        """Fit circle to iris landmarks, return (center, radius)."""
```

**Iris circle fitting**: Use cv2.fitEllipse or algebraic circle fit on the iris landmark points (indices 468–477 for left, 473–477 for right in MediaPipe's 478-point model).

---

### 5.4 Head Pose Estimation

**File**: `src/gaze_estimation/pose/head_pose.py`

**Purpose**: Estimate head rotation and translation using solvePnP on a subset of mesh landmarks.

```python
@dataclass
class HeadPose:
    rvec: np.ndarray          # Rotation vector (3,)
    tvec: np.ndarray          # Translation vector (3,)
    euler_angles: np.ndarray  # [yaw, pitch, roll] in degrees
    rotation_matrix: np.ndarray  # (3, 3)

@dataclass
class PosePacket:
    timestamp: float
    frame: np.ndarray
    mesh_468: np.ndarray
    iris_478: np.ndarray | None
    left_iris_center: tuple[float, float] | None
    right_iris_center: tuple[float, float] | None
    left_iris_radius: float | None
    right_iris_radius: float | None
    head_pose: HeadPose | None
    confidence: float

class HeadPoseEstimator(threading.Thread):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        name: str = "pose_thread"
    ):
        ...

    def run(self) -> None:
        """For each MeshPacket, compute head pose."""

    def _solve_pnp(self, mesh_468: np.ndarray) -> HeadPose | None:
        """Select canonical landmarks, run solvePnP, convert to euler."""

    @staticmethod
    def _rotation_to_euler(rvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector to [yaw, pitch, roll] degrees."""
```

---

### 5.5 3D Gaze Geometry

**File**: `src/gaze_estimation/gaze/gaze_geometry.py`

**Purpose**: Compute 3D gaze rays from iris centers and head pose, estimate gaze yaw/pitch.

```python
@dataclass
class GazeRay:
    origin: np.ndarray    # (3,) 3D point in camera frame (the head translation)
    direction: np.ndarray # (3,) unit vector in the HEAD frame (eye-in-head)

@dataclass
class GazePacket:
    timestamp: float
    gaze_ray_left: GazeRay | None   # head frame
    gaze_ray_right: GazeRay | None  # head frame
    gaze_yaw: float       # Average gaze yaw (degrees), CAMERA frame, kappa applied
    gaze_pitch: float     # Average gaze pitch (degrees), CAMERA frame, kappa applied
    features: dict        # Full feature vector (FEATURE_KEYS), head frame
    head_pose: HeadPose | None
    confidence: float

class GazeGeometryEstimator(threading.Thread):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        kappa_yaw: float = 0.0,    # Set during calibration
        kappa_pitch: float = 0.0,
        eyeball_radius: float = 12.0,  # mm
        name: str = "gaze_thread"
    ):
        ...

    def run(self) -> None:
        """For each PosePacket, compute gaze rays and features."""

    def _compute_gaze_ray(
        self, iris_center_2d, head_pose: HeadPose
    ) -> GazeRay | None:
        """Undistort iris center, back-project to 3D ray, transform to head frame."""

    def _ray_to_angles(self, ray: np.ndarray) -> tuple[float, float]:
        """Convert 3D direction to (yaw, pitch) in degrees."""

    def _extract_features(
        self, mesh_468, iris_478, head_pose, gaze_rays
    ) -> dict:
        """Build full FEATURE_KEYS feature dictionary."""
```

---

### 5.6 Calibration Engine

**File**: `src/gaze_estimation/calibration/calibration_engine.py`

**Purpose**: Orchestrate the 25-point animated calibration session. Collect features + labels.

```python
@dataclass
class CalibrationSample:
    features: dict        # FEATURE_KEYS dict
    screen_x: float      # Target screen X (pixels)
    screen_y: float      # Target screen Y (pixels)
    timestamp: float

@dataclass
class CalibrationResult:
    samples: list[CalibrationSample]
    mlp_weights_path: str | None
    kappa_yaw: float
    kappa_pitch: float
    eyeball_radius: float
    bias_map: np.ndarray | None
    timestamp: float
    screen_resolution: tuple[int, int]

class CalibrationEngine:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        grid_cols: int = 5,
        grid_rows: int = 5,
        samples_per_target: int = 120,
        target_duration_sec: float = 2.0,
        pulse_animation: bool = True,
        circular_motion: bool = True,
        circular_radius_px: int = 15,
    ):
        ...

    def generate_targets(self) -> list[tuple[float, float]]:
        """Generate 25 (x, y) screen targets in a 5×5 grid."""

    def run(
        self,
        gaze_queue: queue.Queue,
        on_target_change: callable,   # Callback when target moves
        on_progress: callable,        # Callback for progress bar
    ) -> CalibrationResult:
        """
        For each target:
          1. Call on_target_change(target) — UI moves the dot.
          2. Wait 0.3s (user finds target).
          3. Collect samples_per_target samples from gaze_queue.
          4. Filter outliers (samples > 2 std from mean).
        After all targets:
          1. Estimate kappa angle (geometric vs target).
          2. Fit eyeball radius.
          3. Train MLP.
          4. Initialize bias map.
        """

    def _estimate_kappa(
        self, samples: list[CalibrationSample]
    ) -> tuple[float, float]:
        """Compare geometric gaze angles to target angles to estimate kappa."""

    def _fit_eyeball_radius(
        self, samples: list[CalibrationSample]
    ) -> float:
        """Fit eyeball radius from iris radius vs gaze angle relationship."""
```

---

### 5.7 Personalized MLP Mapping

**File**: `src/gaze_estimation/model/mlp.py`

**Purpose**: Define the tiny MLP that maps features → screen coordinates.

```python
import torch
import torch.nn as nn

class GazeMLP(nn.Module):
    """
    Input: 34 features (see FEATURE_KEYS)
    Hidden: 64 → 128 → 64
    Output: 2 (screen_x, screen_y) normalized to [0, 1]
    """
    def __init__(self, input_dim: int = 34, hidden_dims: list[int] = [64, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, 2))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**File**: `src/gaze_estimation/model/trainer.py`

```python
class MLPTrainer:
    def __init__(
        self,
        input_dim: int = 34,
        hidden_dims: list[int] = [64, 128, 64],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 32,
        device: str = "cpu",  # "cpu" or "cuda"
    ):
        ...

    def train(
        self, X: np.ndarray, Y: np.ndarray
    ) -> GazeMLP:
        """
        X: (N, 34) feature matrix
        Y: (N, 2) screen coordinates normalized to [0, 1]
        Returns trained GazeMLP.
        Uses early stopping with 10% validation split.
        """

    def save(self, model: GazeMLP, path: str) -> None:
        """Save model weights + config."""

    def load(self, path: str) -> GazeMLP:
        """Load model from path."""
```

---

### 5.8 ONNX Inference

**File**: `src/gaze_estimation/model/onnx_inference.py`

```python
import onnxruntime as ort
import numpy as np

class ONNXInference:
    def __init__(self, model_path: str, device: str = "CPU"):
        providers = ["CPUExecutionProvider"]
        if device == "CUDA":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        features: (1, 34) float32
        returns: (1, 2) float32 — screen_x, screen_y in [0, 1]
        """
        return self.session.run(None, {self.input_name: features})[0]
```

---

### 5.9 Unscented Kalman Filter

**File**: `src/gaze_estimation/filtering/unscented_kalman.py`

**Purpose**: Smooth gaze predictions with a 6-state UKF (x, y, vx, vy, ax, ay).

```python
import numpy as np

class UnscentedKalmanFilter:
    """
    State: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]
    Handles nonlinear motion, predicts trajectory, handles dropped frames.
    """
    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        ...

    def predict(self, dt: float) -> None:
        """Predict next state using constant-acceleration model."""

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with new (x, y) measurement. Returns filtered state."""

    def get_state(self) -> np.ndarray:
        """Return current state vector."""

    def coast(self, dt: float) -> np.ndarray:
        """Predict-only step for dropped frames. Returns predicted state."""
```

---

### 5.10 One Euro Filter

**File**: `src/gaze_estimation/filtering/one_euro.py`

**Purpose**: Adaptive low-pass filter — stable when stationary, responsive when moving.

```python
import math

class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,     # Lower = more smoothing at low speed
        beta: float = 0.015,          # Higher = less smoothing at high speed
        d_cutoff: float = 1.0,        # Derivative cutoff
    ):
        ...

    def filter(self, value: float, timestamp: float) -> float:
        """Apply One Euro filtering to a scalar value."""

    def filter_2d(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        """Apply to (x, y) pair."""

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        """Compute alpha for exponential smoothing."""
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
```

---

### 5.11 Fixation Detector

**File**: `src/gaze_estimation/filtering/fixation_detector.py`

**Purpose**: Detect fixation, saccade, blink, and lost states. Adjust smoothing accordingly.

```python
from enum import Enum

class GazeState(Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    BLINK = "blink"
    LOST = "lost"

@dataclass
class FixationInfo:
    state: GazeState
    velocity: float          # px/s
    duration: float          # seconds in current state
    fixation_point: tuple[float, float] | None  # Centroid if fixating

class FixationDetector:
    def __init__(
        self,
        fixation_velocity_threshold: float = 100.0,  # px/s
        saccade_velocity_threshold: float = 500.0,  # px/s
        blink_ear_threshold: float = 0.15,           # Eye aspect ratio
        min_fixation_duration: float = 0.1,         # seconds
    ):
        ...

    def update(
        self,
        x: float,
        y: float,
        timestamp: float,
        left_ear: float,
        right_ear: float,
    ) -> FixationInfo:
        """
        Update state machine with new observation.
        Returns current FixationInfo.
        """

    def get_smoothing_multiplier(self) -> float:
        """
        During fixation: return 2.0 (increase smoothing).
        During saccade: return 0.3 (reduce smoothing, be responsive).
        During blink/lost: return 5.0 (heavy smoothing / hold).
        """
```

---

### 5.12 Online Adaptation

**File**: `src/gaze_estimation/adaptation/adaptation_buffer.py`

```python
from collections import deque

class AdaptationBuffer:
    def __init__(self, max_size: int = 5000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, features: dict, cursor_x: float, cursor_y: float) -> None:
        """Store (features, cursor_position) pair from implicit calibration."""

    def get_batch(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) arrays for training. If n=None, return all."""

    def count_new_since_last_train(self) -> int:
        """Track how many new samples accumulated since last retrain."""

    def mark_trained(self) -> None:
        """Reset new sample counter."""
```

**File**: `src/gaze_estimation/adaptation/online_trainer.py`

```python
class OnlineTrainer:
    def __init__(
        self,
        mlp: GazeMLP,
        buffer: AdaptationBuffer,
        retrain_threshold: int = 50,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
    ):
        ...

    def maybe_retrain(self) -> bool:
        """
        If buffer has >= retrain_threshold new samples:
          1. Get new batch.
          2. Fine-tune MLP for 1-3 epochs.
          3. Mark trained.
          4. Return True.
        Otherwise return False.
        """
```

---

### 5.13 Bias Map

**File**: `src/gaze_estimation/correction/bias_map.py`

```python
class BiasMap:
    """
    2D grid storing mean error (bias) at each screen region.
    Grid resolution: 40 columns × 20 rows.
    """
    def __init__(self, cols: int = 40, rows: int = 20, smoothing: float = 1.0):
        self.cols = cols
        self.rows = rows
        self.bias_x = np.zeros((rows, cols), dtype=np.float32)
        self.bias_y = np.zeros((rows, cols), dtype=np.float32)
        self.counts = np.zeros((rows, cols), dtype=np.int32)

    def add_sample(self, screen_x: float, screen_y: float, error_x: float, error_y: float) -> None:
        """Add an observed error at a screen location."""

    def get_correction(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Get interpolated bias correction at a screen location."""

    def smooth(self) -> None:
        """Apply Gaussian smoothing to the bias map."""

    def save(self, path: str) -> None:
        """Save bias map to .npz."""

    def load(self, path: str) -> None:
        """Load bias map from .npz."""
```

---

### 5.14 Profile Manager

**File**: `src/gaze_estimation/profile/profile_manager.py`

```python
class ProfileManager:
    def __init__(self, profiles_dir: str = "profiles"):
        ...

    def create_profile(self, user_id: str, screen_resolution: tuple[int, int], camera_name: str) -> str:
        """Create new profile directory, return profile path."""

    def save_calibration(self, user_id: str, result: CalibrationResult) -> None:
        """Save calibration result (MLP weights, kappa, bias map, metadata)."""

    def load_profile(self, user_id: str) -> dict | None:
        """Load profile: returns dict with mlp, kappa, bias_map, metadata."""

    def list_profiles(self) -> list[str]:
        """List all user profile IDs."""

    def needs_recalibration(self, user_id: str, current_camera: str, current_screen: tuple[int, int]) -> bool:
        """Check if full recalibration is needed (camera/screen changed)."""

    def needs_quick_calibration(self, user_id: str, days_since_last: int = 7) -> bool:
        """Check if 5-point quick calibration is recommended."""
```

**Profile JSON schema** (`src/gaze_estimation/profile/schema.py`):

```python
from pydantic import BaseModel
from typing import Optional

class UserProfile(BaseModel):
    user_id: str
    screen_resolution: tuple[int, int]
    camera_name: str
    camera_intrinsics: list[list[float]]  # 3x3 matrix
    dist_coeffs: list[float]
    calibration_samples: int
    mlp_weights_path: str
    kappa_yaw: float
    kappa_pitch: float
    eyeball_radius: float
    bias_map_path: str | None
    last_session: str  # ISO datetime
    calibration_grid: tuple[int, int]  # (cols, rows)
    mlp_config: dict   # input_dim, hidden_dims
```

---

### 5.15 Pipeline Orchestrator

**File**: `src/gaze_estimation/pipeline/pipeline.py`

```python
class GazeEstimationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.queues = {name: queue.Queue(maxsize=2) for name in [
            "frame", "face", "mesh", "pose", "gaze", "prediction"
        ]}
        self.stop_event = threading.Event()
        self.stages: list[StageThread] = []

    def start(self) -> None:
        """Initialize all stages, start all threads."""

    def stop(self) -> None:
        """Set stop_event, join all threads."""

    def get_latest_estimate(self) -> GazeEstimate | None:
        """Non-blocking read from prediction queue. Returns None if no new data."""

    def start_calibration(self, screen_width: int, screen_height: int) -> CalibrationResult:
        """Run 25-point calibration. Blocks until complete."""

    def start_quick_calibration(self) -> CalibrationResult:
        """Run 5-point quick recalibration."""

    def feed_click(self, cursor_x: float, cursor_y: float) -> None:
        """Feed a mouse click for implicit calibration."""
```

---

## 6. Data Schemas

### 6.1 Core Data Packets

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

# === Packets flow through the pipeline ===

@dataclass
class FramePacket:
    timestamp: float
    frame: np.ndarray          # (H, W, 3) BGR
    frame_id: int

@dataclass
class FacePacket:
    timestamp: float
    frame: np.ndarray
    face_bbox: tuple[int, int, int, int] | None
    detection_confidence: float
    face_landmarks_2d: np.ndarray | None  # (6, 2)

@dataclass
class HeadPose:
    rvec: np.ndarray          # (3,)
    tvec: np.ndarray          # (3,)
    euler_angles: np.ndarray  # [yaw, pitch, roll] degrees
    rotation_matrix: np.ndarray  # (3, 3)

@dataclass
class MeshPacket:
    timestamp: float
    frame: np.ndarray
    face_bbox: tuple[int, int, int, int] | None
    mesh_468: np.ndarray         # (468, 2)
    iris_478: np.ndarray | None  # (478, 2)
    left_iris_center: tuple[float, float] | None
    right_iris_center: tuple[float, float] | None
    left_iris_radius: float | None
    right_iris_radius: float | None
    confidence: float

@dataclass
class PosePacket:
    timestamp: float
    frame: np.ndarray
    mesh_468: np.ndarray
    iris_478: np.ndarray | None
    left_iris_center: tuple[float, float] | None
    right_iris_center: tuple[float, float] | None
    left_iris_radius: float | None
    right_iris_radius: float | None
    head_pose: HeadPose | None
    confidence: float

@dataclass
class GazeRay:
    origin: np.ndarray    # (3,)
    direction: np.ndarray # (3,) unit vector

@dataclass
class GazePacket:
    timestamp: float
    gaze_ray_left: GazeRay | None
    gaze_ray_right: GazeRay | None
    gaze_yaw: float
    gaze_pitch: float
    features: dict        # FEATURE_KEYS → float
    head_pose: HeadPose | None
    confidence: float

@dataclass
class PredictionPacket:
    timestamp: float
    screen_x: float       # pixels
    screen_y: float       # pixels
    confidence: float
    raw_features: dict
    source: str           # "mlp" | "geometric" | "hold"

@dataclass
class GazeEstimate:
    timestamp: float
    screen_x: float           # Filtered screen X
    screen_y: float           # Filtered screen Y
    raw_x: float              # Pre-filter X
    raw_y: float              # Pre-filter Y
    velocity: float           # px/s
    confidence: float
    fixation_state: GazeState
    fixation_duration: float  # seconds
    source: str               # "mlp" | "geometric" | "hold"
    latency_ms: float         # End-to-end latency for this frame

@dataclass
class CalibrationSample:
    features: dict
    screen_x: float
    screen_y: float
    timestamp: float

@dataclass
class CalibrationResult:
    samples: list[CalibrationSample]
    mlp_weights_path: str | None
    kappa_yaw: float
    kappa_pitch: float
    eyeball_radius: float
    bias_map: np.ndarray | None
    timestamp: float
    screen_resolution: tuple[int, int]
```

### 6.2 Feature Vector Definition

```python
FEATURE_KEYS = [
    "gaze_yaw_left", "gaze_pitch_left",
    "gaze_yaw_right", "gaze_pitch_right",
    "gaze_yaw_avg", "gaze_pitch_avg",
    "head_yaw", "head_pitch", "head_roll",
    "head_tx", "head_ty", "head_tz",
    "left_iris_x", "left_iris_y",
    "right_iris_x", "right_iris_y",
    "left_iris_radius", "right_iris_radius",
    "left_ear", "right_ear",
    "left_pupil_to_inner_x", "left_pupil_to_inner_y",
    "left_pupil_to_outer_x", "left_pupil_to_outer_y",
    "right_pupil_to_inner_x", "right_pupil_to_inner_y",
    "right_pupil_to_outer_x", "right_pupil_to_outer_y",
    "face_x", "face_y", "face_width", "face_height",
    "interpupil_distance",
    "landmark_confidence",
]
# Total: 34 features
FEATURE_DIM = len(FEATURE_KEYS)  # 34
```

---

## 7. Calibration Protocol

### 7.1 Full 25-Point Calibration

**Grid**: 5×5 evenly spaced targets covering 10%–90% of screen in both dimensions.

```
(10%,10%)  (30%,10%)  (50%,10%)  (70%,10%)  (90%,10%)
(10%,30%)  (30%,30%)  (50%,30%)  (70%,30%)  (90%,30%)
(10%,50%)  (30%,50%)  (50%,50%)  (70%,50%)  (90%,50%)
(10%,70%)  (30%,70%)  (50%,70%)  (70%,70%)  (90%,70%)
(10%,90%)  (30%,90%)  (50%,90%)  (70%,90%)  (90%,90%)
```

**Target order**: Randomized to prevent systematic drift from user anticipation.

**Per-target sequence**:
1. Target appears at position (fade-in, 200ms)
2. Target pulses (scale animation, 400ms)
3. Small circular motion (radius 15px, 400ms) — helps the eye lock on
4. **Data collection**: 2 seconds at 60 FPS → 120 samples per target
5. Target fades out (200ms), next target appears

**Total samples**: 25 × 120 = 3000 samples (after outlier removal: ~2400–2700)

**Outlier removal**: For each target, compute mean + std of gaze predictions. Reject samples > 2σ from mean. If >30% of samples are rejected for a target, flag it for re-collection.

### 7.2 Quick 5-Point Recalibration

**Points**: Center + 4 corners at 15% / 85% margins.

```
    ●           ●


        ●


    ●           ●
```

**Per-target**: 1 second, 60 samples → 300 total samples total.

**Usage**: Fine-tune existing MLP weights rather than training from scratch. Adjusts for small drift (lighting, head position, glasses).

### 7.3 Implicit Calibration (Click-Based)

**Trigger**: User clicks, double-clicks, or presses a button.

**Assumption**: At the moment of a deliberate click, gaze ≈ cursor position (±50px).

**Process**:
1. On click event, grab the latest `GazePacket` from `gaze_queue`.
2. Store `(features, cursor_x, cursor_y)` into `AdaptationBuffer`.
3. If buffer has ≥50 new samples, trigger `OnlineTrainer.maybe_retrain()`.
4. Retraining: fine-tune MLP for 1–3 epochs on new batch only.

**Filtering**: Ignore clicks during rapid mouse movement (velocity > 500 px/s), as these are likely not gaze-directed.

### 7.4 Kappa Angle Estimation

During full calibration, for each sample where the user looks at a known target:

1. Compute the **geometric** gaze yaw/pitch from the 3D gaze ray (optical axis).
2. Compute the **target** yaw/pitch from the target screen position and head pose.
3. The difference is the kappa angle for that sample.
4. Average across all samples (after outlier removal) to get `(kappa_yaw, kappa_pitch)`.

```python
def estimate_kappa(samples: list[CalibrationSample]) -> tuple[float, float]:
    kappa_yaws = []
    kappa_pitchs = []
    for s in samples:
        target_yaw, target_pitch = screen_to_angles(s.screen_x, s.screen_y, s.head_pose)
        optical_yaw = s.features["gaze_yaw_avg"]
        optical_pitch = s.features["gaze_pitch_avg"]
        kappa_yaws.append(target_yaw - optical_yaw)
        kappa_pitchs.append(target_pitch - optical_pitch)
    kappa_yaw = np.median(kappa_yaws)  # Median is robust to outliers
    kappa_pitch = np.median(kappa_pitchs)
    return kappa_yaw, kappa_pitch
```

---

## 8. Model Architecture

### 8.1 MLP Definition

```
Input:  34 features (float32)
Layer 1: Linear(34, 64)  → ReLU → BatchNorm1d(64)
Layer 2: Linear(64, 128) → ReLU → BatchNorm1d(128)
Layer 3: Linear(128, 64) → ReLU → BatchNorm1d(64)
Output: Linear(64, 2)    → Sigmoid  → (screen_x_norm, screen_y_norm) in [0, 1]
```

**Parameter count**: ~14K parameters. Training time: <200ms for 3000 samples on CPU.

### 8.2 Training Pipeline

```python
# 1. Collect calibration data (3000 samples)
# 2. Normalize features (z-score, store mean/std)
# 3. Normalize labels to [0, 1] (divide by screen resolution)
# 4. Split: 90% train, 10% validation
# 5. Train:
#    - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
#    - Loss: MSELoss
#    - Epochs: 200 (early stopping, patience=20)
#    - Batch size: 32
# 6. Export to ONNX
# 7. Save weights + normalizer stats + config
```

### 8.3 ONNX Export

```python
def export_to_onnx(model: GazeMLP, path: str, input_dim: int = 34) -> None:
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy_input, path,
        input_names=["features"],
        output_names=["screen_coords"],
        dynamic_axes={"features": {0: "batch"}, "screen_coords": {0: "batch"}},
        opset_version=17,
    )
```

### 8.4 TensorRT (Optional GPU Acceleration)

```python
# After ONNX export:
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
#
# Inference via TensorRT Python API or ONNX Runtime with TensorRT execution provider.
```

### 8.5 Geometric Fallback Model

If no calibration data exists yet, or confidence is low:

```python
def geometric_predict(gaze_ray: GazeRay, head_pose: HeadPose, kappa: tuple, screen_plane) -> tuple[float, float]:
    """
    1. Apply kappa compensation to optical axis.
    2. Transform visual axis to camera frame using head pose.
    3. Intersect ray with screen plane.
    4. Return (screen_x, screen_y) in pixels.
    """
```

---

## 9. Configuration System

### 9.1 Default Config (`src/gaze_estimation/config/default_config.yaml`)

```yaml
camera:
  index: 0
  width: 1280
  height: 720
  fps: 60
  auto_exposure: true

detection:
  model: "mediapipe_short"  # "mediapipe_short" | "mediapipe_full" | "yolov8"
  detection_interval: 5     # Full detection every N frames
  min_confidence: 0.5

mesh:
  refine_iris: true
  max_num_faces: 1
  min_detection_confidence: 0.5
  static_image_mode: false

pose:
  solvepnp_method: "SOLVEPNP_ITERATIVE"
  use_ransac: false

gaze:
  eyeball_radius_mm: 12.0
  kappa_yaw: 0.0      # Set during calibration
  kappa_pitch: 0.0

calibration:
  grid_cols: 5
  grid_rows: 5
  samples_per_target: 120
  target_duration_sec: 2.0
  pulse_animation: true
  circular_motion: true
  circular_radius_px: 15
  outlier_sigma_threshold: 2.0

quick_calibration:
  points: 5
  samples_per_target: 60
  target_duration_sec: 1.0

mlp:
  input_dim: 34
  hidden_dims: [64, 128, 64]
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 200
  batch_size: 32
  early_stopping_patience: 20

inference:
  backend: "onnx"    # "onnx" | "torch" | "tensorrt"
  device: "CPU"      # "CPU" | "CUDA"
  fallback_to_geometric: true
  confidence_threshold: 0.7

filtering:
  ukf:
    process_noise: 1.0
    measurement_noise: 5.0
    alpha: 0.001
    beta: 2.0
    kappa: 0.0
  one_euro:
    min_cutoff: 1.0
    beta: 0.015
    d_cutoff: 1.0
  fixation:
    fixation_velocity_threshold: 100.0   # px/s
    saccade_velocity_threshold: 500.0    # px/s
    blink_ear_threshold: 0.15
    min_fixation_duration: 0.1           # seconds

adaptation:
  buffer_max_size: 5000
  retrain_threshold: 50
  retrain_lr: 0.0001
  retrain_epochs: 2
  retrain_batch_size: 16
  click_velocity_threshold: 500.0  # Ignore clicks above this mouse velocity

bias_map:
  cols: 40
  rows: 20
  smoothing_sigma: 1.0

profile:
  profiles_dir: "profiles"
  auto_save: true
  quick_calib_interval_days: 7

logging:
  level: "INFO"
  log_latency: true
  log_fps: true
  save_debug_frames: false
```

### 9.2 Config Loader

```python
# src/gaze_estimation/config/config.py
from dataclasses import dataclass, field
from typing import Any
import yaml

@dataclass
class Config:
    """Top-level configuration loaded from YAML."""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, dot_key: str, default=None):
        """Access nested keys: config.get('camera.width', 1280)"""
        keys = dot_key.split(".")
        val = self.raw
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
```

---

## 10. Dependencies & Environment

### 10.1 `requirements.txt`

```
# Core
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
torch>=2.0.0
onnxruntime>=1.15.0
onnx>=1.14.0

# Utilities
pyyaml>=6.0
pydantic>=2.0.0
scipy>=1.10.0

# GUI (for calibration display)
pygame>=2.5.0
# or: tkinter (stdlib, for simpler UI)

# Optional GPU
# tensorrt>=8.6.0  # Install separately per CUDA version
```

### 10.2 `requirements-dev.txt`

```
-r requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
mypy>=1.0.0
ruff>=0.1.0
black>=23.0.0
```

### 10.3 Hardware Targets

| Tier | CPU | GPU | Expected FPS | Notes |
|------|-----|-----|-------------|-------|
| Minimum | Intel i5-8xxx | None | 30 | MediaPipe CPU, ONNX CPU |
| Recommended | Intel i7-12xxx | None | 60 | MediaPipe CPU, ONNX CPU |
| High-end | Ryzen 7 7xxx | RTX 3060+ | 120+ | TensorRT, CUDA pipeline |

---

## 11. Testing & Evaluation

### 11.1 Metrics

```python
def angular_error(pred_yaw, pred_pitch, true_yaw, true_pitch):
    """Compute angular error in degrees using spherical law of cosines."""
    pred = np.array([np.cos(np.radians(pred_pitch)) * np.cos(np.radians(pred_yaw)),
                     np.cos(np.radians(pred_pitch)) * np.sin(np.radians(pred_yaw)),
                     np.sin(np.radians(pred_pitch))])
    true = np.array([np.cos(np.radians(true_pitch)) * np.cos(np.radians(true_yaw)),
                     np.cos(np.radians(true_pitch)) * np.sin(np.radians(true_yaw)),
                     np.sin(np.radians(true_pitch))])
    cos_angle = np.dot(pred, true)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def screen_error_px(pred_xy, true_xy):
    """Euclidean distance in pixels."""
    return np.linalg.norm(np.array(pred_xy) - np.array(true_xy))
```

### 11.2 Test Categories

#### Unit Tests

| Test File | What It Tests |
|-----------|--------------|
| `test_camera.py` | Camera init, frame capture, timestamp monotonicity |
| `test_face_detector.py` | Detection on synthetic face images, no-face case |
| `test_face_mesh.py` | Mesh extraction, landmark count (468/478), iris center accuracy |
| `test_head_pose.py` | solvePnP correctness, euler angle conversion, known-pose test |
| `test_gaze_geometry.py` | Gaze ray computation, ray-to-angle, kappa compensation |
| `test_gaze_features.py` | Feature vector: correct keys, no NaN, correct normalization |
| `test_calibration_engine.py` | Target generation, sample collection, outlier removal |
| `test_mlp.py` | Model forward pass, output shape, output range [0, 1] |
| `test_mlp_trainer.py` | Training convergence on synthetic data, early stopping |
| `test_unscented_kalman.py` | Prediction step, update step, coasting, convergence |
| `test_one_euro.py` | Smoothing at zero velocity, responsiveness at high velocity |
| `test_fixation_detector.py` | State transitions, velocity thresholds, blink detection |
| `test_bias_map.py` | Sample insertion, interpolation, smoothing, save/load |
| `test_profile_manager.py` | Create, save, load, list profiles, recalibration logic |

#### Integration Tests

| Test File | What It Tests |
|-----------|--------------|
| `test_pipeline_integration.py` | Full pipeline with recorded video, end-to-end packet flow |
| `test_accuracy_benchmark.py` | Accuracy against ground-truth dataset |

#### Accuracy Benchmark

**Ground truth dataset**: Record a video of a user looking at known screen positions (e.g., a 10×10 grid of targets with timestamps). The system processes the video offline and compares predictions to ground truth.

```python
# scripts/benchmark_accuracy.py
def run_benchmark(video_path: str, ground_truth_csv: str, config: Config):
    """
    1. Load video and ground truth CSV (timestamp, screen_x, screen_y).
    2. Run pipeline on each frame.
    3. For each ground truth timestamp, find nearest prediction.
    4. Compute: mean angular error, mean pixel error, 95th percentile error.
    5. Report per-region error (bias map validation).
    """
```

### 11.3 Test Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def synthetic_frame():
    """480x640 BGR image with a drawn face."""
    ...

@pytest.fixture
def mock_mesh_landmarks():
    """468 2D landmarks for a frontal face."""
    ...

@pytest.fixture
def mock_head_pose():
    """HeadPose with known yaw=0, pitch=0, roll=0."""
    ...

@pytest.fixture
def calibration_data():
    """3000 synthetic CalibrationSamples for MLP training."""
    ...
```

---

## 12. Performance Optimization

### 12.1 CPU Path

```
Camera → MediaPipe (CPU) → OpenCV solvePnP → ONNX Runtime (CPU) → Python filters
```

**Optimizations**:
- Use `cv2.UMat` for OpenCV operations (OpenCL acceleration)
- Set ONNX Runtime intra-op threads to physical core count
- Use `numpy` pre-allocated buffers for feature extraction (avoid per-frame allocation)
- Batch mesh extraction if MediaPipe supports it
- Pre-compute undistort maps for camera

### 12.2 GPU Path

```
Camera → MediaPipe (GPU) → CUDA solvePnP → ONNX Runtime (CUDA/TensorRT) → CUDA filters
```

**Optimizations**:
- MediaPipe GPU delegate
- ONNX Runtime with CUDA execution provider
- TensorRT FP16 inference for MLP
- Keep data on GPU between stages (avoid CPU↔GPU transfers)

### 12.3 Latency Profiling

```python
# src/gaze_estimation/utils/timing.py
class LatencyProfiler:
    """Track per-stage latency and end-to-end latency."""
    def mark(self, stage: str, timestamp: float) -> None: ...
    def get_report(self) -> dict[str, float]: ...
    # Example output:
    # {
    #   "camera": 2.1,
    #   "detection": 5.3,
    #   "mesh": 8.2,
    #   "pose": 1.1,
    #   "gaze": 0.8,
    #   "inference": 0.5,
    #   "filter": 0.3,
    #   "total": 18.3
    # }
```

### 12.4 Backpressure & Frame Dropping

- All queues have `maxsize=2`. If full, the oldest item is dropped (non-blocking put).
- This ensures the system always processes the most recent frame, preventing latency buildup.
- The camera thread runs at native FPS; downstream threads process as fast as possible.

```python
def put_or_drop(queue, item):
    try:
        queue.put_nowait(item)
    except queue.Full:
        try:
            queue.get_nowait()  # Drop oldest
            queue.put_nowait(item)
        except queue.Empty:
            pass
```

---

## 13. Implementation Roadmap

> **Historical roadmap — every phase below is implemented.** The checkboxes were
> never ticked as work progressed, so they are not a status indicator; see the
> banner at the top of this document and the test suite for what actually exists.

### Phase 1: Core Pipeline (Week 1–2)

```
- [ ] Project scaffold: folder structure, setup.py, pyproject.toml
- [ ] Config system: YAML loader, default config
- [ ] Camera capture thread + FramePacket
- [ ] Face detection (MediaPipe) + FacePacket
- [ ] Face mesh + iris extraction + MeshPacket
- [ ] Head pose estimation (solvePnP) + PosePacket
- [ ] Pipeline orchestrator (threads + queues)
- [ ] Latency profiler
- [ ] Unit tests for all above
```

**Milestone**: Live face mesh + head pose overlay on webcam feed at 30+ FPS.

### Phase 2: 3D Gaze Geometry (Week 2–3)

```
- [ ] Gaze ray computation (undistort → back-project → head frame)
- [ ] Gaze yaw/pitch extraction
- [ ] Feature extraction (all 34 FEATURE_KEYS)
- [ ] Kappa angle estimation
- [ ] Eyeball radius fitting
- [ ] Geometric gaze-to-screen prediction (fallback model)
- [ ] Unit tests for gaze geometry
```

**Milestone**: Geometric gaze prediction on screen (uncalibrated, ~5° error).

### Phase 3: Calibration System (Week 3–4)

```
- [ ] CalibrationEngine: 25-point animated calibration
- [ ] CalibrationUI: fullscreen target display with animation
- [ ] CalibrationSample collection + outlier removal
- [ ] CalibrationResult serialization
- [ ] Quick 5-point recalibration
- [ ] Unit tests for calibration engine
```

**Milestone**: User can complete calibration, data is collected and stored.

### Phase 4: Personalized MLP (Week 4–5)

```
- [ ] GazeMLP PyTorch model
- [ ] MLPTrainer: training pipeline with early stopping
- [ ] ONNX export
- [ ] ONNX inference wrapper
- [ ] MLP prediction in pipeline
- [ ] Profile save/load (weights, normalizer, kappa, config)
- [ ] Unit tests for MLP + trainer
```

**Milestone**: Post-calibration accuracy <3° error, MLP runs in <1ms.

### Phase 5: Temporal Filtering (Week 5–6)

```
- [ ] UnscentedKalmanFilter implementation
- [ ] OneEuroFilter implementation
- [ ] FixationDetector state machine
- [ ] Adaptive smoothing (fixation-aware)
- [ ] Integration into pipeline
- [ ] Unit tests for all filters
```

**Milestone**: Smooth, stable gaze predictions with fixation/saccade awareness.

### Phase 6: Online Adaptation + Error Map (Week 6–7)

```
- [ ] AdaptationBuffer: store click-based samples
- [ ] OnlineTrainer: incremental MLP fine-tuning
- [ ] ImplicitCalibration: mouse click hook
- [ ] BiasMap: 2D spatial error correction
- [ ] BiasMap interpolation and smoothing
- [ ] Integration tests
```

**Milestone**: System improves accuracy over time with usage.

### Phase 7: Profile Management + Polish (Week 7–8)

```
- [ ] ProfileManager: full CRUD for user profiles
- [ ] Recalibration logic (full vs quick)
- [ ] Camera intrinsics calibration script
- [ ] Visualization overlay (debug mode)
- [ ] Accuracy benchmark script
- [ ] Latency benchmark script
- [ ] Documentation: README, usage guide
- [ ] Integration tests with recorded video
```

**Milestone**: Production-ready system with user profiles, benchmarks, and documentation.

### Phase 8: GPU Acceleration (Optional, Week 9)

```
- [ ] MediaPipe GPU delegate
- [ ] TensorRT export + inference
- [ ] GPU-resident pipeline (minimize CPU↔GPU transfer)
- [ ] Benchmark CPU vs GPU
```

**Milestone**: 120+ FPS on GPU, <10ms latency.

---

## Appendix A: Key Formulas Reference

### A.1 Euler Angle Conversion

```python
def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → [yaw, pitch, roll] in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        yaw = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    return np.array([yaw, pitch, roll])
```

### A.2 Eye Aspect Ratio (EAR)

```python
def eye_aspect_ratio(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1..p6 are the 6 eye landmarks.
    """
    p = landmarks[eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)
```

### A.3 Screen-to-Angle Conversion

```python
def screen_to_angles(screen_x, screen_y, screen_width, screen_height, distance_mm):
    """
    Convert screen pixel position to gaze angles assuming user is at
    `distance_mm` from screen center.
    """
    # Convert pixels to mm (assuming 96 DPI or measured PPI)
    # For simplicity, use a fixed mm-per-pixel ratio
    mm_per_px = 0.3  # Approximate for 1080p at 24"
    x_mm = (screen_x - screen_width / 2) * mm_per_px
    y_mm = (screen_y - screen_height / 2) * mm_per_px
    yaw = np.degrees(np.arctan2(x_mm, distance_mm))
    pitch = np.degrees(np.arctan2(y_mm, distance_mm))
    return yaw, pitch
```

---

## Appendix B: MediaPipe Landmark Indices

### B.1 solvePnP Reference Points

```python
# MediaPipe Face Mesh landmark indices for solvePnP
SOLVEPNP_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
```

### B.2 Eye Landmarks (for EAR)

```python
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
```

### B.3 Iris Landmarks

```python
# MediaPipe 478-point model iris landmarks
LEFT_IRIS_INDICES = list(range(468, 473))   # 468-472
RIGHT_IRIS_INDICES = list(range(473, 478))  # 473-477
```

---

## Appendix C: Entry Points

### C.1 Main Tracker

```python
# scripts/run_tracker.py
"""
Real-time gaze tracking entry point.
Usage: python scripts/run_tracker.py [--config configs/my_config.yaml] [--user user_001]
"""
from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
from gaze_estimation.config.config import Config
from gaze_estimation.profile.profile_manager import ProfileManager

def main():
    config = Config.from_yaml("configs/my_config.yaml")
    profile_mgr = ProfileManager(config.get("profile.profiles_dir"))

    pipeline = GazeEstimationPipeline(config)
    pipeline.start()

    # Load user profile if exists
    # Run calibration if needed
    # Display gaze overlay

    try:
        while True:
            estimate = pipeline.get_latest_estimate()
            if estimate:
                print(f"Gaze: ({estimate.screen_x:.1f}, {estimate.screen_y:.1f}) "
                      f"State: {estimate.fixation_state} "
                      f"Conf: {estimate.confidence:.2f}")
    except KeyboardInterrupt:
        pipeline.stop()
```

### C.2 Calibration Session

```python
# scripts/run_calibration.py
"""
Standalone calibration session.
Usage: python scripts/run_calibration.py [--user user_001] [--quick]
"""
```

### C.3 Benchmark

```python
# scripts/benchmark_accuracy.py
"""
Evaluate accuracy against ground truth.
Usage: python scripts/benchmark_accuracy.py --video data/test_video.mp4 --gt data/ground_truth.csv
"""
```

---

*End of Document*
