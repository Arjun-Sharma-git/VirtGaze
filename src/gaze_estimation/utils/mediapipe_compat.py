"""MediaPipe face detection / face mesh across the legacy and Tasks APIs.

MediaPipe 0.10.30 removed ``mediapipe.solutions`` (FaceDetection, FaceMesh);
newer releases, the only ones published for Python 3.13+, offer the Tasks API
(``FaceDetector``, ``FaceLandmarker``) instead, which needs a model file.

The factories below return the legacy solution object when it exists, and
otherwise an adapter around the Tasks API whose ``process(rgb)`` result has the
same shape the pipeline already reads:

* detection: ``.detections[i].score[0]``,
  ``.location_data.relative_bounding_box`` and ``.relative_keypoints``
* mesh: ``.multi_face_landmarks[i].landmark[j].x / .y`` (normalised)

Model files live in ``models/`` at the repository root (or in the directory
named by ``GAZE_MODELS_DIR``) and are downloaded from Google's model storage
on first use when missing.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np

from gaze_estimation.utils.logging import get_logger

_logger = get_logger("utils.mediapipe_compat")

_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
FACE_DETECTOR_MODEL = (
    "blaze_face_short_range.tflite",
    f"{_MODEL_BASE}/face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite",
)
FACE_LANDMARKER_MODEL = (
    "face_landmarker.task",
    f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
)


def has_legacy_solutions() -> bool:
    """True when the installed MediaPipe still ships ``mediapipe.solutions``."""
    import mediapipe as mp

    return hasattr(mp, "solutions")


def models_dir() -> Path:
    """Directory holding the MediaPipe Tasks model files."""
    env = os.environ.get("GAZE_MODELS_DIR")
    if env:
        return Path(env)
    # src/gaze_estimation/utils/ -> repository root
    return Path(__file__).resolve().parents[3] / "models"


def ensure_model(model: tuple) -> str:
    """Return the local path of *model*, downloading it once if missing."""
    filename, url = model
    path = models_dir() / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _logger.info("Downloading MediaPipe model %s", url)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return str(path)


def _mp_image(rgb: np.ndarray) -> Any:
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class _TasksFaceDetection:
    """``FaceDetector`` (Tasks API) behind the legacy ``FaceDetection`` interface."""

    def __init__(self, min_detection_confidence: float) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(FACE_DETECTOR_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def process(self, rgb: np.ndarray) -> SimpleNamespace:
        h, w = rgb.shape[:2]
        result = self._detector.detect(_mp_image(rgb))
        detections: List[SimpleNamespace] = []
        for det in sorted(
            result.detections,
            key=lambda d: d.categories[0].score if d.categories else 0.0,
            reverse=True,
        ):
            box = det.bounding_box  # pixels
            score = det.categories[0].score if det.categories else 0.0
            detections.append(
                SimpleNamespace(
                    score=[score],
                    location_data=SimpleNamespace(
                        relative_bounding_box=SimpleNamespace(
                            xmin=box.origin_x / w,
                            ymin=box.origin_y / h,
                            width=box.width / w,
                            height=box.height / h,
                        ),
                        relative_keypoints=[
                            SimpleNamespace(x=kp.x, y=kp.y) for kp in (det.keypoints or [])
                        ],
                    ),
                )
            )
        return SimpleNamespace(detections=detections)

    def close(self) -> None:
        self._detector.close()


class _TasksFaceMesh:
    """``FaceLandmarker`` (Tasks API) behind the legacy ``FaceMesh`` interface.

    The landmarker always returns the 478-point model (mesh + iris), which is
    what ``FaceMesh(refine_landmarks=True)`` produced.
    """

    def __init__(
        self,
        max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        static_image_mode: bool,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._video = not static_image_mode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(FACE_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.VIDEO if self._video else vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, rgb: np.ndarray) -> SimpleNamespace:
        image = _mp_image(rgb)
        if self._video:
            # VIDEO mode needs strictly increasing timestamps.
            ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        faces = [SimpleNamespace(landmark=list(lms)) for lms in result.face_landmarks]
        return SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self) -> None:
        self._landmarker.close()


def create_face_detection(model_selection: int, min_detection_confidence: float) -> Any:
    """Legacy ``FaceDetection`` when available, else the Tasks adapter.

    The Tasks API only publishes the short-range model, so ``model_selection``
    (full range) is ignored there.
    """
    if has_legacy_solutions():
        import mediapipe as mp

        return mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )
    if model_selection != 0:
        _logger.warning("MediaPipe Tasks has no full-range face detector; using short range")
    return _TasksFaceDetection(min_detection_confidence)


def create_face_mesh(
    max_num_faces: int,
    refine_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    static_image_mode: bool,
) -> Any:
    """Legacy ``FaceMesh`` when available, else the Tasks adapter."""
    if has_legacy_solutions():
        import mediapipe as mp

        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )
    return _TasksFaceMesh(
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        static_image_mode=static_image_mode,
    )
