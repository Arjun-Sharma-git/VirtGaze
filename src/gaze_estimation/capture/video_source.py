"""Video file source: replays a recorded video for offline testing/benchmarking."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2

from gaze_estimation.pipeline.schemas import FramePacket
from gaze_estimation.pipeline.thread_base import StageThread


class VideoFileSource(StageThread):
    """Replays a video file as a stream of FramePackets.

    Useful for offline testing, benchmarking, and accuracy evaluation without
    a live camera.

    Args:
        video_path:     Path to the video file (.mp4, .avi, …).
        output_queue:   Queue to push FramePackets onto.
        stop_event:     Shared stop event.
        loop:           If True, restart from the beginning when the file ends.
        realtime:       If True, sleep to match the video's original FPS.
                        If False, run as fast as possible (benchmarking).
        name:           Thread name.
    """

    def __init__(
        self,
        video_path: str,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        loop: bool = False,
        realtime: bool = True,
        name: str = "video_source_thread",
    ) -> None:
        super().__init__(
            input_queue=None,
            output_queue=output_queue,
            stop_event=stop_event,
            name=name,
        )
        self._video_path = video_path
        self._loop = loop
        self._realtime = realtime
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._frame_interval: float = 1.0 / 30.0  # Default; updated on open

    def _setup(self) -> None:
        self._open()

    def _teardown(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def process(self, _item: object) -> None:
        """Read the next frame from the file and emit a FramePacket."""
        if self._cap is None:
            self.stop()
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                return
            else:
                self._logger.info("End of video file: %s", self._video_path)
                self.stop()
                return

        packet = FramePacket(
            timestamp=time.time(),
            frame=frame,
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        self.emit(packet)

        if self._realtime:
            time.sleep(self._frame_interval)

    # ── Private ───────────────────────────────────────────────────────────

    def _open(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self._video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._logger.info(
            "Opened video: %s  %.0f FPS  %d frames", self._video_path, fps, total
        )
        self._cap = cap
