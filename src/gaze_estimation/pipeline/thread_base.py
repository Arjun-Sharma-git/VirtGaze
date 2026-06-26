"""Base class for all pipeline stage threads."""
from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

from gaze_estimation.utils.logging import get_logger
from gaze_estimation.utils.timing import FPSCounter, StageTimer


def put_or_drop(q: queue.Queue, item: object) -> None:
    """Put *item* on *q*, dropping the oldest item if the queue is full.

    This prevents unbounded queue growth while ensuring the most recent
    data is always available downstream.
    """
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()          # Discard oldest
            q.put_nowait(item)
        except queue.Empty:
            pass


class StageThread(threading.Thread, ABC):
    """Abstract base for a single pipeline stage running on its own thread.

    Subclasses must implement :meth:`process` which receives one item from
    ``input_queue``, does work, and should call ``self.emit(result)`` to push
    data to the next stage.

    Thread lifecycle::

        thread.start()   # begins run() loop
        thread.stop()    # signals stop_event, join externally if needed
    """

    def __init__(
        self,
        input_queue: Optional[queue.Queue],
        output_queue: Optional[queue.Queue],
        stop_event: threading.Event,
        name: str = "stage_thread",
        timeout: float = 0.1,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._timeout = timeout
        self._logger = get_logger(name)
        self._fps = FPSCounter(window=60)
        self._timer = StageTimer()

    # ── Public API ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the thread to stop."""
        self.stop_event.set()

    def emit(self, item: object) -> None:
        """Push *item* to the output queue, dropping oldest if full."""
        if self.output_queue is not None:
            put_or_drop(self.output_queue, item)

    @property
    def fps(self) -> float:
        """Current processing FPS for this stage."""
        return self._fps.get()

    @property
    def last_ms(self) -> float:
        """Latency of the most recent processing call (ms)."""
        return self._timer.last_ms

    # ── Abstract ───────────────────────────────────────────────────────────

    @abstractmethod
    def process(self, item: object) -> None:
        """Process one item from the input queue.

        Call ``self.emit(result)`` to forward to the next stage.
        If nothing should be emitted (e.g. confidence too low), simply return.
        """

    # ── Internal ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop: read from input_queue, call process(), emit output."""
        self._logger.info("Thread started")
        self._setup()
        try:
            while not self.stop_event.is_set():
                if self.input_queue is None:
                    # Source stage — generate data without an input queue
                    self._timer.start()
                    self.process(None)
                    self._timer.stop()
                    self._fps.tick()
                    continue

                try:
                    item = self.input_queue.get(timeout=self._timeout)
                except queue.Empty:
                    continue

                self._timer.start()
                try:
                    self.process(item)
                except Exception as exc:
                    self._logger.exception("Error processing item: %s", exc)
                finally:
                    self._timer.stop()
                    self._fps.tick()
        finally:
            self._teardown()
            self._logger.info("Thread stopped")

    def _setup(self) -> None:
        """Override to perform one-time initialisation inside the thread."""

    def _teardown(self) -> None:
        """Override to perform cleanup when the thread exits."""
