"""OverlayRenderer: draws gaze point, face mesh, and debug info on BGR frames."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from gaze_estimation.pipeline.schemas import GazeEstimate, GazeState, MeshPacket


class OverlayRenderer:
    """Draw gaze/debug overlays on a BGR frame (in-place).

    All drawing is done on a copy of the frame if ``copy=True``
    (default), or directly in-place if ``copy=False``.

    Usage::

        renderer = OverlayRenderer()
        display_frame = renderer.render(frame, estimate, mesh_packet)
        cv2.imshow("Gaze", display_frame)
    """

    # Colours (BGR)
    COLOR_GAZE = (0, 255, 0)       # Green  — gaze dot
    COLOR_GAZE_RAW = (0, 150, 255) # Orange — raw (pre-filter) projection
    COLOR_MESH = (80, 80, 80)      # Dark gray — mesh
    COLOR_IRIS = (255, 100, 50)    # Blue-ish — iris circle
    COLOR_FPS = (200, 200, 50)
    COLOR_STATE = {
        GazeState.FIXATION: (0, 220, 0),
        GazeState.SACCADE:  (0, 100, 255),
        GazeState.BLINK:    (0, 0, 220),
        GazeState.LOST:     (100, 100, 100),
    }

    def __init__(
        self,
        draw_mesh: bool = False,
        draw_iris: bool = True,
        draw_axes: bool = False,
        dot_radius: int = 12,
        draw_fps: bool = True,
        draw_latency: bool = True,
        copy: bool = True,
    ) -> None:
        self._draw_mesh = draw_mesh
        self._draw_iris = draw_iris
        self._draw_axes = draw_axes
        self._dot_radius = dot_radius
        self._draw_fps = draw_fps
        self._draw_latency = draw_latency
        self._copy = copy

    def render(
        self,
        frame: np.ndarray,
        estimate: Optional[GazeEstimate] = None,
        mesh_packet: Optional[MeshPacket] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Render all enabled overlays and return the annotated frame."""
        out = frame.copy() if self._copy else frame
        h, w = out.shape[:2]

        # ── Face mesh ──────────────────────────────────────────────────────
        if self._draw_mesh and mesh_packet is not None and mesh_packet.mesh_468 is not None:
            for pt in mesh_packet.mesh_468:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 1, self.COLOR_MESH, -1)

        # ── Iris circles ───────────────────────────────────────────────────
        if self._draw_iris and mesh_packet is not None:
            for center, radius in [
                (mesh_packet.left_iris_center, mesh_packet.left_iris_radius),
                (mesh_packet.right_iris_center, mesh_packet.right_iris_radius),
            ]:
                if center is not None and radius is not None:
                    cv2.circle(
                        out,
                        (int(center[0]), int(center[1])),
                        int(max(radius, 1)),
                        self.COLOR_IRIS, 1,
                    )

        # ── Gaze point overlay (on the camera preview frame) ───────────────
        # Note: screen coordinates != frame pixel coords;
        # we scale the screen gaze to [0,1] and re-project onto frame
        if estimate is not None and estimate.confidence > 0.3:
            # Approximate: display gaze as fraction of frame
            gx = int(np.clip(estimate.screen_x / 1920 * w, 0, w - 1))
            gy = int(np.clip(estimate.screen_y / 1080 * h, 0, h - 1))

            state_color = self.COLOR_STATE.get(estimate.fixation_state, self.COLOR_GAZE)

            # Outer ring (state colour)
            cv2.circle(out, (gx, gy), self._dot_radius + 4, state_color, 2)
            # Inner filled dot
            cv2.circle(out, (gx, gy), self._dot_radius, self.COLOR_GAZE, -1)

        # ── HUD text ───────────────────────────────────────────────────────
        if self._draw_fps and fps > 0.0:
            cv2.putText(
                out, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_FPS, 1, cv2.LINE_AA,
            )
        if self._draw_latency and latency_ms > 0.0:
            cv2.putText(
                out, f"Lat: {latency_ms:.1f}ms",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_FPS, 1, cv2.LINE_AA,
            )
        if estimate is not None:
            state_txt = estimate.fixation_state.value if estimate.fixation_state else "–"
            conf_txt = f"Conf: {estimate.confidence:.2f}  State: {state_txt}"
            cv2.putText(
                out, conf_txt,
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

        return out
