"""BiasMap: 2D spatial error-correction grid for gaze predictions."""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import gaussian_filter


class BiasMap:
    """A 2-D grid that stores and applies location-dependent bias corrections.

    After calibration and some usage, the system accumulates mean prediction
    errors per screen region.  This map adds a correction term to predictions
    based on their screen location, reducing systematic spatial errors.

    Grid layout: (rows × cols) cells covering the full screen.
    Bilinear interpolation is used between grid cells.

    Args:
        cols:      Number of horizontal grid cells (default 40).
        rows:      Number of vertical grid cells (default 20).
        smoothing: Gaussian smoothing sigma applied after batch updates.
    """

    def __init__(
        self,
        cols: int = 40,
        rows: int = 20,
        smoothing: float = 1.0,
    ) -> None:
        self.cols = cols
        self.rows = rows
        self._smoothing = smoothing

        # Bias arrays in pixels (correction to add)
        self.bias_x = np.zeros((rows, cols), dtype=np.float32)
        self.bias_y = np.zeros((rows, cols), dtype=np.float32)
        # Accumulated bilinear weights per node (float, because samples are
        # splatted onto the 4 surrounding nodes).
        self.counts = np.zeros((rows, cols), dtype=np.float32)

    # ── Sample accumulation ───────────────────────────────────────────────

    def _node_coords(self, screen_x: float, screen_y: float, screen_width: int, screen_height: int):
        """Map a screen position to fractional grid-node coordinates.

        Node ``k`` sits at ``k / (n - 1)`` (i.e. the grid spans the full screen
        with nodes on both edges) — the same convention used by
        :meth:`get_correction`.
        """
        gx = float(np.clip(screen_x / screen_width * (self.cols - 1), 0.0, self.cols - 1))
        gy = float(np.clip(screen_y / screen_height * (self.rows - 1), 0.0, self.rows - 1))
        return gx, gy

    def add_sample(
        self,
        screen_x: float,
        screen_y: float,
        error_x: float,
        error_y: float,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Add one observed error at a screen location.

        The error is distributed over the four grid nodes surrounding the
        sample using the same bilinear weights as :meth:`get_correction`, so
        reading the map back at the same location reproduces the observed
        error instead of an attenuated version of it.

        Args:
            screen_x, screen_y: Predicted gaze position in pixels.
            error_x, error_y:   True − predicted (pixels).
            screen_width/height: Screen resolution for normalisation.
        """
        if self.cols < 2 or self.rows < 2:
            raise ValueError("BiasMap requires at least a 2x2 grid")

        gx, gy = self._node_coords(screen_x, screen_y, screen_width, screen_height)

        x0 = int(np.clip(np.floor(gx), 0, self.cols - 2))
        y0 = int(np.clip(np.floor(gy), 0, self.rows - 2))
        x1, y1 = x0 + 1, y0 + 1
        tx, ty = gx - x0, gy - y0

        weighted_nodes = (
            ((y0, x0), (1.0 - tx) * (1.0 - ty)),
            ((y0, x1), tx * (1.0 - ty)),
            ((y1, x0), (1.0 - tx) * ty),
            ((y1, x1), tx * ty),
        )

        for (row, col), weight in weighted_nodes:
            if weight <= 0.0:
                continue
            # Weighted running average: mean += w * (value - mean) / W_total
            total = float(self.counts[row, col]) + weight
            self.bias_x[row, col] += weight * (error_x - self.bias_x[row, col]) / total
            self.bias_y[row, col] += weight * (error_y - self.bias_y[row, col]) / total
            self.counts[row, col] = total

    def build_from_calibration(
        self,
        pred_xs: np.ndarray,
        pred_ys: np.ndarray,
        true_xs: np.ndarray,
        true_ys: np.ndarray,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Populate the map from calibration ground-truth pairs.

        Args:
            pred_xs, pred_ys: Predicted gaze positions (pixels).
            true_xs, true_ys: Ground-truth target positions (pixels).
        """
        for px, py, tx, ty in zip(pred_xs, pred_ys, true_xs, true_ys):
            self.add_sample(px, py, tx - px, ty - py, screen_width, screen_height)
        self.smooth()

    # ── Correction ────────────────────────────────────────────────────────

    def get_correction(
        self,
        screen_x: float,
        screen_y: float,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> tuple:
        """Get bilinearly interpolated bias correction at (screen_x, screen_y).

        Returns:
            (dx, dy) correction to ADD to the raw prediction.
        """
        # Map to fractional grid coordinates (nodes at k / (n - 1))
        gx, gy = self._node_coords(screen_x, screen_y, screen_width, screen_height)

        # Bilinear interpolation
        x0 = int(np.clip(np.floor(gx), 0, self.cols - 2))
        y0 = int(np.clip(np.floor(gy), 0, self.rows - 2))
        x1 = x0 + 1
        y1 = y0 + 1

        tx = gx - x0
        ty = gy - y0

        bx = (
            (1 - tx) * (1 - ty) * self.bias_x[y0, x0]
            + tx * (1 - ty) * self.bias_x[y0, x1]
            + (1 - tx) * ty * self.bias_x[y1, x0]
            + tx * ty * self.bias_x[y1, x1]
        )
        by_ = (
            (1 - tx) * (1 - ty) * self.bias_y[y0, x0]
            + tx * (1 - ty) * self.bias_y[y0, x1]
            + (1 - tx) * ty * self.bias_y[y1, x0]
            + tx * ty * self.bias_y[y1, x1]
        )
        return float(bx), float(by_)

    def apply(
        self,
        screen_x: float,
        screen_y: float,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> tuple:
        """Apply bias correction.  Returns corrected (screen_x, screen_y)."""
        dx, dy = self.get_correction(screen_x, screen_y, screen_width, screen_height)
        return screen_x + dx, screen_y + dy

    # ── Smoothing ─────────────────────────────────────────────────────────

    def smooth(self) -> None:
        """Apply Gaussian smoothing to reduce grid noise."""
        self.bias_x = gaussian_filter(self.bias_x, sigma=self._smoothing).astype(np.float32)
        self.bias_y = gaussian_filter(self.bias_y, sigma=self._smoothing).astype(np.float32)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to a .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            bias_x=self.bias_x,
            bias_y=self.bias_y,
            counts=self.counts,
            meta=np.array([self.cols, self.rows]),
        )

    def load(self, path: str) -> None:
        """Load from a .npz file."""
        data = np.load(path)
        self.bias_x = data["bias_x"].astype(np.float32)
        self.bias_y = data["bias_y"].astype(np.float32)
        self.counts = data["counts"].astype(np.float32)
        meta = data.get("meta")
        if meta is not None:
            self.cols = int(meta[0])
            self.rows = int(meta[1])

    def reset(self) -> None:
        """Zero out the bias map."""
        self.bias_x[:] = 0.0
        self.bias_y[:] = 0.0
        self.counts[:] = 0
