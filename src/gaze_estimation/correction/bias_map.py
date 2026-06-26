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
        self.counts = np.zeros((rows, cols), dtype=np.int32)

    # ── Sample accumulation ───────────────────────────────────────────────

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

        Args:
            screen_x, screen_y: Predicted gaze position in pixels.
            error_x, error_y:   True − predicted (pixels).
            screen_width/height: Screen resolution for normalisation.
        """
        col = int(np.clip(screen_x / screen_width * self.cols, 0, self.cols - 1))
        row = int(np.clip(screen_y / screen_height * self.rows, 0, self.rows - 1))

        # Running average
        n = self.counts[row, col] + 1
        self.bias_x[row, col] += (error_x - self.bias_x[row, col]) / n
        self.bias_y[row, col] += (error_y - self.bias_y[row, col]) / n
        self.counts[row, col] = n

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
        # Map to fractional grid coordinates
        gx = screen_x / screen_width * (self.cols - 1)
        gy = screen_y / screen_height * (self.rows - 1)

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
        self.counts = data["counts"].astype(np.int32)
        meta = data.get("meta")
        if meta is not None:
            self.cols = int(meta[0])
            self.rows = int(meta[1])

    def reset(self) -> None:
        """Zero out the bias map."""
        self.bias_x[:] = 0.0
        self.bias_y[:] = 0.0
        self.counts[:] = 0
