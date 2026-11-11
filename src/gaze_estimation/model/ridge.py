"""Ridge regression from landmark features to screen position.

The model EyeTrax uses by default (standardised features + ridge, see
:mod:`gaze_estimation.model.landmark_features`), written in plain numpy so it
needs no extra dependency.  Training is a closed-form solve, so refitting on
every few implicit-calibration clicks is cheap.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class RidgeGazeModel:
    """Standardise features, then ridge-regress normalised screen ``(x, y)``.

    The training set is kept so that clicks collected later (implicit
    calibration) can be added and the model refitted on everything.

    Args:
        alpha: L2 penalty on the standardised features.  The features are
               ~500 highly correlated landmark coordinates, so a strong
               penalty generalises far better: on a 25-point webcam
               calibration, leave-one-target-out error was ~200 px at 1.0
               and ~120 px at 100.
    """

    DEFAULT_ALPHA = 100.0

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = float(alpha)
        self.screen_width = 1920
        self.screen_height = 1080
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._coef: Optional[np.ndarray] = None  # (D, 2)
        self._intercept: Optional[np.ndarray] = None  # (2,)
        self._X = np.zeros((0, 0), dtype=np.float32)
        self._Y = np.zeros((0, 2), dtype=np.float32)  # screen pixels
        # Drift correction (normalised screen units) added to every prediction;
        # set by the quick recalibration when the head has moved since the
        # full calibration.
        self.offset = np.zeros(2, dtype=np.float64)

    @property
    def is_trained(self) -> bool:
        return self._coef is not None

    @property
    def n_samples(self) -> int:
        return int(self._X.shape[0])

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        Y_px: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> "RidgeGazeModel":
        """Fit on features ``X`` (N, D) and screen positions ``Y_px`` (N, 2) in pixels."""
        X = np.asarray(X, dtype=np.float32)
        Y_px = np.asarray(Y_px, dtype=np.float32).reshape(-1, 2)
        if X.ndim != 2 or len(X) != len(Y_px) or len(X) < 2:
            raise ValueError("need at least 2 samples with matching X and Y")
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self._X, self._Y = X, Y_px
        self._solve()
        return self

    def add_samples(self, X: np.ndarray, Y_px: np.ndarray) -> None:
        """Append samples (e.g. gaze-directed clicks); call :meth:`refit` to use them."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        Y_px = np.asarray(Y_px, dtype=np.float32).reshape(-1, 2)
        # Predictions add the drift offset, so store the target the fit itself
        # must hit for "fit + offset" to land on Y.
        Y_px = Y_px - (self.offset * [self.screen_width, self.screen_height]).astype(np.float32)
        if self._X.size == 0:
            self._X, self._Y = X, Y_px
        else:
            self._X = np.vstack([self._X, X])
            self._Y = np.vstack([self._Y, Y_px])

    def refit(self) -> None:
        """Refit on every stored sample."""
        if len(self._X) >= 2:
            self._solve()

    def _solve(self) -> None:
        X = self._X.astype(np.float64)
        Y = self._Y.astype(np.float64) / np.array([self.screen_width, self.screen_height])
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        self._std = np.where(std < 1e-8, 1.0, std)
        Xs = (X - self._mean) / self._std
        y_mean = Y.mean(axis=0)
        # (Xs^T Xs + alpha I) W = Xs^T (Y - mean): intercept unpenalised
        A = Xs.T @ Xs + self.alpha * np.eye(Xs.shape[1])
        self._coef = np.linalg.solve(A, Xs.T @ (Y - y_mean))
        self._intercept = y_mean

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return normalised screen positions ``(N, 2)`` in ``[0, 1]`` (unclamped)."""
        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("ridge model is not trained")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return ((X - self._mean) / self._std) @ self._coef + self._intercept + self.offset

    def correct_drift(self, X: np.ndarray, Y_px: np.ndarray) -> np.ndarray:
        """Set :attr:`offset` so predictions on ``X`` centre on ``Y_px``.

        Uses the median residual per target (robust to stray samples), then the
        median over targets, so one badly fixated target cannot drag it.
        Returns the new offset in pixels.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y_px = np.asarray(Y_px, dtype=np.float64).reshape(-1, 2)
        scale = np.array([self.screen_width, self.screen_height], dtype=np.float64)
        self.offset = np.zeros(2)
        resid = Y_px / scale - self.predict(X)
        per_target = [
            np.median(resid[(Y_px == t).all(axis=1)], axis=0) for t in np.unique(Y_px, axis=0)
        ]
        self.offset = np.median(np.array(per_target), axis=0)
        return self.offset * scale

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(
            tmp,
            alpha=self.alpha,
            screen=np.array([self.screen_width, self.screen_height]),
            X=self._X,
            Y=self._Y,
            offset=self.offset,
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, alpha: Optional[float] = None) -> "RidgeGazeModel":
        """Load the stored samples and refit (with *alpha*, if given)."""
        with np.load(path, allow_pickle=False) as data:
            model = cls(alpha=float(data["alpha"]) if alpha is None else float(alpha))
            model.screen_width, model.screen_height = (int(v) for v in data["screen"])
            model._X = data["X"].astype(np.float32)
            model._Y = data["Y"].astype(np.float32)
            if "offset" in data.files:
                model.offset = data["offset"].astype(np.float64)
        model.refit()
        return model
