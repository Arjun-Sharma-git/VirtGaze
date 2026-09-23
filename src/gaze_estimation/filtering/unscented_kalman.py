"""Unscented Kalman Filter for gaze position smoothing.

State vector: [x, y, vx, vy, ax, ay]  (position, velocity, acceleration)
Measurement:  [x, y]
"""
from __future__ import annotations

import numpy as np

# Bounds for the fixation-driven measurement-noise scaling
_MIN_MEASUREMENT_SCALE = 0.05
_MAX_MEASUREMENT_SCALE = 10.0


class UnscentedKalmanFilter:
    """6-state UKF (position, velocity, acceleration) for 2-D gaze.

    Handles dropped frames via the ``coast`` method and provides smooth
    predictions even during fast saccades.

    Args:
        process_noise:      Scalar Q scaling for process uncertainty.
        measurement_noise:  Scalar R scaling for measurement uncertainty.
        alpha:              UKF spread (typically 1e-3).
        beta:               UKF distribution parameter (2.0 for Gaussian).
        kappa:              UKF secondary scaling (typically 0).
    """

    STATE_DIM = 6
    MEASURE_DIM = 2

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        n = self.STATE_DIM
        lam = alpha ** 2 * (n + kappa) - n
        self._lambda = lam

        # Weights
        self._Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self._Wm[0] = lam / (n + lam)
        self._Wc = self._Wm.copy()
        self._Wc[0] += (1 - alpha ** 2 + beta)

        # State and covariance
        self._x = np.zeros(n)
        self._P = np.eye(n) * 100.0      # Large initial uncertainty
        self._initialised = False

        # Noise matrices
        self._Q = np.eye(n) * process_noise
        self._base_measurement_noise = float(measurement_noise)
        self._measurement_scale = 1.0
        self._R = np.eye(self.MEASURE_DIM) * self._base_measurement_noise

    # ── Public API ─────────────────────────────────────────────────────────

    def set_measurement_scale(self, scale: float) -> None:
        """Scale the measurement-noise covariance ``R``.

        Used to adapt smoothing to the current gaze state: a larger scale makes
        the filter trust the measurement less (heavier smoothing), a smaller
        one makes it more responsive during saccades.  *scale* is clamped to a
        sane range.
        """
        scale = float(
            min(max(scale, _MIN_MEASUREMENT_SCALE), _MAX_MEASUREMENT_SCALE)
        )
        if scale == self._measurement_scale:
            return
        self._measurement_scale = scale
        self._R = np.eye(self.MEASURE_DIM) * (
            self._base_measurement_noise * scale
        )

    @property
    def measurement_scale(self) -> float:
        """Current measurement-noise scale (1.0 = configured value)."""
        return self._measurement_scale

    def predict(self, dt: float) -> None:
        """Predict the next state using a constant-acceleration model."""
        if not self._initialised:
            return

        sigma = self._sigma_points()
        n = self.STATE_DIM

        # Propagate sigma points through the motion model
        sigma_pred = np.zeros_like(sigma)
        for i in range(2 * n + 1):
            sigma_pred[:, i] = self._motion_model(sigma[:, i], dt)

        x_pred = sigma_pred @ self._Wm
        P_pred = self._Q.copy()
        for i in range(2 * n + 1):
            diff = sigma_pred[:, i] - x_pred
            P_pred += self._Wc[i] * np.outer(diff, diff)

        self._x = x_pred
        self._P = P_pred

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with new (x, y) measurement.

        Args:
            measurement: (2,) array [screen_x, screen_y].

        Returns:
            Filtered state position (2,).
        """
        if not self._initialised:
            self._x[:2] = measurement
            self._initialised = True
            return measurement.copy()

        n = self.STATE_DIM
        sigma = self._sigma_points()

        # Map sigma points into measurement space
        z_sigma = sigma[:2, :]           # Take only x, y rows
        z_pred = z_sigma @ self._Wm

        # Innovation covariance
        S = self._R.copy()
        Pxz = np.zeros((n, self.MEASURE_DIM))
        for i in range(2 * n + 1):
            dz = z_sigma[:, i] - z_pred
            dx = sigma[:, i] - self._x
            S += self._Wc[i] * np.outer(dz, dz)
            Pxz += self._Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)
        innov = measurement - z_pred
        self._x = self._x + K @ innov
        self._P = self._P - K @ S @ K.T

        return self._x[:2].copy()

    def get_state(self) -> np.ndarray:
        """Return current state vector [x, y, vx, vy, ax, ay]."""
        return self._x.copy()

    def get_position(self) -> np.ndarray:
        """Return filtered (x, y) position."""
        return self._x[:2].copy()

    def coast(self, dt: float) -> np.ndarray:
        """Predict-only step for frames with no measurement (dropped frames).

        Returns the predicted position.
        """
        self.predict(dt)
        return self._x[:2].copy()

    def reset(self) -> None:
        """Reset to initial uninitialised state."""
        n = self.STATE_DIM
        self._x = np.zeros(n)
        self._P = np.eye(n) * 100.0
        self._initialised = False
        self.set_measurement_scale(1.0)

    # ── Private ────────────────────────────────────────────────────────────

    def _sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points from current state and covariance."""
        n = self.STATE_DIM
        try:
            L = np.linalg.cholesky((n + self._lambda) * self._P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + self._lambda) * (self._P + np.eye(n) * 1e-6))

        sigma = np.zeros((n, 2 * n + 1))
        sigma[:, 0] = self._x
        for i in range(n):
            sigma[:, i + 1] = self._x + L[:, i]
            sigma[:, n + i + 1] = self._x - L[:, i]
        return sigma

    @staticmethod
    def _motion_model(state: np.ndarray, dt: float) -> np.ndarray:
        """Constant-acceleration motion model."""
        x, y, vx, vy, ax, ay = state
        return np.array([
            x + vx * dt + 0.5 * ax * dt ** 2,
            y + vy * dt + 0.5 * ay * dt ** 2,
            vx + ax * dt,
            vy + ay * dt,
            ax,
            ay,
        ])
