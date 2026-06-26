"""MLPTrainer: trains GazeMLP on calibration data with early stopping."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.pipeline.schemas import FEATURE_DIM, FEATURE_KEYS
from gaze_estimation.utils.device import get_torch_device


class MLPTrainer:
    """Train or fine-tune a :class:`GazeMLP` on (features, screen_coords) data.

    Normalises features with z-score and labels to [0, 1].
    Uses early stopping with a held-out validation split.

    Args:
        input_dim:               Feature dimension, default 34.
        hidden_dims:             MLP layer widths, default [64, 128, 64].
        learning_rate:           Adam LR.
        weight_decay:            Adam L2 penalty.
        epochs:                  Max training epochs.
        batch_size:              Mini-batch size.
        early_stopping_patience: Stop if val loss doesn't improve for N epochs.
        device:                  "cpu", "cuda", "rocm", or "auto".
                                 "rocm" maps to "cuda" (PyTorch HIP backend).
                                 "auto" selects the best available device.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 32,
        early_stopping_patience: int = 20,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 128, 64]
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        # Normalise device string: "ROCM" → resolved via get_torch_device
        self.device = device

        # Normalisation statistics (set during training)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.screen_width: int = 1920
        self.screen_height: int = 1080

    # ── Main API ───────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        screen_width: int = 1920,
        screen_height: int = 1080,
        existing_model: Optional[GazeMLP] = None,
    ) -> GazeMLP:
        """Train a GazeMLP.

        Args:
            X:              (N, 34) feature matrix (raw, un-normalised).
            Y:              (N, 2) screen coordinates in pixels.
            screen_width:   Screen resolution width (for label normalisation).
            screen_height:  Screen resolution height.
            existing_model: Fine-tune this model instead of creating a new one.

        Returns:
            Trained GazeMLP (in eval mode, on CPU).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Normalise features
        X_norm = self._normalise_features(X, fit=True)

        # Normalise labels to [0, 1]
        Y_norm = Y.copy().astype(np.float32)
        Y_norm[:, 0] /= max(screen_width, 1)
        Y_norm[:, 1] /= max(screen_height, 1)
        Y_norm = np.clip(Y_norm, 0.0, 1.0)

        # Train / val split (90% / 10%)
        n = len(X_norm)
        idx = np.random.permutation(n)
        split = max(1, int(0.9 * n))
        train_idx, val_idx = idx[:split], idx[split:]

        X_tr = torch.from_numpy(X_norm[train_idx])
        Y_tr = torch.from_numpy(Y_norm[train_idx])
        X_val = torch.from_numpy(X_norm[val_idx])
        Y_val = torch.from_numpy(Y_norm[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # get_torch_device handles "ROCM", "CUDA", "CPU", "auto" uniformly
        dev = get_torch_device(self.device)
        if existing_model is not None:
            model = existing_model.to(dev)
        else:
            model = GazeMLP(self.input_dim, self.hidden_dims).to(dev)

        optimiser = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                optimiser.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimiser.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(dev))
                val_loss = loss_fn(val_pred, Y_val.to(dev)).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval().cpu()
        return model

    def fine_tune(
        self,
        model: GazeMLP,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 3,
        lr: Optional[float] = None,
    ) -> GazeMLP:
        """Fine-tune an existing model on a small batch of new data."""
        return self.train(
            X, Y,
            self.screen_width, self.screen_height,
            existing_model=model,
        )

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, model: GazeMLP, path: str) -> None:
        """Save model weights + normaliser statistics + config to a .pt file."""
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "hidden_dims": list(model.hidden_dims),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
            },
            path,
        )

    def load(self, path: str) -> Tuple["GazeMLP", "MLPTrainer"]:
        """Load a GazeMLP and restore trainer normalisation stats.

        Returns (model, trainer_with_stats).
        """
        import torch
        ckpt = torch.load(path, map_location="cpu")
        model = GazeMLP(
            input_dim=ckpt.get("input_dim", FEATURE_DIM),
            hidden_dims=ckpt.get("hidden_dims", [64, 128, 64]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        trainer = MLPTrainer(
            input_dim=model.input_dim,
            hidden_dims=list(model.hidden_dims),
        )
        trainer.feature_mean = ckpt.get("feature_mean")
        trainer.feature_std = ckpt.get("feature_std")
        trainer.screen_width = ckpt.get("screen_width", 1920)
        trainer.screen_height = ckpt.get("screen_height", 1080)
        return model, trainer

    # ── Feature normalisation ──────────────────────────────────────────────

    def _normalise_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = X.astype(np.float32)
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-8
        if self.feature_mean is not None and self.feature_std is not None:
            return (X - self.feature_mean) / self.feature_std
        return X

    def normalise(self, X: np.ndarray) -> np.ndarray:
        """Normalise a feature matrix using the stored statistics."""
        return self._normalise_features(X, fit=False)

    def features_dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert a FEATURE_KEYS dict to a normalised (1, 34) float32 array."""
        raw = np.array(
            [features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32
        ).reshape(1, -1)
        return self._normalise_features(raw, fit=False)
