"""MLPTrainer: trains GazeMLP on calibration data with early stopping."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from gaze_estimation.model.mlp import GazeMLP
from gaze_estimation.pipeline.schemas import FEATURE_DIM, FEATURE_KEYS
from gaze_estimation.utils.device import get_torch_device
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("model.trainer")


def _to_tensor(array: Optional[np.ndarray]):
    """Convert a normalisation-statistics array to a float32 tensor.

    Tensors (unlike raw numpy arrays) are accepted by the ``weights_only=True``
    unpickler used by :meth:`MLPTrainer.load`.
    """
    if array is None:
        return None
    import torch

    return torch.as_tensor(np.asarray(array, dtype=np.float32))


def _from_checkpoint(value: object) -> Optional[np.ndarray]:
    """Convert a checkpoint value (tensor or array) back to a float32 array."""
    if value is None:
        return None
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


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

    @classmethod
    def from_config(cls, config) -> MLPTrainer:
        """Build a trainer from a :class:`~gaze_estimation.config.config.Config`.

        Reads the ``mlp`` section (architecture + optimisation settings) and
        ``inference.device``.  The dataset-dependent ``epochs``/``batch_size``
        actually used are still chosen by :meth:`train`/:meth:`fine_tune`; these
        values are the configured ceiling.
        """
        mlp = config.section("mlp")
        hidden = mlp.get("hidden_dims")
        return cls(
            input_dim=int(mlp.get("input_dim", FEATURE_DIM)),
            hidden_dims=list(hidden) if hidden else None,
            learning_rate=float(mlp.get("learning_rate", 1e-3)),
            weight_decay=float(mlp.get("weight_decay", 1e-4)),
            epochs=int(mlp.get("epochs", 200)),
            batch_size=int(mlp.get("batch_size", 32)),
            early_stopping_patience=int(mlp.get("early_stopping_patience", 20)),
            device=str(config.get("inference.device", "cpu")),
        )

    # ── Main API ───────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        screen_width: int = 1920,
        screen_height: int = 1080,
        existing_model: Optional[GazeMLP] = None,
        refit_normalisation: bool = True,
    ) -> GazeMLP:
        """Train a GazeMLP.

        Args:
            X:              (N, 34) feature matrix (raw, un-normalised).
            Y:              (N, 2) screen coordinates in pixels.
            screen_width:   Screen resolution width (for label normalisation).
            screen_height:  Screen resolution height.
            existing_model: Fine-tune this model instead of creating a new one.
            refit_normalisation: Recompute the feature z-score statistics from
                            *X*.  Must be ``False`` when fine-tuning on a small
                            batch, otherwise the statistics the live model
                            depends on are overwritten with biased ones.

        Returns:
            Trained GazeMLP (in eval mode, on CPU).
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Only fit the normaliser on a full (re)calibration, or when no
        # statistics exist yet.
        fit_norm = refit_normalisation or self.feature_mean is None or self.feature_std is None
        X_norm = self._normalise_features(X, fit=fit_norm)

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

        # BatchNorm1d refuses to train on a batch of one, and the 90 % split can
        # leave a one-sample tail batch: 55 samples → 49 for training, which a
        # batch size of 16 cuts as 16 + 16 + 16 + 1.  Dropping that tail removes
        # the problem at the cost of one sample per epoch.  Without this the run
        # raises "Expected more than 1 value per channel", which OnlineTrainer
        # catches and logs — so click adaptation would fail silently for those
        # sample counts (e.g. the 16-sample fine-tune batch: 19, 37, 55, 73 …).
        n_train = len(X_tr)
        train_loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=n_train > 1 and n_train % self.batch_size == 1,
        )

        # get_torch_device handles "ROCM", "CUDA", "CPU", "auto" uniformly
        dev = get_torch_device(self.device)
        if existing_model is not None:
            model = existing_model.to(dev)
        else:
            model = GazeMLP(self.input_dim, self.hidden_dims).to(dev)

        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
        """Fine-tune an existing model on a small batch of new data.

        The stored feature-normalisation statistics are **kept** (not refit),
        and *lr* (when given) overrides the trainer learning rate for the
        duration of the fine-tune only.
        """
        orig_epochs = self.epochs
        orig_lr = self.lr
        self.epochs = epochs
        if lr is not None:
            self.lr = lr
        try:
            return self.train(
                X,
                Y,
                self.screen_width,
                self.screen_height,
                existing_model=model,
                refit_normalisation=False,
            )
        finally:
            self.epochs = orig_epochs
            self.lr = orig_lr

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, model: GazeMLP, path: str) -> None:
        """Save model weights + normaliser statistics + config to a .pt file.

        Normalisation statistics are stored as tensors (not raw numpy arrays)
        so the checkpoint can be read back with
        ``torch.load(..., weights_only=True)``.
        """
        import torch

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "hidden_dims": list(model.hidden_dims),
                "feature_mean": _to_tensor(self.feature_mean),
                "feature_std": _to_tensor(self.feature_std),
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
            },
            path,
        )

    def load(self, path: str, config=None) -> Tuple[GazeMLP, MLPTrainer]:
        """Load a GazeMLP and restore trainer normalisation stats.

        Args:
            path:   Checkpoint written by :meth:`save`.
            config: Optional :class:`~gaze_estimation.config.config.Config`.
                    When given, the returned trainer is built with
                    :meth:`from_config` so the configured optimisation settings
                    (LR, epochs, …) survive the round-trip; the architecture is
                    still taken from the checkpoint.

        Returns (model, trainer_with_stats).
        """
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Legacy checkpoints stored raw numpy arrays, which the
            # weights_only unpickler refuses.  Only fall back for trusted files.
            _logger.warning("Falling back to weights_only=False for legacy checkpoint: %s", path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = GazeMLP(
            input_dim=ckpt.get("input_dim", FEATURE_DIM),
            hidden_dims=ckpt.get("hidden_dims", [64, 128, 64]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        if config is not None:
            trainer = type(self).from_config(config)
            trainer.input_dim = model.input_dim
            trainer.hidden_dims = list(model.hidden_dims)
        else:
            trainer = MLPTrainer(
                input_dim=model.input_dim,
                hidden_dims=list(model.hidden_dims),
            )
        trainer.feature_mean = _from_checkpoint(ckpt.get("feature_mean"))
        trainer.feature_std = _from_checkpoint(ckpt.get("feature_std"))
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
        raw = np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32).reshape(
            1, -1
        )
        return self._normalise_features(raw, fit=False)
