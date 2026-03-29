"""
Probe implementations for predicting confidence from activations.

Three probe types:
  1. Ridge Regression — linear, with alpha grid search + isotonic calibration
  2. MLP — 2-hidden-layer network with weight decay search
  3. Fisher LDA — IID-corrected (whitened) mass-mean direction

All probes:
  - Input:  activation vector from one layer (hidden_dim,)
  - Output: scalar confidence estimate in [0, 1]
  - Train on empirical_accuracy (continuous [0, 1])
  - Calibrated via isotonic regression post-hoc
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ===================================================================
# Probe 1: Ridge Regression
# ===================================================================

class RidgeProbe:
    """Linear ridge regression with alpha grid search and isotonic calibration."""

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        n_folds: int = 5,
        seed: int = 42,
    ):
        if alpha_grid is None:
            alpha_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.seed = seed
        self.model: Ridge | None = None
        self.calibrator: IsotonicRegression | None = None
        self.mu: NDArray | None = None
        self.sigma: NDArray | None = None
        self.best_alpha: float | None = None

    def _zscore_fit(self, X: NDArray) -> None:
        self.mu = X.mean(axis=0)
        self.sigma = np.maximum(X.std(axis=0), 1e-12)

    def _zscore(self, X: NDArray) -> NDArray:
        return (X - self.mu) / self.sigma

    def fit(self, X_train: NDArray, y_train: NDArray) -> None:
        """Train with alpha grid search using K-fold CV."""
        self._zscore_fit(X_train)
        Z = self._zscore(X_train)

        N = len(Z)
        rng = np.random.default_rng(self.seed)
        fold_ids = rng.integers(0, self.n_folds, size=N)

        best_r2 = -np.inf
        for alpha in self.alpha_grid:
            oof_preds = np.zeros(N)
            for fold in range(self.n_folds):
                train_mask = fold_ids != fold
                val_mask = fold_ids == fold
                reg = Ridge(alpha=alpha, fit_intercept=True)
                reg.fit(Z[train_mask], y_train[train_mask])
                oof_preds[val_mask] = reg.predict(Z[val_mask])

            ss_res = np.sum((y_train - oof_preds) ** 2)
            ss_tot = np.sum((y_train - y_train.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                self.best_alpha = alpha

        # Refit on all data with best alpha
        self.model = Ridge(alpha=self.best_alpha, fit_intercept=True)
        self.model.fit(Z, y_train)

        # Isotonic calibration
        raw_preds = self.model.predict(Z)
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw_preds, y_train)

    def predict(self, X: NDArray) -> NDArray:
        """Return calibrated confidence predictions."""
        Z = self._zscore(X)
        raw = self.model.predict(Z)
        return self.calibrator.predict(raw)

    def predict_raw(self, X: NDArray) -> NDArray:
        """Return uncalibrated raw predictions."""
        Z = self._zscore(X)
        return self.model.predict(Z)


# ===================================================================
# Probe 2: MLP with Weight Decay Search
# ===================================================================

class _MLPNetwork(nn.Module):
    """MLP architecture: input → hidden layers → 1 (sigmoid)."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPProbe:
    """MLP probe with weight decay grid search and isotonic calibration.

    Default architecture: input → [512, 256] → 1 (sigmoid).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        weight_decay_grid: list[float] | None = None,
        lr: float = 1e-3,
        batch_size: int = 256,
        num_epochs: int = 100,
        patience: int = 30,
        seed: int = 42,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256]
        if weight_decay_grid is None:
            weight_decay_grid = [0, 0.1, 1.0, 5.0, 10.0, 25.0]
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.weight_decay_grid = weight_decay_grid
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.model: _MLPNetwork | None = None
        self.calibrator: IsotonicRegression | None = None
        self.mu: NDArray | None = None
        self.sigma: NDArray | None = None
        self.best_wd: float | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _zscore_fit(self, X: NDArray) -> None:
        self.mu = X.mean(axis=0)
        self.sigma = np.maximum(X.std(axis=0), 1e-12)

    def _zscore(self, X: NDArray) -> NDArray:
        return (X - self.mu) / self.sigma

    def _train_one(
        self, X_train: NDArray, y_train: NDArray,
        X_val: NDArray, y_val: NDArray, weight_decay: float,
    ) -> tuple[_MLPNetwork, float]:
        """Train a single MLP with given weight decay. Returns (model, val_loss)."""
        torch.manual_seed(self.seed)
        input_dim = X_train.shape[1]
        model = _MLPNetwork(input_dim, self.hidden_dims, self.dropout).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        for _ in range(self.num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        model.load_state_dict(best_state)
        return model, best_val_loss

    def fit(self, X_train: NDArray, y_train: NDArray) -> None:
        """Train with weight decay grid search."""
        self._zscore_fit(X_train)
        Z = self._zscore(X_train)

        N = len(Z)
        rng = np.random.default_rng(self.seed)
        val_mask = rng.random(N) < 0.2
        train_mask = ~val_mask

        best_val_loss = float("inf")
        for wd in self.weight_decay_grid:
            model, val_loss = self._train_one(
                Z[train_mask], y_train[train_mask],
                Z[val_mask], y_train[val_mask],
                weight_decay=wd,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_wd = wd
                self.model = model

        # Refit on all data with best WD (90/10 for early stopping)
        val_mask2 = rng.random(N) < 0.1
        self.model, _ = self._train_one(
            Z[~val_mask2], y_train[~val_mask2],
            Z[val_mask2], y_train[val_mask2],
            weight_decay=self.best_wd,
        )

        # Isotonic calibration
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(
                torch.tensor(Z, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw_preds, y_train)

    def predict(self, X: NDArray) -> NDArray:
        Z = self._zscore(X)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(
                torch.tensor(Z, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        return self.calibrator.predict(raw)

    def predict_raw(self, X: NDArray) -> NDArray:
        Z = self._zscore(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(Z, dtype=torch.float32).to(self.device)
            ).cpu().numpy()


# ===================================================================
# Probe 3: Fisher LDA (IID-corrected)
# ===================================================================

class FisherLDAProbe:
    """IID-corrected probe using Fisher Linear Discriminant Analysis.

    Computes the Fisher direction:
        w = Σ_w^{-1} (μ_correct − μ_incorrect)

    where Σ_w is the pooled within-class covariance (with shrinkage
    regularization for high-dimensional activations).

    Projects activations onto w and calibrates via isotonic regression.
    """

    def __init__(
        self,
        n_bins: int = 5,
        shrinkage: float = 0.1,
        seed: int = 42,
    ):
        self.n_bins = n_bins
        self.shrinkage = shrinkage
        self.seed = seed
        self.direction: NDArray | None = None
        self.calibrator: IsotonicRegression | None = None
        self.mu: NDArray | None = None
        self.sigma: NDArray | None = None

    def fit(self, X_train: NDArray, y_train: NDArray) -> None:
        """Fit Fisher LDA direction with shrinkage regularization.

        Steps:
          1. Bin training examples by gold confidence into K bins
          2. Compute mass-mean direction from bin centroids
          3. Compute pooled within-class covariance Σ_w
          4. Apply whitening: w = Σ_w^{-1} @ θ_mm
          5. Normalize and fit isotonic calibrator
        """
        N, D = X_train.shape
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Step 1: Compute bin means and mass-mean direction
        bin_means, bin_counts, bin_labels, bin_indices = [], [], [], []
        for k in range(self.n_bins):
            lo, hi = bin_edges[k], bin_edges[k + 1]
            mask = (y_train >= lo) & (y_train <= hi) if k == self.n_bins - 1 \
                else (y_train >= lo) & (y_train < hi)
            if mask.sum() < 2:
                continue
            bin_means.append(X_train[mask].mean(axis=0))
            bin_counts.append(mask.sum())
            bin_labels.append(bin_centers[k])
            bin_indices.append(np.where(mask)[0])

        bin_means_arr = np.stack(bin_means)
        bin_counts_arr = np.array(bin_counts)
        bin_labels_arr = np.array(bin_labels)

        # Mass-mean direction via weighted least squares
        w = bin_counts_arr / bin_counts_arr.sum()
        label_mean = np.average(bin_labels_arr, weights=w)
        labels_centered = bin_labels_arr - label_mean
        denom = np.sum(w * labels_centered ** 2) + 1e-12
        theta_mm = (w[:, None] * labels_centered[:, None] * bin_means_arr).sum(axis=0) / denom

        # Step 2: Pooled within-class covariance
        X_centered = np.zeros_like(X_train, dtype=np.float64)
        for k, idxs in enumerate(bin_indices):
            X_centered[idxs] = X_train[idxs] - bin_means[k]

        all_bin_idxs = np.concatenate(bin_indices)
        X_c = X_centered[all_bin_idxs]
        n_binned = len(all_bin_idxs)

        Sigma = (X_c.T @ X_c) / n_binned
        trace_sigma = np.trace(Sigma)

        # Regularized covariance
        Sigma_reg = (
            (1 - self.shrinkage) * Sigma
            + self.shrinkage * (trace_sigma / D) * np.eye(D)
        )

        # Step 3: Whitened direction
        self.direction = np.linalg.solve(Sigma_reg, theta_mm)
        norm = np.linalg.norm(self.direction) + 1e-12
        self.direction = (self.direction / norm).astype(np.float32)

        # Store z-scoring parameters (identity here; z-scoring done externally)
        self.mu = np.zeros(D, dtype=np.float32)
        self.sigma = np.ones(D, dtype=np.float32)

        # Isotonic calibration
        projections = X_train @ self.direction
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(projections, y_train)

    def predict(self, X: NDArray) -> NDArray:
        projections = X @ self.direction
        return self.calibrator.predict(projections)

    def predict_raw(self, X: NDArray) -> NDArray:
        return X @ self.direction

    def get_direction(self) -> NDArray:
        return self.direction.copy()
