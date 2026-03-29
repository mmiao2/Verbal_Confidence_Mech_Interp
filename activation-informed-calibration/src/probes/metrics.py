"""
Evaluation metrics for confidence calibration probes.

Metrics:
  - Acc:   Accuracy of confidence-based binary prediction (threshold 0.5)
  - AUROC: Discrimination of correct vs. incorrect via confidence scores
  - Brier: Mean squared error between confidence and binary correctness
  - ECE:   Expected calibration error (binned)
  - R²:    Coefficient of determination against empirical accuracy
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


def compute_r2(y_true: NDArray, y_pred: NDArray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def compute_ece(probs: NDArray, labels: NDArray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Bins examples by predicted confidence, compares mean confidence to
    mean binary correctness within each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins, right=False) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def compute_brier(probs: NDArray, labels: NDArray) -> float:
    """Brier score: mean((prob - label)²)."""
    return float(np.mean((probs - labels) ** 2))


def compute_auroc(probs: NDArray, labels: NDArray) -> float:
    """AUROC for binary correct/incorrect classification."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def compute_all_metrics(
    confidence: NDArray,
    binary_correct: NDArray,
    empirical_accuracy: NDArray | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute the full metric suite.

    Args:
        confidence: Probe or verbalized confidence in [0, 1].
        binary_correct: Binary correctness labels {0, 1}.
        empirical_accuracy: If provided, also compute R² against it.
        n_bins: Number of bins for ECE.

    Returns:
        Dictionary with Acc, AUROC, Brier, ECE, and optionally R².
    """
    p = np.clip(confidence, 0, 1)
    bc = binary_correct.astype(float)

    predicted_correct = (p >= 0.5).astype(float)
    acc = float(np.mean(predicted_correct == bc))

    metrics = {
        "Acc": acc,
        "AUROC": compute_auroc(p, bc),
        "Brier": compute_brier(p, bc),
        "ECE": compute_ece(p, bc, n_bins),
    }

    if empirical_accuracy is not None:
        metrics["R2"] = compute_r2(empirical_accuracy, p)

    return metrics
