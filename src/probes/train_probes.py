"""
Step 5: Train probes on collected activations.

Three probe types predicting empirical_accuracy from layer activations:
  (1) Ridge Regression — alpha grid search, best by validation R²
  (2) 2-Layer MLP      — weight decay + output penalty grid search
  (3) Fisher LDA       — shrinkage grid search, isotonic calibration

Data split: 80/10/10 at question level, stratified by (level, type).
All N seeds of a question stay in the same split.

Usage:
    python -m src.probes.train_probes --model llama_base --split train
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config, activations_path, probes_path
from src.probes.metrics import compute_r2, compute_ece, compute_brier, compute_auroc


# ============================================================
# Metric suite
# ============================================================

def all_metrics(
    y_pred: np.ndarray,
    emp_acc: np.ndarray,
    binary_correct: np.ndarray,
) -> dict[str, float]:
    p = np.clip(y_pred, 0, 1)
    return {
        "R2": compute_r2(emp_acc, y_pred),
        "ECE": compute_ece(p, binary_correct),
        "Brier": compute_brier(p, binary_correct),
        "AUROC": compute_auroc(p, binary_correct),
        "Pearson_r": float(pearsonr(emp_acc, y_pred)[0]) if len(emp_acc) > 1 else 0.0,
        "MAE": float(np.mean(np.abs(emp_acc - y_pred))),
    }


# ============================================================
# Stratified question-level split
# ============================================================

def stratified_question_split(
    question_idx: np.ndarray,
    question_level: np.ndarray,
    question_type: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split at question level, stratified by (level, type).

    Returns boolean train_mask, val_mask, test_mask over the full row array.
    """
    rng = np.random.default_rng(seed)
    unique_q = np.unique(question_idx)
    q_to_first_row = {q: np.where(question_idx == q)[0][0] for q in unique_q}

    strata: dict[tuple, list] = defaultdict(list)
    for q in unique_q:
        row = q_to_first_row[q]
        key = (str(question_level[row]), str(question_type[row]))
        strata[key].append(q)

    train_qs, val_qs, test_qs = set(), set(), set()
    for _, questions in strata.items():
        qs = np.array(questions)
        rng.shuffle(qs)
        n_test = max(1, int(len(qs) * test_frac))
        n_val = max(1, int(len(qs) * val_frac))
        test_qs.update(qs[:n_test].tolist())
        val_qs.update(qs[n_test:n_test + n_val].tolist())
        train_qs.update(qs[n_test + n_val:].tolist())

    train_mask = np.isin(question_idx, list(train_qs))
    val_mask = np.isin(question_idx, list(val_qs))
    test_mask = np.isin(question_idx, list(test_qs))
    return train_mask, val_mask, test_mask


# ============================================================
# (1) Ridge Regression
# ============================================================

def train_ridge(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
) -> dict:
    mu = X_train.mean(axis=0)
    sigma = np.maximum(X_train.std(axis=0), 1e-12)
    Z_tr, Z_va, Z_te = (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma

    best_alpha, best_val_r2, best_model = None, -1e9, None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Z_tr, y_train)
        val_r2 = compute_r2(y_val, model.predict(Z_va))
        print(f"    alpha={alpha:10.4f}  val_R2={val_r2:.6f}")
        if val_r2 > best_val_r2:
            best_val_r2, best_alpha, best_model = val_r2, alpha, model

    return {
        "model": best_model, "mu": mu, "sigma": sigma, "best_alpha": best_alpha,
        "train_pred": best_model.predict(Z_tr),
        "val_pred": best_model.predict(Z_va),
        "test_pred": best_model.predict(Z_te),
    }


# ============================================================
# (2) 2-Layer MLP
# ============================================================

class _TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple = (512, 256), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.extend([nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)])
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    weight_decay_grid: tuple = (0, 0.1, 1.0, 5.0, 10.0, 25.0),
    output_penalty_grid: tuple = (0, 0.01, 0.1, 0.25),
    hidden_dims: tuple = (512, 256),
    dropout: float = 0.2,
    lr: float = 1e-3, batch_size: int = 256,
    epochs: int = 100, patience: int = 30,
    seed: int = 42, device: str = "cuda",
) -> dict:
    mu = X_train.mean(axis=0)
    sigma = np.maximum(X_train.std(axis=0), 1e-12)
    Z_tr, Z_va, Z_te = (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma

    # Internal 80/20 split for early stopping
    rng = np.random.default_rng(seed)
    n = len(Z_tr)
    perm = rng.permutation(n)
    n_es = int(n * 0.2)
    es_idx, tr_idx = perm[:n_es], perm[n_es:]

    best_wd, best_op, best_val_r2, best_state = None, None, -1e9, None

    for wd in weight_decay_grid:
        for op in output_penalty_grid:
            torch.manual_seed(seed)
            model = _TwoLayerMLP(Z_tr.shape[1], hidden_dims, dropout).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            criterion = nn.MSELoss()

            ds = TensorDataset(torch.FloatTensor(Z_tr[tr_idx]), torch.FloatTensor(y_train[tr_idx]))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            best_es_loss, pat_cnt, best_ms = float("inf"), 0, None
            for _ in range(epochs):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    if op > 0:
                        loss = loss + op * torch.mean(pred ** 2)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    ep = model(torch.FloatTensor(Z_tr[es_idx]).to(device))
                    el = criterion(ep, torch.FloatTensor(y_train[es_idx]).to(device)).item()
                scheduler.step(el)

                if el < best_es_loss:
                    best_es_loss, pat_cnt = el, 0
                    best_ms = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    pat_cnt += 1
                if pat_cnt >= patience:
                    break

            model.load_state_dict(best_ms)
            model.eval()
            with torch.no_grad():
                vp = model(torch.FloatTensor(Z_va).to(device)).cpu().numpy()
            val_r2 = compute_r2(y_val, vp)
            print(f"    wd={wd:6.2f}  op={op:5.3f}  val_R2={val_r2:.6f}")

            if val_r2 > best_val_r2:
                best_val_r2, best_wd, best_op, best_state = val_r2, wd, op, best_ms

    final_model = _TwoLayerMLP(Z_tr.shape[1], hidden_dims, dropout).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()
    with torch.no_grad():
        train_pred = final_model(torch.FloatTensor(Z_tr).to(device)).cpu().numpy()
        val_pred = final_model(torch.FloatTensor(Z_va).to(device)).cpu().numpy()
        test_pred = final_model(torch.FloatTensor(Z_te).to(device)).cpu().numpy()

    return {
        "model_state": best_state, "mu": mu, "sigma": sigma,
        "best_weight_decay": best_wd, "best_output_penalty": best_op,
        "hidden_dims": hidden_dims,
        "train_pred": train_pred, "val_pred": val_pred, "test_pred": test_pred,
    }


# ============================================================
# (3) Fisher LDA
# ============================================================

def train_fisher_lda(
    X_train: np.ndarray, y_emp_train: np.ndarray, bc_train: np.ndarray,
    X_val: np.ndarray, y_emp_val: np.ndarray,
    X_test: np.ndarray, y_emp_test: np.ndarray,
    shrinkage_grid: tuple = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0),
) -> dict | None:
    """Fisher LDA: find the direction maximally separating correct/incorrect.

    w = (Sw + λI)^{-1} (μ_correct - μ_incorrect), then isotonic calibrate
    projection scores to empirical accuracy.
    """
    mu = X_train.mean(axis=0)
    sigma = np.maximum(X_train.std(axis=0), 1e-12)
    Z_tr, Z_va, Z_te = (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma

    mask_1 = bc_train == 1
    mask_0 = bc_train == 0
    if not mask_1.any() or not mask_0.any():
        print("    [WARN] Only one class present — skipping Fisher LDA")
        return None

    mu_1 = Z_tr[mask_1].mean(axis=0)
    mu_0 = Z_tr[mask_0].mean(axis=0)
    diff = mu_1 - mu_0

    # Within-class scatter
    X1_c = Z_tr[mask_1] - mu_1
    X0_c = Z_tr[mask_0] - mu_0
    N = len(Z_tr)
    Sw = (X1_c.T @ X1_c + X0_c.T @ X0_c) / N

    d = Z_tr.shape[1]
    best_lam, best_val_r2, best_w, best_iso = None, -1e9, None, None

    for lam in shrinkage_grid:
        Sw_reg = Sw + lam * np.eye(d, dtype=Sw.dtype)
        w = np.linalg.solve(Sw_reg, diff)
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm

        scores_tr = Z_tr @ w
        scores_va = Z_va @ w
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(scores_tr, y_emp_train)
        val_r2 = compute_r2(y_emp_val, iso.predict(scores_va))
        print(f"    lambda={lam:10.6f}  val_R2={val_r2:.6f}")

        if val_r2 > best_val_r2:
            best_val_r2, best_lam, best_w, best_iso = val_r2, lam, w.copy(), iso

    return {
        "w": best_w, "mu": mu, "sigma": sigma, "best_shrinkage": best_lam,
        "isotonic": best_iso,
        "train_pred": best_iso.predict(Z_tr @ best_w),
        "val_pred": best_iso.predict(Z_va @ best_w),
        "test_pred": best_iso.predict(Z_te @ best_w),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes on activations.")
    parser.add_argument("--model", default="llama_base", help="Model key")
    parser.add_argument("--split", default="train", help="Data split")
    parser.add_argument("--config", default=None)
    parser.add_argument("--npz", default=None, help="Override activations NPZ path")
    parser.add_argument("--out_dir", default=None, help="Override output directory")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    npz_path = Path(args.npz) if args.npz else activations_path(cfg, args.model, args.split)
    out_dir = Path(args.out_dir) if args.out_dir else probes_path(cfg, args.model, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {npz_path} ...")
    store = np.load(npz_path, allow_pickle=True)
    acts = store["activations"].astype(np.float32)
    q_idx = store["question_idx"]
    binary_correct = store["binary_correct"].astype(np.float32)
    emp_acc = store["empirical_accuracy"].astype(np.float32)
    q_level = store["question_level"]
    q_type = store["question_type"]

    print(f"  Shape: {acts.shape}")
    print(f"  Binary correct rate: {binary_correct.mean():.4f}")

    # Stratified split
    train_mask, val_mask, test_mask = stratified_question_split(
        q_idx, q_level, q_type,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )

    X_tr, X_va, X_te = acts[train_mask], acts[val_mask], acts[test_mask]
    y_tr, y_va, y_te = emp_acc[train_mask], emp_acc[val_mask], emp_acc[test_mask]
    bc_tr, bc_va, bc_te = binary_correct[train_mask], binary_correct[val_mask], binary_correct[test_mask]

    print(f"  Train: {train_mask.sum()} rows, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    results = {}

    # (1) Ridge
    print(f"\n{'='*60}\n(1) Ridge Regression\n{'='*60}")
    ridge_out = train_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te)
    results["ridge"] = {
        "best_alpha": ridge_out["best_alpha"],
        "test_metrics": all_metrics(ridge_out["test_pred"], y_te, bc_te),
    }
    with open(out_dir / "ridge_probe.pkl", "wb") as f:
        pickle.dump({"model": ridge_out["model"], "mu": ridge_out["mu"],
                      "sigma": ridge_out["sigma"], "best_alpha": ridge_out["best_alpha"]}, f)

    # (2) MLP
    print(f"\n{'='*60}\n(2) 2-Layer MLP\n{'='*60}")
    mlp_out = train_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=args.seed, device=args.device)
    results["mlp"] = {
        "best_weight_decay": mlp_out["best_weight_decay"],
        "best_output_penalty": mlp_out["best_output_penalty"],
        "test_metrics": all_metrics(mlp_out["test_pred"], y_te, bc_te),
    }
    with open(out_dir / "mlp_probe.pkl", "wb") as f:
        pickle.dump({"model_state": mlp_out["model_state"], "mu": mlp_out["mu"],
                      "sigma": mlp_out["sigma"], "hidden_dims": mlp_out["hidden_dims"]}, f)

    # (3) Fisher LDA
    print(f"\n{'='*60}\n(3) Fisher LDA\n{'='*60}")
    fisher_out = train_fisher_lda(X_tr, y_tr, bc_tr, X_va, y_va, X_te, y_te)
    if fisher_out is not None:
        results["fisher_lda"] = {
            "best_shrinkage": fisher_out["best_shrinkage"],
            "test_metrics": all_metrics(fisher_out["test_pred"], y_te, bc_te),
        }
        with open(out_dir / "fisher_lda_probe.pkl", "wb") as f:
            pickle.dump({"w": fisher_out["w"], "mu": fisher_out["mu"],
                          "sigma": fisher_out["sigma"], "isotonic": fisher_out["isotonic"],
                          "best_shrinkage": fisher_out["best_shrinkage"]}, f)

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    header = f"{'Probe':<20s} {'R2':>10s} {'ECE':>10s} {'Brier':>10s} {'AUROC':>10s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["test_metrics"]
        print(f"{name:<20s} {m['R2']:10.6f} {m['ECE']:10.6f} {m['Brier']:10.6f} {m['AUROC']:10.6f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
