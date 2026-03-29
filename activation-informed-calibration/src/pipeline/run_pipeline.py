"""
Step 5: Two-stage pipeline — probe calibration + adaptive steering.

Stage 1 (Probe):
  - Load training activations and empirical accuracy
  - Train a Ridge probe with isotonic calibration
  - Predict confidence for test questions

Stage 2 (Steer):
  - Load the question-level CAA steering vector
  - Alpha sweep on validation set: build transfer function α → confidence
  - Invert probe predictions through transfer function → per-question α
  - Steered generation on test set with adaptive α per question
  - Evaluate: ECE, Brier, accuracy

Usage:
    python -m src.pipeline.run_pipeline --model qwen_instruct \\
        --activations outputs/activations/activations_qwen_instruct_train.npz \\
        --vector outputs/steering/question_caa_qwen_instruct_train.pt \\
        --completions outputs/completions/completions_qwen_instruct_test.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config, get_model_config
from src.probes.metrics import compute_all_metrics
from src.steering.hooks import SteeringHook, add_hook, get_layer_module
from src.steering.steering_utils import parse_confidence_from_completion
from src.steering.steer_generate import generate_batch


def stratified_split(
    levels: np.ndarray,
    types: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split at the question level.

    Stratifies by (level, type) combinations.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)
    n = len(levels)

    strata = np.array([f"{l}_{t}" for l, t in zip(levels, types)])
    unique_strata = np.unique(strata)

    train_idx, val_idx, test_idx = [], [], []

    for stratum in unique_strata:
        mask = strata == stratum
        indices = np.where(mask)[0]
        rng.shuffle(indices)

        n_s = len(indices)
        n_val = max(1, int(n_s * val_frac))
        n_test = max(1, int(n_s * test_frac))

        val_idx.extend(indices[:n_val])
        test_idx.extend(indices[n_val : n_val + n_test])
        train_idx.extend(indices[n_val + n_test :])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def build_transfer_function(
    alphas: list[float],
    mean_confs: list[float],
) -> PchipInterpolator | None:
    """Build monotonic alpha → confidence transfer function.

    Filters to monotonically increasing (alpha, confidence) pairs
    and returns a PCHIP interpolant. Returns inverse: confidence → alpha.
    """
    pairs = sorted(zip(alphas, mean_confs))
    mono_a, mono_c = [pairs[0][0]], [pairs[0][1]]

    for a, c in pairs[1:]:
        if c > mono_c[-1]:
            mono_a.append(a)
            mono_c.append(c)

    if len(mono_a) < 3:
        return None

    return PchipInterpolator(mono_c, mono_a)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage pipeline.")
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--activations", required=True, help="Training activations NPZ")
    parser.add_argument("--vector", required=True, help="Steering vector .pt path")
    parser.add_argument("--completions", required=True, help="Test completions JSON")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_seeds", type=int, default=25, help="Seeds for steered eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    pipe_cfg = cfg["pipeline"]
    steer_cfg = cfg["steering"]

    layer = args.layer if args.layer is not None else steer_cfg["layer"]

    # ----------------------------------------------------------------
    # Stage 1: Load data + train probe
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STAGE 1: Probe training")
    print("=" * 60)

    print(f"Loading activations from {args.activations}")
    store = np.load(args.activations, allow_pickle=True)
    acts = store["activations"].astype(np.float32)
    question_idx = store["question_idx"]
    emp_acc = store["empirical_accuracy"].astype(np.float32)
    q_levels = store["question_level"]
    q_types = store["question_type"]

    # Aggregate to question level
    unique_qids = np.unique(question_idx)
    n_questions = len(unique_qids)
    q_emp_acc = np.zeros(n_questions)
    q_mean_acts = np.zeros((n_questions, acts.shape[1]), dtype=np.float32)
    q_level_arr = np.empty(n_questions, dtype=object)
    q_type_arr = np.empty(n_questions, dtype=object)

    for i, qid in enumerate(unique_qids):
        mask = question_idx == qid
        q_emp_acc[i] = emp_acc[mask][0]
        q_mean_acts[i] = acts[mask].mean(axis=0)
        q_level_arr[i] = q_levels[mask][0]
        q_type_arr[i] = q_types[mask][0]

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(
        q_level_arr, q_type_arr,
        val_frac=cfg["probes"]["val_frac"],
        test_frac=cfg["probes"]["test_frac"],
        seed=cfg["probes"]["seed"],
    )
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Z-score normalization
    mu = q_mean_acts[train_idx].mean(axis=0)
    sigma = np.maximum(q_mean_acts[train_idx].std(axis=0), 1e-12)
    Z_train = (q_mean_acts[train_idx] - mu) / sigma
    Z_val = (q_mean_acts[val_idx] - mu) / sigma
    Z_test = (q_mean_acts[test_idx] - mu) / sigma

    # Ridge regression with grid search
    alpha_grid = cfg["probes"]["ridge"]["alpha_grid"]
    best_alpha, best_r2 = None, -np.inf

    for alpha in alpha_grid:
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(Z_train, q_emp_acc[train_idx])
        preds = reg.predict(Z_val)
        ss_res = np.sum((q_emp_acc[val_idx] - preds) ** 2)
        ss_tot = np.sum((q_emp_acc[val_idx] - q_emp_acc[val_idx].mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    print(f"Best ridge alpha: {best_alpha} (val R²={best_r2:.4f})")

    # Refit on train, calibrate on val
    probe = Ridge(alpha=best_alpha, fit_intercept=True)
    probe.fit(Z_train, q_emp_acc[train_idx])

    raw_val = probe.predict(Z_val)
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(raw_val, q_emp_acc[val_idx])

    # Test predictions
    raw_test = probe.predict(Z_test)
    probe_preds = iso_reg.predict(raw_test)

    test_metrics = compute_all_metrics(q_emp_acc[test_idx], probe_preds)
    print(f"Probe test metrics: {test_metrics}")

    # ----------------------------------------------------------------
    # Stage 2: Alpha sweep + steered generation
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Alpha sweep + steered evaluation")
    print("=" * 60)

    # Load model
    print(f"Loading model {model_cfg['hf_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    hf_model.eval()

    # Load steering vector
    print(f"Loading steering vector from {args.vector}")
    sv_data = torch.load(args.vector, map_location="cpu", weights_only=True)
    sv_unit = sv_data["sv_normalized"].float()
    layer_norm = 130.0
    sv_scaled = (sv_unit * layer_norm).to(hf_model.device, dtype=torch.bfloat16)

    # Load test prompts
    print(f"Loading test completions from {args.completions}")
    with open(args.completions) as f:
        test_data = json.load(f)

    # Map test_idx back to data items
    test_prompts = [test_data[int(unique_qids[i])]["formatted_prompt"] for i in test_idx]
    test_gold = [test_data[int(unique_qids[i])]["gold_answer"] for i in test_idx]

    # Alpha sweep on validation questions
    val_prompts = [test_data[int(unique_qids[i])]["formatted_prompt"] for i in val_idx]

    lo, hi = pipe_cfg["sweep_alphas_range"]
    step = pipe_cfg["sweep_alphas_step"]
    sweep_alphas = [round(a * step, 1) for a in range(int(lo / step), int(hi / step) + 1)]

    target_module = get_layer_module(hf_model, layer)
    batch_size = steer_cfg["batch_size"]

    sweep_results = []
    print(f"Sweeping {len(sweep_alphas)} alphas on {len(val_prompts)} val questions...")

    for alpha in sweep_alphas:
        hook = SteeringHook(sv_scaled, alpha, injection_mode="answer_per_token")
        confs = []

        for i in range(0, len(val_prompts), batch_size):
            batch = val_prompts[i : i + batch_size]
            with add_hook(target_module, hook):
                completions = generate_batch(
                    hf_model, tokenizer, batch,
                    max_new_tokens=50, temperature=1.0,
                )
            for comp in completions:
                c = parse_confidence_from_completion(comp)
                if c is not None:
                    confs.append(c)

        mean_conf = float(np.mean(confs)) if confs else float("nan")
        parse_rate = len(confs) / len(val_prompts)
        sweep_results.append({
            "alpha": alpha,
            "mean_conf": mean_conf,
            "parse_rate": parse_rate,
            "n_valid": len(confs),
        })
        print(f"  α={alpha:+5.1f}: conf={mean_conf:.4f}, parse={parse_rate:.1%}")

    # Build transfer function
    valid_sweeps = [s for s in sweep_results if not np.isnan(s["mean_conf"])]
    sweep_alphas_valid = [s["alpha"] for s in valid_sweeps]
    sweep_confs_valid = [s["mean_conf"] for s in valid_sweeps]

    inv_interp = build_transfer_function(sweep_alphas_valid, sweep_confs_valid)

    if inv_interp is None:
        print("WARNING: Could not build monotonic transfer function. Using alpha=0.")
        adaptive_alphas = np.zeros(len(test_idx))
    else:
        conf_range = (min(sweep_confs_valid), max(sweep_confs_valid))
        alpha_range = (min(sweep_alphas_valid), max(sweep_alphas_valid))
        print(f"Transfer function: conf range {conf_range}, alpha range {alpha_range}")

        # Map probe predictions → adaptive alphas
        clipped_preds = np.clip(probe_preds, conf_range[0], conf_range[1])
        adaptive_alphas = inv_interp(clipped_preds)
        adaptive_alphas = np.clip(adaptive_alphas, alpha_range[0], alpha_range[1])
        print(f"Adaptive alphas: mean={adaptive_alphas.mean():.3f}, "
              f"range=[{adaptive_alphas.min():.1f}, {adaptive_alphas.max():.1f}]")

    # Discretize alphas and run steered generation
    bin_step = 0.1
    discrete_alphas = np.round(adaptive_alphas / bin_step) * bin_step

    # Group test questions by discretized alpha
    alpha_groups: dict[float, list[int]] = {}
    for i, a in enumerate(discrete_alphas):
        alpha_groups.setdefault(float(a), []).append(i)

    n_test = len(test_idx)
    steered_confs = np.full((n_test, args.n_seeds), np.nan)
    unsteered_confs = np.full((n_test, args.n_seeds), np.nan)

    print(f"\nRunning steered generation ({args.n_seeds} seeds, {n_test} questions)...")

    for seed in range(args.n_seeds):
        torch.manual_seed(seed + 1)

        # Steered
        for alpha_val, indices in alpha_groups.items():
            hook = SteeringHook(sv_scaled, alpha_val, injection_mode="answer_per_token")
            batch_prompts = [test_prompts[i] for i in indices]

            for b_start in range(0, len(batch_prompts), batch_size):
                b_end = min(b_start + batch_size, len(batch_prompts))
                batch = batch_prompts[b_start:b_end]
                b_indices = indices[b_start:b_end]

                with add_hook(target_module, hook):
                    completions = generate_batch(
                        hf_model, tokenizer, batch,
                        max_new_tokens=50, temperature=1.0,
                    )
                for j, comp in enumerate(completions):
                    c = parse_confidence_from_completion(comp)
                    steered_confs[b_indices[j], seed] = c if c is not None else np.nan

        # Unsteered
        for b_start in range(0, n_test, batch_size):
            b_end = min(b_start + batch_size, n_test)
            batch = test_prompts[b_start:b_end]
            completions = generate_batch(
                hf_model, tokenizer, batch,
                max_new_tokens=50, temperature=1.0,
            )
            for j, comp in enumerate(completions):
                c = parse_confidence_from_completion(comp)
                unsteered_confs[b_start + j, seed] = c if c is not None else np.nan

        print(f"  Seed {seed + 1}/{args.n_seeds} done")

    # Aggregate and evaluate
    steered_mean = np.nanmean(steered_confs, axis=1)
    unsteered_mean = np.nanmean(unsteered_confs, axis=1)
    gold = q_emp_acc[test_idx]

    # Mask out questions with no valid confidences
    steered_valid = ~np.isnan(steered_mean)
    unsteered_valid = ~np.isnan(unsteered_mean)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if steered_valid.sum() > 0:
        steered_metrics = compute_all_metrics(gold[steered_valid], steered_mean[steered_valid])
        print(f"Steered (adaptive):  {steered_metrics}")

    if unsteered_valid.sum() > 0:
        unsteered_metrics = compute_all_metrics(gold[unsteered_valid], unsteered_mean[unsteered_valid])
        print(f"Unsteered:           {unsteered_metrics}")

    print(f"Probe:               {test_metrics}")

    # Save
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["steering_dir"]) / "pipeline" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    results_out = {
        "model": args.model,
        "layer": layer,
        "n_test": n_test,
        "n_seeds": args.n_seeds,
        "probe_metrics": test_metrics,
        "sweep_results": sweep_results,
        "adaptive_alphas": adaptive_alphas.tolist(),
    }

    if steered_valid.sum() > 0:
        results_out["steered_metrics"] = steered_metrics
    if unsteered_valid.sum() > 0:
        results_out["unsteered_metrics"] = unsteered_metrics

    with open(out_dir / "pipeline_results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)

    np.savez_compressed(
        out_dir / "pipeline_confs.npz",
        steered_confs=steered_confs,
        unsteered_confs=unsteered_confs,
        probe_preds=probe_preds,
        gold=gold,
        adaptive_alphas=adaptive_alphas,
    )

    print(f"\nSaved pipeline results to {out_dir}")


if __name__ == "__main__":
    main()
