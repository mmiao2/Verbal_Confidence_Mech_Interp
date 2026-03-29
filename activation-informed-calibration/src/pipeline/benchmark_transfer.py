"""
Cross-benchmark transfer evaluation.

Trains a probe on one benchmark (e.g., MATH train) and evaluates
the steered generation pipeline on a different benchmark, testing
whether the learned confidence direction transfers across tasks.

Usage:
    python -m src.pipeline.benchmark_transfer \
        --probe_activations outputs/activations/activations_qwen_instruct_train.npz \
        --eval_completions outputs/completions/completions_qwen_instruct_test.json \
        --vector outputs/steering/question_caa_qwen_instruct_train.pt \
        --model qwen_instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config, get_model_config
from src.probes.metrics import compute_all_metrics
from src.steering.hooks import SteeringHook, add_hook, get_layer_module
from src.steering.steering_utils import parse_confidence_from_completion
from src.steering.steer_generate import generate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-benchmark transfer evaluation.")
    parser.add_argument("--probe_activations", required=True,
                        help="Training activations NPZ (source benchmark)")
    parser.add_argument("--eval_completions", required=True,
                        help="Evaluation completions JSON (target benchmark)")
    parser.add_argument("--eval_activations", default=None,
                        help="Eval activations NPZ (for probe transfer)")
    parser.add_argument("--vector", required=True, help="Steering vector .pt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    steer_cfg = cfg["steering"]
    layer = args.layer if args.layer is not None else steer_cfg["layer"]

    # Train probe on source benchmark
    print("Loading source activations...")
    store = np.load(args.probe_activations, allow_pickle=True)
    acts = store["activations"].astype(np.float32)
    emp_acc = store["empirical_accuracy"].astype(np.float32)
    question_idx = store["question_idx"]

    # Aggregate to question level
    unique_qids = np.unique(question_idx)
    q_acts = np.zeros((len(unique_qids), acts.shape[1]), dtype=np.float32)
    q_acc = np.zeros(len(unique_qids))
    for i, qid in enumerate(unique_qids):
        mask = question_idx == qid
        q_acts[i] = acts[mask].mean(axis=0)
        q_acc[i] = emp_acc[mask][0]

    # Z-score and train probe
    mu = q_acts.mean(axis=0)
    sigma = np.maximum(q_acts.std(axis=0), 1e-12)
    Z = (q_acts - mu) / sigma

    probe = Ridge(alpha=10.0, fit_intercept=True)
    probe.fit(Z, q_acc)
    raw_train = probe.predict(Z)
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(raw_train, q_acc)

    print(f"Source probe trained on {len(unique_qids)} questions")

    # If eval activations provided, run probe transfer
    if args.eval_activations:
        print(f"\nLoading eval activations from {args.eval_activations}")
        eval_store = np.load(args.eval_activations, allow_pickle=True)
        eval_acts = eval_store["activations"].astype(np.float32)
        eval_emp_acc = eval_store["empirical_accuracy"].astype(np.float32)
        eval_qidx = eval_store["question_idx"]

        eval_uq = np.unique(eval_qidx)
        eq_acts = np.zeros((len(eval_uq), eval_acts.shape[1]), dtype=np.float32)
        eq_acc = np.zeros(len(eval_uq))
        for i, qid in enumerate(eval_uq):
            mask = eval_qidx == qid
            eq_acts[i] = eval_acts[mask].mean(axis=0)
            eq_acc[i] = eval_emp_acc[mask][0]

        Z_eval = (eq_acts - mu) / sigma
        raw_eval = probe.predict(Z_eval)
        probe_preds = iso_reg.predict(raw_eval)

        transfer_metrics = compute_all_metrics(eq_acc, probe_preds)
        print(f"Transfer probe metrics: {transfer_metrics}")

    # Save
    out_path = Path(args.output) if args.output else (
        Path(cfg["paths"]["steering_dir"]) / "transfer" / f"transfer_{args.model}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "source_n_questions": len(unique_qids),
    }
    if args.eval_activations:
        results["transfer_metrics"] = transfer_metrics

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved transfer results to {out_path}")


if __name__ == "__main__":
    main()
