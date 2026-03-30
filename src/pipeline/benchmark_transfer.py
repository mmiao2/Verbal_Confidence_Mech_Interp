"""
Cross-benchmark steering transfer evaluation.

Tests whether steering vectors trained on MATH transfer to other benchmarks
(MMLU, TriviaQA, TruthfulQA), as described in Section 3.6 of the paper.

Two transfer modes:
  (a) Probe transfer: Train probe on source activations, evaluate on target
  (b) Steering transfer: Apply MATH-trained steering vector during
      confidence-only generation on target benchmark questions

Usage:
    # Probe transfer only
    python -m src.pipeline.benchmark_transfer \
        --probe_activations outputs/activations/activations_llama_base_train.npz \
        --eval_activations outputs/activations/activations_llama_base_mmlu.npz \
        --vector outputs/steering/question_caa_llama_base_train.pt \
        --model llama_base

    # Steering transfer (generates confidence-only completions on target)
    python -m src.pipeline.benchmark_transfer \
        --vector outputs/steering/question_caa_llama_base_train.pt \
        --model llama_base --benchmark mmlu \
        --benchmark_data outputs/data/mmlu_test.jsonl \
        --mode steer --alphas -1.0 -0.5 0 0.5 1.0
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config, get_model_config
from src.probes.metrics import compute_all_metrics
from src.steering.hooks import SteeringHook, add_hook, get_layer_module
from src.steering.steer_generate import generate_batch


# ============================================================
# Benchmark-specific confidence-only prompts (Section 3.6)
# ============================================================

MATH_CONF_PROMPT = (
    "Read the following problem and rate how confident you are "
    "that you could solve it correctly.\n"
    "Do not attempt to solve the problem. Only provide your "
    "confidence as a number from 0 to 100.\n\n"
    "Problem: {problem}\n\n"
    "Confidence:"
)

QA_CONF_PROMPT = (
    "Read the following question and rate how confident you are "
    "that you could answer it correctly.\n"
    "Do not attempt to answer the question. Only provide your "
    "confidence as a number from 0 to 100.\n\n"
    "Question: {question}\n\n"
    "Confidence:"
)

MMLU_CONF_PROMPT = (
    "Read the following question and rate how confident you are "
    "that you could answer it correctly.\n"
    "Do not attempt to answer the question. Only provide your "
    "confidence as a number from 0 to 100.\n\n"
    "Question: {question}\n\n"
    "Choices:\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n\n"
    "Confidence:"
)


def build_benchmark_prompt(q: dict, benchmark: str) -> str:
    """Build a confidence-only prompt for a benchmark question."""
    if benchmark == "math":
        return MATH_CONF_PROMPT.format(problem=q["question"])
    elif benchmark == "mmlu":
        return MMLU_CONF_PROMPT.format(
            question=q["question"],
            choice_a=q["choices"][0],
            choice_b=q["choices"][1],
            choice_c=q["choices"][2],
            choice_d=q["choices"][3],
        )
    else:
        return QA_CONF_PROMPT.format(question=q["question"])


def parse_confidence_from_completion(text: str) -> float | None:
    """Parse a confidence value (0-100 scale -> 0-1) from generated text."""
    if not text or not text.strip():
        return None
    numbers = re.findall(r"\d+\.?\d*", text.strip()[:200])
    if not numbers:
        return None
    val = float(numbers[0])
    if val > 1.0 and val <= 100:
        return val / 100.0
    elif 0.0 <= val <= 1.0:
        return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-benchmark steering transfer evaluation."
    )
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--vector", required=True, help="Steering vector .pt")
    parser.add_argument("--mode", default="probe", choices=["probe", "steer", "both"],
                        help="Transfer mode: probe only, steering only, or both")
    parser.add_argument("--benchmark", default="mmlu",
                        choices=["mmlu", "triviaqa", "truthfulqa"],
                        help="Target benchmark for steering transfer")
    parser.add_argument("--benchmark_data", default=None,
                        help="Path to target benchmark JSONL")
    parser.add_argument("--probe_activations", default=None,
                        help="Training activations NPZ (source benchmark)")
    parser.add_argument("--eval_activations", default=None,
                        help="Eval activations NPZ (for probe transfer)")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75],
                        help="Alpha values for steering sweep")
    parser.add_argument("--max_questions", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap prompts with chat template (for instruct models)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    steer_cfg = cfg["steering"]
    layer = args.layer if args.layer is not None else steer_cfg["layer"]
    batch_size = args.batch_size if args.batch_size is not None else steer_cfg["batch_size"]

    results = {"model": args.model, "benchmark": args.benchmark, "layer": layer}

    # ================================================================
    # Probe transfer (optional)
    # ================================================================
    if args.mode in ("probe", "both") and args.probe_activations:
        print("=" * 60)
        print("PROBE TRANSFER")
        print("=" * 60)

        store = np.load(args.probe_activations, allow_pickle=True)
        acts = store["activations"].astype(np.float32)
        emp_acc = store["empirical_accuracy"].astype(np.float32)
        question_idx = store["question_idx"]

        unique_qids = np.unique(question_idx)
        q_acts = np.zeros((len(unique_qids), acts.shape[1]), dtype=np.float32)
        q_acc = np.zeros(len(unique_qids))
        for i, qid in enumerate(unique_qids):
            mask = question_idx == qid
            q_acts[i] = acts[mask].mean(axis=0)
            q_acc[i] = emp_acc[mask][0]

        mu = q_acts.mean(axis=0)
        sigma = np.maximum(q_acts.std(axis=0), 1e-12)
        Z = (q_acts - mu) / sigma

        probe = Ridge(alpha=10.0, fit_intercept=True)
        probe.fit(Z, q_acc)
        raw_train = probe.predict(Z)
        iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_reg.fit(raw_train, q_acc)

        print(f"Source probe trained on {len(unique_qids)} questions")
        results["source_n_questions"] = len(unique_qids)

        if args.eval_activations:
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
            results["probe_transfer_metrics"] = transfer_metrics

    # ================================================================
    # Steering transfer (Section 3.6)
    # ================================================================
    if args.mode in ("steer", "both"):
        print("\n" + "=" * 60)
        print(f"STEERING TRANSFER: {args.benchmark.upper()}")
        print("=" * 60)

        if args.benchmark_data is None:
            raise ValueError("--benchmark_data required for steering transfer")

        # Load benchmark questions
        questions = []
        with open(args.benchmark_data) as f:
            for line in f:
                questions.append(json.loads(line))
        if len(questions) > args.max_questions:
            questions = questions[:args.max_questions]
        print(f"Loaded {len(questions)} {args.benchmark} questions")

        # Build confidence-only prompts
        prompts = [build_benchmark_prompt(q, args.benchmark) for q in questions]

        # Load model
        print(f"Loading model {model_cfg['hf_id']}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["hf_id"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if args.use_chat_template:
            print("Applying chat template...")
            wrapped = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                wrapped.append(text)
            prompts = wrapped

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Load steering vector
        print(f"Loading steering vector from {args.vector}")
        sv_data = torch.load(args.vector, map_location="cpu", weights_only=True)
        sv_unit = sv_data["sv_normalized"].float()
        layer_norm = steer_cfg.get("layer_norm", 22.91)
        sv_scaled = (sv_unit * layer_norm).to(model.device, dtype=torch.bfloat16)

        target_module = get_layer_module(model, layer)

        # Sweep alphas
        alphas_sorted = sorted(set(args.alphas))
        sweep_results = []
        print(f"Sweeping {len(alphas_sorted)} alphas: {alphas_sorted}")

        for alpha in alphas_sorted:
            confs = []
            use_hook = abs(alpha) > 1e-6

            if use_hook:
                hook = SteeringHook(sv_scaled, alpha, injection_mode="answer_per_token")

            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                if use_hook:
                    with add_hook(target_module, hook):
                        completions = generate_batch(
                            model, tokenizer, batch,
                            max_new_tokens=50, temperature=1.0,
                        )
                else:
                    completions = generate_batch(
                        model, tokenizer, batch,
                        max_new_tokens=50, temperature=1.0,
                    )

                for comp in completions:
                    c = parse_confidence_from_completion(comp)
                    confs.append(c)

            valid_confs = [c for c in confs if c is not None]
            mean_conf = float(np.mean(valid_confs)) if valid_confs else None
            parse_rate = len(valid_confs) / len(prompts) if prompts else 0

            sweep_results.append({
                "alpha": alpha,
                "mean_confidence": mean_conf,
                "n_valid": len(valid_confs),
                "parse_rate": parse_rate,
            })

            cs = f"{mean_conf:.4f}" if mean_conf is not None else "N/A"
            print(f"  alpha={alpha:+5.2f}: conf={cs}, "
                  f"parse={len(valid_confs)}/{len(prompts)} ({parse_rate:.1%})")

        results["steering_sweep"] = sweep_results
        results["n_questions"] = len(questions)

        del model
        torch.cuda.empty_cache()

    # Save
    out_path = Path(args.output) if args.output else (
        Path(cfg["paths"]["steering_dir"])
        / "transfer"
        / f"transfer_{args.model}_{args.benchmark}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved transfer results to {out_path}")


if __name__ == "__main__":
    main()
