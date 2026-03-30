"""
Step 4: Steered generation with CAA steering vectors.

Applies a question-level CAA steering vector during autoregressive
generation via a forward hook on the target layer. Supports two modes:

  - baseline: Unsteered HF generation (greedy or sampled)
  - steer:    Sweep over alpha values, applying the steering hook
               in answer_per_token mode (steer only during decoding)

For each alpha, generates completions, parses confidence, and records
metrics. Supports:
  - Full answer generation (--prompt_type answer, default)
  - Confidence-only generation (--prompt_type confidence_only, faster)
  - Instruct model steering (--use_chat_template --model_override)

Usage:
    # Baseline (no steering)
    python -m src.steering.steer_generate --model llama_base \\
        --split test --mode baseline

    # Steered generation with alpha sweep
    python -m src.steering.steer_generate --model llama_base \\
        --split test --mode steer \\
        --vector outputs/steering/question_caa_llama_base_train.pt \\
        --alphas -2.0 -1.0 -0.5 0.5 1.0 2.0

    # Confidence-only steering (Section 3.4)
    python -m src.steering.steer_generate --model llama_base \\
        --split test --mode steer --prompt_type confidence_only \\
        --vector outputs/steering/question_caa_llama_base_train.pt

    # Instruct model with base vector (Section 3.5)
    python -m src.steering.steer_generate --model llama_base \\
        --split test --mode steer --prompt_type confidence_only \\
        --model_override meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --use_chat_template \\
        --vector outputs/steering/question_caa_llama_base_train.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config, get_model_config, completions_path
from src.steering.hooks import SteeringHook, add_hook, get_layer_module
from src.steering.steering_utils import (
    parse_confidence_from_completion,
    parse_answer_from_completion,
    check_math_answer,
)


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> list[str]:
    """Generate completions for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated portion
    input_len = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        gen_tokens = output[input_len:]
        completions.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))

    return completions


def run_steered_generation(
    model,
    tokenizer,
    prompts: list[str],
    gold_answers: list[str],
    steering_vec: torch.Tensor,
    alpha: float,
    layer: int,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    batch_size: int = 4,
) -> dict:
    """Run steered generation on a batch of prompts at a given alpha.

    Returns dict with per-question results.
    """
    target_module = get_layer_module(model, layer)
    hook = SteeringHook(
        steering_vec=steering_vec,
        alpha=alpha,
        injection_mode="answer_per_token",
    )

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_gold = gold_answers[i : i + batch_size]

        with add_hook(target_module, hook):
            completions = generate_batch(
                model, tokenizer, batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        for j, (comp, gold) in enumerate(zip(completions, batch_gold)):
            conf = parse_confidence_from_completion(comp)
            ans = parse_answer_from_completion(comp)
            correct = check_math_answer(ans, gold) if ans and gold else False

            results.append({
                "completion": comp,
                "parsed_confidence": conf,
                "parsed_answer": ans,
                "correct": correct,
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Steered generation with CAA.")
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--mode", required=True, choices=["baseline", "steer"])
    parser.add_argument("--vector", default=None, help="Path to steering vector .pt")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values for sweep")
    parser.add_argument("--layer", type=int, default=None, help="Override steering layer")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=1, help="Seeds per question")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap prompts with tokenizer chat template (for instruct models)")
    parser.add_argument("--model_override", default=None,
                        help="Override HF model ID (e.g. use instruct model with base vector)")
    parser.add_argument("--prompt_type", default="answer",
                        choices=["answer", "confidence_only"],
                        help="Prompt type: full answer or confidence-only (for steering eval)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    steer_cfg = cfg["steering"]

    layer = args.layer if args.layer is not None else steer_cfg["layer"]
    batch_size = args.batch_size if args.batch_size is not None else steer_cfg["batch_size"]

    # Load data
    comp_path = completions_path(cfg, args.model, args.split)
    print(f"Loading data from {comp_path}")
    with open(comp_path) as f:
        data = json.load(f)

    # Build prompts based on prompt_type
    hf_id = args.model_override if args.model_override else model_cfg["hf_id"]

    if args.prompt_type == "confidence_only":
        from src.utils.prompts import PURE_CONFIDENCE_PROMPT
        prompts = [PURE_CONFIDENCE_PROMPT.format(problem=item["question"]) for item in data]
        max_new_tokens = 50  # confidence-only needs far fewer tokens
    else:
        prompts = [item["formatted_prompt"] for item in data]

    gold_answers = [item["gold_answer"] for item in data]
    print(f"  {len(prompts)} questions (prompt_type={args.prompt_type})")

    # Load model
    print(f"Loading model {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Wrap prompts with chat template if requested (for instruct models)
    if args.use_chat_template:
        print("Applying chat template to prompts...")
        wrapped = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            wrapped.append(text)
        prompts = wrapped

    # Override max_new_tokens for confidence-only
    if args.prompt_type == "confidence_only":
        max_new_tokens = 50
    else:
        max_new_tokens = args.max_new_tokens or steer_cfg["max_new_tokens"]

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_results = {}

    if args.mode == "baseline":
        print("Running baseline (unsteered) generation...")
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_gold = gold_answers[i : i + batch_size]

            completions = generate_batch(
                model, tokenizer, batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
            )
            for comp, gold in zip(completions, batch_gold):
                conf = parse_confidence_from_completion(comp)
                ans = parse_answer_from_completion(comp)
                correct = check_math_answer(ans, gold) if ans and gold else False
                results.append({
                    "completion": comp,
                    "parsed_confidence": conf,
                    "parsed_answer": ans,
                    "correct": correct,
                })

            if i % (batch_size * 10) == 0:
                print(f"  {i}/{len(prompts)}")

        confs = [r["parsed_confidence"] for r in results if r["parsed_confidence"] is not None]
        parse_rate = len(confs) / len(results) if results else 0
        print(f"  Parse rate: {parse_rate:.1%}, mean conf: {np.mean(confs):.4f}" if confs else "  No confidences parsed")

        all_results["baseline"] = results

    elif args.mode == "steer":
        if args.vector is None:
            raise ValueError("--vector required for steer mode")

        print(f"Loading steering vector from {args.vector}")
        sv_data = torch.load(args.vector, map_location="cpu", weights_only=True)
        sv_raw = sv_data["sv_normalized"].float()

        # Scale to match layer activation norms
        layer_norm = steer_cfg.get("layer_norm", 22.91)
        sv_scaled = (sv_raw * layer_norm).to(model.device, dtype=torch.bfloat16)

        alphas = args.alphas if args.alphas else steer_cfg["alphas"]
        print(f"Sweeping {len(alphas)} alpha values: {alphas}")

        for alpha in alphas:
            print(f"\n--- alpha = {alpha:+.1f} ---")
            results = run_steered_generation(
                model, tokenizer, prompts, gold_answers,
                steering_vec=sv_scaled,
                alpha=alpha,
                layer=layer,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )

            confs = [r["parsed_confidence"] for r in results if r["parsed_confidence"] is not None]
            corrects = [r["correct"] for r in results]
            parse_rate = len(confs) / len(results) if results else 0
            acc = np.mean(corrects)

            print(f"  Parse rate: {parse_rate:.1%}")
            if confs:
                print(f"  Mean conf:  {np.mean(confs):.4f}")
            print(f"  Accuracy:   {acc:.4f}")

            all_results[f"alpha_{alpha:+.1f}"] = results

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            Path(cfg["paths"]["steering_dir"])
            / f"steered_{args.model}_{args.split}_{args.mode}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip raw completions to save space (keep first 500 chars)
    for key in all_results:
        for r in all_results[key]:
            r["completion"] = r["completion"][:500]

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
