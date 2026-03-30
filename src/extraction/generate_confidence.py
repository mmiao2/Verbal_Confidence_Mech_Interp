"""
Step 2b: Generate pure-confidence completions for CAA steering vectors.

For each question and each confidence level note (Table 5):
  - Build a confidence-only prompt with the level note appended
  - Generate N completions at T=1.0
  - Parse the verbalized confidence from each completion

This produces the contrastive confidence data needed to compute
question-level CAA steering vectors (Section 2.2).

Usage:
    python -m src.extraction.generate_confidence --model llama_base --split train
    python -m src.extraction.generate_confidence --model llama_base --split train \
        --levels level_2_very_cautious level_8_vanilla_no_note
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.config import load_config, get_model_config
from src.utils.prompts import (
    PURE_CONFIDENCE_PROMPT,
    CONFIDENCE_LEVELS,
    LEVEL_KEYS,
    build_confidence_prompt,
)


def parse_confidence_from_completion(text: str) -> float | None:
    """Parse a confidence value (0-100 scale → 0-1) from generated text."""
    if not text or not text.strip():
        return None
    # Look for first number in the completion
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
        description="Generate pure-confidence completions for CAA vector computation."
    )
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--levels", nargs="+", default=None,
                        help="Subset of confidence level keys (default: all)")
    parser.add_argument("--n_seeds", type=int, default=None,
                        help="Seeds per (question, level) pair (default: from config)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    gen_cfg = cfg["generation"]
    vllm_cfg = cfg["vllm"]

    n_seeds = args.n_seeds if args.n_seeds else gen_cfg["n_seeds"]
    levels = args.levels if args.levels else LEVEL_KEYS

    # Validate level keys
    for lev in levels:
        if lev not in CONFIDENCE_LEVELS:
            raise ValueError(f"Unknown confidence level: {lev}. Valid: {LEVEL_KEYS}")

    # Load questions
    data_path = Path(cfg["paths"]["data_dir"]) / f"math_{args.split}.jsonl"
    questions = []
    with open(data_path) as f:
        for line in f:
            questions.append(json.loads(line))

    if args.max_questions and len(questions) > args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions from {data_path}")
    print(f"Confidence levels ({len(levels)}): {levels}")
    print(f"Seeds per (question, level): {n_seeds}")

    # Build prompts: one per (question, level)
    all_prompts: list[str] = []
    prompt_meta: list[dict] = []

    for q in questions:
        for lev in levels:
            prompt = build_confidence_prompt(q["question"], lev)
            all_prompts.append(prompt)
            prompt_meta.append({
                "question_idx": q.get("question_idx", q.get("id", "")),
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "level": q.get("level", ""),
                "type": q.get("type", ""),
                "confidence_level": lev,
            })

    n_prompts = len(all_prompts)
    print(f"Total prompts: {n_prompts} ({len(questions)} questions × {len(levels)} levels)")

    # For instruct models, wrap with chat template
    is_instruct = model_cfg["is_instruct"]
    if is_instruct:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
        wrapped = []
        for p in all_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            wrapped.append(text)
        all_prompts = wrapped
        print("Applied chat template for instruct model")

    # Initialize vLLM
    print(f"Initializing vLLM with {model_cfg['hf_id']}...")
    llm = LLM(
        model=model_cfg["hf_id"],
        tensor_parallel_size=vllm_cfg["tensor_parallel_size"],
        dtype="auto",
        max_model_len=vllm_cfg["max_model_len"],
        trust_remote_code=True,
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
    )

    # Generate completions for each seed
    all_completions: dict[int, list] = {i: [] for i in range(n_prompts)}

    for seed in range(1, n_seeds + 1):
        print(f"\n--- Seed {seed}/{n_seeds} ---")
        params = SamplingParams(
            n=1,
            temperature=gen_cfg["temperature"],
            max_tokens=50,  # confidence-only: short output
            seed=seed,
        )
        outputs = llm.generate(all_prompts, params)

        n_parsed = 0
        for idx, output in enumerate(outputs):
            raw_text = output.outputs[0].text
            conf = parse_confidence_from_completion(raw_text)
            if conf is not None:
                n_parsed += 1
            all_completions[idx].append({
                "seed": seed,
                "raw_text": raw_text,
                "parsed_confidence": conf,
            })

        print(f"  Parsed: {n_parsed}/{n_prompts} ({n_parsed / n_prompts:.1%})")

    # Assemble results
    results = []
    for idx in range(n_prompts):
        comps = all_completions[idx]
        confs = [c["parsed_confidence"] for c in comps if c["parsed_confidence"] is not None]
        mean_conf = float(np.mean(confs)) if confs else None

        results.append({
            **prompt_meta[idx],
            "formatted_prompt": all_prompts[idx],
            "completions": comps,
            "n_seeds": len(comps),
            "n_parsed": len(confs),
            "mean_confidence": mean_conf,
        })

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            Path(cfg["paths"]["completions_dir"])
            / f"confidence_{args.model}_{args.split}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    parsed_rates = []
    for lev in levels:
        lev_results = [r for r in results if r["confidence_level"] == lev]
        lev_confs = [r["mean_confidence"] for r in lev_results if r["mean_confidence"] is not None]
        mean = float(np.mean(lev_confs)) if lev_confs else float("nan")
        parsed_rates.append(len(lev_confs) / len(lev_results) if lev_results else 0)
        desc = CONFIDENCE_LEVELS[lev]["description"]
        print(f"  {lev:35s}: mean_conf={mean:.4f}  parse={len(lev_confs)}/{len(lev_results)}  ({desc})")

    print(f"\nSaved {len(results)} entries to {out_path}")
    print(f"Overall parse rate: {np.mean(parsed_rates):.1%}")


if __name__ == "__main__":
    main()
