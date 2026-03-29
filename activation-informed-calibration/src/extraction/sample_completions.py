"""
Step 2: Generate N completions per question using vLLM.

For each question and each seed (1..N_SEEDS):
  - Generate one completion at T=1.0
  - Parse the answer and check correctness against gold

Per-question output:
  - empirical_accuracy = fraction correct across seeds
  - formatted_prompt stored for exact token reconstruction in Step 3

Usage:
    python -m src.extraction.sample_completions --model qwen_instruct --split train
    python -m src.extraction.sample_completions --model qwen_base --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.config import load_config, get_model_config, completions_path
from src.utils.answer_extraction import get_parse_fn, get_check_fn
from src.utils.prompts import get_answer_prompt, get_question_field


def load_questions(path: Path) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def build_formatted_prompts(
    questions: list[dict],
    prompt_template: str,
    q_field: str,
    is_instruct: bool,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Build final prompt strings sent to the model.

    For instruct models, wraps content in the chat template so that
    generation (vLLM) and forward-pass (HF) see identical tokens.
    """
    formatted = []
    for q in questions:
        content = prompt_template.format(**{q_field: q["question"]})
        if is_instruct:
            messages = [{"role": "user", "content": content}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = content
        formatted.append(text)
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate completions with vLLM.")
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Output JSON path (auto if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    gen_cfg = cfg["generation"]
    vllm_cfg = cfg["vllm"]
    is_instruct = model_cfg["is_instruct"]

    # Load questions
    data_path = Path(cfg["paths"]["data_dir"]) / f"math_{args.split}.jsonl"
    questions = load_questions(data_path)
    print(f"Loaded {len(questions)} questions from {data_path}")

    # Output path
    out_path = Path(args.output) if args.output else completions_path(cfg, args.model, args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prompt template
    benchmark_name = cfg["benchmark"]["name"]
    prompt_template = get_answer_prompt(benchmark_name, is_instruct)
    q_field = get_question_field(benchmark_name)
    parse_fn = get_parse_fn(benchmark_name)
    check_fn = get_check_fn(benchmark_name)

    # Build formatted prompts
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    formatted_prompts = build_formatted_prompts(
        questions, prompt_template, q_field, is_instruct, tokenizer,
    )

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
    all_completions: dict[int, list] = {i: [] for i in range(len(questions))}

    for seed in gen_cfg["seeds"]:
        print(f"\n--- Seed {seed}/{gen_cfg['n_seeds']} ---")
        params = SamplingParams(
            n=1,
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_new_tokens"],
            seed=seed,
        )
        outputs = llm.generate(formatted_prompts, params)

        n_correct = 0
        for q_idx, output in enumerate(outputs):
            raw_text = output.outputs[0].text
            parsed = parse_fn(raw_text)
            gold = questions[q_idx]["gold_answer"]
            correct = bool(check_fn(parsed, gold))
            if correct:
                n_correct += 1
            all_completions[q_idx].append({
                "seed": seed,
                "raw_text": raw_text,
                "parsed_answer": str(parsed),
                "correct": correct,
            })

        print(f"  Accuracy: {n_correct}/{len(questions)} ({n_correct / len(questions):.3f})")

    # Assemble per-question results
    results = []
    for q_idx, q in enumerate(questions):
        comps = all_completions[q_idx]
        n_correct = sum(c["correct"] for c in comps)
        empirical_acc = n_correct / len(comps)
        results.append({
            "question_idx": q_idx,
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "level": q.get("level", ""),
            "type": q.get("type", ""),
            "formatted_prompt": formatted_prompts[q_idx],
            "completions": comps,
            "n_correct": n_correct,
            "n_seeds": len(comps),
            "empirical_accuracy": empirical_acc,
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    accs = [r["empirical_accuracy"] for r in results]
    print(f"\n{'='*60}")
    print(f"Summary: {args.model} / {args.split}")
    print(f"  N questions:          {len(results)}")
    print(f"  N seeds per question: {gen_cfg['n_seeds']}")
    print(f"  Mean empirical acc:   {np.mean(accs):.4f}")
    print(f"  All correct (1.0):    {sum(1 for a in accs if a == 1.0)}")
    print(f"  All wrong   (0.0):    {sum(1 for a in accs if a == 0.0)}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
