"""
Step 3: Extract hidden-state activations from completions.

For each (question, seed) pair from Step 2:
  1. Reconstruct the full token sequence: formatted_prompt + raw_completion
  2. Run a single forward pass with HF Transformers
  3. Extract the target-layer hidden state at the last token position

The hidden state at position T in a full forward pass is identical to the one
produced during autoregressive generation (causal attention mask), so this is
equivalent to extracting the activation at the last generated token.

Uses left-padding so the last position in every batch element is the last
real token.

Output NPZ fields (all length N_questions * N_seeds):
  - activations:        (N, hidden_dim) float16
  - question_idx:       (N,) int32
  - seed:               (N,) int32
  - binary_correct:     (N,) int8
  - empirical_accuracy: (N,) float32
  - gold_answer:        (N,) object
  - generated_answer:   (N,) object
  - question_text:      (N,) object
  - question_level:     (N,) object
  - question_type:      (N,) object
  - completion_length:  (N,) int32

Usage:
    python -m src.extraction.extract_activations --model llama_base --split train
    python -m src.extraction.extract_activations --model llama_base --split train --layer 21
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config, get_model_config, completions_path, activations_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations from completions.")
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=None, help="Override extraction batch size")
    parser.add_argument("--layer", type=int, default=None, help="Override extraction layer")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Output NPZ path (auto if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = get_model_config(cfg, args.model)
    ext_cfg = cfg["extraction"]

    layer = args.layer if args.layer is not None else ext_cfg["layer"]
    batch_size = args.batch_size if args.batch_size is not None else ext_cfg["batch_size"]

    # Load completions from Step 2
    comp_path = completions_path(cfg, args.model, args.split)
    print(f"Loading completions from {comp_path}...")
    with open(comp_path) as f:
        data = json.load(f)

    n_questions = len(data)
    n_seeds = len(data[0]["completions"])
    n_total = n_questions * n_seeds
    print(f"  {n_questions} questions x {n_seeds} seeds = {n_total} activations")

    # Build flat lists: one entry per (question, seed)
    sequences: list[str] = []
    meta_question_idx: list[int] = []
    meta_seed: list[int] = []
    meta_binary_correct: list[int] = []
    meta_empirical_accuracy: list[float] = []
    meta_gold_answer: list[str] = []
    meta_generated_answer: list[str] = []
    meta_question_text: list[str] = []
    meta_question_level: list[str] = []
    meta_question_type: list[str] = []

    for item in data:
        prompt = item["formatted_prompt"]
        for comp in item["completions"]:
            sequences.append(prompt + comp["raw_text"])
            meta_question_idx.append(item["question_idx"])
            meta_seed.append(comp["seed"])
            meta_binary_correct.append(int(comp["correct"]))
            meta_empirical_accuracy.append(item["empirical_accuracy"])
            meta_gold_answer.append(item["gold_answer"])
            meta_generated_answer.append(comp["parsed_answer"])
            meta_question_text.append(item["question"])
            meta_question_level.append(item.get("level", ""))
            meta_question_type.append(item.get("type", ""))

    # Load tokenizer + model
    print(f"Loading model {model_cfg['hf_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Register forward hook on target layer
    target_layer = model.model.layers[layer]
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    hook_handle = target_layer.register_forward_hook(hook_fn)

    # Compute completion token lengths
    print("Computing completion token lengths...")
    meta_completion_length: list[int] = []
    for item in data:
        for comp in item["completions"]:
            toks = tokenizer.encode(comp["raw_text"], add_special_tokens=False)
            meta_completion_length.append(len(toks))

    # Process in batches with left-padding
    max_model_len = cfg["vllm"]["max_model_len"]
    all_activations: list[np.ndarray] = []

    print(f"Extracting layer-{layer} activations (batch_size={batch_size})...")
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_seqs = sequences[batch_start:batch_end]

        if batch_start % (batch_size * 50) == 0:
            print(f"  {batch_start}/{n_total} ({100 * batch_start / n_total:.1f}%)")

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_model_len,
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        # With left-padding, the last position is always the last real token
        last_hidden = captured["hidden"][:, -1, :].cpu().numpy().astype(np.float16)
        all_activations.append(last_hidden)

    hook_handle.remove()

    # Concatenate and save
    activations = np.concatenate(all_activations, axis=0)

    out_path = Path(args.output) if args.output else activations_path(cfg, args.model, args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        activations=activations,
        question_idx=np.array(meta_question_idx, dtype=np.int32),
        seed=np.array(meta_seed, dtype=np.int32),
        binary_correct=np.array(meta_binary_correct, dtype=np.int8),
        empirical_accuracy=np.array(meta_empirical_accuracy, dtype=np.float32),
        gold_answer=np.array(meta_gold_answer, dtype=object),
        generated_answer=np.array(meta_generated_answer, dtype=object),
        question_text=np.array(meta_question_text, dtype=object),
        question_level=np.array(meta_question_level, dtype=object),
        question_type=np.array(meta_question_type, dtype=object),
        completion_length=np.array(meta_completion_length, dtype=np.int32),
    )

    print(f"\nSaved activations to {out_path}")
    print(f"  Shape: {activations.shape}")
    print(f"  Binary correct rate: {np.mean(meta_binary_correct):.4f}")
    print(f"  Mean empirical acc:  {np.mean(meta_empirical_accuracy):.4f}")


if __name__ == "__main__":
    main()
