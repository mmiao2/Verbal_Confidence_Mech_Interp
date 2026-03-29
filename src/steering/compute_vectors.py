"""
Step 3b: Compute question-level CAA steering vectors.

For each question that has completions in both high-confidence and
low-confidence bins, compute a per-question contrastive vector:

    δ_q = mean(activations[q, high_conf]) − mean(activations[q, low_conf])

The global steering vector is the average over all qualifying questions:

    sv = mean(δ_q for all qualifying q)

This question-level control cancels out question-difficulty confounds,
isolating the direction associated with confidence variation.

Usage:
    python -m src.steering.compute_vectors \
        --activations outputs/activations/activations_qwen_instruct_train.npz \
        --confidences outputs/completions/completions_qwen_instruct_train.json \
        --output outputs/steering/question_caa_layer24.pt

    python -m src.steering.compute_vectors \
        --activations outputs/activations/activations_qwen_instruct_train.npz \
        --confidences outputs/completions/completions_qwen_instruct_train.json \
        --low_threshold 0.25 --high_threshold 0.75
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from src.steering.steering_utils import parse_confidence_from_completion


def compute_question_caa(
    activations: np.ndarray,
    question_idx: np.ndarray,
    confidences: np.ndarray,
    low_threshold: float = 0.25,
    high_threshold: float = 0.75,
) -> dict[str, np.ndarray | dict]:
    """Compute question-level CAA steering vector.

    Args:
        activations: (N, hidden_dim) activation matrix.
        question_idx: (N,) question index per sample.
        confidences: (N,) verbalized confidence in [0, 1] per sample.
        low_threshold: Upper bound for low-confidence bin.
        high_threshold: Lower bound for high-confidence bin.

    Returns:
        Dictionary with:
          - sv_raw: (hidden_dim,) raw steering vector (unnormalized)
          - sv_normalized: (hidden_dim,) L2-normalized steering vector
          - meta: dict with computation statistics
    """
    # Group samples by question
    question_groups: dict[int, list[int]] = defaultdict(list)
    for i, qid in enumerate(question_idx):
        if not np.isnan(confidences[i]):
            question_groups[int(qid)].append(i)

    # Compute per-question delta vectors
    deltas: list[np.ndarray] = []
    n_low_total, n_high_total = 0, 0
    n_qualifying = 0

    for qid, indices in question_groups.items():
        confs = confidences[indices]
        low_mask = confs < low_threshold
        high_mask = confs > high_threshold

        if low_mask.sum() < 1 or high_mask.sum() < 1:
            continue

        low_indices = [indices[i] for i in range(len(indices)) if low_mask[i]]
        high_indices = [indices[i] for i in range(len(indices)) if high_mask[i]]

        mu_high = activations[high_indices].mean(axis=0)
        mu_low = activations[low_indices].mean(axis=0)
        delta = mu_high - mu_low

        deltas.append(delta)
        n_qualifying += 1
        n_low_total += len(low_indices)
        n_high_total += len(high_indices)

    if not deltas:
        raise ValueError(
            f"No qualifying questions found with thresholds "
            f"low<{low_threshold}, high>{high_threshold}. "
            f"Check that confidences are in [0, 1]."
        )

    # Average per-question deltas
    sv_raw = np.stack(deltas).mean(axis=0).astype(np.float32)
    sv_norm = float(np.linalg.norm(sv_raw))
    sv_normalized = (sv_raw / (sv_norm + 1e-12)).astype(np.float32)

    # Compute intra-question cosine similarity for reliability check
    if len(deltas) > 1:
        delta_stack = np.stack(deltas)
        delta_norms = np.linalg.norm(delta_stack, axis=1, keepdims=True) + 1e-12
        delta_unit = delta_stack / delta_norms
        cosines = delta_unit @ sv_normalized
        intra_cosine = float(np.mean(cosines))
    else:
        intra_cosine = 1.0

    meta = {
        "n_total_questions": len(question_groups),
        "n_qualifying": n_qualifying,
        "n_low_instances": n_low_total,
        "n_high_instances": n_high_total,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "sv_norm": sv_norm,
        "intra_question_cosine": intra_cosine,
    }

    print(f"  Qualifying questions: {n_qualifying}/{len(question_groups)}")
    print(f"  Low-conf instances:   {n_low_total}")
    print(f"  High-conf instances:  {n_high_total}")
    print(f"  SV norm (raw):        {sv_norm:.4f}")
    print(f"  Intra-Q cosine:       {intra_cosine:.4f}")

    return {
        "sv_raw": sv_raw,
        "sv_normalized": sv_normalized,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute question-level CAA steering vectors."
    )
    parser.add_argument(
        "--activations", required=True,
        help="Path to activations NPZ from extract_activations",
    )
    parser.add_argument(
        "--confidences", required=True,
        help="Path to completions JSON (with verbalized confidence) or "
             "a pre-parsed confidence NPZ",
    )
    parser.add_argument("--low_threshold", type=float, default=0.25)
    parser.add_argument("--high_threshold", type=float, default=0.75)
    parser.add_argument("--output", default=None, help="Output .pt path")
    args = parser.parse_args()

    # Load activations
    print(f"Loading activations from {args.activations}")
    store = np.load(args.activations, allow_pickle=True)
    acts = store["activations"].astype(np.float32)
    question_idx = store["question_idx"]

    # Load or parse confidences
    conf_path = Path(args.confidences)
    if conf_path.suffix == ".npz":
        conf_store = np.load(conf_path, allow_pickle=True)
        confidences = conf_store["confidences"].astype(np.float32)
    elif conf_path.suffix == ".json":
        print(f"Parsing confidences from {conf_path}")
        with open(conf_path) as f:
            data = json.load(f)
        confidences = []
        for item in data:
            for comp in item["completions"]:
                text = comp.get("raw_text", "")
                conf = parse_confidence_from_completion(text)
                confidences.append(conf if conf is not None else float("nan"))
        confidences = np.array(confidences, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported confidence file format: {conf_path.suffix}")

    assert len(acts) == len(confidences), (
        f"Activation count ({len(acts)}) != confidence count ({len(confidences)})"
    )

    n_valid = int(np.sum(~np.isnan(confidences)))
    print(f"  Valid confidences: {n_valid}/{len(confidences)}")

    # Compute question-level CAA
    print("Computing question-level CAA steering vector...")
    result = compute_question_caa(
        acts, question_idx, confidences,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        act_path = Path(args.activations)
        out_path = act_path.parent.parent / "steering" / f"question_caa_{act_path.stem.replace('activations_', '')}.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "sv_raw": torch.from_numpy(result["sv_raw"]),
            "sv_normalized": torch.from_numpy(result["sv_normalized"]),
            "meta": result["meta"],
        },
        out_path,
    )
    print(f"\nSaved steering vector to {out_path}")


if __name__ == "__main__":
    main()
