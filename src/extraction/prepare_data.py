"""
Step 1: Download MATH dataset and prepare JSONL files.

Downloads MATH-lighteval train (~7.5k) and test (~5k) splits.
Extracts gold answers from \\boxed{} in the solution field.
Saves standardised JSONL with level/type metadata.

Usage:
    python -m src.extraction.prepare_data
    python -m src.extraction.prepare_data --config configs/custom.yaml
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.config import load_config
from src.utils.answer_extraction import extract_boxed_answer


def save_split(ds, split_name: str, out_path: Path, cfg: dict) -> None:
    """Save a HuggingFace dataset split to JSONL."""
    benchmark = cfg["benchmark"]
    q_field = benchmark["question_field"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, ex in enumerate(ds):
            raw_solution = ex[benchmark["answer_field"]]
            boxed = extract_boxed_answer(raw_solution)
            gold = boxed if boxed is not None else raw_solution

            record = {
                "id": f"math_{split_name}_{i:06d}",
                "question_idx": i,
                "question": ex[q_field],
                "gold_answer": gold,
                "solution": raw_solution,
                "level": ex.get("level", ""),
                "type": ex.get("type", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(ds)} examples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare MATH dataset.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark = cfg["benchmark"]
    hf_id = benchmark["hf_id"]
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Train split
    print(f"Loading train split: {hf_id} / {benchmark['train_split']}...")
    ds_train = load_dataset(hf_id, split=benchmark["train_split"])
    print(f"  Train set size: {len(ds_train)}")
    save_split(ds_train, "train", data_dir / "math_train.jsonl", cfg)

    # Test split
    print(f"Loading test split: {hf_id} / {benchmark['test_split']}...")
    ds_test = load_dataset(hf_id, split=benchmark["test_split"])
    print(f"  Test set size: {len(ds_test)}")
    save_split(ds_test, "test", data_dir / "math_test.jsonl", cfg)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
