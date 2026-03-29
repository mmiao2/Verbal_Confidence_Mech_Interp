#!/bin/bash
#SBATCH --job-name=steer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/steering_%j.log

set -euo pipefail

MODEL=${1:-qwen_instruct}
ACTIVATIONS=${2:-outputs/activations/activations_${MODEL}_train.npz}
COMPLETIONS_TRAIN=${3:-outputs/completions/completions_${MODEL}_train.json}
COMPLETIONS_TEST=${4:-outputs/completions/completions_${MODEL}_test.json}

echo "=== Step 3b: Compute question-level CAA steering vectors ==="
python -m src.steering.compute_vectors \
    --activations "$ACTIVATIONS" \
    --confidences "$COMPLETIONS_TRAIN"

VECTOR="outputs/steering/question_caa_${MODEL}_train.pt"

echo "=== Step 4: Steered generation (alpha sweep) ==="
python -m src.steering.steer_generate \
    --model "$MODEL" --split test --mode steer \
    --vector "$VECTOR"

echo "=== Step 5: Full two-stage pipeline ==="
python -m src.pipeline.run_pipeline \
    --model "$MODEL" \
    --activations "$ACTIVATIONS" \
    --vector "$VECTOR" \
    --completions "$COMPLETIONS_TEST"

echo "=== Done ==="
