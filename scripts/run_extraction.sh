#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extraction_%j.log

set -euo pipefail

MODEL=${1:-qwen_instruct}
SPLIT=${2:-train}

echo "=== Step 1: Prepare MATH dataset ==="
python -m src.extraction.prepare_data

echo "=== Step 2: Sample completions (vLLM) ==="
python -m src.extraction.sample_completions --model "$MODEL" --split "$SPLIT"

echo "=== Step 3: Extract activations ==="
python -m src.extraction.extract_activations --model "$MODEL" --split "$SPLIT"

echo "=== Done ==="
