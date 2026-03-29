#!/bin/bash
#SBATCH --job-name=probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/probes_%j.log

set -euo pipefail

MODEL=${1:-qwen_instruct}
SPLIT=${2:-train}

echo "=== Training probes ==="
python -m src.probes.train_probes --model "$MODEL" --split "$SPLIT"

echo "=== Done ==="
