# Activation-Informed Calibration

Code for the Mar 2026 paper: **"Closing the Confidence-Faithfulness Gap in Large Language Models"**

We show that linear probes trained on intermediate-layer activations can predict a language model's empirical accuracy, and that contrastive activation addition (CAA) steering vectors derived from verbalized confidence can modulate model confidence at inference time.

## Pipeline Overview

The codebase implements five core components:

```
(1) Activation Extraction     Extract hidden-state activations from model completions
         │
         ▼
(2) Linear Probe Training     Train Ridge / MLP / Fisher LDA probes on activations
         │                    to predict empirical accuracy
         ▼
(3) CAA Steering Vectors      Compute question-level contrastive activation addition
         │                    vectors: mean(high confidence) − mean(low confidence)
         ▼
(4) Steered Generation        Apply steering vectors during decoding via forward hooks
         │                    at target layers (answer_per_token injection)
         ▼
(5) Two-Stage Pipeline        End-to-end: probe → alpha sweep → adaptive steered eval
                              across multiple models and benchmarks
```

## Repository Structure

```
├── configs/
│   └── default.yaml              # Model/dataset/experiment configuration
├── src/
│   ├── extraction/               # (1) Activation extraction
│   │   ├── prepare_data.py       #     Download & prepare MATH dataset
│   │   ├── sample_completions.py #     Generate N completions per question (vLLM)
│   │   └── extract_activations.py#     Extract layer activations via forward hooks
│   ├── probes/                   # (2) Linear probe training
│   │   ├── probe_models.py       #     Ridge, MLP, Fisher LDA probe implementations
│   │   ├── train_probes.py       #     Train & evaluate probes on activations
│   │   └── metrics.py            #     AUROC, ECE, Brier, R² metrics
│   ├── steering/                 # (3-4) CAA steering vectors & steered generation
│   │   ├── compute_vectors.py    #     Question-level CAA vector computation
│   │   ├── hooks.py              #     Steering forward hooks for HF models
│   │   ├── steer_generate.py     #     Steered generation with alpha sweep
│   │   └── steering_utils.py     #     Answer parsing, confidence extraction
│   ├── pipeline/                 # (5) Two-stage end-to-end pipeline
│   │   ├── run_pipeline.py       #     Full pipeline: probe → sweep → steer → eval
│   │   └── benchmark_transfer.py #     Cross-benchmark transfer evaluation
│   └── utils/
│       ├── answer_extraction.py  #     \boxed{} parsing, math answer checking
│       └── prompts.py            #     Prompt templates (base/instruct/confidence)
├── scripts/
│   ├── run_extraction.sh         #     SLURM script: data prep + completion sampling
│   ├── run_probes.sh             #     SLURM script: probe training
│   └── run_steering.sh           #     SLURM script: steering experiments
├── requirements.txt
└── README.md
```

## Supported Models

| Model | HuggingFace ID | Layers | Hidden Dim |
|-------|----------------|--------|------------|
| Qwen 2.5-7B | `Qwen/Qwen2.5-7B` | 28 | 3584 |
| Qwen 2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 |
| Llama 3.1-8B | `meta-llama/Llama-3.1-8B` | 32 | 4096 |
| Llama 3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 32 | 4096 |
| Mistral 7B v0.1 | `mistralai/Mistral-7B-v0.1` | 32 | 4096 |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4096 |
| DeepSeek LLM 7B | `deepseek-ai/deepseek-llm-7b-base` | 30 | 4096 |
| DeepSeek LLM 7B Chat | `deepseek-ai/deepseek-llm-7b-chat` | 30 | 4096 |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data and extract activations

```bash
# Download MATH dataset
python -m src.extraction.prepare_data

# Generate 50 completions per question (requires GPU + vLLM)
python -m src.extraction.sample_completions --model llama_base --split train

# Extract layer-24 activations
python -m src.extraction.extract_activations --model llama_base --split train
```

### 3. Train probes

```bash
python -m src.probes.train_probes --model llama_base --split train
```

### 4. Compute steering vectors and run steered generation

```bash
# Compute question-level CAA vector (high confidence − low confidence)
python -m src.steering.compute_vectors \
    --activations outputs/activations/activations_llama_base_train.npz \
    --confidences outputs/completions/completions_llama_base_train.json

# Steered generation with alpha sweep
python -m src.steering.steer_generate --model llama_base --split test \
    --mode steer --vector outputs/steering/question_caa_llama_base_train.pt
```

### 5. Run full two-stage pipeline

```bash
python -m src.pipeline.run_pipeline --model llama_base \
    --activations outputs/activations/activations_llama_base_train.npz \
    --vector outputs/steering/question_caa_llama_base_train.pt \
    --completions outputs/completions/completions_llama_base_test.json
```

## Method Details

### Activation Extraction
We generate *N*=50 completions per question at temperature *T*=1.0, compute empirical accuracy as the fraction correct, and extract the hidden-state activation at the last token position from a target layer (default: layer 24 for Llama 3.1-8B) via a forward hook on the full prompt+completion sequence.

### Linear Probes
Three probe architectures predict empirical accuracy from single-layer activations:
- **Ridge regression** with alpha grid search and isotonic calibration
- **MLP** (512-256-1) with weight decay search and early stopping
- **Fisher LDA** using binned mass-mean direction, within-class scatter with shrinkage, and isotonic calibration

### Question-Level CAA Steering Vectors
For each question with completions spanning both high and low verbalized confidence:

1. Compute per-question delta: `delta_q = mean(act[q, high_conf]) - mean(act[q, low_conf])`
2. Average across qualifying questions: `sv = mean(delta_q)`
3. Normalize to unit vector

This question-level control cancels question-difficulty confounds, isolating the direction associated with confidence variation. High/low thresholds default to 0.75/0.25.

### Steered Generation
During autoregressive decoding, a forward hook at the target layer adds `alpha * sv` to the residual stream at every decode step (`answer_per_token` mode). The hook skips the prefill pass entirely, steering only the generation phase. Positive alpha pushes toward higher confidence; negative toward lower.

### Two-Stage Pipeline
1. **Stage 1 (Probe)**: Train ridge probe on training activations, calibrate with isotonic regression on validation set
2. **Stage 2 (Steer)**: Sweep alpha on validation to build a transfer function (alpha -> confidence), invert probe predictions through it to get per-question adaptive alphas, generate on test set

## Citation

```bibtex
@misc{miao2026closingconfidencefaithfulnessgaplarge,
      title={Closing the Confidence-Faithfulness Gap in Large Language Models}, 
      author={Miranda Muqing Miao and Lyle Ungar},
      year={2026},
      eprint={2603.25052},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.25052}, 
}
```


