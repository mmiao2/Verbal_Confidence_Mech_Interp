"""
Global configuration loader.

Reads configs/default.yaml and exposes all settings as a nested dictionary.
Individual scripts can override via CLI arguments.
"""

from pathlib import Path
from typing import Any

import yaml


_CONFIG_CACHE: dict[str, Any] | None = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration. Caches after first call."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    paths = cfg.get("paths", {})
    for key, val in paths.items():
        if not Path(val).is_absolute():
            paths[key] = str(PROJECT_ROOT / val)

    # Auto-fill generation seeds if not specified
    gen = cfg.get("generation", {})
    if gen.get("seeds") is None:
        gen["seeds"] = list(range(1, gen.get("n_seeds", 50) + 1))

    _CONFIG_CACHE = cfg
    return cfg


def get_model_config(cfg: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Get configuration for a specific model."""
    models = cfg["models"]
    if model_key not in models:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {list(models.keys())}"
        )
    return models[model_key]


# --- Path helpers ---

def sampled_path(cfg: dict[str, Any], split: str) -> Path:
    return Path(cfg["paths"]["data_dir"]) / f"math_{split}.jsonl"


def completions_path(cfg: dict[str, Any], model_key: str, split: str) -> Path:
    return Path(cfg["paths"]["completions_dir"]) / f"completions_{model_key}_{split}.json"


def activations_path(cfg: dict[str, Any], model_key: str, split: str) -> Path:
    return Path(cfg["paths"]["activations_dir"]) / f"activations_{model_key}_{split}.npz"


def probes_path(cfg: dict[str, Any], model_key: str, split: str) -> Path:
    return Path(cfg["paths"]["probes_dir"]) / f"{model_key}_{split}"
