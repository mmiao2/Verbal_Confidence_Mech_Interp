"""
Steering forward hooks for HuggingFace transformer models.

Provides hooks that add a scaled steering vector to the residual stream
at a target layer during autoregressive generation.

Two injection modes:
  - answer_per_token: Steer only during decoding (skip prefill)
  - both: Steer last token during prefill + every token during decoding
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn as nn


class SteeringHook:
    """Forward hook that adds α * steering_vector to the residual stream.

    Args:
        steering_vec: Steering vector tensor, shape (hidden_dim,).
        alpha: Scaling coefficient. Positive = toward "confident/correct".
        injection_mode: One of "answer_per_token", "prompt_last_token", "both".
    """

    VALID_MODES = ("prompt_last_token", "answer_per_token", "both")

    def __init__(
        self,
        steering_vec: torch.Tensor,
        alpha: float,
        injection_mode: str = "answer_per_token",
    ):
        if injection_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {injection_mode}. Must be one of {self.VALID_MODES}")
        self.steering_vec = steering_vec
        self.alpha = alpha
        self.injection_mode = injection_mode
        self.prefill_calls = 0
        self.gen_calls = 0

    def __call__(self, module: nn.Module, input: tuple, output: tuple | torch.Tensor):
        hidden = output[0] if isinstance(output, tuple) else output
        is_prefill = hidden.shape[1] > 1

        if self.injection_mode == "prompt_last_token":
            if not is_prefill:
                return output
            steered = hidden.clone()
            steered[:, -1, :] += self.alpha * self.steering_vec
            self.prefill_calls += 1

        elif self.injection_mode == "answer_per_token":
            if is_prefill:
                return output
            steered = hidden + self.alpha * self.steering_vec.unsqueeze(0).unsqueeze(0)
            self.gen_calls += 1

        elif self.injection_mode == "both":
            if is_prefill:
                steered = hidden.clone()
                steered[:, -1, :] += self.alpha * self.steering_vec
                self.prefill_calls += 1
            else:
                steered = hidden + self.alpha * self.steering_vec.unsqueeze(0).unsqueeze(0)
                self.gen_calls += 1
        else:
            return output

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def stats(self) -> dict[str, int]:
        return {"prefill_calls": self.prefill_calls, "gen_calls": self.gen_calls}


class NormMatchedSteeringHook:
    """Norm-matched activation steering hook.

    Instead of simple addition, scales the steering vector to match the
    norm of the original activation at each position:
        steered = original + (v_hat * ||original|| * α)

    where v_hat is the unit-normalized steering vector.
    """

    def __init__(
        self,
        steering_vec: torch.Tensor,
        alpha: float,
    ):
        self.steering_vec_normed = torch.nn.functional.normalize(
            steering_vec, dim=-1
        ).detach()
        self.alpha = alpha

    def __call__(self, module: nn.Module, input: tuple, output: tuple | torch.Tensor):
        hidden = output[0] if isinstance(output, tuple) else output
        norms = hidden.norm(dim=-1, keepdim=True)
        steered = hidden + self.steering_vec_normed * norms * self.alpha

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


@contextlib.contextmanager
def add_hook(module: nn.Module, hook: Callable):
    """Context manager to temporarily register a forward hook."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_layer_module(model, layer: int, use_lora: bool = False) -> nn.Module:
    """Get the residual stream submodule at the given layer.

    Supports Qwen, Llama, Mistral, DeepSeek, Gemma architectures.
    """
    model_name = model.config._name_or_path.lower()
    if use_lora:
        base = model.base_model.model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        base = model.model
    else:
        raise ValueError(f"Cannot find layers for model: {model.config._name_or_path}")
    return base.layers[layer]
