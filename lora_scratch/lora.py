"""From-scratch implementation of Low-Rank Adaptation (LoRA).

The core idea: instead of fine-tuning the full weight matrix W of a linear
layer, freeze W and learn a low-rank update

    delta_W = B @ A        (B is d_out x r, A is r x d_in, r << d_in, d_out)

so that the adapted forward pass is

    y = x @ W^T + b + (alpha / r) * (x @ A^T @ B^T)

A is initialized with small random values and B is initialized to exactly
zero, so delta_W = B @ A = 0 at initialization and the adapter starts as a
mathematically exact no-op: the adapted layer computes precisely the same
output as the frozen base layer until any training happens.

During training only A and B receive gradients. W (and the bias) are
wrapped so that `requires_grad = False`, which means autograd never
populates `.grad` for them and no optimizer step changes them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A linear layer with a frozen base weight and a trainable low-rank
    adapter, matching nn.Linear's (in_features, out_features) convention.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the original weight and bias. They remain nn.Parameters
        # (so state_dict / device moves behave normally) but requires_grad
        # is turned off, so autograd skips them entirely during backward.
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable low-rank adapter matrices.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays all zeros -> delta_W = B @ A = 0 at init.

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def delta_weight(self) -> torch.Tensor:
        """The effective low-rank update to W, shape (out_features, in_features)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge_into_linear(self) -> nn.Linear:
        """Return a plain nn.Linear with the adapter folded into the weight,
        useful for sanity-checking numerical equivalence."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged.weight.copy_(self.weight + self.delta_weight())
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        return merged

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def apply_lora(module: nn.Module, target_attr_names: list[str], r: int = 8, alpha: float = 16.0) -> None:
    """In-place: replace each named nn.Linear submodule of `module` (given by
    attribute name, e.g. "ffn_in") with a LoRALinear wrapping it."""
    for name in target_attr_names:
        base = getattr(module, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear, got {type(base)}")
        setattr(module, name, LoRALinear(base, r=r, alpha=alpha))


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
