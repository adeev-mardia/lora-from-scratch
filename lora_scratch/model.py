"""A small transformer-style text classifier built entirely from raw
nn.Linear layers and tensor ops (no prebuilt transformer block, no
PEFT-aware architecture). This is the base model that gets pretrained on
Task A and then adapted to Task B via frozen/full-fine-tune/LoRA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Single-head self-attention implemented from the q/k/v/out
    projections directly, each a plain nn.Linear (the layers LoRA will
    later target)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) with 1 for real tokens, 0 for padding.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_model)  # (B, T, T)
        pad = (mask == 0).unsqueeze(1)  # (B, 1, T) broadcasts over query dim
        scores = scores.masked_fill(pad, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)  # rows that are all -inf (shouldn't happen) -> 0
        attended = weights @ v
        return self.out_proj(attended)


class FeedForward(nn.Module):
    """Two-layer MLP, the other natural LoRA target in a transformer block."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_out(self.act(self.ffn_in(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TextClassifier(nn.Module):
    """Embedding -> stack of TransformerBlocks -> masked mean pool -> linear
    classification head."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        d_ff: int = 256,
        n_layers: int = 2,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def backbone_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).float()
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        # Masked mean pooling over the sequence dimension.
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        return pooled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone_forward(input_ids)
        return self.head(pooled)

    def replace_head(self, num_classes: int) -> None:
        """Swap in a freshly initialized classification head for a new task
        with a different label space (Task B)."""
        self.head = nn.Linear(self.d_model, num_classes)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = trainable

    def lora_target_modules(self):
        """Yield (block, attr_name) pairs for every nn.Linear that LoRA
        adapters get attached to: the attention output projection and both
        feedforward layers of every transformer block."""
        for block in self.blocks:
            yield block.attn, "q_proj"
            yield block.attn, "v_proj"
            yield block.attn, "out_proj"
            yield block.ffn, "ffn_in"
            yield block.ffn, "ffn_out"
