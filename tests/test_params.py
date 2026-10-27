"""Verify the trainable-parameter-count reduction LoRA achieves on the full
model (backbone + head) is computed correctly, and that it's substantial."""

import copy

import torch

from lora_scratch.engine import build_lora_model, param_report
from lora_scratch.model import TextClassifier


def _make_model():
    torch.manual_seed(0)
    return TextClassifier(
        vocab_size=200, num_classes=5, d_model=32, d_ff=64, n_layers=2, max_len=16
    )


def test_full_finetune_trains_every_parameter():
    model = _make_model()
    model.set_backbone_trainable(True)
    model.replace_head(3)
    report = param_report(model)
    assert report["trainable"] == report["total"]


def test_frozen_baseline_trains_only_the_head():
    model = _make_model()
    model.set_backbone_trainable(False)
    model.replace_head(3)
    report = param_report(model)
    head_params = sum(p.numel() for p in model.head.parameters())
    assert report["trainable"] == head_params
    assert report["trainable"] < report["total"]


def test_lora_trainable_params_far_fewer_than_full_finetune():
    base = _make_model()
    full_model = copy.deepcopy(base)
    full_model.set_backbone_trainable(True)
    full_model.replace_head(3)
    full_report = param_report(full_model)

    lora_model = copy.deepcopy(base)
    lora_model = build_lora_model(lora_model, num_classes=3, r=4, alpha=8.0)
    lora_report = param_report(lora_model)

    assert lora_report["trainable"] < full_report["trainable"]
    reduction = 1 - (lora_report["trainable"] / full_report["trainable"])
    assert reduction > 0.5, "LoRA should train well under half the parameters of full fine-tuning here"


def test_lora_only_adapter_and_head_params_trainable():
    model = _make_model()
    model = build_lora_model(model, num_classes=3, r=4, alpha=8.0)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or name.startswith("head."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"
