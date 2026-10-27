"""Training and evaluation loops shared across the three comparison runs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .datasets import Corpus
from .lora import apply_lora, count_total_params, count_trainable_params
from .model import TextClassifier
from .vocab import Vocab


def encode_corpus(corpus: Corpus, vocab: Vocab, max_len: int):
    ids = torch.tensor([vocab.encode(t, max_len) for t in corpus.texts], dtype=torch.long)
    labels = torch.tensor(corpus.labels, dtype=torch.long)
    return ids, labels


def iterate_batches(ids: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool = True, generator=None):
    n = ids.size(0)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield ids[idx], labels[idx]


@torch.no_grad()
def evaluate(model: TextClassifier, ids: torch.Tensor, labels: torch.Tensor, batch_size: int = 64) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch_ids, batch_labels in iterate_batches(ids, labels, batch_size, shuffle=False):
        logits = model(batch_ids)
        loss = loss_fn(logits, batch_labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def train(
    model: TextClassifier,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    val_ids: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_ids, batch_labels in iterate_batches(train_ids, train_labels, batch_size, generator=generator):
            optimizer.zero_grad()
            logits = model(batch_ids)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        val_metrics = evaluate(model, val_ids, val_labels)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / n_batches,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        if verbose:
            print(
                f"epoch {epoch:2d}  train_loss={record['train_loss']:.4f}  "
                f"val_loss={record['val_loss']:.4f}  val_acc={record['val_accuracy']:.4f}"
            )
    return history


def build_lora_model(base_model: TextClassifier, num_classes: int, r: int, alpha: float) -> TextClassifier:
    """Attach LoRA adapters to the backbone's target linears, freeze
    everything else in the backbone, replace the head with a fresh
    trainable one for the new task."""
    for module, attr in base_model.lora_target_modules():
        apply_lora(module, [attr], r=r, alpha=alpha)
    base_model.set_backbone_trainable(False)
    # set_backbone_trainable already froze the raw weights/biases inside
    # LoRALinear too (they were created with requires_grad=False anyway),
    # but lora_A / lora_B must stay trainable -- re-enable them explicitly.
    for module, attr in base_model.lora_target_modules():
        lora_layer = getattr(module, attr)
        lora_layer.lora_A.requires_grad_(True)
        lora_layer.lora_B.requires_grad_(True)
    base_model.replace_head(num_classes)
    return base_model


def param_report(model: TextClassifier) -> dict:
    return {
        "trainable": count_trainable_params(model),
        "total": count_total_params(model),
    }
