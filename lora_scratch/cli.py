"""Command-line entry point: pretrain the base model on Task A, then run the
three-way comparison (frozen / full fine-tune / LoRA) on Task B, printing a
summary table and the parameter-count comparison."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from .datasets import generate_review_corpus, generate_topic_corpus, train_val_split
from .engine import build_lora_model, encode_corpus, evaluate, param_report, train
from .model import TextClassifier
from .vocab import Vocab

MAX_LEN = 24
D_MODEL = 128
D_FF = 256
N_LAYERS = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def pretrain_base(seed: int, epochs: int, verbose: bool):
    corpus = generate_topic_corpus()
    train_corpus, val_corpus = train_val_split(corpus)
    vocab = Vocab(train_corpus.texts)

    train_ids, train_labels = encode_corpus(train_corpus, vocab, MAX_LEN)
    val_ids, val_labels = encode_corpus(val_corpus, vocab, MAX_LEN)

    model = TextClassifier(
        vocab_size=len(vocab),
        num_classes=len(corpus.label_names),
        d_model=D_MODEL,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
    )
    history = train(
        model, train_ids, train_labels, val_ids, val_labels,
        epochs=epochs, seed=seed, verbose=verbose,
    )
    final = evaluate(model, val_ids, val_labels)
    return model, vocab, history, final, corpus.label_names


def run_comparison(base_model: TextClassifier, vocab: Vocab, seed: int, epochs: int, r: int, alpha: float, verbose: bool):
    review_corpus = generate_review_corpus()
    train_corpus, val_corpus = train_val_split(review_corpus)
    train_ids, train_labels = encode_corpus(train_corpus, vocab, MAX_LEN)
    val_ids, val_labels = encode_corpus(val_corpus, vocab, MAX_LEN)
    num_classes = len(review_corpus.label_names)

    results = {}

    # (a) frozen baseline: freeze the whole pretrained backbone, train only
    # a freshly initialized head.
    set_seed(seed)
    frozen_model = copy.deepcopy(base_model)
    frozen_model.set_backbone_trainable(False)
    frozen_model.replace_head(num_classes)
    train(frozen_model, train_ids, train_labels, val_ids, val_labels, epochs=epochs, seed=seed, verbose=verbose)
    results["frozen"] = {
        "metrics": evaluate(frozen_model, val_ids, val_labels),
        "params": param_report(frozen_model),
    }

    # (b) full fine-tune: every parameter, including the backbone, is
    # trainable -- the gold-standard upper bound.
    set_seed(seed)
    full_model = copy.deepcopy(base_model)
    full_model.set_backbone_trainable(True)
    full_model.replace_head(num_classes)
    train(full_model, train_ids, train_labels, val_ids, val_labels, epochs=epochs, seed=seed, verbose=verbose)
    results["full_finetune"] = {
        "metrics": evaluate(full_model, val_ids, val_labels),
        "params": param_report(full_model),
    }

    # (c) LoRA: freeze the backbone's original weights, attach trainable
    # low-rank adapters to the attention and feedforward linears, train
    # those plus a fresh head.
    set_seed(seed)
    lora_model = copy.deepcopy(base_model)
    lora_model = build_lora_model(lora_model, num_classes, r=r, alpha=alpha)
    train(lora_model, train_ids, train_labels, val_ids, val_labels, epochs=epochs, seed=seed, verbose=verbose)
    results["lora"] = {
        "metrics": evaluate(lora_model, val_ids, val_labels),
        "params": param_report(lora_model),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train the base model and run the LoRA comparison.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=== Pretraining base model on Task A (topic classification) ===")
    base_model, vocab, pretrain_history, pretrain_final, topic_labels = pretrain_base(
        args.seed, args.pretrain_epochs, args.verbose
    )
    print(f"Pretraining finished. Final val accuracy on Task A: {pretrain_final['accuracy']:.4f}")
    print(f"Vocabulary size: {len(vocab)}")

    print("\n=== Running three-way comparison on Task B (review sentiment) ===")
    results = run_comparison(
        base_model, vocab, args.seed, args.finetune_epochs, args.lora_r, args.lora_alpha, args.verbose
    )

    print("\n=== Results ===")
    header = f"{'Method':<16}{'Val Accuracy':<16}{'Val Loss':<12}{'Trainable Params':<20}{'Total Params':<14}"
    print(header)
    print("-" * len(header))
    for name in ["frozen", "full_finetune", "lora"]:
        m = results[name]["metrics"]
        p = results[name]["params"]
        print(f"{name:<16}{m['accuracy']:<16.4f}{m['loss']:<12.4f}{p['trainable']:<20,}{p['total']:<14,}")

    full_trainable = results["full_finetune"]["params"]["trainable"]
    lora_trainable = results["lora"]["params"]["trainable"]
    reduction = 1 - (lora_trainable / full_trainable)
    print(f"\nLoRA trains {lora_trainable:,} parameters vs {full_trainable:,} for full fine-tuning "
          f"({reduction * 100:.2f}% fewer trainable parameters).")

    if args.json_out:
        payload = {
            "pretrain_val_accuracy": pretrain_final["accuracy"],
            "vocab_size": len(vocab),
            "results": {
                name: {
                    "val_accuracy": results[name]["metrics"]["accuracy"],
                    "val_loss": results[name]["metrics"]["loss"],
                    "trainable_params": results[name]["params"]["trainable"],
                    "total_params": results[name]["params"]["total"],
                }
                for name in results
            },
            "trainable_param_reduction_vs_full": reduction,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote metrics to {args.json_out}")


if __name__ == "__main__":
    main()
