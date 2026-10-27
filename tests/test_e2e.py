"""End-to-end test: pretrain a tiny model on a small fixture task, then
confirm that LoRA fine-tuning on a shifted task actually improves accuracy
over the frozen baseline. This is the sanity check that the from-scratch
LoRA implementation produces real, measurable transfer learning, not just
mathematically-correct-but-useless adapters."""

import random

import torch

from lora_scratch.datasets import Corpus, train_val_split
from lora_scratch.engine import build_lora_model, encode_corpus, evaluate, train
from lora_scratch.model import TextClassifier
from lora_scratch.vocab import Vocab

MAX_LEN = 12


def _fixture_corpus(label_words, n_per_class=40, seed=0) -> Corpus:
    """A tiny synthetic fixture task: each class is built from sentences
    containing one of a small set of class-specific keywords, scattered
    among shared filler words, so the task is learnable but not trivial."""
    rng = random.Random(seed)
    filler = ["the", "quick", "report", "today", "said", "team", "later", "again", "really", "quite"]
    texts, labels = [], []
    for label_idx, words in enumerate(label_words):
        for _ in range(n_per_class):
            n_filler = rng.randint(2, 4)
            sentence_words = [rng.choice(filler) for _ in range(n_filler)]
            sentence_words.insert(rng.randint(0, len(sentence_words)), rng.choice(words))
            texts.append(" ".join(sentence_words))
            labels.append(label_idx)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return Corpus(list(texts), list(labels), [f"class{i}" for i in range(len(label_words))])


def test_lora_finetuning_improves_over_frozen_baseline():
    torch.manual_seed(0)

    # Task A fixture (pretraining) and Task B fixture (shifted domain: a
    # disjoint set of class-signaling keywords and a different number of
    # classes). The vocabulary is built over both tasks' training text,
    # the same way a real tokenizer is built to cover the vocabulary of
    # everything it will later see, pretraining and fine-tuning alike.
    task_a = _fixture_corpus([["alpha", "aleph"], ["bravo", "beta"], ["charlie", "gamma"]], seed=1)
    train_a, val_a = train_val_split(task_a, seed=2)
    task_b = _fixture_corpus([["delta", "dawn"], ["echo", "east"]], seed=3)
    train_b, val_b = train_val_split(task_b, seed=4)
    vocab = Vocab(train_a.texts + train_b.texts)

    train_ids_a, train_labels_a = encode_corpus(train_a, vocab, MAX_LEN)
    val_ids_a, val_labels_a = encode_corpus(val_a, vocab, MAX_LEN)

    model = TextClassifier(
        vocab_size=len(vocab), num_classes=3, d_model=32, d_ff=64, n_layers=2, max_len=MAX_LEN
    )
    train(model, train_ids_a, train_labels_a, val_ids_a, val_labels_a, epochs=15, lr=3e-3, seed=0)

    train_ids_b, train_labels_b = encode_corpus(train_b, vocab, MAX_LEN)
    val_ids_b, val_labels_b = encode_corpus(val_b, vocab, MAX_LEN)

    import copy

    frozen_model = copy.deepcopy(model)
    frozen_model.set_backbone_trainable(False)
    frozen_model.replace_head(2)
    train(frozen_model, train_ids_b, train_labels_b, val_ids_b, val_labels_b, epochs=15, lr=3e-3, seed=0)
    frozen_acc = evaluate(frozen_model, val_ids_b, val_labels_b)["accuracy"]

    lora_model = copy.deepcopy(model)
    lora_model = build_lora_model(lora_model, num_classes=2, r=4, alpha=8.0)
    train(lora_model, train_ids_b, train_labels_b, val_ids_b, val_labels_b, epochs=15, lr=3e-3, seed=0)
    lora_acc = evaluate(lora_model, val_ids_b, val_labels_b)["accuracy"]

    assert lora_acc >= frozen_acc, (
        f"LoRA fine-tuning (acc={lora_acc:.3f}) should be at least as good as the "
        f"frozen baseline (acc={frozen_acc:.3f}) on the shifted fixture task."
    )
    assert lora_acc > 0.6, "LoRA fine-tuning should reach reasonable accuracy on this learnable fixture task."
