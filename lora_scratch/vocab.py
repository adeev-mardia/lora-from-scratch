"""A minimal whitespace/punctuation tokenizer and vocabulary built from the
training corpus itself (no external tokenizer or pretrained embeddings)."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z']+|[.,!?;]")

PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class Vocab:
    def __init__(self, texts: list[str], min_freq: int = 1, max_size: int = 20000):
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        tokens = [tok for tok, freq in counter.most_common(max_size) if freq >= min_freq]
        self.itos = [PAD, UNK] + tokens
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(text)]
        ids = ids[:max_len]
        ids = ids + [self.stoi[PAD]] * (max_len - len(ids))
        return ids
