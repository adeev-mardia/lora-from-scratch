# lora-from-scratch

A from-scratch implementation of **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, with a real, measured three-way comparison against full fine-tuning and a frozen baseline.

Everything here -- the transformer-style classifier, the LoRA adapter math, the two training corpora, and the training/evaluation loops -- is implemented directly with raw `torch.nn.Linear` layers and tensor operations. No `peft` library, no pretrained checkpoint download, no hub dataset. The point of the project is to show *why* LoRA works and *how much* it saves, by building the mechanism itself and measuring it honestly on a real (if small) transfer-learning problem.

## What LoRA actually is

Fine-tuning a large model by updating every weight is expensive: for a linear layer with weight matrix `W` of shape `(d_out, d_in)`, a full update touches `d_out * d_in` parameters. LoRA's observation is that the *update* a fine-tuning run needs to make to `W`, `Delta W`, tends to have much lower intrinsic rank than `W` itself -- the task-specific adjustment lives in a small subspace.

So instead of learning a full-rank `Delta W`, LoRA factors it as the product of two small matrices:

```
Delta W = B * A          where A is (r, d_in), B is (d_out, r), and r << min(d_in, d_out)
```

The adapted forward pass becomes:

```
y = x * W^T + b + (alpha / r) * x * A^T * B^T
```

`W` and `b` are **frozen** (`requires_grad=False`); only `A` and `B` are trained. The `alpha / r` scaling factor controls how strongly the adapter contributes relative to its rank, so `r` can be changed without re-tuning other hyperparameters.

Two initialization details make this work correctly:

- **`A` is initialized randomly** (Kaiming-uniform, same scheme `nn.Linear` itself uses).
- **`B` is initialized to exactly zero.**

Since `Delta W = B*A` and `B = 0`, the adapter computes a mathematically exact no-op at initialization: `LoRALinear(x) == Linear(x)` for every input, bit for bit (see `tests/test_lora.py::test_lora_is_noop_at_initialization`). Training can then only ever *add* new behavior on top of the frozen model, never destabilize it from a random starting point.

A subtler consequence of this init scheme, also covered by the tests: on the very first backward pass, `dL/dA` is exactly zero (it flows through `B`, which is zero), while `dL/dB` is already nonzero (it flows through the randomly-initialized `A`). After the first optimizer step moves `B` away from zero, `A` starts receiving real gradients too. Both matrices converge; the implementation doesn't special-case this, it falls straight out of the chain rule.

The implementation lives in `lora_scratch/lora.py`, in the `LoRALinear` class -- it wraps an existing `nn.Linear`, clones and freezes its weight/bias, and adds the trainable `lora_A` / `lora_B` parameters alongside it.

## The model and the task

**Base model** (`lora_scratch/model.py`, `TextClassifier`): a small transformer-style text classifier built entirely from `nn.Embedding` and `nn.Linear` -- token + position embeddings, two transformer blocks (each a single-head self-attention module with separate `q_proj`/`k_proj`/`v_proj`/`out_proj` linears, plus a two-layer feedforward block), masked mean pooling, and a linear classification head. Roughly 340K parameters total at the default size (`d_model=128`, `d_ff=256`, 2 layers).

**Task A -- pretraining (topic classification)**: a 5-way news/article-style topic classification task (sports, technology, finance, cooking, politics). The corpus (`lora_scratch/datasets.py::generate_topic_corpus`) is built from hand-written sentence templates and vocabulary banks per topic, combined programmatically into ~1,100 distinct sentences. The base model is trained from scratch on this task until it reaches strong accuracy.

**Task B -- fine-tuning (review sentiment, a genuine domain shift)**: a 3-way product-review sentiment task (negative/neutral/positive), written in a completely different register -- first-person reviews about everyday products rather than third-person news reporting -- with a different number of classes and largely different vocabulary. This is the actual transfer problem: can the Task-A-pretrained backbone be adapted to Task B cheaply?

Because the label space changes (5 classes -> 3 classes), every method gets a freshly initialized classification head; what differs between methods is which *backbone* parameters, if any, get to move.

LoRA adapters are attached to five linear layers per transformer block: `q_proj`, `v_proj`, `out_proj` (attention) and `ffn_in`, `ffn_out` (feedforward) -- see `TextClassifier.lora_target_modules`.

## The three-way comparison

All three methods start from the exact same Task-A-pretrained backbone and are fine-tuned on Task B with identical data, optimizer, learning rate, and number of epochs -- the only thing that changes is which parameters are trainable:

| Method | Backbone | Head | What's trainable |
|---|---|---|---|
| **Frozen baseline** | frozen | trainable | only the new head |
| **Full fine-tune** | trainable | trainable | everything |
| **LoRA (from scratch)** | frozen + low-rank adapters (`r=8`, `alpha=16`) | trainable | adapters + head |

### Real results from an actual run

(ran as: python -m lora_scratch.cli --pretrain-epochs 15 --finetune-epochs 15, seed 0 -- the CLI defaults)

Task A pretraining reached **100% validation accuracy** before any fine-tuning began (vocabulary size 551, built from the training corpus itself).

| Method | Val. Accuracy (Task B) | Val. Loss (Task B) | Trainable Params | Total Params |
|---|---|---|---|---|
| Frozen baseline | 0.4924 | 1.0182 | 387 | 339,203 |
| Full fine-tune | 1.0000 | 0.0004 | 339,203 | 339,203 |
| LoRA (from scratch) | 0.9924 | 0.0127 | 24,963 | 363,779 |

**LoRA comes within 0.76 points of full fine-tuning's accuracy on Task B while training 92.64% fewer parameters** (24,963 vs. 339,203 trainable parameters) -- and both comfortably beat the frozen baseline, which sits near chance-plus-a-bit (49.24%, versus 33% chance on 3 classes) because a frozen backbone's representations, learned for a disjoint 5-way topic task, aren't well suited to 3-way sentiment without being allowed to adapt. Note that LoRA's *total* parameter count (363,779) is slightly larger than the full model's (339,203): the adapters are additive, not a replacement, which is exactly why the trainable-parameter comparison -- not the total-parameter comparison -- is LoRA's real selling point.

The frozen baseline's weak transfer is itself informative: it shows the backbone genuinely needs to move to solve Task B, so LoRA's near-full-fine-tuning accuracy with a small fraction of the trainable parameters is a real, non-trivial result rather than an artifact of an easy task.

Re-running with different seeds or epoch counts (via `--seed`, `--pretrain-epochs`, `--finetune-epochs`) shifts the exact numbers but preserves the ordering: `frozen < lora ~= full_finetune` on accuracy, and `lora << full_finetune` on trainable parameters. The full run above (pretraining + all three fine-tuning regimes, CPU only) takes a little over 7 minutes on a single CPU core.

## Project layout

```
lora_scratch/
  lora.py       LoRALinear and the from-scratch low-rank adapter math
  model.py      TextClassifier -- embeddings, transformer blocks, head
  datasets.py   Task A / Task B corpus generation
  vocab.py      Tokenizer and vocabulary built from the training corpus
  engine.py     Train/eval loops, LoRA model construction, param counting
  cli.py        Command-line entry point running the full comparison
tests/
  test_lora.py    LoRA no-op init, gradient flow, param counting
  test_params.py  Trainable-parameter-reduction correctness
  test_e2e.py     End-to-end: LoRA fine-tuning beats the frozen baseline
```

## Reproducing the results

Install with `pip install -e .`, then run as a plain Python module invocation: python -m lora_scratch.cli --seed 0

Useful flags: `--lora-r` / `--lora-alpha` (adapter rank and scaling), `--verbose` (per-epoch loss/accuracy), `--json-out metrics.json` (dump the final numbers).

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 13 tests pass. They check, in order of how central they are to the project:

- The LoRA forward pass is an exact no-op at initialization (`B = 0`).
- Only `lora_A`/`lora_B` ever receive gradients; the frozen base weight's gradient stays `None` throughout training, including the real quirk that `A`'s gradient is exactly zero on the very first step (because it flows through `B`, which starts at zero) and becomes nonzero afterward.
- An optimizer step never changes the frozen base weight.
- Merging the adapter into a plain `nn.Linear` reproduces the adapter's forward pass exactly.
- The trainable-parameter-count reduction calculation is correct for the frozen, full, and LoRA configurations.
- An end-to-end fixture task where LoRA fine-tuning measurably outperforms the frozen baseline.

## Limitations

This is a small, from-scratch model (~340K parameters) trained on a constructed corpus of a few thousand sentences, not a pretrained large language model fine-tuned on a production dataset. That's a deliberate scope choice to keep the whole pipeline -- pretraining, both fine-tuning regimes, and every test -- runnable from a clean environment in a few minutes with no external downloads, so the LoRA mechanism itself, not infrastructure, is what's being demonstrated.

The core math and the qualitative conclusion generalize directly to large pretrained transformers: in production LoRA setups (e.g., fine-tuning a multi-billion-parameter LLM), the trainable-parameter savings are typically far larger than the ~93% measured here, because the frozen base model's parameter count grows much faster than the handful of low-rank adapters needed to adapt it -- which is precisely why LoRA is the standard parameter-efficient fine-tuning technique for models at that scale.
