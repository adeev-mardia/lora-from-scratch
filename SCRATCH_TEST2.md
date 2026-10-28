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
