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
