"""Verify the core LoRA mechanics: no-op initialization and correct
gradient flow (only A/B get gradients, the frozen base weight does not)."""

import torch
import torch.nn as nn

from lora_scratch.lora import LoRALinear, apply_lora, count_total_params, count_trainable_params


def test_lora_is_noop_at_initialization():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    expected = base(x)

    lora_layer = LoRALinear(base, r=4, alpha=8.0)
    actual = lora_layer(x)

    assert torch.allclose(actual, expected, atol=1e-6), (
        "LoRA-adapted output must exactly match the frozen base layer's "
        "output at initialization, since B is initialized to zero."
    )


def test_delta_weight_is_zero_at_init_and_nonzero_after_training_B():
    torch.manual_seed(1)
    base = nn.Linear(10, 6)
    lora_layer = LoRALinear(base, r=3, alpha=6.0)

    delta = lora_layer.delta_weight()
    assert torch.allclose(delta, torch.zeros_like(delta))

    with torch.no_grad():
        lora_layer.lora_B.add_(0.1)
    delta_after = lora_layer.delta_weight()
    assert not torch.allclose(delta_after, torch.zeros_like(delta_after))


def test_merge_into_linear_matches_adapter_forward():
    torch.manual_seed(2)
    base = nn.Linear(12, 5)
    lora_layer = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora_layer.lora_B.add_(torch.randn_like(lora_layer.lora_B) * 0.05)

    x = torch.randn(7, 12)
    out_adapter = lora_layer(x)
    merged = lora_layer.merge_into_linear()
    out_merged = merged(x)

    assert torch.allclose(out_adapter, out_merged, atol=1e-5)


def test_only_lora_params_receive_gradients():
    """Only A and B ever receive gradients; the frozen base weight/bias
    never do. Note a real mathematical subtlety of LoRA's init scheme:
    since B starts at exactly zero, d(loss)/dA is also exactly zero on the
    very first backward pass (the chain rule through B @ A makes dA's
    gradient proportional to B). B itself does get a nonzero gradient
    immediately, since d(loss)/dB flows through A, which is randomly
    initialized and nonzero. Once an optimizer step makes B nonzero, A
    starts receiving nonzero gradients too, which this test also checks."""
    torch.manual_seed(3)
    base = nn.Linear(16, 8)
    lora_layer = LoRALinear(base, r=4, alpha=8.0)

    x = torch.randn(5, 16, requires_grad=False)
    target = torch.randn(5, 8)
    out = lora_layer(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    assert lora_layer.weight.grad is None, "Frozen base weight must not receive a gradient."
    if lora_layer.bias is not None:
        assert lora_layer.bias.grad is None, "Frozen base bias must not receive a gradient."
    assert lora_layer.lora_A.grad is not None
    assert lora_layer.lora_B.grad is not None
    # At init, B == 0 so dL/dA is exactly zero by the chain rule through B;
    # dL/dB is nonzero since it flows through the randomly-initialized A.
    assert torch.all(lora_layer.lora_A.grad == 0)
    assert torch.any(lora_layer.lora_B.grad != 0)

    optimizer = torch.optim.SGD(
        [p for p in lora_layer.parameters() if p.requires_grad], lr=0.5
    )
    optimizer.step()
    optimizer.zero_grad()

    out2 = lora_layer(x)
    loss2 = torch.nn.functional.mse_loss(out2, target)
    loss2.backward()

    assert lora_layer.weight.grad is None
    assert torch.any(lora_layer.lora_A.grad != 0), (
        "Once B is nonzero after one optimizer step, A should start receiving gradients."
    )
    assert torch.any(lora_layer.lora_B.grad != 0)


def test_weight_requires_grad_is_false():
    base = nn.Linear(6, 6)
    lora_layer = LoRALinear(base, r=2, alpha=4.0)
    assert lora_layer.weight.requires_grad is False
    assert lora_layer.lora_A.requires_grad is True
    assert lora_layer.lora_B.requires_grad is True


def test_optimizer_step_does_not_change_frozen_weight():
    torch.manual_seed(4)
    base = nn.Linear(8, 4)
    lora_layer = LoRALinear(base, r=2, alpha=4.0)
    original_weight = lora_layer.weight.clone()

    optimizer = torch.optim.SGD(
        [p for p in lora_layer.parameters() if p.requires_grad], lr=0.1
    )
    x = torch.randn(6, 8)
    target = torch.randn(6, 4)
    for _ in range(5):
        optimizer.zero_grad()
        out = lora_layer(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

    assert torch.equal(lora_layer.weight, original_weight)
    assert not torch.equal(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B))


def test_apply_lora_replaces_named_linear():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(10, 10)

    block = Block()
    apply_lora(block, ["proj"], r=2, alpha=4.0)
    assert isinstance(block.proj, LoRALinear)


def test_param_counting_helpers():
    base = nn.Linear(100, 50)
    lora_layer = LoRALinear(base, r=5, alpha=10.0)
    total = count_total_params(lora_layer)
    trainable = count_trainable_params(lora_layer)

    expected_total = 100 * 50 + 50 + 5 * 100 + 50 * 5
    expected_trainable = 5 * 100 + 50 * 5

    assert total == expected_total
    assert trainable == expected_trainable
    assert trainable < total
