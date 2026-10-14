import torch
import torch.nn as nn
from src.lora_layer import LpLoRALinear
from src.solver import lp_lora_threshold_solver


def test_initialization():
    in_features = 32
    out_features = 64
    r = 8
    layer = LpLoRALinear(in_features, out_features, r=r, lora_alpha=16.0)

    # Check base layer is frozen
    for p in layer.base_layer.parameters():
        assert not p.requires_grad, "Base layer parameters must be frozen"

    # Check LoRA parameters are trainable
    assert layer.weight_a.requires_grad
    assert layer.weight_b.requires_grad
    assert layer.weight_a.shape == (r, in_features)
    assert layer.weight_b.shape == (out_features, r)

    # Effective rank is initially r
    assert layer.effective_rank == r

    # Output initially equals base layer output because B is initialized to 0
    x = torch.randn(4, in_features)
    out = layer(x)
    base_out = layer.base_layer(x)
    assert torch.allclose(out, base_out, atol=1e-6), "Initial output must match base layer output"
    print("test_initialization passed!")


def test_forward_backward():
    in_features = 16
    out_features = 16
    r = 4
    layer = LpLoRALinear(in_features, out_features, r=r)

    x = torch.randn(2, in_features)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Base layer must not receive gradients
    for p in layer.base_layer.parameters():
        assert p.grad is None, "Base layer must not accumulate gradients"

    # LoRA params receive gradients
    assert layer.weight_a.grad is not None
    assert layer.weight_b.grad is not None

    # Test gradient accumulator
    layer.update_gradient_accumulator(beta=0.9)
    assert not torch.allclose(layer.grad_mean_a, torch.zeros_like(layer.grad_mean_a))
    assert not torch.allclose(layer.grad_mean_b, torch.zeros_like(layer.grad_mean_b))
    print("test_forward_backward passed!")


def test_component_norms_and_pruning():
    in_features = 16
    out_features = 16
    r = 4
    layer = LpLoRALinear(in_features, out_features, r=r)

    # Populate dummy gradients and update accumulator
    x = torch.randn(2, in_features)
    out = layer(x)
    out.sum().backward()
    layer.update_gradient_accumulator(beta=0.0)

    # Get component norms s and t
    s, t = layer.get_component_norms(M=10.0)
    assert s.shape == (r,)
    assert t.shape == (r,)
    assert (s > 0).all()
    assert (t > 0).all()

    # Run solver on these norms
    keep_mask, alpha, beta = lp_lora_threshold_solver(s, t, M=10.0, lam=0.05, p=0.5)

    # Apply the mask
    initial_rank = layer.effective_rank
    layer.apply_mask(keep_mask)

    assert layer.effective_rank == int(keep_mask.sum().item())
    assert layer.effective_rank <= initial_rank

    # Check pruned rows of A and cols of B are exact zero
    pruned_indices = (~keep_mask).nonzero(as_tuple=True)[0]
    for idx in pruned_indices:
        assert torch.all(layer.weight_a[idx] == 0.0)
        assert torch.all(layer.weight_b[:, idx] == 0.0)

    # Forward pass after pruning works seamlessly
    out_after = layer(x)
    assert out_after.shape == (2, out_features)
    print(f"test_component_norms_and_pruning passed! (Rank pruned from {initial_rank} -> {layer.effective_rank})")


def test_from_linear():
    base = nn.Linear(20, 40, bias=True)
    lora_layer = LpLoRALinear.from_linear(base, r=8, lora_alpha=16.0)

    assert lora_layer.in_features == 20
    assert lora_layer.out_features == 40
    assert lora_layer.r == 8
    assert lora_layer.base_layer is base

    x = torch.randn(3, 20)
    assert torch.allclose(lora_layer(x), base(x))
    print("test_from_linear passed!")


if __name__ == "__main__":
    test_initialization()
    test_forward_backward()
    test_component_norms_and_pruning()
    test_from_linear()
    print("All Phase 2 tests passed successfully!")

