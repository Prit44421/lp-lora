import torch
import torch.nn as nn
from src.lora_layer import LpLoRALinear
from src.allocator import LpLoRAAllocator


class TinyMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=8, r=8):
        super().__init__()
        self.fc1 = LpLoRALinear(in_dim, hidden_dim, r=r)
        self.act = nn.ReLU()
        self.fc2 = LpLoRALinear(hidden_dim, out_dim, r=r)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_allocator_discovery():
    model = TinyMLP(r=8)
    allocator = LpLoRAAllocator(model, M=10.0, lam=0.01)

    assert len(allocator.layers) == 2
    assert allocator.initial_total_rank == 16
    assert allocator.total_effective_rank == 16
    print("test_allocator_discovery passed!")


def test_allocator_gradient_tracking():
    model = TinyMLP(r=8)
    allocator = LpLoRAAllocator(model, M=10.0, lam=0.01, beta=0.9)

    x = torch.randn(4, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()

    allocator.update_gradients()

    for _, layer in allocator.layers:
        assert not torch.allclose(layer.grad_mean_a, torch.zeros_like(layer.grad_mean_a))
        assert not torch.allclose(layer.grad_mean_b, torch.zeros_like(layer.grad_mean_b))
    print("test_allocator_gradient_tracking passed!")


def test_allocator_prune_fixed_lambda():
    model = TinyMLP(r=8)
    allocator = LpLoRAAllocator(model, M=10.0, lam=0.1, beta=0.0)

    # Simulate gradient update
    x = torch.randn(4, 16)
    model(x).sum().backward()
    allocator.update_gradients()

    # Prune with lambda
    dist = allocator.prune(lam=0.2)
    assert allocator.total_effective_rank <= allocator.initial_total_rank
    assert isinstance(dist, dict)
    assert len(dist) == 2
    print(f"test_allocator_prune_fixed_lambda passed! Remaining rank: {allocator.total_effective_rank}/16")


def test_allocator_target_budget_search():
    model = TinyMLP(r=8)  # Total 16 ranks
    target_rank = 8  # 50% budget
    allocator = LpLoRAAllocator(model, M=10.0, target_total_rank=target_rank, beta=0.0)

    # Simulate backward pass
    x = torch.randn(4, 16)
    model(x).sum().backward()
    allocator.update_gradients()

    dist = allocator.prune()
    assert allocator.total_effective_rank <= target_rank + 1  # within search tolerance
    print(f"test_allocator_target_budget_search passed! Hit target budget: {allocator.total_effective_rank} (target was {target_rank})")


if __name__ == "__main__":
    test_allocator_discovery()
    test_allocator_gradient_tracking()
    test_allocator_prune_fixed_lambda()
    test_allocator_target_budget_search()
    print("All Phase 3 allocator tests passed successfully!")
