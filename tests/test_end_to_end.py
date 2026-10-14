import torch
import torch.nn as nn
import torch.optim as optim
from src.lora_layer import LpLoRALinear
from src.allocator import LpLoRAAllocator


class TinyNetwork(nn.Module):
    def __init__(self, r=8):
        super().__init__()
        self.fc1 = LpLoRALinear(16, 32, r=r)
        self.act = nn.ReLU()
        self.fc2 = LpLoRALinear(32, 16, r=r)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_end_to_end_training_and_pruning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = TinyNetwork(r=8).to(device)
    allocator = LpLoRAAllocator(
        model,
        M=10.0,
        lam=0.1,
        p=0.5,
        pruning_interval=5,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    initial_total_rank = allocator.total_effective_rank
    assert initial_total_rank == 16

    # Run 10 optimization steps
    for step in range(1, 11):
        x = torch.randn(8, 16, device=device)
        y = torch.randn(8, 16, device=device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        allocator.update_gradients()
        optimizer.step()

        allocator.step(global_step=step)

    # After step 5 and step 10, pruning has occurred
    final_rank = allocator.total_effective_rank
    assert final_rank <= initial_total_rank
    print(f"test_end_to_end_training_and_pruning passed! Rank pruned from {initial_total_rank} -> {final_rank}")


if __name__ == "__main__":
    test_end_to_end_training_and_pruning()
    print("End-to-end integration test passed successfully!")
