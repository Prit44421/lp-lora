import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.allocator import replace_linear_with_lplora, LpLoRAAllocator

class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        h = torch.relu(self.q_proj(x) + self.v_proj(x))
        return x + self.fc2(torch.relu(self.fc1(h)))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training with lp-LoRA on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    torch.manual_seed(42)
    d_model = 512
    model = ToyTransformerBlock(d_model=d_model).to(device)

    # Create a synthetic teacher update where:
    # - q_proj needs high rank (rank 12)
    # - v_proj needs almost no adaptation (rank 2)
    # - fc1 / fc2 need medium rank (rank 6)
    teacher_q = torch.randn(d_model, 12, device=device) @ torch.randn(12, d_model, device=device) * 0.5
    teacher_v = torch.randn(d_model, 2, device=device) @ torch.randn(2, d_model, device=device) * 0.05

    # Wrap model with LpLoRALinear (initial rank r = 16 everywhere -> 64 total bricks)
    model = replace_linear_with_lplora(model, target_modules=("q_proj", "v_proj", "fc1", "fc2"), r=16)

    # Only optimize trainable LoRA parameters (A and B)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-3)

    # Initialize the lp-LoRA Allocator (target: prune 50% of total ranks -> from 64 down to 32)
    total_steps = 200
    allocator = LpLoRAAllocator(
        model,
        total_steps=total_steps,
        M=10.0,
        p=0.5,
        prune_every=50,
        init_warmup_ratio=0.1,
        final_warmup_ratio=0.7,
        target_rank_ratio=0.5,
    )

    print(f"Initial Ranks per Layer: {allocator.get_rank_summary()} (Total: {allocator.initial_total_rank})")

    # Training loop
    for step in range(1, 201):
        x = torch.randn(64, d_model, device=device)
        # Target output has heavy adaptation in q_proj, weak in v_proj
        with torch.no_grad():
            target = x + (x @ teacher_q.T) + (x @ teacher_v.T)

        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()

        # --- lp-LoRA Step (updates gradient EMA & prunes on pruning steps) ---
        allocator.step()

        optimizer.step()

        if step % 50 == 0:
            ranks = allocator.get_rank_summary()
            total_alive = sum(ranks.values())
            print(f"Step {step:3d} | Loss: {loss.item():.5f} | Ranks: {ranks} (Total Alive: {total_alive}/64)")

if __name__ == "__main__":
    main()