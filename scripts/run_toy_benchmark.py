"""
Phase 4: Toy Benchmark & Verification Script for lp-LoRA.

This script proves the automatic rank allocation capability of lp-LoRA on a synthetic
task where the ground-truth adaptation has a low intrinsic rank (e.g. true rank = 2).
The student adapter starts with an over-parameterized rank (e.g. r = 16).
lp-LoRA automatically detects and prunes the redundant ranks down to the true rank.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.lora_layer import LpLoRALinear
from src.allocator import LpLoRAAllocator


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticLowRankTask:
    """
    Synthetic dataset where y = (W_base + B* @ A*) x + noise.
    The true adaptation rank r_star is small (e.g. 2).
    """

    def __init__(self, in_features: int = 64, out_features: int = 64, r_star: int = 2, device: str = "cpu"):
        self.in_features = in_features
        self.out_features = out_features
        self.r_star = r_star
        self.device = device

        # Fixed base weight
        self.w_base = torch.randn(out_features, in_features, device=device) * 0.1

        # Ground truth low-rank update (rank = r_star)
        a_star = torch.randn(r_star, in_features, device=device)
        b_star = torch.randn(out_features, r_star, device=device)
        self.delta_w = (b_star @ a_star) * (1.0 / r_star)

        self.w_target = self.w_base + self.delta_w

    def sample_batch(self, batch_size: int = 64):
        x = torch.randn(batch_size, self.in_features, device=self.device)
        y = x @ self.w_target.t() + 0.01 * torch.randn(batch_size, self.out_features, device=self.device)
        return x, y


def run_benchmark(
    in_features: int = 64,
    out_features: int = 64,
    r_star: int = 2,
    init_r: int = 16,
    num_steps: int = 1000,
    prune_interval: int = 200,
    lr: float = 3e-3,
    lam: float = 0.05,
    M: float = 10.0,
    p: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(42)
    print(f"=== Running lp-LoRA Toy Benchmark on {device} ===")
    print(f"Configuration: in={in_features}, out={out_features}, true_rank={r_star}, initial_lora_rank={init_r}")
    print(f"Prune Interval: every {prune_interval} steps | lambda={lam}, M={M}, p={p}\n")

    task = SyntheticLowRankTask(in_features=in_features, out_features=out_features, r_star=r_star, device=device)

    # Base layer initialized with task's base weight (frozen)
    base_linear = nn.Linear(in_features, out_features, bias=False).to(device)
    base_linear.weight.data.copy_(task.w_base)

    # Wrap with LpLoRALinear with initial rank = init_r
    model = LpLoRALinear.from_linear(base_linear, r=init_r, lora_alpha=float(init_r)).to(device)

    # lp-LoRA Allocator
    allocator = LpLoRAAllocator(
        model,
        M=M,
        lam=lam,
        p=p,
        pruning_interval=prune_interval,
        beta=0.9,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"{'Step':>6} | {'MSE Loss':>12} | {'Active Rank':>12} | {'Action':>18}")
    print("-" * 56)

    for step in range(1, num_steps + 1):
        x, y = task.sample_batch(batch_size=64)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # 1. Update smoothed gradients
        allocator.update_gradients()

        optimizer.step()

        # 2. Check for periodic pruning
        if allocator.is_pruning_step(step):
            rank_before = allocator.total_effective_rank
            allocator.prune()
            rank_after = allocator.total_effective_rank
            action_str = f"Pruned {rank_before} -> {rank_after}"
            print(f"{step:6d} | {loss.item():12.6f} | {rank_after:12d} | {action_str:>18}")
        elif step % 100 == 0:
            print(f"{step:6d} | {loss.item():12.6f} | {allocator.total_effective_rank:12d} | {'Training...':>18}")

    print("-" * 56)
    final_rank = allocator.total_effective_rank
    print(f"Benchmark Complete! Ground-truth rank: {r_star} | lp-LoRA final rank: {final_rank}")
    print(f"Redundant ranks eliminated: {init_r - final_rank} / {init_r - r_star}")
    return final_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--init_r", type=int, default=16)
    parser.add_argument("--true_r", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.05)
    args = parser.parse_args()

    run_benchmark(num_steps=args.steps, init_r=args.init_r, r_star=args.true_r, lam=args.lam)
