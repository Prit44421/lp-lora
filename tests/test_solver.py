import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.solver import lp_lora_threshold_solver

device = "cuda" if torch.cuda.is_available() else "cpu"
s = torch.tensor([0.02, 0.05, 0.15, 0.40, 1.20], device=device)
t = torch.tensor([0.03, 0.04, 0.20, 0.50, 1.00], device=device)

keep, alpha, beta = lp_lora_threshold_solver(s, t, M=10.0, lam=0.1, p=0.5)

for i in range(len(s)):
    print(f"Brick {i+1} (s={s[i]:.2f}, t={t[i]:.2f}) -> keep={bool(keep[i])}, alpha={alpha[i]:.3f}, beta={beta[i]:.3f}")