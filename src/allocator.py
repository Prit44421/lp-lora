import torch
import torch.nn as nn
from src.lora_layer import LpLoRALinear
from src.solver import lp_lora_threshold_solver

def replace_linear_with_lplora(
    model: nn.Module,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj", "query", "value", "fc1", "fc2"),
    r: int = 16,
    lora_alpha: float = 16.0,
):
    """Recursively replaces matching nn.Linear layers in any model with LpLoRALinear."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, LpLoRALinear(module, r=r, lora_alpha=lora_alpha))
        else:
            replace_linear_with_lplora(module, target_modules, r=r, lora_alpha=lora_alpha)
    return model


class LpLoRAAllocator:
    """
    Implements Algorithm 1 (lp-LoRA Rank Allocation via Implicit Thresholding).
    """
    def __init__(
        self,
        model: nn.Module,
        M: float = 10.0,
        p: float = 0.5,
        lam: float = 0.01,
        prune_every: int = 50,
        warmup_steps: int = 50,
        target_rank_ratio: float = None,  # e.g., 0.5 to prune 50% of total ranks
    ):
        self.model = model
        self.M = M
        self.p = p
        self.lam = lam
        self.prune_every = prune_every
        self.warmup_steps = warmup_steps
        self.target_rank_ratio = target_rank_ratio
        self.step_count = 0

        # Collect all LpLoRALinear layers in the model
        self.lora_layers = {
            name: mod for name, mod in model.named_modules() if isinstance(mod, LpLoRALinear)
        }
        self.initial_total_rank = sum(layer.r for layer in self.lora_layers.values())

    @torch.no_grad()
    def step(self):
        """
        Call this right after loss.backward() and BEFORE optimizer.step().
        """
        self.step_count += 1

        # 1. Always update smoothed gradients \bar{G}_A and \bar{G}_B
        for layer in self.lora_layers.values():
            layer.update_gradient_ema()

        # 2. Check if current step is a designated pruning step
        if self.step_count >= self.warmup_steps and (self.step_count % self.prune_every == 0):
            self.prune()

    @torch.no_grad()
    def prune(self):
        """Collects all (s_i, t_i) across the model and runs the 2D solver to prune weak bricks."""
        s_list, t_list, active_list = [], [], []
        for layer in self.lora_layers.values():
            s, t = layer.get_nominal_norms(M=self.M)
            s_list.append(s)
            t_list.append(t)
            active_list.append(layer.rank_mask > 0)

        all_s = torch.cat(s_list)
        all_t = torch.cat(t_list)
        all_active = torch.cat(active_list)

        # Determine lambda (either fixed or binary-searched to match target_rank_ratio)
        lam_to_use = self.lam
        if self.target_rank_ratio is not None:
            target_k = max(1, int(self.initial_total_rank * self.target_rank_ratio))
            lam_to_use = self._find_lambda_for_budget(all_s, all_t, all_active, target_k)

        # Solve 2D subproblem for every brick in the entire neural network at once
        keep_mask, _, _ = lp_lora_threshold_solver(
            all_s, all_t, M=self.M, lam=lam_to_use, p=self.p
        )
        keep_mask = keep_mask & all_active

        # Distribute updated masks back to each layer
        idx = 0
        for layer in self.lora_layers.values():
            r = layer.r
            layer.apply_pruning_mask(keep_mask[idx : idx + r])
            idx += r

    @torch.no_grad()
    def _find_lambda_for_budget(self, all_s, all_t, all_active, target_k: int) -> float:
        """Binary searches for the exact \lambda that leaves `target_k` bricks alive."""
        low, high = 0.0, 10.0
        # Expand upper bound if needed
        for _ in range(10):
            keep, _, _ = lp_lora_threshold_solver(all_s, all_t, M=self.M, lam=high, p=self.p)
            if (keep & all_active).sum().item() <= target_k:
                break
            high *= 2.0

        # 20 steps of binary search on lambda
        for _ in range(20):
            mid = 0.5 * (low + high)
            keep, _, _ = lp_lora_threshold_solver(all_s, all_t, M=self.M, lam=mid, p=self.p)
            alive = (keep & all_active).sum().item()
            if alive > target_k:
                low = mid  # Need higher tax to kill more bricks
            else:
                high = mid
        return high

    def get_rank_summary(self) -> dict:
        """Returns a dictionary of {layer_name: effective_rank}."""
        return {name: layer.effective_rank for name, layer in self.lora_layers.items()}