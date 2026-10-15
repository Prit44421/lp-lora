import torch
import torch.nn as nn
from src.lora_layer import LpLoRALinear
from src.solver import lp_lora_threshold_solver

def replace_linear_with_lplora(
    model: nn.Module,
    target_modules=("query_proj", "key_proj", "value_proj", "dense", "q_proj", "k_proj", "v_proj", "o_proj"),
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
):
    """Recursively replaces matching nn.Linear layers in any model with LpLoRALinear."""
    for name, module in list(model.named_children()):
        # Skip classification heads
        if "classifier" in name or "pooler" in name or "score" in name:
            continue
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(
                model,
                name,
                LpLoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            )
        else:
            replace_linear_with_lplora(
                module, target_modules, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
    return model


class LpLoRAAllocator:
    """
    Implements Algorithm 1 (lp-LoRA Rank Allocation via Implicit Thresholding)
    with gradual warmup-to-final budget scheduling.
    """
    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        M: float = 10.0,
        p: float = 0.5,
        prune_every: int = 50,
        init_warmup_ratio: float = 0.1,
        final_warmup_ratio: float = 0.7,
        target_rank_ratio: float = 0.5,
    ):
        self.model = model
        self.total_steps = total_steps
        self.M = M
        self.p = p
        self.prune_every = prune_every
        self.init_warmup = int(total_steps * init_warmup_ratio)
        self.final_warmup = int(total_steps * final_warmup_ratio)
        self.target_rank_ratio = target_rank_ratio
        self.step_count = 0

        self.lora_layers = {
            name: mod for name, mod in model.named_modules() if isinstance(mod, LpLoRALinear)
        }
        self.initial_total_rank = sum(layer.r for layer in self.lora_layers.values())

    def get_current_target_ratio(self) -> float:
        """Gradually schedules the rank budget from 1.0 down to target_rank_ratio."""
        if self.step_count < self.init_warmup:
            return 1.0
        if self.step_count >= self.final_warmup:
            return self.target_rank_ratio
        # Cubic schedule (standard in AdaLoRA / PEFT rank allocation)
        progress = (self.step_count - self.init_warmup) / max(1, (self.final_warmup - self.init_warmup))
        current_ratio = self.target_rank_ratio + (1.0 - self.target_rank_ratio) * ((1.0 - progress) ** 3)
        return current_ratio

    @torch.no_grad()
    def step(self):
        """Follows Algorithm 1: aggregate gradients every step, run thresholding on pruning steps."""
        self.step_count += 1

        for layer in self.lora_layers.values():
            layer.update_gradient_ema()

        if (
            self.init_warmup <= self.step_count <= self.final_warmup + self.prune_every
            and self.step_count % self.prune_every == 0
        ):
            self.prune()

    @torch.no_grad()
    def prune(self):
        s_list, t_list, active_list = [], [], []
        for layer in self.lora_layers.values():
            s, t = layer.get_nominal_norms(M=self.M)
            s_list.append(s)
            t_list.append(t)
            active_list.append(layer.rank_mask > 0)

        all_s = torch.cat(s_list)
        all_t = torch.cat(t_list)
        all_active = torch.cat(active_list)

        curr_ratio = self.get_current_target_ratio()
        target_k = max(1, int(round(self.initial_total_rank * curr_ratio)))
        current_alive = int(all_active.sum().item())

        if current_alive <= target_k:
            return

        lam_to_use = self._find_lambda_for_budget(all_s, all_t, all_active, target_k)
        keep_mask, _, _ = lp_lora_threshold_solver(
            all_s, all_t, M=self.M, lam=lam_to_use, p=self.p
        )
        keep_mask = keep_mask & all_active

        idx = 0
        for layer in self.lora_layers.values():
            r = layer.r
            layer.apply_pruning_mask(keep_mask[idx : idx + r])
            idx += r

    @torch.no_grad()
    def _find_lambda_for_budget(self, all_s, all_t, all_active, target_k: int) -> float:
        low, high = 0.0, 1e-2
        for _ in range(25):
            keep, _, _ = lp_lora_threshold_solver(all_s, all_t, M=self.M, lam=high, p=self.p)
            if (keep & all_active).sum().item() <= target_k:
                break
            high *= 2.0

        for _ in range(25):
            mid = 0.5 * (low + high)
            keep, _, _ = lp_lora_threshold_solver(all_s, all_t, M=self.M, lam=mid, p=self.p)
            alive = (keep & all_active).sum().item()
            if alive > target_k:
                low = mid
            else:
                high = mid
        return high

    def get_rank_summary(self) -> dict:
        return {name: layer.effective_rank for name, layer in self.lora_layers.items()}