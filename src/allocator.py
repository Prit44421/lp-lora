from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.lora_layer import LpLoRALinear
from src.solver import lp_lora_threshold_solver


class LpLoRAAllocator:
    """
    Algorithm 1: Rank Allocator & Pruning Scheduler for lp-LoRA.

    Coordinates periodic rank allocation across all LpLoRALinear layers in a model.
    1. Tracks smoothed gradients during regular training steps.
    2. Collects (s_i, t_i) norms across all adapter layers during pruning steps.
    3. Solves the global thresholding problem (either with a fixed lambda or via
       binary search over lambda to hit a target rank budget b).
    4. Applies layer-wise survival masks to prune redundant rank-one components.
    """

    def __init__(
        self,
        model: nn.Module,
        M: float = 10.0,
        lam: Optional[float] = 0.01,
        p: float = 0.5,
        target_rank_ratio: Optional[float] = None,
        target_total_rank: Optional[int] = None,
        pruning_steps: Optional[List[int]] = None,
        pruning_interval: Optional[int] = None,
        beta: float = 0.9,
    ):
        """
        Args:
            model: PyTorch model containing LpLoRALinear layers.
            M: Proximal step size constant (default 10.0 in paper).
            lam: Fixed regularization penalty lambda (if not using budget search).
            p: lp sparsity exponent 0 < p < 1 (default 0.5 in paper).
            target_rank_ratio: Target fraction of total initial rank (e.g. 0.5 for 50%).
            target_total_rank: Exact target total rank across all layers.
            pruning_steps: Explicit list of step numbers on which to execute pruning.
            pruning_interval: Periodic interval (every N steps) to execute pruning.
            beta: Momentum / smoothing factor for running gradients.
        """
        self.model = model
        self.M = M
        self.lam = lam
        self.p = p
        self.target_rank_ratio = target_rank_ratio
        self.target_total_rank = target_total_rank
        self.pruning_steps = set(pruning_steps) if pruning_steps is not None else set()
        self.pruning_interval = pruning_interval
        self.beta = beta

        # Discover all LpLoRALinear layers with their module names
        self.layers: List[Tuple[str, LpLoRALinear]] = []
        for name, module in model.named_modules():
            if isinstance(module, LpLoRALinear):
                self.layers.append((name, module))

        if not self.layers:
            raise ValueError("No LpLoRALinear layers found in the provided model.")

        self.initial_total_rank = sum(layer.r for _, layer in self.layers)

        # Resolve target rank budget if specified
        if self.target_total_rank is None and self.target_rank_ratio is not None:
            self.target_total_rank = max(1, int(round(self.initial_total_rank * self.target_rank_ratio)))

    def is_pruning_step(self, step: int) -> bool:
        """Checks if the given training step is designated as a pruning step."""
        if step in self.pruning_steps:
            return True
        if self.pruning_interval is not None and self.pruning_interval > 0:
            return (step > 0) and (step % self.pruning_interval == 0)
        return False

    def update_gradients(self):
        """
        Called after loss.backward() and before optimizer.step() at each training step
        to update smoothed gradient tracking buffers across all LpLoRALinear layers.
        """
        for _, layer in self.layers:
            layer.update_gradient_accumulator(beta=self.beta)

    def step(self, global_step: int) -> Optional[Dict[str, int]]:
        """
        Convenience hook called at the end of each training step.
        If global_step is a designated pruning step, runs pruning and returns
        the updated rank distribution. Otherwise returns None.
        """
        if self.is_pruning_step(global_step):
            return self.prune()
        return None

    def prune(
        self,
        lam: Optional[float] = None,
        target_rank: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Executes Algorithm 1 pruning inspection across all layers:
        1. Collects s and t norms from every layer.
        2. Solves thresholding problem (fixed lambda or binary search for target_rank).
        3. Applies masks to prune redundant components.
        4. Resets gradient accumulators for the next interval.

        Returns:
            Dict mapping layer names to their new effective rank.
        """
        target_rank = target_rank if target_rank is not None else self.target_total_rank
        lam = lam if lam is not None else self.lam

        # 1. Collect all s and t norms across all layers
        layer_s_list = []
        layer_t_list = []
        layer_splits = []

        for _, layer in self.layers:
            s, t = layer.get_component_norms(M=self.M)
            layer_s_list.append(s)
            layer_t_list.append(t)
            layer_splits.append(len(s))

        all_s = torch.cat(layer_s_list, dim=0)
        all_t = torch.cat(layer_t_list, dim=0)

        # 2. Determine pruning masks
        if target_rank is not None:
            # Binary search over lambda to hit target rank budget
            keep_mask, final_lam = self._binary_search_lambda(all_s, all_t, target_rank)
        else:
            if lam is None:
                raise ValueError("Either lam or target_rank/target_rank_ratio must be provided.")
            keep_mask, _, _ = lp_lora_threshold_solver(
                all_s, all_t, M=self.M, lam=lam, p=self.p
            )

        # 3. Apply masks back to individual layers
        masks_per_layer = torch.split(keep_mask, layer_splits)
        for (_, layer), mask in zip(self.layers, masks_per_layer):
            layer.apply_mask(mask)
            layer.reset_gradient_accumulator()

        return self.get_rank_distribution()

    def _binary_search_lambda(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        target_rank: int,
        max_iters: int = 25,
        tol: int = 0,
    ) -> Tuple[torch.Tensor, float]:
        """
        Finds the exact lambda that prunes down to target_rank.
        Higher lambda -> more aggressive pruning -> fewer surviving ranks.
        Monotonically decreases surviving rank as lambda increases.
        """
        # Find search bounds for lambda
        low_lam = 1e-7
        high_lam = 100.0

        # Verify low bound doesn't over-prune
        mask_low, _, _ = lp_lora_threshold_solver(s, t, M=self.M, lam=low_lam, p=self.p)
        if mask_low.sum().item() <= target_rank:
            return mask_low, low_lam

        # Expand high bound if necessary
        for _ in range(5):
            mask_high, _, _ = lp_lora_threshold_solver(s, t, M=self.M, lam=high_lam, p=self.p)
            if mask_high.sum().item() <= target_rank:
                break
            high_lam *= 10.0

        best_mask = mask_low
        best_lam = low_lam

        # Binary search for optimal lambda
        for _ in range(max_iters):
            mid_lam = 0.5 * (low_lam + high_lam)
            mask_mid, _, _ = lp_lora_threshold_solver(s, t, M=self.M, lam=mid_lam, p=self.p)
            surviving = int(mask_mid.sum().item())

            if abs(surviving - target_rank) <= tol:
                return mask_mid, mid_lam

            if surviving > target_rank:
                # Need more pruning -> increase lambda
                low_lam = mid_lam
                best_mask = mask_mid
                best_lam = mid_lam
            else:
                # Pruned too much -> decrease lambda
                high_lam = mid_lam
                best_mask = mask_mid
                best_lam = mid_lam

        return best_mask, best_lam

    def get_rank_distribution(self) -> Dict[str, int]:
        """Returns the current effective rank for each LpLoRALinear layer."""
        return {name: layer.effective_rank for name, layer in self.layers}

    @property
    def total_effective_rank(self) -> int:
        """Returns the sum of effective ranks across all adapter layers."""
        return sum(layer.effective_rank for _, layer in self.layers)
