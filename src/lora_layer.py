import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LpLoRALinear(nn.Module):
    """
    Lp-LoRA Linear Layer (Phase 2):
    A drop-in replacement for nn.Linear supporting automatic rank allocation
    via \ell_p regularization.

    Decomposes the weight update \Delta W into rank-1 components:
        \Delta W = \sum_{i=1}^r B_{:, i} A_{i, :} = B A

    Tracks running smoothed gradients (\bar{G}_A, \bar{G}_B) and maintains a
    binary component mask to permanently prune rank-1 components whose energy
    is driven to zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # 1. Base Pretrained Weight (frozen)
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 2. Trainable LoRA Parameters: A in R^{r x in_features}, B in R^{out_features x r}
        self.weight_a = nn.Parameter(torch.empty(r, in_features))
        self.weight_b = nn.Parameter(torch.empty(out_features, r))

        # Dropout on adapter input
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # 3. Buffers for smoothed gradients (\bar{G}_A and \bar{G}_B) used in Algorithm 1
        self.register_buffer("grad_mean_a", torch.zeros(r, in_features))
        self.register_buffer("grad_mean_b", torch.zeros(out_features, r))

        # 4. Binary survival mask for each of the r bricks (True = active, False = pruned)
        self.register_buffer("component_mask", torch.ones(r, dtype=torch.bool))

        # Counter for gradient accumulation steps (if needed for simple averaging)
        self.register_buffer("grad_steps", torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Standard LoRA initialization:
        A ~ Kaiming uniform (or normal), B initialized to 0 so \Delta W = 0 at start.
        """
        nn.init.kaiming_uniform_(self.weight_a, a=math.sqrt(5))
        nn.init.zeros_(self.weight_b)
        self.component_mask.fill_(True)
        self.grad_mean_a.zero_()
        self.grad_mean_b.zero_()
        self.grad_steps.zero_()

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> "LpLoRALinear":
        """
        Factory method to convert an existing nn.Linear module into an LpLoRALinear module.
        """
        return cls(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=base_linear.bias is not None,
            base_layer=base_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            Output = W_pt(x) + (lora_alpha / r) * (B * diag(m) * A) x
        """
        base_out = self.base_layer(x)

        # If all components are pruned, return only base layer output
        if not self.component_mask.any():
            return base_out

        # Adapter path: x -> dropout -> A -> mask -> B -> scaling
        x_dropped = self.dropout(x)
        lora_a = F.linear(x_dropped, self.weight_a)  # shape: [..., r]

        # Zero out inactive rank components directly on intermediate activation
        lora_a = lora_a * self.component_mask.to(dtype=lora_a.dtype)

        lora_out = F.linear(lora_a, self.weight_b) * self.scaling  # shape: [..., out_features]

        return base_out + lora_out

    def update_gradient_accumulator(self, beta: float = 0.9):
        """
        Update the running smoothed gradients \bar{G}_A and \bar{G}_B.
        If beta > 0: Exponential Moving Average: \bar{G} = beta * \bar{G} + (1 - beta) * grad
        If beta == 0: Simple running average over accumulated steps.
        """
        if self.weight_a.grad is not None:
            if beta > 0.0:
                self.grad_mean_a.mul_(beta).add_(self.weight_a.grad, alpha=1.0 - beta)
            else:
                self.grad_mean_a.add_(self.weight_a.grad)

        if self.weight_b.grad is not None:
            if beta > 0.0:
                self.grad_mean_b.mul_(beta).add_(self.weight_b.grad, alpha=1.0 - beta)
            else:
                self.grad_mean_b.add_(self.weight_b.grad)

        self.grad_steps.add_(1)

    def reset_gradient_accumulator(self):
        """Reset gradient tracking buffers."""
        self.grad_mean_a.zero_()
        self.grad_mean_b.zero_()
        self.grad_steps.zero_()

    def get_nominal_matrices(self, M: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute nominal tax-free matrices \tilde{A} and \tilde{B} (Eq. 12-13):
            \tilde{A} = A - (1 / M) * \bar{G}_A
            \tilde{B} = B - (1/ M) * \bar{G}_B
        """
        # Determine the effective gradient to use:
        # If grad_steps > 0 and beta == 0 was used, normalize by grad_steps
        g_a = self.grad_mean_a
        g_b = self.grad_mean_b

        tilde_a = self.weight_a.detach() - (1.0 / M) * g_a
        tilde_b = self.weight_b.detach() - (1.0 / M) * g_b
        return tilde_a, tilde_b

    def get_component_norms(self, M: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute row norms s_i = ||\tilde{a}_i||_2 and column norms t_i = ||\tilde{b}_i||_2
        for each rank component i in {1, ..., r}.

        Returns:
            s: [r] Tensor of row norms for \tilde{A}
            t: [r] Tensor of column norms for \tilde{B}
        """
        tilde_a, tilde_b = self.get_nominal_matrices(M=M)
        # s_i: row norm of \tilde{a}_i (tilde_a is [r, in_features])
        s = torch.norm(tilde_a, p=2, dim=1)
        # t_i: col norm of \tilde{b}_i (tilde_b is [out_features, r])
        t = torch.norm(tilde_b, p=2, dim=0)
        return s, t

    def apply_mask(self, new_mask: torch.Tensor):
        """
        Permanently prune components where new_mask is False.
        Once pruned, components cannot be revived.
        Zeroes out the corresponding rows of A and columns of B,
        as well as their smoothed gradient buffers.
        """
        new_mask = new_mask.bool().to(self.component_mask.device)
        self.component_mask.logical_and_(new_mask)

        # Zero out inactive components permanently
        with torch.no_grad():
            pruned_rows = (~self.component_mask).unsqueeze(1)  # [r, 1]
            pruned_cols = (~self.component_mask).unsqueeze(0)  # [1, r]

            self.weight_a.data.masked_fill_(pruned_rows, 0.0)
            self.weight_b.data.masked_fill_(pruned_cols, 0.0)

            self.grad_mean_a.masked_fill_(pruned_rows, 0.0)
            self.grad_mean_b.masked_fill_(pruned_cols, 0.0)

            if self.weight_a.grad is not None:
                self.weight_a.grad.masked_fill_(pruned_rows, 0.0)
            if self.weight_b.grad is not None:
                self.weight_b.grad.masked_fill_(pruned_cols, 0.0)

    @property
    def effective_rank(self) -> int:
        """Returns the number of active rank components."""
        return int(self.component_mask.sum().item())

