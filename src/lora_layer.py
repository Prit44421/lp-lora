import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LpLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        ema_beta: float = 0.9,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.ema_beta = ema_beta

        # 1. Freeze the original pretrained linear layer
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # 2. Trainable LoRA matrices: A is [r, in_features], B is [out_features, r]
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty((self.out_features, r), device=device, dtype=dtype))

        # Initialize A with Kaiming Uniform, B with tiny non-zero values so gradients flow to both from step 1
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)

        # 3. Buffers for smoothed gradients (\bar{G}_A and \bar{G}_B) and active rank mask
        self.register_buffer("grad_ema_A", torch.zeros_like(self.lora_A))
        self.register_buffer("grad_ema_B", torch.zeros_like(self.lora_B))
        self.register_buffer("rank_mask", torch.ones(r, device=device, dtype=dtype))

    @property
    def effective_rank(self) -> int:
        """Returns how many rank-1 bricks are still alive."""
        return int((self.rank_mask > 0).sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard frozen output: W_pt * x
        result = self.base_layer(x)

        # LoRA branch: (x @ A^T) * mask @ B^T
        # Multiplying by rank_mask zeros out any pruned rank-1 bricks!
        lora_x = self.dropout(x.to(self.lora_A.dtype))
        after_A = F.linear(lora_x, self.lora_A) * self.rank_mask  # shape: [..., r]
        lora_out = F.linear(after_A, self.lora_B) * self.scaling  # shape: [..., out_features]

        return result + lora_out.to(result.dtype)

    @torch.no_grad()
    def update_gradient_ema(self):
        """Updates the running average of gradients (\bar{G}_A, \bar{G}_B) after loss.backward()."""
        if self.lora_A.grad is not None and self.lora_B.grad is not None:
            # Safety guard: skip if gradients contain NaN or Inf
            if not (torch.isfinite(self.lora_A.grad).all() and torch.isfinite(self.lora_B.grad).all()):
                return

            mask_A = self.rank_mask.unsqueeze(1)
            mask_B = self.rank_mask.unsqueeze(0)
            self.lora_A.grad.mul_(mask_A)
            self.lora_B.grad.mul_(mask_B)

            self.grad_ema_A.mul_(self.ema_beta).add_(self.lora_A.grad, alpha=1.0 - self.ema_beta)
            self.grad_ema_B.mul_(self.ema_beta).add_(self.lora_B.grad, alpha=1.0 - self.ema_beta)

    @torch.no_grad()
    def get_nominal_norms(self, M: float = 10.0):
        """
        Phase 1 of the Proximal Step (Eq. 13):
        Computes tax-free matrices \tilde{A} and \tilde{B}, and returns brick lengths s_i and t_i.
        """
        A_tilde = self.lora_A - (1.0 / M) * self.grad_ema_A
        B_tilde = self.lora_B - (1.0 / M) * self.grad_ema_B

        # s_i = ||\tilde{A}_{i,:}||_2 (norm of each row of A_tilde, shape: [r])
        s = torch.norm(A_tilde.float(), p=2, dim=1)
        # t_i = ||\tilde{B}_{:,i}||_2 (norm of each column of B_tilde, shape: [r])
        t = torch.norm(B_tilde.float(), p=2, dim=0)

        # Force already-pruned bricks to have 0 length
        active = self.rank_mask.float() > 0
        s = torch.where(active, s, torch.zeros_like(s))
        t = torch.where(active, t, torch.zeros_like(t))
        return s, t

    @torch.no_grad()
    def apply_pruning_mask(self, keep_mask: torch.Tensor):
        """Permanently zeros out pruned bricks in rank_mask, A, B, and gradient buffers."""
        new_mask = keep_mask.to(self.rank_mask.dtype) * self.rank_mask
        self.rank_mask.copy_(new_mask)

        # Physically zero the weights and EMA buffers of pruned bricks
        self.lora_A.mul_(self.rank_mask.unsqueeze(1))
        self.lora_B.mul_(self.rank_mask.unsqueeze(0))
        self.grad_ema_A.mul_(self.rank_mask.unsqueeze(1))
        self.grad_ema_B.mul_(self.rank_mask.unsqueeze(0))