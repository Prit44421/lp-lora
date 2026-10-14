"""
Phase 4: Transformer Fine-Tuning & Dynamic Rank Allocation Script (GLUE Benchmark).

Applies lp-LoRA to HuggingFace Transformer models (e.g. RoBERTa / DeBERTa / BERT / LLMs).
Replaces linear attention projection layers with LpLoRALinear, periodically allocates
ranks across all layers using LpLoRAAllocator, and logs the layer-wise rank evolution.
"""

import argparse
from typing import List, Optional
import torch
import torch.nn as nn

from src.lora_layer import LpLoRALinear
from src.allocator import LpLoRAAllocator


def apply_lp_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> int:
    """
    Recursively replaces target nn.Linear layers in model with LpLoRALinear.

    Args:
        model: HuggingFace or PyTorch model.
        target_modules: Names of module substrings to replace (e.g. ['query', 'value', 'q_proj', 'v_proj']).
        r: Initial LoRA rank.
        lora_alpha: LoRA scaling alpha.
        lora_dropout: Dropout probability.

    Returns:
        Number of layers replaced.
    """
    if target_modules is None:
        target_modules = ["query", "value", "q_proj", "v_proj", "k_proj", "out_proj"]

    replaced_count = 0

    for name, module in list(model.named_modules()):
        # Check if any target substring matches the module name
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[-1] if "." in name else name

            parent = model.get_submodule(parent_name) if parent_name else model

            # Replace with LpLoRALinear
            lora_module = LpLoRALinear.from_linear(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(parent, child_name, lora_module)
            replaced_count += 1

    return replaced_count


def print_rank_summary(allocator: LpLoRAAllocator, step: int):
    """Prints a formatted summary of layer-wise effective rank."""
    dist = allocator.get_rank_distribution()
    total = allocator.total_effective_rank
    initial = allocator.initial_total_rank
    print(f"\n[Step {step}] Layer-Wise Rank Allocation (Total: {total}/{initial} | {total/initial*100:.1f}%):")
    for name, rank in dist.items():
        bar = "█" * rank + "░" * (allocator.layers[0][1].r - rank)
        print(f"  {name:<45} | Rank: {rank:2d} | [{bar}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="lp-LoRA Transformer Fine-tuning")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--init_r", type=int, default=16)
    parser.add_argument("--target_ratio", type=float, default=0.5, help="Prune down to 50% global rank")
    parser.add_argument("--prune_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.model_name} for task {args.task_name}...")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(args.device)
        replaced = apply_lp_lora_to_model(model, r=args.init_r)
        print(f"Successfully converted {replaced} layers to LpLoRALinear.")

        allocator = LpLoRAAllocator(
            model,
            target_rank_ratio=args.target_ratio,
            pruning_interval=args.prune_interval,
        )
        print_rank_summary(allocator, step=0)

    except ImportError:
        print("Note: transformers library not yet installed. You can install it with: pip install transformers datasets")


if __name__ == "__main__":
    main()
