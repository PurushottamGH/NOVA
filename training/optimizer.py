"""
NovaMind Optimizer Configuration
==================================
Creates an AdamW optimizer with proper parameter groups.

AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update,
which is essential for correct regularization with adaptive learning rates.

Parameter groups:
- Group 1 (with weight decay): All 2D parameters (Linear weights, Embedding weights)
- Group 2 (no weight decay): All 1D parameters (biases, LayerNorm weights/biases)

Why separate groups?
- Weight decay on biases and LayerNorm params can hurt performance
- Only the main weight matrices benefit from L2 regularization
"""

import torch
import bitsandbytes as bnb


def create_optimizer(model, config):
    """
    Create an AdamW optimizer with separate parameter groups for decay/no-decay.
    
    Args:
        model: NovaMind model
        config: NovaMindConfig with learning_rate and weight_decay
    
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []     # Parameters that get weight decay (2D weights)
    no_decay_params = []  # Parameters that DON'T get weight decay (1D: biases, norms)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Apply weight decay only to 2D+ parameters (weight matrices)
        # Skip biases (1D) and LayerNorm parameters
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,  # No weight decay for biases and norms
        },
    ]

    # Print parameter group info
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"[Optimizer] AdamW created:")
    print(f"  Decayed params:    {num_decay:,} ({num_decay/1e6:.2f}M)")
    print(f"  Non-decayed params: {num_no_decay:,}")
    print(f"  Learning rate:     {config.learning_rate}")
    print(f"  Weight decay:      {config.weight_decay}")

    optimizer = bnb.optim.AdamW8bit(
        param_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),    # Beta1=0.9, Beta2=0.95 (GPT-3 settings)
        eps=1e-8,
    )

    return optimizer
