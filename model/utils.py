"""
NovaMind Model Utilities
=========================
Helper functions for weight initialization, model introspection, and FLOP estimation.

- initialize_weights: Apply custom weight init to all layers
- model_summary: Print a clean table of layer names, shapes, and param counts
- estimate_flops: Rough FLOP estimate for one forward pass
"""

import torch
import torch.nn as nn


def initialize_weights(model, config):
    """
    Apply weight initialization to all layers in the model.
    
    Strategy:
    - nn.Linear weights: Normal(0, initializer_range)
    - nn.Embedding weights: Normal(0, initializer_range)
    - nn.LayerNorm: weight=1, bias=0
    - All biases: 0
    
    Args:
        model: The NovaMind model (or any nn.Module)
        config: NovaMindConfig with initializer_range
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def model_summary(model):
    """
    Print a clean summary table of all model parameters.
    Shows layer name, parameter shape, parameter count, and whether trainable.
    """
    header = f"{'Layer Name':<50} {'Shape':<25} {'Params':>12} {'Trainable':>10}"
    sep = "=" * 100
    print(sep)
    print(f"  NovaMind Model Summary")
    print(sep)
    print(header)
    print("-" * 100)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        shape_str = str(list(param.shape))
        num_params = param.numel()
        trainable = "Yes" if param.requires_grad else "No"
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"  {name:<48} {shape_str:<25} {num_params:>12,} {trainable:>10}")

    print("-" * 100)
    print(f"  Total Parameters:     {total_params:>12,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable:        {total_params - trainable_params:>12,}")
    print(sep)

    return {"total": total_params, "trainable": trainable_params}


def estimate_flops(config):
    """
    Rough FLOP estimate for one forward pass through the NovaMind model.
    
    This is an approximation based on the dominant operations:
    1. Embedding lookup: negligible
    2. Attention QKV projections: 3 × 2 × seq × d² per layer
    3. Attention score computation: 2 × seq² × d per layer
    4. Attention output projection: 2 × seq × d² per layer  
    5. FFN: 2 × 2 × seq × d × ff_dim per layer
    6. LM Head: 2 × seq × d × vocab
    
    Where factor of 2 accounts for multiply-add.
    
    Args:
        config: NovaMindConfig
    
    Returns:
        dict with FLOP breakdown and total
    """
    seq = config.context_length
    d = config.embed_dim
    ff = config.feedforward_dim
    v = config.vocab_size
    L = config.num_layers

    # Per-layer attention FLOPs
    # QKV projections: 3 matmuls of (seq, d) @ (d, d) = 3 × 2 × seq × d²
    attn_qkv = 3 * 2 * seq * d * d
    # Score computation: (seq, d) @ (d, seq) = 2 × seq² × d  
    attn_scores = 2 * seq * seq * d
    # Output projection: (seq, d) @ (d, d) = 2 × seq × d²
    attn_out = 2 * seq * d * d
    # Total attention per layer
    attn_per_layer = attn_qkv + attn_scores + attn_out

    # Per-layer FFN FLOPs
    # Two matmuls: (seq, d)@(d, ff) + (seq, ff)@(ff, d) = 2×(2×seq×d×ff)
    ffn_per_layer = 2 * 2 * seq * d * ff

    # Total per layer
    per_layer = attn_per_layer + ffn_per_layer
    all_layers = per_layer * L

    # LM Head: (seq, d) @ (d, vocab) = 2 × seq × d × vocab
    lm_head = 2 * seq * d * v

    total = all_layers + lm_head

    result = {
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "total_per_layer": per_layer,
        "all_layers": all_layers,
        "lm_head": lm_head,
        "total": total,
        "total_gflops": round(total / 1e9, 3),
    }

    print(f"╔══════════════════════════════════════════╗")
    print(f"║     NovaMind FLOP Estimate (1 forward)   ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Attention/layer : {attn_per_layer/1e6:>12.1f}M FLOPs  ║")
    print(f"║  FFN/layer       : {ffn_per_layer/1e6:>12.1f}M FLOPs  ║")
    print(f"║  All {L} layers    : {all_layers/1e6:>12.1f}M FLOPs  ║")
    print(f"║  LM Head         : {lm_head/1e6:>12.1f}M FLOPs  ║")
    print(f"║  TOTAL           : {total/1e9:>12.3f} GFLOPs  ║")
    print(f"╚══════════════════════════════════════════╝")

    return result
