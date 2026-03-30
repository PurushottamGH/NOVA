"""
NovaMind Learning Rate Scheduler
===================================
Cosine annealing schedule with linear warmup.

Schedule:
1. Warmup phase (steps 0 → warmup_steps):
   LR increases linearly from 0 to peak_lr
   
2. Cosine decay phase (steps warmup_steps → max_steps):
   LR follows a cosine curve from peak_lr down to min_lr
   LR(t) = min_lr + 0.5*(peak_lr - min_lr)*(1 + cos(π * progress))
   
   where progress = (step - warmup_steps) / (max_steps - warmup_steps)

Why this schedule?
- Warmup prevents training instability in early steps when gradients are large
- Cosine decay provides smooth, gradual LR reduction (better than step decay)
- This is the standard schedule used in GPT-2, GPT-3, LLaMA, etc.
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(optimizer, config, min_lr_ratio=0.1):
    """
    Create a cosine decay scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        config: NovaMindConfig with warmup_steps and max_steps
        min_lr_ratio: Minimum LR as fraction of peak LR (default: 10%)
    
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps

    def lr_lambda(step):
        """
        Compute the LR multiplier for a given step.
        
        Returns a value between 0 and 1 that is multiplied with the base LR.
        """
        if step < warmup_steps:
            # Linear warmup: scale from 0 to 1 over warmup_steps
            return float(step) / float(max(1, warmup_steps))
        elif step >= max_steps:
            # After max_steps, hold at minimum LR
            return min_lr_ratio
        else:
            # Cosine decay from 1.0 to min_lr_ratio
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            # Cosine annealing formula: 0.5 * (1 + cos(π * progress))
            # This smoothly goes from 1.0 to 0.0 as progress goes from 0 to 1
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale between min_lr_ratio and 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"[Scheduler] Cosine decay with linear warmup:")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Max steps:     {max_steps}")
    print(f"  Min LR ratio:  {min_lr_ratio}")

    return scheduler
