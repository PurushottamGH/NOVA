"""
NovaMind Loss Function
========================
Cross-entropy loss with optional label smoothing.

Label Smoothing (Szegedy et al., 2016):
Instead of training with hard targets (0 or 1), we use soft targets:
- True class gets probability: 1 - smoothing
- All other classes share: smoothing / (vocab_size - 1)

Why label smoothing?
- Prevents the model from becoming overconfident (logits → ±infinity)
- Acts as regularization, improving generalization
- Results in better calibrated probability estimates
- Typical values: 0.0 (none) to 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_perplexity(
    loss: torch.Tensor,
) -> float:  # FIXED: added perplexity helper — easier to interpret than raw loss
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss). Lower is better.
    - PPL ~1: perfect prediction
    - PPL ~vocab_size: random guessing

    Args:
        loss: Scalar cross-entropy loss tensor

    Returns:
        Perplexity as a float
    """
    clamped = torch.clamp(loss.detach(), max=20.0)  # Cap to prevent overflow
    return torch.exp(clamped).item()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
        ignore_index: Token ID to ignore in loss computation (e.g., PAD)
    """

    def __init__(
        self, smoothing: float = 0.05, ignore_index: int = 0
    ):  # FIXED: default smoothing=0.05 to match trainer usage
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model output of shape (batch * seq_len, vocab_size)
            targets: Target token IDs of shape (batch * seq_len,)

        Returns:
            Scalar loss tensor
        """
        if self.smoothing == 0.0:
            # Standard cross-entropy without smoothing
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)

        vocab_size = logits.size(-1)

        # Create smoothed target distribution
        # FIXED: denominator must be (vocab_size - 1) not vocab_size for correct smoothing math
        # The true class gets (1 - smoothing), the remaining (vocab_size - 1) classes
        # each get smoothing / (vocab_size - 1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (vocab_size - 1)
            )  # FIXED: was / vocab_size
            # Set the true class to (1 - smoothing)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1), 1.0 - self.smoothing
            )  # FIXED: simplified — no need to add back smoothing/vocab

            # Zero out positions where target is the ignore index (PAD)
            mask = (targets == self.ignore_index).unsqueeze(1)
            smooth_targets.masked_fill_(mask, 0.0)

        # Compute KL divergence loss (equivalent to cross-entropy with soft targets)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Average only over non-ignored positions
        non_pad_mask = targets != self.ignore_index
        if non_pad_mask.sum() == 0:
            return loss.sum() * 0.0  # Avoid div by zero
        loss = loss[non_pad_mask].mean()

        return loss


def create_loss_fn(config, smoothing: float = 0.05):  # FIXED: default matches trainer usage
    """
    Create the loss function for training.

    Args:
        config: NovaMindConfig with pad_token_id
        smoothing: Label smoothing factor

    Returns:
        Loss function (callable)
    """
    loss_fn = LabelSmoothingCrossEntropy(
        smoothing=smoothing,
        ignore_index=config.pad_token_id,  # FIXED: verified this uses config, not hardcoded 0
    )
    print(f"[Loss] Cross-entropy with smoothing={smoothing}, ignore_index={config.pad_token_id}")
    return loss_fn
