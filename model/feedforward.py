"""
NovaMind Feed-Forward Network Block
=====================================
Implements the position-wise feed-forward network (FFN) used in each
transformer decoder block.

Architecture:
    FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))
    
    Linear_1: embed_dim → feedforward_dim   (expansion)
    GELU:     non-linear activation
    Linear_2: feedforward_dim → embed_dim   (projection back)

Why GELU over ReLU?
    - GELU (Gaussian Error Linear Unit) was proposed by Hendrycks & Gimpel (2016)
    - Unlike ReLU which hard-zeros negative values, GELU applies a smooth gating:
      GELU(x) = x * Φ(x), where Φ is the cumulative distribution of N(0,1)
    - This means small negative values aren't completely killed — they're smoothly
      dampened, preserving gradient flow
    - GELU is used in GPT-2, GPT-3, BERT, and most modern LLMs
    - Empirically, GELU leads to faster convergence and better final loss than ReLU

Tensor shapes (documented at every step):
    Input:  (batch, seq_len, embed_dim)
    After Linear_1: (batch, seq_len, feedforward_dim)
    After GELU: (batch, seq_len, feedforward_dim)
    After Linear_2: (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
"""

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    
    This is applied independently to each position in the sequence.
    It's the "thinking" part of each transformer layer — attention gathers
    information, and the FFN processes it.
    
    The expansion ratio (feedforward_dim / embed_dim) is typically 4x,
    giving the network a "bottleneck" structure that first expands the
    representation to a higher dimension for richer computation,
    then compresses it back.
    
    Args:
        config: NovaMindConfig with embed_dim, feedforward_dim, dropout
    """
    
    def __init__(self, config):
        super().__init__()
        
        # === First linear layer: expand from embed_dim to feedforward_dim ===
        # This up-projection increases the dimensionality, allowing the model
        # to compute in a richer feature space
        self.linear_1 = nn.Linear(
            config.embed_dim,        # Input: embed_dim (e.g., 256)
            config.feedforward_dim   # Output: feedforward_dim (e.g., 1024)
        )
        
        # === GELU activation function ===
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # Smoother than ReLU, allows small negative gradient flow
        # Using the default 'none' approximation for mathematical exactness
        self.activation = nn.GELU()
        
        # === Second linear layer: project back from feedforward_dim to embed_dim ===
        # This down-projection compresses the expanded representation back
        # to the model's hidden dimension
        self.linear_2 = nn.Linear(
            config.feedforward_dim,  # Input: feedforward_dim (e.g., 1024)
            config.embed_dim         # Output: embed_dim (e.g., 256)
        )
        
        # === Dropout for regularization ===
        # Applied after the activation, before the second linear layer
        # Prevents co-adaptation of features in the expanded space
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Step 1: Up-project to higher dimension
        # (batch, seq_len, embed_dim) → (batch, seq_len, feedforward_dim)
        x = self.linear_1(x)  # (batch, seq_len, feedforward_dim)
        
        # Step 2: Apply GELU non-linearity
        # GELU smoothly gates the values — unlike ReLU which sharply zeros negatives
        # (batch, seq_len, feedforward_dim) → (batch, seq_len, feedforward_dim)
        x = self.activation(x)  # (batch, seq_len, feedforward_dim)
        
        # Step 3: Apply dropout in the expanded space
        # (batch, seq_len, feedforward_dim) → (batch, seq_len, feedforward_dim)
        x = self.dropout(x)  # (batch, seq_len, feedforward_dim)
        
        # Step 4: Down-project back to embed_dim
        # (batch, seq_len, feedforward_dim) → (batch, seq_len, embed_dim)
        x = self.linear_2(x)  # (batch, seq_len, embed_dim)
        
        return x  # (batch, seq_len, embed_dim)
