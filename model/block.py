"""
NovaMind Transformer Decoder Block
====================================
A single transformer decoder block with Pre-Norm architecture.

Pre-Norm vs Post-Norm:
    Post-Norm (original transformer, Vaswani 2017):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Pre-Norm (used here, GPT-2 style):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Why Pre-Norm is better for training stability:
    - In Post-Norm, the residual path passes through LayerNorm, which can
      dampen the gradient signal and make deep networks harder to train
    - In Pre-Norm, the residual connection is a direct identity path — gradients
      flow freely through the skip connection without any normalization barrier
    - This makes training more stable, especially for deeper models (6+ layers)
    - GPT-2, GPT-3, and most modern LLMs use Pre-Norm

Block Structure:
    1. LayerNorm → Multi-Head Causal Self-Attention → Residual Add
    2. LayerNorm → Feed-Forward Network → Residual Add

Tensor shapes throughout:
    Input:  (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
"""

import torch
import torch.nn as nn

from model.attention import MultiHeadCausalSelfAttention
from model.feedforward import FeedForwardBlock


class TransformerDecoderBlock(nn.Module):
    """
    A single transformer decoder block with pre-norm residual connections.

    The model stacks num_layers of these blocks to create the full architecture.
    Each block attends to prior context (via causal attention) and then processes
    the gathered information (via feed-forward network).

    Args:
        config: NovaMindConfig with embed_dim, norm_eps, and all sub-module configs
    """

    def __init__(self, config):
        super().__init__()

        # === Layer Normalization 1 (before attention) ===
        self.ln_1 = nn.LayerNorm(
            config.embed_dim,  # Normalize across this dimension
            eps=config.norm_eps,  # Numerical stability epsilon (e.g., 1e-5)
        )

        # === Multi-Head Causal Self-Attention ===
        self.attention = MultiHeadCausalSelfAttention(config)

        # === Layer Normalization 2 (before feed-forward) ===
        self.ln_2 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)

        # === Feed-Forward Network ===
        self.ffn = FeedForwardBlock(config)

    def forward(
        self, x: torch.Tensor, past_kv=None, use_cache=False
    ):  # FIXED: added past_kv and use_cache to propagate KV cache
        """
        Forward pass of one transformer decoder block.

        Pre-Norm architecture with residual connections:
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            past_kv: Optional cached (K, V) from previous forward pass
            use_cache: Whether to return updated KV cache

        Returns:
            x: Output tensor of shape (batch, seq_len, embed_dim)
            present_kv: Updated KV cache if use_cache=True, else None
        """
        # ===========================================================
        # Sub-block 1: Self-Attention with Pre-Norm Residual
        # ===========================================================
        normed_for_attn = self.ln_1(x)  # (batch, seq_len, embed_dim)

        # FIXED: attention now returns (output, kv_cache) tuple
        attn_output, present_kv = self.attention(
            normed_for_attn, past_kv=past_kv, use_cache=use_cache
        )  # (batch, seq_len, embed_dim)

        x = x + attn_output  # (batch, seq_len, embed_dim) — residual connection

        # ===========================================================
        # Sub-block 2: Feed-Forward Network with Pre-Norm Residual
        # ===========================================================
        normed_for_ffn = self.ln_2(x)  # (batch, seq_len, embed_dim)
        ffn_output = self.ffn(normed_for_ffn)  # (batch, seq_len, embed_dim)
        x = x + ffn_output  # (batch, seq_len, embed_dim) — residual connection

        return x, present_kv  # FIXED: now returns (output, kv_cache) tuple
