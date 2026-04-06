"""
NovaMind Multi-Head Causal Self-Attention
=========================================
Implements scaled dot-product multi-head causal self-attention entirely from
scratch using pure PyTorch primitives. No nn.MultiheadAttention, no
F.scaled_dot_product_attention — every matrix multiply is explicit.

The causal mask ensures that position i can only attend to positions <= i,
which is essential for autoregressive (left-to-right) language modeling.

Mathematical formulation:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
    - QKV = X @ W_QKV  (combined projection, then split)
    - d_k = head_dim = embed_dim / num_heads
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention built from scratch.

    This module:
    1. Projects input into Q, K, V using a COMBINED linear layer (no bias)
    2. Splits into multiple heads for parallel attention computation
    3. Computes scaled dot-product attention with a causal mask
    4. Concatenates heads and projects back to embed_dim
    5. Supports KV caching for efficient autoregressive inference

    Args:
        config: NovaMindConfig with embed_dim, num_heads, context_length, dropout
    """

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim  # Total embedding dimension (e.g., 256)
        self.num_heads = config.num_heads  # Number of attention heads (e.g., 8)
        self.head_dim = config.embed_dim // config.num_heads  # Per-head dimension (e.g., 32)
        self.dropout_p = config.dropout  # Dropout probability for attention weights

        # Validate that embed_dim is evenly divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )

        # === FIXED: Combined QKV projection instead of separate W_q, W_k, W_v ===
        # Single matmul is more efficient than three separate ones
        # Projects from embed_dim → 3 * embed_dim (Q, K, V concatenated)
        self.W_qkv = nn.Linear(
            self.embed_dim, 3 * self.embed_dim, bias=False
        )  # FIXED: was 3 separate Linear layers

        # === Output projection ===
        # Projects concatenated multi-head output back to embed_dim
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # === Dropout on attention weights ===
        # Applied after softmax, before multiplying by V
        self.attn_dropout = nn.Dropout(config.dropout)

        # === Residual dropout ===
        # Applied to the output projection result
        self.resid_dropout = nn.Dropout(config.dropout)

        # === Causal mask ===
        # FIXED: Use dtype=torch.bool instead of float — saves memory, enables ~mask syntax
        # Lower-triangular matrix of True: position i can only attend to j <= i
        # Shape: (1, 1, context_length, context_length) for broadcasting with attention scores
        causal_mask = (
            torch.tril(
                torch.ones(
                    config.context_length, config.context_length, dtype=torch.bool
                )  # FIXED: was float, now bool
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, context_length, context_length)
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, x: torch.Tensor, past_kv=None, use_cache=False
    ):  # FIXED: added past_kv and use_cache for KV caching
        """
        Forward pass of multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            past_kv: Optional tuple (past_K, past_V) each of shape
                     (batch, num_heads, past_seq_len, head_dim) for cached inference
            use_cache: If True, return present_kv = (K, V) for caching

        Returns:
            output: Tensor of shape (batch, seq_len, embed_dim)
            present_kv: Tuple (K, V) if use_cache=True, else None
        """
        batch_size, seq_len, _embed_dim = x.shape  # (batch, seq_len, embed_dim)

        # ============================================================
        # Step 1: Compute Q, K, V via combined projection
        # FIXED: Single W_qkv matmul replaces 3 separate projections
        # (batch, seq_len, embed_dim) → (batch, seq_len, 3 * embed_dim)
        # ============================================================
        qkv = self.W_qkv(x)  # (batch, seq_len, 3 * embed_dim)
        Q, K, V = qkv.chunk(3, dim=-1)  # Each: (batch, seq_len, embed_dim)

        # ============================================================
        # Step 2: Reshape into multiple heads
        # (batch, seq_len, embed_dim) → (batch, num_heads, seq_len, head_dim)
        # ============================================================
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)

        # ============================================================
        # Step 2b: KV cache — concat past keys/values for inference
        # FIXED: Added KV cache support for efficient autoregressive generation
        # ============================================================
        if past_kv is not None:
            past_K, past_V = past_kv  # Each: (batch, num_heads, past_seq_len, head_dim)
            K = torch.cat(
                [past_K, K], dim=2
            )  # (batch, num_heads, past_seq_len + seq_len, head_dim)
            V = torch.cat(
                [past_V, V], dim=2
            )  # (batch, num_heads, past_seq_len + seq_len, head_dim)

        present_kv = (K, V) if use_cache else None  # FIXED: return cached KV when requested

        kv_seq_len = K.size(2)  # Total key/value sequence length (may include past)

        # ============================================================
        # Step 3: Scaled dot-product attention
        # scores = Q @ K^T / sqrt(head_dim)
        # ============================================================
        scale = math.sqrt(self.head_dim)  # sqrt(d_k) scaling factor
        # Q @ K^T: (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, kv_seq_len)
        #        → (batch, num_heads, seq_len, kv_seq_len)
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / scale
        )  # (batch, num_heads, seq_len, kv_seq_len)

        # ============================================================
        # Step 4: Apply causal mask
        # FIXED: Use bool mask with ~mask instead of float mask with mask == 0
        # ============================================================
        # Slice the pre-computed mask to match current and total sequence lengths
        # When using KV cache, seq_len=1 but we need row 'kv_seq_len-1' of the mask
        start_pos = kv_seq_len - seq_len
        mask = self.causal_mask[
            :, :, start_pos : start_pos + seq_len, :kv_seq_len
        ]  # (1, 1, seq_len, kv_seq_len)

        attention_scores = attention_scores.masked_fill(
            ~mask,  # FIXED: was mask == 0 on float tensor, now ~mask on bool tensor
            float("-inf"),  # -inf → softmax gives 0 probability
        )  # (batch, num_heads, seq_len, kv_seq_len)

        # ============================================================
        # Step 5: Numerical stability + Softmax normalization
        # FIXED: Subtract max before softmax to prevent NaN from large scores
        # ============================================================
        attention_scores = attention_scores - attention_scores.amax(
            dim=-1, keepdim=True
        )  # FIXED: numerical stability — prevents exp() overflow

        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # (batch, num_heads, seq_len, kv_seq_len)

        attention_weights = torch.nan_to_num(
            attention_weights, nan=0.0
        )  # FIXED: guard against any remaining NaN after softmax

        # Apply dropout to attention weights (regularization during training)
        attention_weights = self.attn_dropout(
            attention_weights
        )  # (batch, num_heads, seq_len, kv_seq_len)

        # ============================================================
        # Step 6: Weighted sum of values
        # (batch, num_heads, seq_len, kv_seq_len) @ (batch, num_heads, kv_seq_len, head_dim)
        # → (batch, num_heads, seq_len, head_dim)
        # ============================================================
        attended = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # ============================================================
        # Step 7: Concatenate heads and project output
        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim) → (batch, seq_len, embed_dim)
        # ============================================================
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        )  # (batch, seq_len, embed_dim)

        # Final output projection and residual dropout
        output = self.W_o(attended)  # (batch, seq_len, embed_dim)
        output = self.resid_dropout(output)  # (batch, seq_len, embed_dim)

        return output, present_kv  # FIXED: now returns (output, kv_cache) tuple
