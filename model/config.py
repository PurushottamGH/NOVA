"""
NovaMind Configuration Module
=============================
Central configuration dataclass holding ALL hyperparameters for the NovaMind LLM.
Every other module imports from here — nothing is hardcoded elsewhere.

This includes:
- Model architecture params (embed_dim, num_heads, num_layers, etc.)
- Training params (learning_rate, batch_size, grad_clip, etc.)
- Special token IDs (PAD, BOS, EOS)
- Device selection with auto-detection
"""

import torch
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class NovaMindConfig:
    """
    Complete hyperparameter configuration for the NovaMind language model.
    
    Architecture Parameters:
        vocab_size: Size of the token vocabulary (default: 8000 for BPE)
        embed_dim: Dimensionality of token embeddings and hidden states
        num_heads: Number of attention heads in multi-head attention
        num_layers: Number of stacked transformer decoder blocks
        context_length: Maximum sequence length the model can process
        feedforward_dim: Inner dimension of the feed-forward network (typically 4x embed_dim)
        dropout: Dropout probability applied throughout the model
        activation: Activation function used in feed-forward blocks
        weight_tying: Whether to share weights between token embedding and output projection
        norm_eps: Epsilon for LayerNorm numerical stability
        initializer_range: Standard deviation for weight initialization
    
    Special Token IDs:
        pad_token_id: ID for padding token
        bos_token_id: ID for beginning-of-sequence token
        eos_token_id: ID for end-of-sequence token
    
    Training Parameters:
        learning_rate: Peak learning rate for AdamW optimizer
        weight_decay: L2 regularization coefficient
        warmup_steps: Number of linear warmup steps before cosine decay
        max_steps: Total number of training steps
        batch_size: Number of sequences per training batch
        grad_clip: Maximum gradient norm for gradient clipping
        save_every: Save checkpoint every N steps
        eval_every: Run evaluation every N steps
    
    Device:
        device: Compute device — "auto" detects cuda > mps > cpu
    """
    
    # === Model Architecture ===
    vocab_size: int = 32000
    embed_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    context_length: int = 256
    feedforward_dim: int = 4096
    dropout: float = 0.1
    activation: str = "gelu"
    weight_tying: bool = True
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    # === Special Token IDs ===
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # === Training Hyperparameters ===
    gradient_checkpointing: bool = False  # FIXED: added to reduce VRAM footprint
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 4
    grad_clip: float = 1.0
    accumulation_steps: int = 16    # FIXED: added for gradient accumulation support
    save_every: int = 500
    eval_every: int = 100
    sample_every: int = 1000  # FIXED: added to track generation quality during training
    
    # === Device ===
    device: str = "auto"
    
    def __post_init__(self):
        """Validate configuration and resolve device."""
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.dropout >= 0.0 and self.dropout < 1.0, (
            f"dropout must be in [0, 1), got {self.dropout}"
        )
        assert self.vocab_size > 0, f"vocab_size must be positive, got {self.vocab_size}"
        assert self.context_length > 0, f"context_length must be positive, got {self.context_length}"
        
        # Auto-detect device if set to "auto"
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head: embed_dim // num_heads."""
        return self.embed_dim // self.num_heads
    
    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "NovaMindConfig":
        """Create a NovaMindConfig from a dictionary.
        
        Ignores unknown keys so configs saved from future versions
        don't crash older code.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    def __repr__(self) -> str:
        """Pretty-print the configuration as a readable summary."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         NovaMind Configuration            ║",
            "╠══════════════════════════════════════════╣",
            f"║  vocab_size       : {self.vocab_size:<20} ║",
            f"║  embed_dim        : {self.embed_dim:<20} ║",
            f"║  num_heads        : {self.num_heads:<20} ║",
            f"║  head_dim         : {self.head_dim:<20} ║",
            f"║  num_layers       : {self.num_layers:<20} ║",
            f"║  context_length   : {self.context_length:<20} ║",
            f"║  feedforward_dim  : {self.feedforward_dim:<20} ║",
            f"║  dropout          : {self.dropout:<20} ║",
            f"║  activation       : {self.activation:<20} ║",
            f"║  weight_tying     : {str(self.weight_tying):<20} ║",
            f"║  norm_eps         : {self.norm_eps:<20} ║",
            f"║  init_range       : {self.initializer_range:<20} ║",
            "╠══════════════════════════════════════════╣",
            f"║  learning_rate    : {self.learning_rate:<20} ║",
            f"║  weight_decay     : {self.weight_decay:<20} ║",
            f"║  warmup_steps     : {self.warmup_steps:<20} ║",
            f"║  max_steps        : {self.max_steps:<20} ║",
            f"║  batch_size       : {self.batch_size:<20} ║",
            f"║  grad_clip        : {self.grad_clip:<20} ║",
            f"║  save_every       : {self.save_every:<20} ║",
            f"║  eval_every       : {self.eval_every:<20} ║",
            "╠══════════════════════════════════════════╣",
            f"║  pad_token_id     : {self.pad_token_id:<20} ║",
            f"║  bos_token_id     : {self.bos_token_id:<20} ║",
            f"║  eos_token_id     : {self.eos_token_id:<20} ║",
            f"║  device           : {self.device:<20} ║",
            "╚══════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
