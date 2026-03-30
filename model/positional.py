"""
NovaMind Positional Encoding Module
====================================
Implements two types of positional encoding for the transformer:

1. SinusoidalPositionalEncoding (default):
   Uses the fixed sine/cosine formula from "Attention Is All You Need" (Vaswani et al., 2017).
   Not learned — the patterns are mathematically determined.
   
   Formula:
   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
   
   Where:
   - pos = position index in the sequence (0, 1, 2, ..., seq_len-1)
   - i   = dimension index (0, 1, 2, ..., d_model//2 - 1)
   - d_model = embedding dimension
   
   Why this works:
   - Each dimension of the PE corresponds to a sinusoid of a different frequency
   - The model can learn to attend to relative positions because
     PE(pos+k) can be represented as a linear function of PE(pos)

2. LearnedPositionalEncoding:
   Uses a learnable nn.Embedding table. Each position gets its own learned vector.
   More flexible but doesn't generalize to unseen sequence lengths.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (not learned).
    
    Registered as a buffer so it:
    - Moves to GPU/CPU with the model
    - Is NOT updated by the optimizer
    - Is saved/loaded with the model state dict
    
    Args:
        config: NovaMindConfig with embed_dim, context_length, dropout
    """
    
    def __init__(self, config):
        super().__init__()
        
        embed_dim = config.embed_dim            # d_model: dimensionality of embeddings (e.g., 256)
        max_len = config.context_length          # Maximum sequence length (e.g., 512)
        
        # === Build the positional encoding matrix ===
        # Shape will be (max_len, embed_dim)
        
        # Create a matrix of shape (max_len, embed_dim) filled with zeros
        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
        
        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Unsqueeze to get column vector of shape (max_len, 1) for broadcasting
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Compute the division term: 10000^(2i/d_model)
        # Using the log-space trick for numerical stability:
        # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        # So 1/10000^(2i/d_model) = exp(-2i * log(10000) / d_model)
        
        # Create dimension indices: [0, 2, 4, ..., embed_dim-2] (even indices only)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()          # [0, 2, 4, ..., d-2], shape: (embed_dim/2,)
            * (-math.log(10000.0) / embed_dim)              # Multiply by -log(10000)/d_model
        )  # (embed_dim / 2,)
        
        # === Apply sine to even indices (2i) ===
        # position * div_term broadcasts: (max_len, 1) * (embed_dim/2,) → (max_len, embed_dim/2)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        
        # === Apply cosine to odd indices (2i+1) ===
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        
        # Add batch dimension: (max_len, embed_dim) → (1, max_len, embed_dim)
        # This allows broadcasting when adding to input of shape (batch, seq_len, embed_dim)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        
        # Register as buffer (not a learned parameter)
        self.register_buffer('pe', pe)
        
        # Dropout applied after adding positional encoding
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.
        
        Args:
            x: Token embeddings of shape (batch, seq_len, embed_dim)
        
        Returns:
            Embeddings with positional information added, same shape (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)  # Get the actual sequence length from input
        
        # Slice the PE table to match the sequence length and add to embeddings
        # self.pe[:, :seq_len, :] has shape (1, seq_len, embed_dim)
        # Broadcasting adds it to each item in the batch
        
        # x = x + PE: element-wise addition of positional signal to token embeddings
        x = x + self.pe[:, :seq_len, :]  # (batch, seq_len, embed_dim) + (1, seq_len, embed_dim)
        
        # Apply dropout for regularization (helps prevent overfitting to exact positions)
        x = self.dropout(x)  # (batch, seq_len, embed_dim)
        
        return x  # (batch, seq_len, embed_dim)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using nn.Embedding.
    
    Each position (0, 1, ..., max_len-1) gets its own learned embedding vector.
    These are updated during training via backpropagation.
    
    Pros: More flexible, can learn task-specific positional patterns
    Cons: Cannot generalize to sequence lengths longer than max_len
    
    Args:
        config: NovaMindConfig with embed_dim, context_length, dropout
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Learnable position embedding table
        # Each of the max_len positions maps to an embed_dim-dimensional vector
        self.position_embedding = nn.Embedding(
            config.context_length,   # Number of positions (e.g., 512)
            config.embed_dim         # Embedding dimension (e.g., 256)
        )  # Parameters shape: (context_length, embed_dim)
        
        # Dropout applied after adding positional encoding
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to token embeddings.
        
        Args:
            x: Token embeddings of shape (batch, seq_len, embed_dim)
        
        Returns:
            Embeddings with positional information added, same shape (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)  # Actual sequence length
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # on the same device as x
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        
        # Look up the learned positional embeddings
        pos_emb = self.position_embedding(positions)  # (seq_len, embed_dim)
        
        # Add positional embeddings to token embeddings
        # pos_emb broadcasts across the batch dimension
        x = x + pos_emb  # (batch, seq_len, embed_dim) + (seq_len, embed_dim) → (batch, seq_len, embed_dim)
        
        # Apply dropout
        x = self.dropout(x)  # (batch, seq_len, embed_dim)
        
        return x  # (batch, seq_len, embed_dim)
