"""
NovaMind Model Package
======================
Contains all model architecture components for the NovaMind LLM:
- config: Hyperparameter configuration dataclass
- attention: Multi-head causal self-attention from scratch
- positional: Sinusoidal and learned positional encodings
- feedforward: Feed-forward network with GELU activation
- block: Transformer decoder block with pre-norm architecture
- architecture: Full NovaMind model class
- utils: Weight initialization, model summary, FLOP estimation
"""

from model.architecture import NovaMind
from model.config import NovaMindConfig
