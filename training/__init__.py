"""
NovaMind Training Package
==========================
Complete training pipeline:
- optimizer: AdamW with weight decay and param groups
- scheduler: Cosine LR scheduler with linear warmup
- loss: Cross-entropy with label smoothing
- trainer: Full training loop class
- checkpointing: Save/load/resume checkpoint logic
- train: Entry point script
"""
