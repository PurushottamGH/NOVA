"""
NovaMind Checkpoint System
============================
Save, load, resume, and manage training checkpoints.

Checkpoint contents:
- model_state_dict: Model weights
- optimizer_state_dict: Optimizer state (momenta, etc.)
- scheduler_state_dict: LR scheduler state
- step: Current training step
- loss: Loss at checkpoint time
- config: Model configuration dict
- best_val_loss: Best validation loss seen so far

Usage:
    save_checkpoint(model, optimizer, scheduler, step, loss, config, "weights/")
    step, loss = load_checkpoint("weights/step_1000.pt", model, optimizer, scheduler)
"""

import os
import torch
from pathlib import Path
from typing import Optional, List, Tuple


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    loss: float,
    config,
    path: str,
    best_val_loss: float = float('inf'),
    is_best: bool = False,
):
    """
    Save a training checkpoint.
    
    Args:
        model: NovaMind model
        optimizer: Optimizer
        scheduler: LR scheduler
        step: Current training step
        loss: Current loss value
        config: NovaMindConfig
        path: Directory to save checkpoints
        best_val_loss: Best validation loss seen
        is_best: Whether this is the best model so far
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "config": config.to_dict(),
        "rng_state": torch.get_rng_state(),  # FIXED: save RNG state for exact reproducibility on resume
    }

    # Save step-numbered checkpoint
    checkpoint_path = save_dir / f"step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] Saved: {checkpoint_path} (loss={loss:.4f})")

    # Always save a 'latest' checkpoint for easy resume
    latest_path = save_dir / "latest.pt"
    torch.save(checkpoint, latest_path)

    # Save best model separately
    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] New best model saved (val_loss={best_val_loss:.4f})")


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
) -> Tuple[int, float]:
    """
    Load a checkpoint and restore model/optimizer/scheduler states.
    
    Args:
        path: Path to checkpoint file (.pt)
        model: NovaMind model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
    
    Returns:
        Tuple of (step, loss) from the checkpoint
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"[Checkpoint] Loading: {checkpoint_path}")

    # Load checkpoint, mapping to the current device
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", 0.0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))

    print(f"[Checkpoint] Restored: step={step}, loss={loss:.4f}, best_val_loss={best_val_loss:.4f}")

    return step, loss


def list_checkpoints(checkpoint_dir: str) -> List[dict]:
    """
    List all checkpoints in a directory, sorted by step number.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        Sorted list of dicts with 'path', 'step', 'filename'
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for f in ckpt_dir.glob("step_*.pt"):
        try:
            step = int(f.stem.split("_")[1])
            checkpoints.append({
                "path": str(f),
                "step": step,
                "filename": f.name,
                "size_mb": f.stat().st_size / 1e6,
            })
        except (ValueError, IndexError):
            continue

    checkpoints.sort(key=lambda x: x["step"])
    return checkpoints


def delete_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """
    Delete old checkpoints, keeping only the N most recent plus 'best' and 'latest'.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    if len(checkpoints) <= keep_last_n:
        return  # Nothing to delete

    # Keep the last N checkpoints
    to_delete = checkpoints[:-keep_last_n]

    for ckpt in to_delete:
        os.remove(ckpt["path"])
        print(f"[Checkpoint] Deleted old: {ckpt['filename']}")

    print(f"[Checkpoint] Kept {keep_last_n} most recent checkpoints")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    Prefers 'latest.pt', falls back to highest step number.
    
    Args:
        checkpoint_dir: Directory to search
    
    Returns:
        Path to latest checkpoint, or None if not found
    """
    ckpt_dir = Path(checkpoint_dir)

    # Check for 'latest.pt' first
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return str(latest)

    # Fall back to highest step number
    checkpoints = list_checkpoints(checkpoint_dir)
    if checkpoints:
        return checkpoints[-1]["path"]

    return None
