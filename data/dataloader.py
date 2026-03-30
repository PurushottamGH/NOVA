"""
NovaMind DataLoader Factory
==============================
Creates PyTorch DataLoaders with train/validation split.

Features:
- Configurable train/val split ratio
- Shuffled training data, sequential validation data
- Proper worker count for data loading
- Pin memory for faster GPU transfer

Usage:
    from data.dataloader import create_dataloaders
    train_loader, val_loader = create_dataloaders(
        text_files=["personal_data/book.txt"],
        tokenizer=tokenizer,
        config=config,
    )
"""

import math
from typing import List, Tuple
from pathlib import Path

from torch.utils.data import DataLoader, random_split

from data.dataset import NovaMindDataset


def create_dataloaders(
    text_files: List[str],
    tokenizer,
    config,
    train_split: float = 0.9,
    stride: int = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        text_files: List of paths to text files
        tokenizer: Trained NovaMindTokenizer instance
        config: NovaMindConfig with batch_size and context_length
        train_split: Fraction of data for training (rest is validation)
        stride: Sliding window stride (default: context_length // 2)
        num_workers: Number of data loading workers (0 = main process)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create the full dataset
    dataset = NovaMindDataset(
        text_files=text_files,
        tokenizer=tokenizer,
        context_length=config.context_length,
        stride=stride,
    )

    # Split into train and validation sets
    total_size = len(dataset)
    train_size = math.floor(total_size * train_split)
    val_size = total_size - train_size

    if val_size == 0:
        # If dataset is too small for a split, use the same data for both
        print("[DataLoader] Warning: Dataset too small for split, using same data for train and val")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\n[DataLoader] Split: {train_size} train, {val_size} val")

    # Determine if we should pin memory (only useful with GPU)
    pin_memory = config.device == "cuda"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,           # Shuffle training data for better generalization
        num_workers=num_workers,
        pin_memory=pin_memory,  # Speed up CPU→GPU transfer
        drop_last=True,         # Drop incomplete last batch for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,          # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # Keep all validation samples
    )

    print(f"[DataLoader] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[DataLoader] Batch size: {config.batch_size}")

    return train_loader, val_loader


def get_text_files(data_dir: str = "personal_data") -> List[str]:
    """
    Get all .txt files from a directory.
    
    Args:
        data_dir: Directory to search for .txt files
    
    Returns:
        Sorted list of .txt file paths as strings
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = sorted([str(f) for f in data_path.glob("*.txt")])
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    print(f"[DataLoader] Found {len(txt_files)} text files in {data_dir}")
    return txt_files
