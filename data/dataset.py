"""
NovaMind Dataset
=================
PyTorch Dataset that creates overlapping chunks from tokenized text files
using a sliding window approach.

Each sample is a (input_ids, target_ids) pair where:
- input_ids:  tokens at positions [i, i+1, ..., i+context_length-1]
- target_ids: tokens at positions [i+1, i+2, ..., i+context_length]

This means target = input shifted right by 1 (next-token prediction).

The sliding window with configurable stride controls overlap:
- stride == context_length: no overlap (each token appears once)
- stride == context_length // 2: 50% overlap
- stride == 1: maximum overlap (each token is a new sample)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class NovaMindDataset(Dataset):
    """
    PyTorch Dataset for NovaMind training.

    Loads text files, tokenizes them, and creates overlapping chunks
    using a sliding window for next-token prediction training.

    Supports two calling conventions:
        1. NovaMindDataset(text_files=[...], tokenizer=tok, context_length=512)
        2. NovaMindDataset(data_dir="/path/to/data", context_length=512)

    Args:
        text_files: List of paths to .txt files (optional if data_dir is given)
        tokenizer: NovaMindTokenizer instance (optional — uses whitespace split if None)
        context_length: Number of tokens per chunk (model's context window)
        stride: Step size for the sliding window (controls overlap)
        data_dir: Directory to scan for .txt files (alternative to text_files)
    """

    def __init__(
        self,
        text_files: list[str] | None = None,
        tokenizer=None,
        context_length: int = 512,
        stride: int | None = None,
        data_dir: str | None = None,
    ):
        super().__init__()

        self.context_length = context_length
        self.stride = stride if stride is not None else context_length // 2
        self.chunks = []  # List of (input_ids, target_ids) tuples

        # --- Resolve file list ---
        if data_dir is not None:
            # Scan directory for .txt files (used by train_resume.py)
            data_path = Path(data_dir).resolve()
            self.files = sorted([p for p in data_path.rglob("*.txt") if p.is_file()])
        elif text_files is not None:
            # Use explicit file list (used by train.py / dataloader.py)
            self.files = [Path(f) for f in text_files if Path(f).is_file()]
        else:
            raise ValueError("Either 'text_files' or 'data_dir' must be provided")

        print(f"[Dataset] Found {len(self.files)} valid text files")
        assert len(self.files) > 0, "❌ No valid .txt files found!"

        # --- Tokenize all files ---
        all_token_ids = []

        print(f"[Dataset] Tokenizing {len(self.files)} files...")

        for path in self.files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")
                continue
            if len(text.strip()) < 50:
                continue  # Skip nearly empty files

            # Encode text to token IDs
            if tokenizer is not None:
                token_ids = tokenizer.encode(text)
            else:
                # Fallback: simple whitespace tokenization (for unblocking training)
                token_ids = list(range(len(text.split())))
            all_token_ids.extend(token_ids)

        if not all_token_ids:
            raise ValueError("No tokens produced from the provided text files")

        total_tokens = len(all_token_ids)

        # Convert to tensor for efficient slicing
        token_tensor = torch.tensor(all_token_ids, dtype=torch.long)

        # Create overlapping chunks using sliding window
        # Each chunk needs context_length + 1 tokens (input + 1 shifted target)
        chunk_size = context_length + 1

        print(
            f"[Dataset] Creating chunks (context_length={context_length}, stride={self.stride})..."
        )
        for start in range(0, len(token_tensor) - chunk_size + 1, self.stride):
            chunk = token_tensor[start : start + chunk_size]  # (context_length + 1,)

            input_ids = chunk[:-1]  # First context_length tokens: (context_length,)
            target_ids = chunk[1:]  # Last context_length tokens: (context_length,)

            self.chunks.append((input_ids, target_ids))

        # Print dataset statistics
        coverage = (len(self.chunks) * self.stride) / max(total_tokens, 1) * 100
        print("\n[Dataset] Statistics:")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Total chunks:     {len(self.chunks):,}")
        print(f"  Context length:   {context_length}")
        print(f"  Stride:           {self.stride}")
        print(f"  Coverage:         {min(coverage, 100):.1f}%")

    def __len__(self) -> int:
        """Total number of training chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int):
        """
        Get a single training sample.

        Returns:
            Tuple of (input_ids, target_ids), each of shape (context_length,)
        """
        input_ids, target_ids = self.chunks[idx]
        return input_ids, target_ids  # (context_length,), (context_length,)
