"""
NovaMind Tokenizer
===================
Main tokenizer class that wraps the BPE engine with a clean API.

Features:
- Train on multiple text files
- Encode text → token IDs (with BOS/EOS)
- Decode token IDs → text (skipping special tokens)
- Batch encode and pad for model input
- Save/load vocabulary and merges as JSON
"""

import json
from pathlib import Path

import torch

from tokenizer.bpe import BPETrainer
from tokenizer.special_tokens import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_TO_ID,
    UNK_TOKEN_ID,
)


class NovaMindTokenizer:
    """
    Tokenizer for NovaMind using custom BPE.

    Usage:
        tokenizer = NovaMindTokenizer()
        tokenizer.train(["data/file1.txt", "data/file2.txt"], vocab_size=8000)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
        tokenizer.save("tokenizer/")
        tokenizer = NovaMindTokenizer.load("tokenizer/")
    """

    def __init__(self):
        self.bpe = BPETrainer()
        self._vocab = {}  # token_str → token_id
        self._inverse_vocab = {}  # token_id → token_str
        self._merges = []  # List of (pair_a, pair_b) merge operations
        self._trained = False

    def train(self, text_files: list[str], vocab_size: int = 8000):
        """
        Train the BPE tokenizer on a list of text files.

        Args:
            text_files: List of paths to .txt files
            vocab_size: Target vocabulary size (including special tokens)
        """
        # Combine all text from all files
        all_text = []
        total_chars = 0

        for file_path in text_files:
            path = Path(file_path)
            if not path.exists():
                print(f"[Tokenizer] Warning: {file_path} not found, skipping")
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            all_text.append(text)
            total_chars += len(text)
            print(f"[Tokenizer] Loaded {file_path}: {len(text):,} chars")

        if not all_text:
            raise ValueError("No text files found for training")

        combined_text = " ".join(all_text)
        print(f"[Tokenizer] Total training text: {total_chars:,} characters")

        # FIXED: was vocab_size - NUM_SPECIAL_TOKENS which double-subtracts
        # because BPETrainer.train() already subtracts NUM_SPECIAL_TOKENS internally (line 141)
        self._merges, self._vocab, self._inverse_vocab = self.bpe.train(
            combined_text,
            vocab_size,  # FIXED: pass full vocab_size — BPE handles special token offset
        )

        # Add special tokens to vocabulary
        full_vocab = {}
        for token, tid in SPECIAL_TOKEN_TO_ID.items():
            full_vocab[token] = tid
        for token, tid in self._vocab.items():
            full_vocab[token] = tid

        self._vocab = full_vocab
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}
        self._trained = True

        print(f"[Tokenizer] Final vocabulary size: {len(self._vocab)}")

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs with BOS and EOS.

        Args:
            text: Input text string

        Returns:
            List of token IDs: [BOS, token_1, token_2, ..., token_n, EOS]
        """
        assert self._trained, "Tokenizer not trained yet. Call train() first."

        # Tokenize using BPE
        tokens = self.bpe.tokenize(text)

        # Convert to IDs
        ids = [BOS_TOKEN_ID]  # Prepend BOS
        for token in tokens:
            if token in self._vocab:
                ids.append(self._vocab[token])
            else:
                ids.append(UNK_TOKEN_ID)
        ids.append(EOS_TOKEN_ID)  # Append EOS

        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id_ in ids:
            if id_ in SPECIAL_TOKEN_IDS:
                continue
            if id_ in self._inverse_vocab:
                tokens.append(self._inverse_vocab[id_])
            else:
                tokens.append("")

        result = []
        for token in tokens:
            if token.endswith("</w>"):
                result.append(token[:-4])
                result.append(" ")
            else:
                result.append(token)

        return "".join(result).strip()

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode multiple texts to lists of token IDs.

        Args:
            texts: List of text strings

        Returns:
            List of lists of token IDs
        """
        return [self.encode(text) for text in texts]

    def pad_batch(
        self, encoded_batch: list[list[int]], pad_to_length: int | None = None
    ) -> torch.Tensor:
        """
        Pad a batch of encoded sequences to the same length.

        Args:
            encoded_batch: List of lists of token IDs (from encode_batch)
            pad_to_length: Target length. If None, uses the longest sequence.

        Returns:
            Padded tensor of shape (batch_size, pad_to_length)
        """
        if pad_to_length is None:
            pad_to_length = max(len(seq) for seq in encoded_batch)

        padded = []
        for seq in encoded_batch:
            if len(seq) >= pad_to_length:
                padded.append(seq[:pad_to_length])
            else:
                padded.append(seq + [PAD_TOKEN_ID] * (pad_to_length - len(seq)))

        return torch.tensor(padded, dtype=torch.long)  # (batch, pad_to_length)

    def save(self, path: str):
        """
        Save vocabulary and merges to JSON files.

        Creates:
            {path}/vocab.json — token→ID mapping
            {path}/merges.json — ordered merge operations
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        # Save merges as list of [pair_a, pair_b]
        merges_list = [[a, b] for a, b in self._merges]
        with open(save_dir / "merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

        print(
            f"[Tokenizer] Saved to {save_dir} (vocab: {len(self._vocab)}, merges: {len(self._merges)})"
        )

    @classmethod
    def load(cls, path: str) -> "NovaMindTokenizer":
        """
        Load a tokenizer from saved JSON files.

        Args:
            path: Directory containing vocab.json and merges.json

        Returns:
            Loaded NovaMindTokenizer ready for encoding/decoding
        """
        load_dir = Path(path)

        tokenizer = cls()

        # Load vocabulary
        with open(load_dir / "vocab.json", encoding="utf-8") as f:
            tokenizer._vocab = json.load(f)

        # Load merges
        with open(load_dir / "merges.json", encoding="utf-8") as f:
            merges_list = json.load(f)
        tokenizer._merges = [(a, b) for a, b in merges_list]

        # Rebuild inverse vocab
        tokenizer._inverse_vocab = {int(v): k for k, v in tokenizer._vocab.items()}

        # Rebuild BPE engine state
        tokenizer.bpe.merges = tokenizer._merges
        int_vocab = {k: int(v) for k, v in tokenizer._vocab.items()}
        tokenizer.bpe.vocab = int_vocab
        tokenizer.bpe.inverse_vocab = {v: k for k, v in int_vocab.items()}

        tokenizer._trained = True
        print(f"[Tokenizer] Loaded from {load_dir} (vocab: {len(tokenizer._vocab)})")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self._vocab)

    def __len__(self) -> int:
        """Returns vocabulary size."""
        return self.vocab_size
