"""
NovaMind Special Token Definitions
====================================
Defines the special tokens used by the NovaMind tokenizer and model.

These tokens are reserved at the beginning of the vocabulary:
  ID 0: [PAD] — Padding token for batching sequences of different lengths
  ID 1: [BOS] — Beginning of sequence, prepended to every input
  ID 2: [EOS] — End of sequence, appended to mark where text ends
  ID 3: [UNK] — Unknown token for characters not in the vocabulary

Convention: Special tokens use IDs 0-3, real vocabulary starts at ID 4.
"""

# Special token strings
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# Special token IDs (must match config.py)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# Number of reserved special tokens
NUM_SPECIAL_TOKENS = 4

# Mapping: token string → ID
SPECIAL_TOKEN_TO_ID = {
    PAD_TOKEN: PAD_TOKEN_ID,
    BOS_TOKEN: BOS_TOKEN_ID,
    EOS_TOKEN: EOS_TOKEN_ID,
    UNK_TOKEN: UNK_TOKEN_ID,
}

# Mapping: ID → token string
ID_TO_SPECIAL_TOKEN = {v: k for k, v in SPECIAL_TOKEN_TO_ID.items()}

# Set of all special token IDs (for quick lookup during decoding)
SPECIAL_TOKEN_IDS = set(SPECIAL_TOKEN_TO_ID.values())
