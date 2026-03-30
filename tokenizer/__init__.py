"""
NovaMind Tokenizer Package
===========================
Custom BPE tokenizer built entirely from scratch — no tiktoken, no sentencepiece.
- bpe: Core Byte Pair Encoding algorithm
- special_tokens: PAD, BOS, EOS, UNK definitions
- tokenizer: Main tokenizer class with train/encode/decode/save/load
"""
from tokenizer.tokenizer import NovaMindTokenizer
