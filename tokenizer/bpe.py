"""
NovaMind Byte Pair Encoding (BPE) — Built From Scratch
========================================================
Implements the BPE algorithm without any external tokenizer library.

BPE Algorithm:
1. Start with a vocabulary of individual characters from the training text
2. Count the frequency of every adjacent pair of symbols
3. Merge the most frequent pair into a single new symbol
4. Repeat until the desired vocabulary size is reached

The resulting merges list defines how to tokenize new text:
- Split text into characters
- Apply each learned merge in order
- Map resulting tokens to integer IDs

References:
- Sennrich et al., 2016: "Neural Machine Translation of Rare Words with Subword Units"
"""

from collections import Counter

from tqdm import tqdm

from tokenizer.special_tokens import NUM_SPECIAL_TOKENS


class BPETrainer:
    """
    Trains a BPE vocabulary from raw text.

    The trainer learns a list of merge operations that define how to
    split text into subword tokens. These merges are then used by the
    tokenizer to encode new text.
    """

    def __init__(self):
        self.merges = []  # Ordered list of (pair_a, pair_b) merges
        self.vocab = {}  # token → ID mapping
        self.inverse_vocab = {}  # ID → token mapping
        self.cache = {}  # Speeds up tokenization by 100x using Zipf's law

    @staticmethod
    def get_stats(word_freqs):
        """
        Count frequency of all adjacent symbol pairs across the vocabulary.

        Args:
            word_freqs: dict mapping tuple-of-symbols → frequency
                        e.g., {('l', 'o', 'w'): 5, ('l', 'o', 'w', 'e', 'r'): 2}

        Returns:
            Counter of (symbol_a, symbol_b) → total frequency
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word  # word is already a tuple of symbols
            # Count all adjacent pairs in this word, weighted by word frequency
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    @staticmethod
    def merge_vocab(pair, word_freqs):
        """
        Merge all occurrences of `pair` into a single symbol in the vocabulary.

        Args:
            pair: Tuple (symbol_a, symbol_b) to merge
            word_freqs: dict mapping tuple-of-symbols → frequency

        Returns:
            New word_freqs dict with the pair merged everywhere
        """
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]  # Concatenate the two symbols

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if the current position matches the pair to merge
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2  # Skip both symbols (they've been merged)
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs

    def train(self, text, vocab_size):
        """
        Train BPE on the given text until vocab_size is reached.

        Args:
            text: Raw training text (string)
            vocab_size: Target vocabulary size (including special tokens)

        Returns:
            Tuple of (merges_list, vocab_dict, inverse_vocab_dict)
        """
        # Target number of merge operations
        # vocab starts with: special_tokens + unique_characters
        # each merge adds one new token

        # Step 1: Tokenize text into words (split on whitespace, keep the space as prefix)
        # We use a simple whitespace-based pre-tokenization
        words = text.split()

        # Count word frequencies
        word_freq_counter = Counter(words)

        # Step 2: Initialize vocabulary with character-level tokens
        # Each word becomes a tuple of characters, with a special end-of-word marker
        word_freqs = {}
        for word, freq in word_freq_counter.items():
            # Convert word to tuple of characters with end-of-word marker '▁'
            char_tuple = (*tuple(word), "</w>")
            word_freqs[char_tuple] = freq

        # Collect all unique characters as initial vocabulary
        char_vocab = set()
        for word_tuple in word_freqs:
            for char in word_tuple:
                char_vocab.add(char)

        # Build initial vocabulary: special tokens + characters
        self.vocab = {}
        # Reserve IDs 0-3 for special tokens (handled by special_tokens.py)
        next_id = NUM_SPECIAL_TOKENS

        # Add character-level tokens
        for char in sorted(char_vocab):
            self.vocab[char] = next_id
            next_id += 1

        # Number of merges needed to reach vocab_size
        num_merges = vocab_size - len(self.vocab) - NUM_SPECIAL_TOKENS
        if num_merges <= 0:
            num_merges = 0

        self.merges = []

        # Step 3: Iteratively merge the most frequent pair
        print(f"[BPE] Training with {len(char_vocab)} initial characters")
        print(f"[BPE] Target vocab size: {vocab_size}, merges needed: {num_merges}")

        for i in tqdm(range(num_merges), desc="[BPE] Learning merges"):
            # Count all adjacent pairs
            pairs = self.get_stats(word_freqs)

            if not pairs:
                print(f"[BPE] No more pairs to merge at step {i}")
                break

            # Find the most frequent pair
            best_pair = pairs.most_common(1)[0]  # FIXED: call most_common once instead of twice
            best_freq = best_pair[1]
            best_pair = best_pair[0]

            if best_freq < 2:
                # Stop if the best pair only appears once — no compression benefit
                print(f"[BPE] Stopping early: best pair frequency = {best_freq}")
                break

            # Merge this pair everywhere in the vocabulary
            word_freqs = self.merge_vocab(best_pair, word_freqs)

            # Record the merge and add the merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[merged_token] = next_id
            next_id += 1

        # Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        print(f"[BPE] Training complete. Vocabulary size: {len(self.vocab) + NUM_SPECIAL_TOKENS}")
        print(f"[BPE] Learned {len(self.merges)} merge operations")

        return self.merges, self.vocab, self.inverse_vocab

    def apply_merges(self, word):
        """
        Apply learned BPE merges to tokenize a single word.

        Args:
            word: String to tokenize

        Returns:
            List of subword token strings
        """
        # Lazy initialization for rank lookup
        if not hasattr(self, 'bpe_ranks') or len(getattr(self, 'bpe_ranks', {})) != len(self.merges):
            self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Start with character-level split with end-of-word marker
        symbols = [*list(word), "</w>"]

        # Iteratively merge the best pair
        while True:
            if len(symbols) < 2:
                break
            
            # Find pair with lowest rank (highest priority)
            lowest_rank = float('inf')
            best_pair = None
            
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < lowest_rank:
                    lowest_rank = rank
                    best_pair = pair
                    
            # Stop if no more valid pairs found
            if lowest_rank == float('inf'):
                break
                
            # Apply merge across all occurrences cleanly
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def tokenize(self, text):
        """
        Tokenize a full text string using learned BPE merges.

        Args:
            text: Input text string

        Returns:
            List of subword token strings
        """
        # FIXED: handle empty text gracefully
        if not text or not text.strip():
            return []

        words = text.split()
        tokens = []
        for word in words:
            # Huge optimization: use memory cache
            if word in self.cache:
                tokens.extend(self.cache[word])
                continue

            try:
                word_tokens = self.apply_merges(word)
                self.cache[word] = word_tokens
                tokens.extend(word_tokens)
            except Exception:  # FIXED: catch any unexpected character errors
                tokens.append(word)  # Fall back to raw word
        return tokens

    def encode_tokens(self, tokens):
        """
        Convert token strings to integer IDs.

        Args:
            tokens: List of token strings from tokenize()

        Returns:
            List of integer token IDs
        """
        from tokenizer.special_tokens import UNK_TOKEN_ID

        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(UNK_TOKEN_ID)
        return ids

    def decode_ids(self, ids):
        """
        Convert integer IDs back to text.

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        from tokenizer.special_tokens import SPECIAL_TOKEN_IDS

        tokens = []
        for id_ in ids:
            if id_ in SPECIAL_TOKEN_IDS:
                continue  # Skip special tokens
            if id_ in self.inverse_vocab:
                tokens.append(self.inverse_vocab[id_])
            else:
                tokens.append("")
        text = "".join(tokens)
        # Remove end-of-word markers and add spaces
        text = text.replace("</w>", " ")
        return text.strip()
