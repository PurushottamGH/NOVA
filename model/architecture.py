"""
NovaMind Full Model Architecture
==================================
The complete NovaMind language model — a decoder-only transformer built
entirely from scratch using pure PyTorch.

Architecture: Token Embed → PosEnc → N×DecoderBlock → LayerNorm → LM Head → logits

Weight Tying: When enabled, token_embedding and lm_head share the same weight matrix.
This reduces params and acts as regularization (GPT-2, ALBERT style).
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model.config import NovaMindConfig
from model.positional import SinusoidalPositionalEncoding
from model.block import TransformerDecoderBlock


class NovaMind(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.config = config

        # Token Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )

        # Positional Encoding: adds position info to embeddings
        self.positional_encoding = SinusoidalPositionalEncoding(config)

        # Stack of transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.num_layers)
        ])

        # Final LayerNorm before output projection
        self.final_norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)

        # LM Head: (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight  # FIXED: verified — same tensor object, not a copy
            assert self.lm_head.weight is self.token_embedding.weight  # FIXED: assertion proves weight tying shares the tensor

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Normal init for Linear/Embedding, identity init for LayerNorm."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        """
        Forward pass.
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target IDs for loss computation, or None
        Returns:
            (logits, loss) where logits is (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape  # (batch, seq_len)
        assert seq_len <= self.config.context_length

        # Step 1: Token embedding (batch, seq_len) → (batch, seq_len, embed_dim)
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Step 2: Add positional encoding
        x = self.positional_encoding(token_emb)  # (batch, seq_len, embed_dim)

        # Step 3: Pass through all transformer blocks
        for block in self.blocks:
            x, _ = block(x)  # FIXED: updated to handle (output, kv_cache) return from block

        # Step 4: Final layer normalization
        x = self.final_norm(x)  # (batch, seq_len, embed_dim)

        # Step 5: Project to vocab logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Step 6: Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  # (batch*seq_len, vocab_size)
                targets.view(-1),                          # (batch*seq_len,)
                ignore_index=self.config.pad_token_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8,
                 top_k=50, top_p=0.9, repetition_penalty=1.1):
        """
        Autoregressive generation: predict one token at a time.
        Args:
            input_ids: (batch, seq_len) starting tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (higher=more random)
            top_k: keep only top-k logits
            top_p: nucleus sampling threshold
            repetition_penalty: penalize repeated tokens
        Returns:
            (batch, seq_len + generated_tokens)
        """
        self.eval()
        generated = input_ids

        # FIXED: temperature guard — clamp to safe range to prevent division by zero or garbage
        temperature = max(0.1, min(float(temperature), 2.0))  # FIXED: was unguarded, could be 0 or >2

        for _ in range(max_new_tokens):
            # Crop to context window
            seq = generated[:, -self.config.context_length:]  # (batch, ≤context_length)

            logits, _ = self.forward(seq)  # (batch, seq_len, vocab_size)
            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for tok in generated[b].unique():
                        if next_logits[b, tok] > 0:
                            next_logits[b, tok] /= repetition_penalty
                        else:
                            next_logits[b, tok] *= repetition_penalty

            # Temperature scaling
            next_logits = next_logits / temperature  # FIXED: always apply since we clamped temperature >= 0.1

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample
            probs = F.softmax(next_logits, dim=-1)  # (batch, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # FIXED: proper EOS stopping — break if all sequences hit EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return generated

    def count_parameters(self):
        """Returns dict with total/trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total, "trainable": trainable,
            "non_trainable": total - trainable,
            "total_million": round(total / 1e6, 2),
            "trainable_million": round(trainable / 1e6, 2),
        }

    def save(self, path):
        """Save model weights + config to directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:  # FIXED: added encoding='utf-8'
            json.dump(self.config.to_dict(), f, indent=2)
        info = self.count_parameters()
        print(f"[NovaMind] Saved to {save_dir} ({info['total_million']}M params)")

    @classmethod
    def load(cls, path, device="auto"):
        """Load a saved NovaMind model from directory."""
        load_dir = Path(path)
        with open(load_dir / "config.json", "r", encoding="utf-8") as f:  # FIXED: added encoding='utf-8'
            config_dict = json.load(f)
        config = NovaMindConfig.from_dict(config_dict)
        if device != "auto":
            config.device = device
        model = cls(config)
        state = torch.load(load_dir / "model.pt", map_location=config.device, weights_only=True)
        model.load_state_dict(state)
        model.to(config.device)
        model.eval()
        info = model.count_parameters()
        print(f"[NovaMind] Loaded from {load_dir} ({info['total_million']}M params, device={config.device})")
        return model
