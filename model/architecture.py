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
from torch.utils.checkpoint import checkpoint  # FIXED: added for VRAM efficiency
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

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False):
        """
        Forward pass.
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target IDs for loss computation, or None
            past_key_values: list of (K, V) from previous passes for each block
            use_cache: whether to return KV cache for future calls
        Returns:
            (logits, loss) or (logits, past_key_values_out) if use_cache=True
            (logits, loss) where logits is (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape  # (batch, seq_len)
        assert seq_len <= self.config.context_length

        # Step 1: Token embedding (batch, seq_len) → (batch, seq_len, embed_dim)
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Step 2: Add positional encoding
        start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        x = self.positional_encoding(token_emb, start_pos=start_pos)  # (batch, seq_len, embed_dim)

        # Step 3: Pass through all transformer blocks
        past_key_values_out = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self.training and self.config.gradient_checkpointing:
                # Gradient checkpointing trades computation for memory:
                # Blocks are re-computed during the backward pass instead of storing activations.
                # Signature: block(x, past_kv=None, use_cache=False)
                x, present_kv = checkpoint(block, x, None, False, use_reentrant=False)
            else:
                past_kv = past_key_values[i] if past_key_values is not None else None
                x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            
            if use_cache:
                past_key_values_out.append(present_kv)

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

        if use_cache:
            return logits, past_key_values_out
        
        return logits, loss

    @torch.inference_mode()
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

        past_kv = None

        for _ in range(max_new_tokens):
            if past_kv is None:
                # Crop to context window
                seq = generated[:, -self.config.context_length:]  # (batch, ≤context_length)
            else:
                seq = generated[:, -1:]  # (batch, 1)

            logits, past_kv = self.forward(seq, past_key_values=past_kv, use_cache=True)  # (batch, seq_len, vocab_size)
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
        if device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config.device = device
        model = cls(config)
        state = torch.load(load_dir / "model.pt", map_location=config.device, weights_only=True, mmap=True)
        
        # Handle state_dict from old architecture (W_q, W_k, W_v separate)
        keys_to_delete = []
        new_state = {}
        for key in list(state.keys()):
            if key.endswith(".W_q.weight"):
                prefix = key[:-len("W_q.weight")]
                q_w = state[key]
                k_w = state[prefix + "W_k.weight"]
                v_w = state[prefix + "W_v.weight"]
                new_state[prefix + "W_qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                keys_to_delete.extend([key, prefix + "W_k.weight", prefix + "W_v.weight"])
                
        for key in keys_to_delete:
            if key in state:
                del state[key]
        state.update(new_state)

        model.load_state_dict(state)
        model.to(config.device)
        model.eval()
        info = model.count_parameters()
        print(f"[NovaMind] Loaded from {load_dir} ({info['total_million']}M params, device={config.device})")
        return model
