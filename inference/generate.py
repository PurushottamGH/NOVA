"""
NovaMind Token-by-Token Streaming Generation
===============================================
Provides a generator-based interface for text generation that yields
tokens one at a time, enabling streaming responses.

This module works with the sampler module for controlled generation
and provides both batch and streaming generation functions.

Usage:
    # Streaming generation (yields tokens one at a time)
    for token_text in stream_generate(model, tokenizer, "Hello"):
        print(token_text, end="", flush=True)
    
    # Batch generation (returns full text)
    text = generate_text(model, tokenizer, "Hello", max_tokens=100)
"""

import torch
from typing import Generator

from inference.sampler import combined_sample


@torch.inference_mode()
def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> Generator[str, None, None]:
    """
    Generator that yields decoded tokens one at a time for streaming.
    
    Args:
        model: NovaMind model (already on correct device)
        tokenizer: NovaMindTokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
    
    Yields:
        Decoded text for each generated token
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, seq_len)

    generated_ids = prompt_ids.copy()  # Track all generated IDs for repetition penalty
    context_length = model.config.context_length
    eos_id = model.config.eos_token_id

    past_kv = None

    for _ in range(max_new_tokens):
        if past_kv is None:
            # First pass: process the full prompt
            seq = input_ids[:, -context_length:]  # (1, ≤context_length)
        else:
            # Subsequent passes: only process the last generated token
            seq = input_ids[:, -1:]  # (1, 1)

        # Forward pass returning next token logits and updated KV cache
        logits, past_kv = model(seq, past_key_values=past_kv, use_cache=True)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Sample next token using combined strategy
        next_token_id = combined_sample(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        ).item()

        # Check for EOS
        if next_token_id == eos_id:
            break

        # Append to sequence
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        generated_ids.append(next_token_id)

        # Decode the single new token
        # We decode the last 2 tokens and subtract the previous decode to handle
        # multi-character subwords correctly
        token_text = tokenizer.decode([next_token_id])
        yield token_text


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate text from a prompt (non-streaming, returns full result).
    
    Args:
        model: NovaMind model
        tokenizer: NovaMindTokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Complete generated text string
    """
    tokens = []
    for token_text in stream_generate(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    ):
        tokens.append(token_text)

    return prompt + "".join(tokens)
