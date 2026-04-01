"""
NovaMind Token Sampling Strategies
=====================================
Implements various sampling methods for controlling text generation diversity.

Available strategies:
- greedy_sample: Always pick the most probable token (deterministic)
- temperature_sample: Scale logits by temperature before sampling
- top_k_sample: Keep only top-k most probable tokens
- top_p_sample: Nucleus sampling — keep tokens until cumulative prob > p
- apply_repetition_penalty: Reduce probability of already-generated tokens
- combined_sample: Apply all strategies in the correct order

Correct application order:
1. Repetition penalty (modify raw logits)
2. Temperature scaling (control randomness)
3. Top-k filtering (remove unlikely tokens)
4. Top-p filtering (nucleus sampling)
5. Sample from resulting distribution
"""

import torch
import torch.nn.functional as F


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling: always select the token with highest probability.
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
    
    Returns:
        Single token ID (scalar tensor)
    """
    return logits.argmax(dim=-1)  # scalar


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from logits scaled by temperature.
    
    temperature < 1.0: Sharper distribution (more confident, less random)
    temperature = 1.0: No change (standard sampling)
    temperature > 1.0: Flatter distribution (more random, more creative)
    
    Math: P(token) = softmax(logits / temperature)
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
        temperature: Scaling factor (must be > 0)
    
    Returns:
        Sampled token ID (scalar tensor)
    """
    assert temperature > 0, "Temperature must be positive"
    scaled_logits = logits / temperature  # (vocab_size,)
    probs = F.softmax(scaled_logits, dim=-1)  # (vocab_size,)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # scalar


def top_k_sample(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Top-k sampling: keep only the k most probable tokens.
    
    All tokens outside the top-k have their logits set to -inf,
    which gives them zero probability after softmax.
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
        k: Number of top tokens to keep
    
    Returns:
        Filtered logits (vocab_size,) with non-top-k set to -inf
    """
    if k <= 0 or k >= logits.size(-1):
        return logits  # No filtering needed

    # Get the k-th largest value as the threshold
    top_k_values, _ = torch.topk(logits, k)  # (k,)
    threshold = top_k_values[-1]  # k-th largest value (scalar)

    # Set all values below threshold to -inf
    filtered = logits.clone()
    filtered[filtered < threshold] = float('-inf')
    return filtered  # (vocab_size,)


def top_p_sample(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens
    whose cumulative probability mass >= p.
    
    This adaptively adjusts the number of tokens considered:
    - When the model is confident, fewer tokens are kept
    - When uncertain, more tokens are kept
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
        p: Cumulative probability threshold (0.0 to 1.0)
    
    Returns:
        Filtered logits (vocab_size,) with low-prob tokens set to -inf
    """
    if p >= 1.0:
        return logits  # No filtering

    # Sort tokens by probability (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (vocab_size,)
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )  # (vocab_size,) cumulative probabilities

    # Find where cumulative probability exceeds threshold p
    # Mark tokens to remove (cumulative prob > p, except always keep first token)
    sorted_indices_to_remove = cumulative_probs > p  # (vocab_size,) boolean mask
    # Shift right to always keep at least one token
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False  # Always keep the top token

    # Scatter the removal mask back to original indices
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)

    # Set removed tokens to -inf
    filtered = logits.clone()
    filtered[indices_to_remove] = float('-inf')
    return filtered  # (vocab_size,)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list,
    penalty: float = 1.1
) -> torch.Tensor:
    """
    Reduce the probability of tokens that have already been generated.
    
    For each previously generated token:
    - If its logit is positive: divide by penalty (reduce probability)
    - If its logit is negative: multiply by penalty (make more negative)
    
    This discourages the model from repeating itself.
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
        generated_ids: List of previously generated token IDs
        penalty: Repetition penalty factor (1.0 = no penalty)
    
    Returns:
        Modified logits (vocab_size,)
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    modified = logits.clone()
    unique_ids = set(generated_ids)

    for token_id in unique_ids:
        if token_id < modified.size(0):
            if modified[token_id] > 0:
                modified[token_id] = modified[token_id] / penalty
            else:
                modified[token_id] = modified[token_id] * penalty

    return modified  # (vocab_size,)


def combined_sample(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_ids: list = None,
) -> torch.Tensor:
    """
    Apply all sampling strategies in the correct order and sample a token.
    
    Pipeline:
    1. Repetition penalty (on raw logits)
    2. Temperature scaling
    3. Top-k filtering
    4. Top-p filtering
    5. Sample from the resulting distribution
    
    Args:
        logits: (vocab_size,) unnormalized log-probabilities
        temperature: Sampling temperature
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
        generated_ids: List of previously generated token IDs
    
    Returns:
        Sampled token ID (scalar tensor)
    """
    if generated_ids is None:
        generated_ids = []

    # Step 1: Apply repetition penalty
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Step 2: Apply temperature scaling
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Step 3: Apply top-k filtering
    if top_k > 0:
        logits = top_k_sample(logits, k=top_k)

    # Step 4: Apply top-p (nucleus) filtering
    if top_p < 1.0:
        logits = top_p_sample(logits, p=top_p)

    # Step 5: Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)  # (vocab_size,)

    # Handle edge case: all probs are 0 (all filtered out)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.size(0)

    token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # scalar
    return token_id
