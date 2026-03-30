"""
NovaMind Evaluation Module
============================
Evaluates model quality through:
1. Perplexity — measures how well the model predicts the next token
   Lower perplexity = better model. PPL = exp(average cross-entropy loss)
   
2. Sample quality — generates text from various prompts for human inspection

Usage:
    python -m inference.evaluate --model_path weights/final_model --tokenizer_path weights/tokenizer
"""

import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List
from tqdm import tqdm

from model.architecture import NovaMind
from tokenizer.tokenizer import NovaMindTokenizer
from inference.generate import generate_text


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, stride: int = 256) -> float:
    """
    Compute perplexity of the model on a given text.
    
    Perplexity = exp(average negative log-likelihood per token)
    
    Lower perplexity means better prediction:
    - PPL ~1: Perfect prediction
    - PPL ~vocab_size: Random guessing
    - Good small LMs: PPL 20-100
    - Large LLMs: PPL 5-20
    
    Uses a sliding window approach for texts longer than context_length.
    
    Args:
        model: NovaMind model
        tokenizer: NovaMindTokenizer
        text: Text to evaluate on
        stride: Sliding window stride for long texts
    
    Returns:
        Perplexity value (float)
    """
    model.eval()
    device = next(model.parameters()).device
    context_length = model.config.context_length

    # Encode text
    token_ids = tokenizer.encode(text)
    seq_len = len(token_ids)

    if seq_len <= 1:
        return float('inf')

    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    total_nll = 0.0  # Total negative log-likelihood
    total_tokens = 0  # Total tokens evaluated

    # Sliding window evaluation
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + context_length + 1, seq_len)
        chunk = token_tensor[begin:end]

        if len(chunk) <= 1:
            continue

        input_ids = chunk[:-1].unsqueeze(0)   # (1, chunk_len-1)
        target_ids = chunk[1:].unsqueeze(0)    # (1, chunk_len-1)

        # Crop input to context_length
        if input_ids.size(1) > context_length:
            input_ids = input_ids[:, :context_length]
            target_ids = target_ids[:, :context_length]

        # Forward pass
        logits, _ = model(input_ids)  # (1, seq_len, vocab_size)

        # Compute per-token NLL
        log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab_size)

        # Gather the log-prob of each target token
        target_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len)

        # Skip padding tokens
        mask = target_ids != model.config.pad_token_id  # (1, seq_len)
        nll = -(target_log_probs * mask).sum()
        total_nll += nll.item()
        total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float('inf')

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(min(avg_nll, 100))  # Cap to prevent overflow

    return perplexity


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    eval_texts: List[str] = None,
    prompts: List[str] = None,
    device: str = "auto",
):
    """
    Run a comprehensive evaluation of a trained NovaMind model.
    
    Args:
        model_path: Path to saved model directory
        tokenizer_path: Path to saved tokenizer directory
        eval_texts: List of texts to compute perplexity on
        prompts: List of prompts to generate samples from
        device: Device to evaluate on
    """
    print("╔══════════════════════════════════════════╗")
    print("║      NovaMind Model Evaluation           ║")
    print("╚══════════════════════════════════════════╝")

    # Load model and tokenizer
    model = NovaMind.load(model_path, device=device)
    tokenizer = NovaMindTokenizer.load(tokenizer_path)

    # Default evaluation texts
    if eval_texts is None:
        eval_texts = [
            "The study of artificial intelligence has grown rapidly in recent years, with new breakthroughs in natural language processing, computer vision, and reinforcement learning pushing the boundaries of what machines can achieve.",
            "Stars form from clouds of gas and dust in space. When these clouds collapse under gravity, the material at the center heats up and eventually begins nuclear fusion, creating a new star.",
            "Machine learning algorithms learn patterns from data without being explicitly programmed. They can be supervised, where labeled examples guide learning, or unsupervised, where the algorithm discovers structure on its own.",
        ]

    # Default prompts
    if prompts is None:
        prompts = [
            "The future of artificial intelligence",
            "In the beginning of the universe",
            "The most important scientific discovery",
            "When I look at the stars",
            "To build a neural network",
        ]

    # === Perplexity Evaluation ===
    print("\n=== Perplexity Evaluation ===")
    perplexities = []

    for i, text in enumerate(eval_texts):
        ppl = compute_perplexity(model, tokenizer, text)
        perplexities.append(ppl)
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  Text {i+1}: PPL = {ppl:.2f}  |  \"{preview}\"")

    avg_ppl = sum(perplexities) / len(perplexities)
    print(f"\n  Average Perplexity: {avg_ppl:.2f}")

    # === Sample Generation ===
    print("\n=== Sample Generations ===")

    for prompt in prompts:
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Generated: \"{generated[:300]}\"")
        print(f"  {'─' * 60}")

    # === Model Info ===
    params = model.count_parameters()
    print(f"\n=== Model Info ===")
    print(f"  Parameters: {params['total_million']}M total, {params['trainable_million']}M trainable")
    print(f"  Device: {model.config.device}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Context length: {model.config.context_length}")

    return {
        "perplexities": perplexities,
        "average_perplexity": avg_ppl,
        "parameters": params,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NovaMind model")
    parser.add_argument("--model_path", type=str, default="weights/final_model")
    parser.add_argument("--tokenizer_path", type=str, default="weights/tokenizer")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.tokenizer_path, device=args.device)
