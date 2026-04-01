"""
NovaMind Quick Chat — Terminal Interface
==========================================
Talk to Nova directly from the terminal after training.

Usage:
    python scripts/quick_chat.py
    python scripts/quick_chat.py --model_path weights/best.pt --tokenizer_path tokenizer_data
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# 1. Use all CPU cores
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(2)

# 2. INT8 Quantization + 3. Compilation functions
def optimize_model(model):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Quantization only works on CPU
    if device.type == "cpu":
        print("[Nova] Applying dynamic quantization for CPU speedup...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        # On GPU, we use Half Precision (FP16) for a speed boost instead
        print(f"[Nova] Using FP16 acceleration on {device}")
        # Note: model.half() is standard for GPU inference
        model = model.half()

    base_model = model
    try:
        # Only compile if the platform supports it (mostly Linux)
        if hasattr(torch, "compile"):
            compiled_model = torch.compile(model, mode="reduce-overhead")
            # Force evaluation to trigger lazy compilation immediately
            with torch.inference_mode():
                dummy_input = torch.zeros(1, 1, dtype=torch.long, device=device)
                _ = compiled_model(dummy_input)
            print("[Nova] Model compiled for faster inference")
            return compiled_model
        return base_model
    except Exception as e:
        print(f"[Nova] Running without compile (Compiler not available: {type(e).__name__})")
        return base_model

from model.config import NovaMindConfig
from model.architecture import NovaMind
from tokenizer.tokenizer import NovaMindTokenizer
from inference.generate import generate_text


# Nova's system prompt
SYSTEM_PROMPT = (
    "You are Nova, an intelligent personal AI assistant built for Purushottam. "
    "You are knowledgeable in AI, space, astronomy, data science, and software engineering."
)


def format_prompt(user_message: str, history: list) -> str:
    """Build the full prompt with system prompt + history + new message."""
    parts = [f"<|system|>\n{SYSTEM_PROMPT}\n"]
    for turn in history:
        parts.append(f"<|user|>\n{turn['user']}\n")
        parts.append(f"<|assistant|>\n{turn['nova']}\n")
    parts.append(f"<|user|>\n{user_message}\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Chat with Nova in the terminal")
    parser.add_argument("--model_path", type=str, default="weights/final_model",
                        help="Path to saved model directory")
    parser.add_argument("--tokenizer_path", type=str, default="weights/tokenizer",
                        help="Path to saved tokenizer directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    args = parser.parse_args()

    print("=" * 50)
    print("  NOVA — Personal AI Assistant")
    print("  Powered by NovaMind (your own LLM)")
    print("  Type 'exit' or 'quit' to leave")
    print("=" * 50)

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)

    # Load tokenizer
    if (tokenizer_path / "vocab.json").exists():
        print(f"\n[Loading tokenizer from {tokenizer_path}...]")
        tokenizer = NovaMindTokenizer.load(str(tokenizer_path))
    else:
        print(f"\n[WARNING] No tokenizer found at {tokenizer_path}")
        print("[Training a tiny tokenizer on sample text...]")
        tokenizer = NovaMindTokenizer()
        sample = "Hello Nova is a personal AI assistant. Nova answers questions about AI and space."
        temp_file = Path("scripts/_temp_chat.txt")
        temp_file.write_text(sample, encoding="utf-8")
        tokenizer.train([str(temp_file)], vocab_size=200)
        temp_file.unlink(missing_ok=True)

    # Load model
    config_path = model_path / "config.json"
    if config_path.exists():
        print(f"[Loading model from {model_path}...]")
        model = NovaMind.load(str(model_path), device=args.device)
    else:
        print(f"[WARNING] No trained model found at {model_path}")
        print("[Creating random model for demo — responses will be gibberish until trained]")
        config = NovaMindConfig(vocab_size=tokenizer.vocab_size, device=args.device)
        model = NovaMind(config)
        model.to(config.device)

    # Apply optimizations
    model = optimize_model(model)
    print("\n[Nova is ready. Start chatting!]\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNova: Goodbye!")
            break

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print("Nova: Goodbye!")
            break

        if not user_input:
            continue

        # Build prompt
        prompt = format_prompt(user_input, history)

        # Generate response
        try:
            start_time = time.time()
            response = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=200,
                temperature=0.8, top_k=50, top_p=0.9,
                repetition_penalty=1.15,
            )
            elapsed = time.time() - start_time

            # Extract Nova's response
            nova_response = response
            if "<|assistant|>" in nova_response:
                parts = nova_response.rsplit("<|assistant|>", 1)
                if len(parts) > 1:
                    nova_response = parts[1].strip()

            # Stop at "<|user|>" if model continues
            if "<|user|>" in nova_response:
                nova_response = nova_response.split("<|user|>")[0].strip()

            if not nova_response:
                nova_response = "I need more training data to answer that."

        except Exception as e:
            nova_response = f"[Error: {e}]"
            elapsed = 0.0

        print(f"Nova: {nova_response}\n")
        if elapsed > 0:
            print(f"[Nova] Response generated in {elapsed:.1f} seconds\n")
            
        history.append({"user": user_input, "nova": nova_response})


if __name__ == "__main__":
    main()
