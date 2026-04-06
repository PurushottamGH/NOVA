"""
NovaMind Training Entry Point
================================
Main script to run end-to-end training. Orchestrates:
1. Configuration
2. Data loading
3. Tokenizer training (or loading)
4. Model creation
5. Training loop
6. Final evaluation and sample generation

Usage:
    python -m training.train
    python -m training.train --resume
    python -m training.train --data_dir personal_data --vocab_size 8000
"""

import sys
import argparse
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import NovaMindConfig
from model.architecture import NovaMind
from model.utils import model_summary, estimate_flops
from tokenizer.tokenizer import NovaMindTokenizer
from data.dataloader import create_dataloaders, get_text_files
from training.trainer import Trainer
from training.checkpoint import is_checkpoint_stable, resume_training


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train NovaMind from scratch")
    parser.add_argument("--data_dir", type=str, default="personal_data", help="Directory with training .txt files")
    parser.add_argument("--vocab_size", type=int, default=8000, help="BPE vocabulary size")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--new", action="store_true", help="Force new training from scratch")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu/mps")
    parser.add_argument("--checkpoint_dir", type=str, default="weights", help="Checkpoint directory")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer_data", help="Tokenizer save directory")
    args = parser.parse_args()

    # Automatic resume detection
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_file = checkpoint_dir / "checkpoint"
    
    if checkpoint_file.exists() and not args.new:
        args.resume = True
        print(f"  [Resume] Found explicit checkpoint file: {checkpoint_file}")
    elif is_checkpoint_stable(str(checkpoint_dir)) and not args.new:
        args.resume = True
        print(f"  [Auto-Resume] Found existing checkpoint in '{checkpoint_dir}'")
    elif args.new:
        print(f"  [Fresh Start] '--new' flag detected — starting from step 0")
    else:
        print(f"  [Fresh Start] No checkpoint found in '{checkpoint_dir}' — starting from step 0")

    print("╔══════════════════════════════════════════╗")
    print("║       NovaMind Training Pipeline         ║")
    print("║       Built by Purushottam               ║")
    print("╚══════════════════════════════════════════╝")

    # ==========================================
    # Step 1: Configuration
    # ==========================================
    config = NovaMindConfig(
        vocab_size=args.vocab_size,
        device=args.device,
    )
    config.save_every = 10000  # Save checkpoint every 10,000 steps
    
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size

    print(f"\n{config}")

    # ==========================================
    # Step 2: Get training data files
    # ==========================================
    print("\n[Step 2] Loading training data...")
    try:
        text_files = get_text_files(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python -m data.collector' first to download training data.")
        sys.exit(1)

    # ==========================================
    # Step 3: Train or load tokenizer
    # ==========================================
    print("\n[Step 3] Preparing tokenizer...")
    tokenizer_path = Path(args.tokenizer_dir)
    vocab_path = tokenizer_path / "vocab.json"

    if vocab_path.exists():
        print(f"  Loading existing tokenizer from {tokenizer_path}")
        tokenizer = NovaMindTokenizer.load(str(tokenizer_path))
    else:
        print(f"  Training new BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = NovaMindTokenizer()
        tokenizer.train(text_files, vocab_size=args.vocab_size)
        tokenizer.save(str(tokenizer_path))

    # Update config vocab_size to match actual tokenizer
    config.vocab_size = tokenizer.vocab_size
    print(f"  Actual vocab size: {config.vocab_size}")

    # ==========================================
    # Step 4: Create DataLoaders
    # ==========================================
    print("\n[Step 4] Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        text_files=text_files,
        tokenizer=tokenizer,
        config=config,
        train_split=0.9,
    )

    # ==========================================
    # Step 5: Create or load model
    # ==========================================
    print("\n[Step 5] Creating NovaMind model...")
    model = NovaMind(config)
    model_summary(model)
    estimate_flops(config)

    # Optimization: Use torch.compile for 2.0+ performance boost
    if torch.__version__ >= "2.0":
        print("  [Optim] Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ==========================================
    # Step 6: Train!
    # ==========================================
    print("\n[Step 6] Starting training...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    trainer.train()

    # ==========================================
    # Step 7: Final evaluation
    # ==========================================
    print("\n[Step 7] Final evaluation...")
    val_loss = trainer.evaluate()
    import math
    perplexity = math.exp(min(val_loss, 20))
    print(f"  Final Validation Loss: {val_loss:.4f}")
    print(f"  Final Perplexity: {perplexity:.2f}")

    # Generate samples from different prompts
    prompts = [
        "The universe is",
        "Artificial intelligence will",
        "In the year 2050",
        "The most important thing in life",
        "Stars are born when",
    ]
    print("\n[Step 8] Sample generations:")
    for prompt in prompts:
        trainer.generate_sample(prompt_text=prompt, max_tokens=80)

    # Save final model
    model.save(str(Path(args.checkpoint_dir) / "final_model"))  # FIXED: use Path for Windows compat
    tokenizer.save(str(Path(args.checkpoint_dir) / "tokenizer"))  # FIXED: use Path for Windows compat

    print("\n✅ Training complete! Model saved to", args.checkpoint_dir)


if __name__ == "__main__":
    main()
