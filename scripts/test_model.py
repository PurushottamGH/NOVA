"""
NovaMind Full Pipeline Test
==============================
Verifies the entire pipeline works end-to-end:
1. Config loads
2. Tokenizer trains
3. Model creates
4. Forward pass works
5. Loss computes
6. Generation works
7. All systems go

Usage:
    python scripts/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model.config import NovaMindConfig
from model.architecture import NovaMind


def main():
    print("=" * 55)
    print("  NovaMind Full Pipeline Test")
    print("=" * 55)

    # Step 1: Load config
    print("\n[Step 1] Loading config...")
    config = NovaMindConfig(vocab_size=200, context_length=64, num_layers=2, embed_dim=64, num_heads=4)
    print(f"  Config loaded — embed_dim={config.embed_dim}, heads={config.num_heads}, layers={config.num_layers}")
    print("  ✓ OK")

    # Step 2: Create and train tokenizer
    print("\n[Step 2] Training tokenizer on sample text...")
    from tokenizer.tokenizer import NovaMindTokenizer
    tokenizer = NovaMindTokenizer()
    sample_text = (
        "Hello Nova is a personal AI assistant built by Purushottam. "
        "Nova can answer questions about artificial intelligence, space, astronomy, "
        "data science, and software engineering. Nova is powered by NovaMind, "
        "a custom transformer language model trained from scratch using pure PyTorch."
    )
    # Write to temp file for tokenizer training
    temp_dir = Path(__file__).resolve().parent.parent / "scripts" / "_temp"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "sample.txt"
    temp_file.write_text(sample_text, encoding="utf-8")
    tokenizer.train([str(temp_file)], vocab_size=config.vocab_size)
    config.vocab_size = tokenizer.vocab_size  # Sync
    print(f"  Tokenizer trained — vocab_size={tokenizer.vocab_size}")

    # Quick encode/decode test
    ids = tokenizer.encode("Hello Nova")
    decoded = tokenizer.decode(ids)
    print(f"  Encode 'Hello Nova' → {ids[:8]}...")
    print(f"  Decode back → '{decoded}'")
    print("  ✓ OK")

    # Step 3: Create model
    print("\n[Step 3] Creating NovaMind model...")
    model = NovaMind(config)
    params = model.count_parameters()
    print(f"  Model created — {params['total_million']}M params ({params['total']:,} total)")
    print("  ✓ OK")

    # Step 4: Forward pass with dummy data
    print("\n[Step 4] Forward pass (no targets)...")
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, config.vocab_size), f"Wrong shape: {logits.shape}"
    assert loss is None, "Loss should be None without targets"
    assert not torch.isnan(logits).any(), "NaN detected in logits!"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  No NaN: ✓")
    print("  ✓ OK")

    # Step 5: Forward with targets
    print("\n[Step 5] Forward pass (with targets)...")
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, targets)
    assert loss is not None, "Loss should not be None with targets"
    assert not torch.isnan(loss), f"NaN loss detected!"
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ OK")

    # Step 6: Generation
    print("\n[Step 6] Generating tokens...")
    model.eval()
    prompt_ids = tokenizer.encode("Hello")
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(input_tensor, max_new_tokens=20, temperature=1.0, top_k=10)
    assert generated.shape[1] > len(prompt_ids), "No tokens were generated"
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt: 'Hello' ({len(prompt_ids)} tokens)")
    print(f"  Generated: {generated.shape[1]} total tokens")
    print(f"  Decoded: '{decoded[:100]}'")
    print("  ✓ OK")

    # Cleanup temp
    temp_file.unlink(missing_ok=True)
    temp_dir.rmdir()

    print("\n" + "=" * 55)
    print("  ✅ All systems go — NovaMind is ready to train!")
    print("=" * 55)


if __name__ == "__main__":
    main()
