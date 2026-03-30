"""
NovaMind Test Suite
=====================
Comprehensive tests for all major components.
Each test prints PASS/FAIL with clear error messages.

Usage:
    python test_all.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch


def test_config():
    """Test that config loads correctly and all fields have right types."""
    from model.config import NovaMindConfig
    config = NovaMindConfig()
    assert isinstance(config.vocab_size, int), "vocab_size should be int"
    assert isinstance(config.embed_dim, int), "embed_dim should be int"
    assert isinstance(config.num_heads, int), "num_heads should be int"
    assert isinstance(config.num_layers, int), "num_layers should be int"
    assert isinstance(config.context_length, int), "context_length should be int"
    assert isinstance(config.dropout, float), "dropout should be float"
    assert isinstance(config.learning_rate, float), "learning_rate should be float"
    assert isinstance(config.accumulation_steps, int), "accumulation_steps should be int"
    assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
    assert config.head_dim == config.embed_dim // config.num_heads, "head_dim computation wrong"
    print("PASS: test_config")


def test_tokenizer():
    """Train tokenizer on small text, verify encode/decode round-trip."""
    from tokenizer.tokenizer import NovaMindTokenizer

    tokenizer = NovaMindTokenizer()

    # Write sample text to a temp file
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / "sample.txt"
    temp_file.write_text(
        "Hello Nova is a personal AI assistant. Nova answers questions about AI and space. "
        "Nova was built by Purushottam using PyTorch. Nova is learning every day.",
        encoding="utf-8"
    )

    tokenizer.train([str(temp_file)], vocab_size=100)

    # Test encode
    ids = tokenizer.encode("Hello Nova")
    assert isinstance(ids, list), "encode should return list"
    assert len(ids) > 0, "encode should return non-empty list"
    assert all(isinstance(i, int) for i in ids), "all IDs should be ints"

    # Test decode
    text = tokenizer.decode(ids)
    assert isinstance(text, str), "decode should return string"

    # Test save/load round-trip
    save_dir = temp_dir / "tokenizer"
    tokenizer.save(str(save_dir))
    loaded = NovaMindTokenizer.load(str(save_dir))
    ids2 = loaded.encode("Hello Nova")
    assert ids == ids2, f"Save/load round-trip failed: {ids} != {ids2}"

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("PASS: test_tokenizer")


def test_attention():
    """Test attention forward pass, check shape and no NaN."""
    from model.config import NovaMindConfig
    from model.attention import MultiHeadCausalSelfAttention

    config = NovaMindConfig(embed_dim=64, num_heads=4, context_length=32)
    attn = MultiHeadCausalSelfAttention(config)

    x = torch.randn(2, 16, config.embed_dim)
    output, kv_cache = attn(x)
    assert output.shape == (2, 16, config.embed_dim), f"Wrong shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN detected in attention output!"
    assert kv_cache is None, "KV cache should be None when use_cache=False"

    # Test with KV cache
    output2, kv = attn(x, use_cache=True)
    assert kv is not None, "KV cache should not be None when use_cache=True"
    assert len(kv) == 2, "KV cache should be a tuple of (K, V)"

    print("PASS: test_attention")


def test_model():
    """Test full model forward pass with targets."""
    from model.config import NovaMindConfig
    from model.architecture import NovaMind

    config = NovaMindConfig(vocab_size=100, embed_dim=64, num_heads=4,
                            num_layers=2, context_length=32)
    model = NovaMind(config)

    x = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, targets)

    assert logits.shape == (2, 16, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss is not None, "Loss should not be None with targets"
    assert not torch.isnan(loss), f"NaN loss: {loss}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    print("PASS: test_model")


def test_generate():
    """Test autoregressive generation produces tokens."""
    from model.config import NovaMindConfig
    from model.architecture import NovaMind

    config = NovaMindConfig(vocab_size=100, embed_dim=64, num_heads=4,
                            num_layers=2, context_length=32)
    model = NovaMind(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=10)

    assert out.shape[0] == 1, "Batch size should be 1"
    assert out.shape[1] > 8, f"Should have generated tokens, got shape {out.shape}"
    assert out.shape[1] <= 8 + 20, f"Generated too many tokens: {out.shape[1]}"

    print("PASS: test_generate")


def test_loss_fn():
    """Test loss function with label smoothing."""
    from model.config import NovaMindConfig
    from training.loss import create_loss_fn, compute_perplexity

    config = NovaMindConfig(vocab_size=100)
    loss_fn = create_loss_fn(config, smoothing=0.05)

    logits = torch.randn(32, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (32,))
    loss = loss_fn(logits, targets)

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Test perplexity
    ppl = compute_perplexity(loss)
    assert ppl > 0, "Perplexity should be positive"
    assert ppl < 1e10, f"Perplexity too large: {ppl}"

    print("PASS: test_loss_fn")


def test_checkpointing():
    """Test save/load checkpoint round-trip."""
    from model.config import NovaMindConfig
    from model.architecture import NovaMind
    from training.checkpointing import save_checkpoint, load_checkpoint

    config = NovaMindConfig(vocab_size=100, embed_dim=64, num_heads=4,
                            num_layers=2, context_length=32)
    model = NovaMind(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            model, None, None,
            step=42, loss=1.23, config=config,
            path=tmpdir, best_val_loss=0.99,
        )

        # Check files exist
        assert (Path(tmpdir) / "step_42.pt").exists(), "Step checkpoint not saved"
        assert (Path(tmpdir) / "latest.pt").exists(), "Latest checkpoint not saved"

        # Load and verify step
        step, loss = load_checkpoint(
            str(Path(tmpdir) / "step_42.pt"), model
        )
        assert step == 42, f"Step mismatch: expected 42, got {step}"
        assert abs(loss - 1.23) < 0.01, f"Loss mismatch: expected 1.23, got {loss}"

    print("PASS: test_checkpointing")


def test_sampler():
    """Test sampling strategies."""
    from inference.sampler import combined_sample

    logits = torch.randn(100)
    token = combined_sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    assert 0 <= token.item() < 100, f"Token out of range: {token.item()}"

    # Test greedy
    from inference.sampler import greedy_sample
    greedy_token = greedy_sample(logits)
    assert greedy_token.item() == logits.argmax().item(), "Greedy sample should pick argmax"

    print("PASS: test_sampler")


if __name__ == "__main__":
    print("=" * 50)
    print("  NovaMind Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_config,
        test_tokenizer,
        test_attention,
        test_model,
        test_generate,
        test_loss_fn,
        test_checkpointing,
        test_sampler,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} — {e}")
            failed += 1

    print()
    print("=" * 50)
    if failed == 0:
        print(f"  ✅ All {passed} tests passed — NovaMind is healthy!")
    else:
        print(f"  ❌ {failed} test(s) FAILED, {passed} passed")
    print("=" * 50)
