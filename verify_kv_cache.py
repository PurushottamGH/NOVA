
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import NovaMind
from model.config import NovaMindConfig
from tokenizer.tokenizer import NovaMindTokenizer

def verify():
    # 1. Setup config and model
    config = NovaMindConfig(
        vocab_size=100,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        context_length=32
    )
    model = NovaMind(config)
    model.eval()

    # 2. Dummy inputs
    input_ids = torch.randint(0, 100, (1, 10))
    
    # 3. Standard forward pass
    with torch.inference_mode():
        full_logits, _ = model(input_ids)
        target_logits = full_logits[:, -1, :]

    # 4. KV Cache forward pass
    # Pass 1: Get cache from initial prompt
    with torch.inference_mode():
        logits_1, past_kv = model(input_ids, use_cache=True)
        # Pass 2: Use last token + cache
        next_input = torch.randint(0, 100, (1, 1)) # placeholder for now, 
        # but let's test if appending a token matches the full forward.
        
        # To verify correctly:
        # Full forward: [t0, t1, ..., t9] -> logits for t10
        # KV forward: [t0, ..., t8] -> KV cache, then [t9] + cache -> logits for t10
        
        # Standard:
        logits_full, _ = model(input_ids)
        
        # KV Cache:
        _, past_kv = model(input_ids[:, :-1], use_cache=True)
        last_token = input_ids[:, -1:]
        logits_kv, _ = model(last_token, past_key_values=past_kv, use_cache=True)
        
        # Compare
        diff = torch.abs(logits_full[:, -1, :] - logits_kv[:, -1, :]).max().item()
        print(f"Max difference between full forward and KV cache: {diff:.8f}")
        
        if diff < 1e-4:
            print("✅ KV cache verification passed!")
        else:
            print("❌ KV cache verification failed!")

if __name__ == "__main__":
    verify()
