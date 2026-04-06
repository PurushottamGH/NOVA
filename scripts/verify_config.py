import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from model.architecture import NovaMind
    from model.config import NovaMindConfig

    config = NovaMindConfig()
    model = NovaMind(config)

    print("--- NovaMind 200M Config Verification ---")
    print(f"Vocab Size:      {config.vocab_size}")
    print(f"Embed Dim:       {config.embed_dim}")
    print(f"Num Heads:       {config.num_heads}")
    print(f"Num Layers:      {config.num_layers}")
    print(f"Context Length:  {config.context_length}")
    print(f"Feedforward Dim: {config.feedforward_dim}")
    print(f"Batch Size:      {config.batch_size}")
    print(f"Accum Steps:     {config.accumulation_steps}")
    print(f"Effective Batch: {config.batch_size * config.accumulation_steps}")

    params = model.count_parameters()
    print(f"\nTotal Parameters: {params['total_million']}M")
    print(f"Trainable Params: {params['trainable_million']}M")

except Exception as e:
    print(f"Error during verification: {e}")
    sys.exit(1)
