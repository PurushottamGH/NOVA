"""
NovaMind RAG Chat Example
==========================
This script demonstrates how to use the NovaRAG pipeline to augment
NovaMind's knowledge. It:
1. Loads the NovaMind model and tokenizer.
2. Initializes the NovaRAG search index.
3. Indexes local text files for retrieval.
4. Performs a retrieval-augmented query.

Usage:
    python -m inference.chat_rag --checkpoint_dir weights/ --data_dir personal_data/
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.rag import NovaRAG, rag_generate
from model.architecture import NovaMind
from model.config import NovaMindConfig
from tokenizer.tokenizer import NovaMindTokenizer


def main():
    parser = argparse.ArgumentParser(description="NovaMind RAG Chat Example")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="weights", help="Model weights directory"
    )
    parser.add_argument(
        "--data_dir", type=str, default="personal_data", help="Directory with documents to index"
    )
    parser.add_argument("--query", type=str, default="What is NovaMind?", help="Question to ask")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    # 1. Load Config & Model
    print("\n[RAG] Loading NovaMind model...")
    config = NovaMindConfig.from_dict(
        torch.load(f"{args.checkpoint_dir}/latest.pt", weights_only=False)["config"]
    )
    if args.device != "auto":
        config.device = args.device

    model = NovaMind(config)
    # Load weights
    checkpoint = torch.load(
        f"{args.checkpoint_dir}/latest.pt", map_location=config.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    # 2. Load Tokenizer
    tokenizer = NovaMindTokenizer.load("tokenizer_data")

    # 3. Initialize & Index RAG
    print("\n[RAG] Initializing NovaRAG search system...")
    rag = NovaRAG(device=config.device)

    # Get text files to index
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return

    text_files = [str(f) for f in data_path.glob("*.txt")]
    if not text_files:
        print(f"Warning: No text files found in {args.data_dir}. RAG will have no context.")
    else:
        print(f"[RAG] Indexing {len(text_files)} files...")
        rag.add_text_files(text_files, chunk_size=400, overlap=50)

    # 4. Perform Retrieval-Augmented Generation
    print(f"\n[RAG] Prompt: {args.query}")
    print("[RAG] Retrieving context and generating response...")

    # Synchronous generation for the example
    response = rag_generate(
        model=model,
        tokenizer=tokenizer,
        rag=rag,
        query=args.query,
        k=3,
        max_new_tokens=150,
        temperature=0.7,
    )

    print("\n" + "=" * 50)
    print("NOVA Response with RAG:")
    print("=" * 50)
    print(response.split("Answer: ")[-1].strip())
    print("=" * 50)


if __name__ == "__main__":
    main()
