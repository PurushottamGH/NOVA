"""
NovaRAG: Minimal Retrieval-Augmented Generation for NovaMind
============================================================
Provides a local vector database and retrieval logic to augment 
generations with information from external text documents.

Uses:
- sentence-transformers for fast, local embedding generation.
- FAISS (CPU) for efficient vector storage and search.

Flow:
1. Split documents into overlapping chunks.
2. Embed fragments into vectors.
3. Index with FAISS for fast similarity search.
4. Retrieve top-K chunks for a query and inject into prompt.
"""

import os
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer


class NovaRAG:
    """
    Local RAG pipeline for NovaMind models.
    
    Args:
        model_name: Name of the sentence-transformer model (default: all-MiniLM-L6-v2)
        device: Device to run embeddings on ('cuda', 'cpu', 'mps')
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        print(f"[RAG] Initializing embedding model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = []  # List of strings corresponding to indexed vectors
        
        print(f"[RAG] Embedding dimension: {self.embedding_dim}")

    def add_text_files(self, file_paths: List[str], chunk_size: int = 500, overlap: int = 50):
        """
        Ingest text files, chunk them, and add to the vector index.
        """
        all_chunks = []
        for path in file_paths:
            p = Path(path)
            if not p.exists():
                print(f"[RAG] Warning: File not found: {path} — skipping")
                continue
            
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            # Naive chunking (character-based for simplicity)
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size].strip()
                if len(chunk) > 20:  # Avoid tiny fragments
                    chunks.append(chunk)
            
            all_chunks.extend(chunks)
            print(f"[RAG] Ingested '{p.name}': {len(chunks)} chunks")

        if not all_chunks:
            return

        # Embed and add to FAISS
        embeddings = self.encoder.encode(all_chunks, show_progress_bar=True)
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks.extend(all_chunks)
        print(f"[RAG] Total chunks indexed: {len(self.chunks)}")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for top-K most relevant chunks for a query.
        """
        if self.index.ntotal == 0:
            return []
            
        # Embed query
        query_vec = self.encoder.encode([query]).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vec, k)
        
        # Filter out invalid indices and map to chunks
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

    def save(self, path: str):
        """Save index and chunks to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        with open(save_dir / "chunks.txt", 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                # Replace newlines in chunks to preserve one-per-line structure
                safe_chunk = chunk.replace('\n', ' ')
                f.write(f"{safe_chunk}\n")
        print(f"[RAG] Index saved to {save_dir}")

    def load(self, path: str):
        """Load index and chunks from disk."""
        load_dir = Path(path)
        if not (load_dir / "faiss.index").exists():
            raise FileNotFoundError(f"Index file not found in {path}")
            
        self.index = faiss.read_index(str(load_dir / "faiss.index"))
        with open(load_dir / "chunks.txt", 'r', encoding='utf-8') as f:
            self.chunks = [line.strip() for line in f.readlines()]
        print(f"[RAG] Index loaded: {len(self.chunks)} chunks")


def format_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """Construct a prompt with injected context."""
    if not context_chunks:
        return query
        
    context_str = "\n".join([f"- {c}" for c in context_chunks])
    
    prompt = (
        "Use the following pieces of context to answer the user's question.\n"
        "If you don't know the answer based on the context, just say you don't know.\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        f"Question: {query}\n"
        "Answer: "
    )
    return prompt


def rag_generate(
    model, 
    tokenizer, 
    rag: NovaRAG, 
    query: str, 
    k: int = 3,
    stream: bool = False,
    **gen_kwargs
):
    """
    High-level helper to perform RAG-augmented generation.
    """
    # 1. Retrieve
    context = rag.search(query, k=k)
    
    # 2. Construct Prompt
    full_prompt = format_rag_prompt(query, context)
    
    # 3. Generate
    if stream:
        from inference.generate import stream_generate
        return stream_generate(model, tokenizer, full_prompt, **gen_kwargs)
    else:
        from inference.generate import generate_text
        return generate_text(model, tokenizer, full_prompt, **gen_kwargs)
