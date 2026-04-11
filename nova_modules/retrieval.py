"""
Nova Retriever
=====================
RAG (Retrieval-Augmented Generation) module for NovaMind.
Uses FAISS for fast vector similarity search and SentenceTransformer
for dense embeddings to retrieve relevant knowledge chunks.

Usage:
    retriever = NovaRetriever()
    retriever.add_documents(["Nova is a custom transformer.", "PyTorch is used."])
    results = retriever.search("How does Nova work?")
    print(results)
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class NovaRetriever:
    """
    FAISS-backed retrieval engine for Nova's RAG pipeline.

    - Embeds documents using SentenceTransformer (all-MiniLM-L6-v2)
    - Indexes embeddings in a FAISS flat L2 index
    - Supports search, save/load, and folder indexing with chunking
    """

    def __init__(self, index_path: str = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the NovaRetriever.

        Args:
            index_path: Optional path to load a pre-built FAISS index from.
                        If the path exists, loads the index and documents.
                        If None or path doesn't exist, creates an empty index.
            model_name: SentenceTransformer model name (default: all-MiniLM-L6-v2,
                        which produces 384-dimensional embeddings).
        """
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        self.index_path = index_path

        if index_path:
            try:
                self.load(index_path)
            except FileNotFoundError:
                pass  # Start with empty index if path doesn't exist
            except Exception:
                pass  # Gracefully handle any load errors

    def add_documents(self, texts: list[str]) -> int:
        """
        Add documents to the FAISS index.

        Encodes all texts into dense vectors using the SentenceTransformer
        embedder and adds them to the FAISS index. The original texts are
        stored in a parallel list for retrieval.

        Args:
            texts: List of text strings to index.

        Returns:
            Number of documents added.
        """
        try:
            if not texts:
                return 0

            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)

            self.index.add(embeddings)
            self.documents.extend(texts)

            return len(texts)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the index for documents most similar to the query.

        Args:
            query: Search query string.
            top_k: Number of top results to return (default: 5).

        Returns:
            List of dicts with "text" and "score" keys, sorted by score
            (higher = more relevant). Score = 1 / (1 + distance).
            Returns empty list if index is empty.
        """
        try:
            if self.index.ntotal == 0:
                return []

            query_embedding = self.embedder.encode([query], show_progress_bar=False)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Clamp top_k to number of indexed documents
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                score = 1.0 / (1.0 + float(distances[0][i]))
                results.append({
                    "text": self.documents[idx],
                    "score": round(score, 4),
                })

            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        Build a formatted context string from search results for RAG.

        Calls search() and formats the top results into a structured
        block that can be prepended to a model prompt.

        Args:
            query: Search query string.
            top_k: Number of results to include (default: 3).

        Returns:
            Formatted context string with retrieved knowledge.
        """
        try:
            results = self.search(query, top_k=top_k)
            if not results:
                return "[Retrieved Knowledge]\nNo relevant documents found.\n[End of Retrieved Knowledge]"

            lines = ["[Retrieved Knowledge]"]
            for i, r in enumerate(results, 1):
                lines.append(f"--- Result {i} (score: {r['score']:.2f}) ---")
                lines.append(r["text"])
            lines.append("[End of Retrieved Knowledge]")

            return "\n".join(lines)
        except Exception as e:
            return f"[Retrieved Knowledge]\nError: {e}\n[End of Retrieved Knowledge]"

    def save(self, path: str = None):
        """
        Save the FAISS index and documents to disk.

        Args:
            path: Base path for saving (without extension).
                  Saves {path}.faiss and {path}.docs.json.
                  Uses self.index_path if path is None.
        """
        try:
            save_path = path or self.index_path
            if not save_path:
                print("No save path specified.")
                return

            faiss.write_index(self.index, f"{save_path}.faiss")

            with open(f"{save_path}.docs.json", "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            print(f"Saved index ({self.index.ntotal} vectors) to {save_path}")
        except Exception as e:
            print(f"Save error: {e}")

    def load(self, path: str):
        """
        Load a FAISS index and documents from disk.

        Args:
            path: Base path to load from (without extension).
                  Expects {path}.faiss and {path}.docs.json to exist.

        Raises:
            FileNotFoundError: If the index or docs file doesn't exist.
        """
        faiss_path = f"{path}.faiss"
        docs_path = f"{path}.docs.json"

        if not Path(faiss_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not Path(docs_path).exists():
            raise FileNotFoundError(f"Documents file not found: {docs_path}")

        try:
            self.index = faiss.read_index(faiss_path)

            with open(docs_path, encoding="utf-8") as f:
                self.documents = json.load(f)

            self.index_path = path
            print(f"Loaded index ({self.index.ntotal} vectors) from {path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise e
        except Exception as e:
            print(f"Load error: {e}")

    def chunk_text(self, text: str, chunk_size: int = 512,
                   overlap: int = 64) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk.
            chunk_size: Maximum characters per chunk (default: 512).
            overlap: Number of overlapping characters between consecutive
                     chunks (default: 64).

        Returns:
            List of chunk strings.
        """
        try:
            if not text:
                return []

            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = start + chunk_size
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append(chunk)
                start += chunk_size - overlap

            return chunks
        except Exception as e:
            print(f"Chunking error: {e}")
            return []

    def index_from_folder(self, folder_path: str,
                          extensions: list[str] = None) -> int:
        """
        Recursively index all matching files from a folder.

        Walks the folder tree, reads files matching the given extensions,
        chunks each file into 512-character overlapping chunks, and adds
        all chunks to the FAISS index.

        Args:
            folder_path: Path to the folder to index.
            extensions: List of file extensions to include
                        (default: [".txt", ".md", ".py"]).

        Returns:
            Total number of chunks indexed.
        """
        if extensions is None:
            extensions = [".txt", ".md", ".py"]

        try:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"Folder not found: {folder_path}")
                return 0

            all_chunks = []

            for ext in extensions:
                for file_path in folder.rglob(f"*{ext}"):
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        if content.strip():
                            chunks = self.chunk_text(content)
                            # Prefix each chunk with source file info
                            for chunk in chunks:
                                prefixed = f"[Source: {file_path.name}]\n{chunk}"
                                all_chunks.append(prefixed)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

            if all_chunks:
                added = self.add_documents(all_chunks)
                print(f"Indexed {added} chunks from {folder_path}")
                return added

            print(f"No matching files found in {folder_path}")
            return 0
        except Exception as e:
            print(f"Folder indexing error: {e}")
            return 0


if __name__ == "__main__":
    r = NovaRetriever()
    r.add_documents([
        "Nova is a custom transformer.",
        "NovaMind uses PyTorch.",
        "Blender uses bpy for scripting.",
    ])
    results = r.search("How does Nova work?")
    print(results)
