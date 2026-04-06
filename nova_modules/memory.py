import os
from datetime import datetime, timedelta

import chromadb


class NovaMemory:
    """
    Persistent long-term memory for Nova using ChromaDB.
    Remembers conversations and key facts across separate sessions.

    Features:
        - Automatic TTL (conversations expire after configurable days)
        - Maximum capacity limit with oldest-first eviction
        - Periodic cleanup on store operations
    """

    # Default TTL: 30 days
    DEFAULT_TTL_DAYS = 30
    # Maximum number of conversation turns to keep
    MAX_ENTRIES = 5000
    # Cleanup frequency: run cleanup every N store operations
    CLEANUP_INTERVAL = 50

    def __init__(
        self,
        persist_dir: str = "nova_memory",
        ttl_days: int | None = None,
        max_entries: int | None = None,
    ):
        """
        Initialize the memory engine.

        Args:
            persist_dir: Directory for ChromaDB persistence.
            ttl_days: Days before conversations expire (default: 30).
            max_entries: Maximum stored conversations (default: 5000).
        """
        # Step 1: ChromaDB Cache Fix
        # Prevent repetitive 79MB downloads by pinning the cache directory
        CHROMA_CACHE = os.path.expanduser("~/.cache/chroma")
        os.makedirs(CHROMA_CACHE, exist_ok=True)
        os.environ["CHROMA_CACHE_DIR"] = CHROMA_CACHE

        self.ttl_days = ttl_days if ttl_days is not None else self.DEFAULT_TTL_DAYS
        self.max_entries = max_entries if max_entries is not None else self.MAX_ENTRIES
        self._store_count = 0

        # Step 2: Resilient Initialization
        try:
            # Ensure the persistence directory exists
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="nova_conversations", metadata={"hnsw:space": "cosine"}
            )
            self._available = True
            print(
                f"[NovaMemory] Persistent memory connected at {persist_dir} "
                f"(TTL={self.ttl_days}d, max={self.max_entries})"
            )
        except Exception as e:
            print(f"[NovaMemory] Warning: ChromaDB unavailable: {e}")
            self._available = False
            self.collection = None

    def _cleanup_expired(self):
        """Remove conversations older than TTL and enforce max capacity."""
        if not self._available:
            return

        try:
            count = self.collection.count()
            if count == 0:
                return

            # Fetch all metadata to find expired entries
            cutoff = datetime.now() - timedelta(days=self.ttl_days)
            cutoff_iso = cutoff.isoformat()

            # Get all entries sorted by timestamp (oldest first)
            all_results = self.collection.get(
                include=["metadatas"],
            )

            ids_to_delete = []
            if all_results and all_results.get("metadatas"):
                for i, metadata in enumerate(all_results["metadatas"]):
                    ts = metadata.get("timestamp", "")
                    if ts < cutoff_iso:
                        ids_to_delete.append(all_results["ids"][i])

            # Delete expired entries
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(f"[NovaMemory] Cleaned up {len(ids_to_delete)} expired entries")

            # Enforce max capacity — delete oldest if over limit
            remaining = self.collection.count()
            if remaining > self.max_entries:
                excess = remaining - self.max_entries
                # Get oldest entries to delete
                oldest = self.collection.get(
                    include=["metadatas"],
                    limit=excess,
                )
                if oldest and oldest.get("ids"):
                    self.collection.delete(ids=oldest["ids"])
                    print(f"[NovaMemory] Enforced capacity: removed {excess} oldest entries")

        except Exception as e:
            # Don't fail store operations if cleanup fails
            print(f"[NovaMemory] Cleanup warning: {e}")

    def remember(self, user_msg: str, nova_response: str):
        """
        Store a conversation turn in the vector database.
        Automatically cleans up expired entries periodically.
        """
        if not self._available:
            return

        # Use timestamp as unique ID
        doc_id = f"conv_{datetime.now().timestamp()}"

        self.collection.add(
            documents=[f"User: {user_msg}\nNova: {nova_response}"],
            ids=[doc_id],
            metadatas=[{"timestamp": datetime.now().isoformat(), "type": "conversation"}],
        )

        # Periodic cleanup
        self._store_count += 1
        if self._store_count >= self.CLEANUP_INTERVAL:
            self._store_count = 0
            self._cleanup_expired()

    def recall(self, query: str, n_results: int = 3) -> list:
        """
        Retrieve relevant past conversations based on a search query.
        """
        if not self._available or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query], n_results=min(n_results, self.collection.count())
        )

        return results["documents"][0] if results["documents"] else []

    def inject_memory(self, query: str) -> str:
        """
        Format retrieved memories for direct injection into a prompt.

        Args:
            query: The query to search for.

        Returns:
            A formatted string block containing context from past conversations.
        """
        memories = self.recall(query)
        if not memories:
            return ""

        memory_text = "\n".join(f"- {m}" for m in memories)
        return f"\n[Relevant past context:\n{memory_text}]\n"

    def clear(self):
        """Delete all stored conversations."""
        if not self._available:
            return
        try:
            all_results = self.collection.get()
            if all_results and all_results.get("ids"):
                self.collection.delete(ids=all_results["ids"])
                print("[NovaMemory] All memories cleared")
        except Exception as e:
            print(f"[NovaMemory] Clear failed: {e}")

    @property
    def size(self) -> int:
        """Return the number of stored conversations."""
        if not self._available:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0
