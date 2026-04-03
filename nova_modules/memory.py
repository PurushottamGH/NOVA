import chromadb
from datetime import datetime
import os

class NovaMemory:
    """
    Persistent long-term memory for Nova using ChromaDB.
    Remembers conversations and key facts across separate sessions.
    """
    
    def __init__(self, persist_dir: str = "nova_memory"):
        """
        Initialize the memory engine.
        
        Args:
            persist_dir: Directory to store the ChromaDB database.
        """
        # Ensure the persistence directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="nova_conversations",
            metadata={"hnsw:space": "cosine"}
        )
    
    def remember(self, user_msg: str, nova_response: str):
        """
        Store a conversation turn in the vector database.
        
        Args:
            user_msg: The message from the user.
            nova_response: The response from Nova.
        """
        # Use timestamp as unique ID
        doc_id = f"conv_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[f"User: {user_msg}\nNova: {nova_response}"],
            ids=[doc_id],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "type": "conversation"
            }]
        )
    
    def recall(self, query: str, n_results: int = 3) -> list:
        """
        Retrieve relevant past conversations based on a search query.
        
        Args:
            query: The search text (usually the current user message).
            n_results: Number of memories to retrieve.
            
        Returns:
            List of relevant document strings.
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        return results['documents'][0] if results['documents'] else []
    
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
