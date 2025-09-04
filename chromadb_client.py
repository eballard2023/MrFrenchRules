try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    CHROMADB_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"ChromaDB not available: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None
    DefaultEmbeddingFunction = None

import uuid
from typing import List, Dict, Optional

# ChromaDB Persistent Client Setup
DB_PATH = "chroma_db_data"

class ChromaDBClient:
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            print("ChromaDB not available - skipping initialization")
            self.client = None
            self.collection = None
            return
            
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.client.get_or_create_collection(
                name="mrfrench-ai-coach",
                embedding_function=DefaultEmbeddingFunction()
            )
            print(f"ChromaDB client initialized successfully at: {DB_PATH}")
        except Exception as e:
            print(f"ChromaDB initialization failed: {e}")
            self.client = None
            self.collection = None
    
    def store_conversation(self, session_id: str, conversation: List[Dict]) -> bool:
        """Store conversation in ChromaDB with embeddings"""
        if not self.collection:
            return False
            
        try:
            # Convert conversation to text
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation
            ])
            
            # Store in ChromaDB
            self.collection.add(
                documents=[conversation_text],
                metadatas=[{
                    "session_id": session_id,
                    "message_count": len(conversation),
                    "source": "interview_chat"
                }],
                ids=[f"session_{session_id}"]
            )
            print(f"Successfully stored conversation for session: {session_id}")
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    def search_similar_conversations(self, query: str, n_results: int = 3) -> Optional[List[Dict]]:
        """Search for similar conversations"""
        if not self.collection:
            return None
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return None

# Global ChromaDB client instance
chromadb_client = ChromaDBClient()