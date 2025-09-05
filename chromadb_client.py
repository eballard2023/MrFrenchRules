# Disable ChromaDB due to version 0.4.22 embedding function bug
CHROMADB_AVAILABLE = False
chromadb = None
print("⚠️ ChromaDB disabled - using Supabase only")

import uuid
from typing import List, Dict, Optional

# ChromaDB Persistent Client Setup
DB_PATH = "chroma_db_data"

class ChromaDBClient:
    def __init__(self):
        self.client = None
        self.collection = None
        print("❌ ChromaDB disabled - all operations will return False")
    
    def add_conversation(self, session_id: str, role: str, content: str, metadata: Dict = None) -> bool:
        """Add single conversation message with metadata"""
        if not self.client or not self.collection:
            print(f"⚠️ ChromaDB: Client or collection not available (client: {self.client is not None}, collection: {self.collection is not None})")
            return False
            
        try:
            message_id = str(uuid.uuid4())
            base_metadata = {
                "session_id": session_id,
                "role": role,
                "source": "interview_chat"
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Ensure content is not empty
            if not content or not content.strip():
                print(f"⚠️ ChromaDB: Empty content for {role} message, skipping")
                return False
            
            self.collection.add(
                documents=[f"{role}: {content}"],
                metadatas=[base_metadata],
                ids=[message_id]
            )
            print(f"✅ ChromaDB: Added {role} message for session {session_id}")
            return True
        except Exception as e:
            print(f"❌ ChromaDB: Error adding message: {e}")
            print(f"❌ ChromaDB error type: {type(e).__name__}")
            return False
    
    def store_conversation(self, session_id: str, conversation: List[Dict], expert_info: Dict = None) -> bool:
        """Store full conversation in ChromaDB with expert info"""
        if not self.client or not self.collection:
            print(f"⚠️ ChromaDB: Client or collection not available for storing conversation")
            return False
            
        try:
            # Convert conversation to text
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation if msg.get('content', '').strip()
            ])
            
            if not conversation_text.strip():
                print(f"⚠️ ChromaDB: Empty conversation text for session {session_id}")
                return False
            
            metadata = {
                "session_id": session_id,
                "message_count": len(conversation),
                "source": "interview_chat"
            }
            
            if expert_info:
                metadata.update({
                    "expert_name": expert_info.get("expert_name", "Unknown"),
                    "expert_email": expert_info.get("expert_email", "unknown@example.com"),
                    "expertise_area": expert_info.get("expertise_area", "General")
                })
            
            # Store in ChromaDB
            self.collection.add(
                documents=[conversation_text],
                metadatas=[metadata],
                ids=[f"session_{session_id}"]
            )
            print(f"✅ ChromaDB: Stored conversation for session {session_id}")
            return True
        except Exception as e:
            print(f"❌ ChromaDB: Error storing conversation: {e}")
            print(f"❌ ChromaDB error type: {type(e).__name__}")
            return False
    
    def get_conversations_with_experts(self) -> List[Dict]:
        """Get all conversations with expert information - disabled"""
        return []
    
    def get_session_conversation(self, session_id: str) -> Optional[List[Dict]]:
        """Get conversation history for a specific session"""
        if not self.client or not self.collection:
            print(f"⚠️ ChromaDB: Client or collection not available for getting session conversation")
            return None
            
        try:
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if not results or not results.get('documents'):
                print(f"⚠️ ChromaDB: No conversation found for session {session_id}")
                return None
            
            # Parse conversation messages
            conversation = []
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                if metadata.get('role') in ['user', 'assistant']:
                    # Extract content from "role: content" format
                    content = doc.split(': ', 1)[1] if ': ' in doc else doc
                    conversation.append({
                        'role': metadata['role'],
                        'content': content
                    })
            
            # Sort by timestamp if available, otherwise maintain order
            return conversation
        except Exception as e:
            print(f"❌ ChromaDB: Error getting session conversation: {e}")
            print(f"❌ ChromaDB error type: {type(e).__name__}")
            return None
    
    def search_similar_conversations(self, query: str, n_results: int = 3) -> Optional[List[Dict]]:
        """Search for similar conversations"""
        if not self.client or not self.collection:
            print(f"⚠️ ChromaDB: Client or collection not available for searching")
            return None
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ ChromaDB: Error searching conversations: {e}")
            print(f"❌ ChromaDB error type: {type(e).__name__}")
            return None

# Global ChromaDB client instance
chromadb_client = ChromaDBClient()