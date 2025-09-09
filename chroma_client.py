"""
ChromaDB client for document storage and vector similarity search
Optimized vector database for AI Coach Interview System
"""

import os
from dotenv import load_dotenv
import logging
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ChromaDocumentClient:
    def __init__(self):
        """Initialize ChromaDB client with persistent storage"""
        self.client = None
        self.collection = None
        self.connected = False
        
        # Load environment variables from .env if present
        load_dotenv()
        
        # Initialize ChromaDB with persistent storage
        try:
            # Use persistent storage in ./chroma_db directory
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get the documents collection (we'll pass embeddings ourselves)
            self.collection = self.client.get_or_create_collection(
                name="interview_documents",
                metadata={"description": "Document chunks for AI Coach interviews"}
            )

            # Initialize OpenAI embedding client (required)
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise RuntimeError("Please provide an OpenAI API key. You can set OPENAI_API_KEY in the environment.")
            self.openai_client = OpenAI(api_key=openai_key)
            self.embedding_model = "text-embedding-3-small"
            
            self.connected = True
            logger.info("✅ ChromaDB connected successfully")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB connection failed: {e}")
            self.connected = False

    def add_document_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document chunk to ChromaDB
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            metadata: Document metadata (session_id, title, page, etc.)
        
        Returns:
            bool: Success status
        """
        try:
            if not self.connected:
                logger.warning("ChromaDB not connected - cannot add chunk")
                return False
            
            # Check if chunk already exists to avoid duplicates
            try:
                existing = self.collection.get(ids=[chunk_id])
                if existing['ids']:
                    logger.info(f"📄 Chunk {chunk_id} already exists in ChromaDB, skipping")
                    return True
            except Exception:
                pass  # Continue if check fails
            
            # Compute embedding using OpenAI and add with explicit embeddings
            embedding = self._embed_texts([content])[0]
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[chunk_id],
                embeddings=[embedding]
            )
            
            logger.info(f"📄 Added chunk {chunk_id} to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add chunk {chunk_id}: {e}")
            return False


    def get_document_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get document statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with document statistics
        """
        try:
            if not self.connected:
                return {"total_chunks": 0, "documents": []}
            
            # Get all chunks for this session
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            total_chunks = len(results['ids']) if results['ids'] else 0
            
            # Extract unique document titles
            documents = []
            if results['metadatas']:
                titles_seen = set()
                for metadata in results['metadatas']:
                    title = metadata.get('title', 'Unknown')
                    if title not in titles_seen:
                        documents.append({
                            'title': title,
                            'doc_type': metadata.get('doc_type', 'unknown'),
                            'upload_status': 'completed'
                        })
                        titles_seen.add(title)
            
            return {
                "total_chunks": total_chunks,
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats for session {session_id}: {e}")
            return {"total_chunks": 0, "documents": []}

    def delete_session_documents(self, session_id: str) -> bool:
        """
        Delete all documents for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            if not self.connected:
                return False
            
            # Get all IDs for this session
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted {len(results['ids'])} chunks for session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents for session {session_id}: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        try:
            if not self.connected:
                return {"connected": False, "error": "Not connected"}
            
            count = self.collection.count()
            return {
                "connected": True,
                "total_chunks": count,
                "collection_name": "interview_documents",
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {e}")
            return {"connected": False, "error": str(e)}

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings using OpenAI"""
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        resp = self.openai_client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

# Global ChromaDB client instance
chroma_client = None

def get_chroma_client() -> ChromaDocumentClient:
    """Get or create the global ChromaDB client instance"""
    global chroma_client
    if chroma_client is None:
        chroma_client = ChromaDocumentClient()
    return chroma_client
