"""
Chroma Cloud-backed memory service for chat recall.
Uses per-session/per-user collections for storing and retrieving conversation memories.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VECTOR_STORE_CACHE: Dict[str, Any] = {}
CHROMA_LOCK = asyncio.Lock()
_CHROMA_CLIENT = None


def _get_chroma_client() -> chromadb.CloudClient:
    """Initialize and return a Chroma Cloud client."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")
    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        raise ValueError(
            "Missing Chroma Cloud credentials. Set CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE in .env"
        )
    _CHROMA_CLIENT = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )
    logger.info("Chroma Cloud client initialized for memory service")
    return _CHROMA_CLIENT


def _get_current_time_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _get_user_collection(user_id: str):
    """Get or create a Chroma collection for the user/thread."""
    safe_id = md5(user_id.encode("utf-8")).hexdigest()
    collection_name = f"mrfrench_memory_{safe_id}"

    if user_id in VECTOR_STORE_CACHE:
        return VECTOR_STORE_CACHE[user_id]

    client = _get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
        logger.debug(f"Loaded collection for {user_id}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "description": f"Memory for thread/user {user_id}",
            },
        )
        logger.info(f"Created collection for {user_id}")

    VECTOR_STORE_CACHE[user_id] = collection
    return collection


async def save_memory(user_id: str, role: str, text: str) -> None:
    """Save a chat message to Chroma Cloud memory."""
    async with CHROMA_LOCK:
        collection = _get_user_collection(user_id)

    doc_id = md5(f"{user_id}:{role}:{text}".encode()).hexdigest()
    metadata = {
        "user_id": user_id,
        "role": role,
        "timestamp": _get_current_time_str(),
    }

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            ),
        )
        logger.debug(f"Saved memory for user_id={user_id}, role={role}")
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            async with CHROMA_LOCK:
                VECTOR_STORE_CACHE.pop(user_id, None)
                collection = _get_user_collection(user_id)
            await loop.run_in_executor(
                None,
                lambda: collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[doc_id],
                ),
            )
        else:
            raise


async def search_memories(
    user_id: str,
    role: str,
    query: str,
    k: int = 30,
) -> List[dict]:
    """
    Search recalled memories for the user/thread by query.
    Returns list of dicts with keys: page_content, metadata.
    """
    async with CHROMA_LOCK:
        collection = _get_user_collection(user_id)

    loop = asyncio.get_event_loop()
    # Only filter by role when it's "user" or "assistant"; otherwise recall all session memories
    where_filter = {"role": role} if role in ("user", "assistant") else None

    try:
        expanded_queries = [
            query,
            query.lower(),
            query.replace("?", "").replace("!", "").strip(),
        ]
        all_results = []
        seen_ids = set()

        for q in expanded_queries:
            if not q:
                continue
            kwargs = {
                "query_texts": [q],
                "n_results": min(k * 3, 100),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = await loop.run_in_executor(
                None,
                lambda kw=kwargs: collection.query(**kw),
            )

            if results and results.get("documents") and results["documents"][0]:
                for i, content in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    dedup_key = meta.get("timestamp", "") + content[:50]
                    if dedup_key in seen_ids:
                        continue
                    seen_ids.add(dedup_key)
                    distance = (
                        results["distances"][0][i]
                        if results.get("distances")
                        else 0.0
                    )
                    all_results.append({
                        "content": content,
                        "metadata": meta,
                        "distance": distance,
                    })

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []

        for r in all_results:
            content_lower = r["content"].lower()
            distance = r["distance"]
            semantic_score = max(0, 2.0 - distance)
            content_words = set(content_lower.split())
            keyword_score = len(query_words & content_words) * 0.3
            phrase_bonus = 1.0 if query_lower in content_lower else 0.0
            ts = r["metadata"].get("timestamp", "")
            recency_bonus = 0.1 if ts else 0.0
            final_score = semantic_score + keyword_score + phrase_bonus + recency_bonus
            if distance < 2.0 or keyword_score > 0:
                scored.append({
                    "page_content": r["content"],
                    "metadata": {**r["metadata"], "distance": distance, "score": final_score},
                    "score": final_score,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        out = [{"page_content": s["page_content"], "metadata": s["metadata"]} for s in scored[:k]]
        logger.info(f"Retrieved {len(out)} memories for user_id={user_id}")
        return out

    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            async with CHROMA_LOCK:
                VECTOR_STORE_CACHE.pop(user_id, None)
            return []
        logger.error(f"Memory search failed for {user_id}: {e}")
        return []


async def get_recent_dialogues(thread_id: str, limit: int = 10) -> List[str]:
    """
    Fetch recent dialogue messages for the thread.
    Returns list of message strings, oldest to newest.
    """
    async with CHROMA_LOCK:
        collection = _get_user_collection(thread_id)

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: collection.get(include=["documents", "metadatas"]),
        )

        if not results or not results.get("documents"):
            return []

        docs = []
        for i, content in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            ts_str = meta.get("timestamp", "")
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")[:19])
                except Exception:
                    ts = datetime.min
            docs.append((ts, content))

        docs.sort(key=lambda x: x[0])
        return [d[1] for d in docs[-limit:]]

    except Exception as e:
        logger.warning(f"Failed to get recent dialogues for {thread_id}: {e}")
        return []


async def clear_user_memory(user_id: str) -> None:
    """Clear all memories for a user/thread."""
    safe_id = md5(user_id.encode("utf-8")).hexdigest()
    collection_name = f"mrfrench_memory_{safe_id}"
    loop = asyncio.get_event_loop()
    try:
        client = _get_chroma_client()
        await loop.run_in_executor(
            None,
            lambda: client.delete_collection(name=collection_name),
        )
        async with CHROMA_LOCK:
            VECTOR_STORE_CACHE.pop(user_id, None)
        logger.info(f"Cleared memory for {user_id}")
    except Exception as e:
        logger.warning(f"Failed to clear memory for {user_id}: {e}")


def init_memory_client() -> None:
    """
    Initialize the Chroma Cloud client for memory service.
    Safe to call at startup; a no-op if already initialized.
    """
    try:
        _get_chroma_client()
        logger.info("Memory service Chroma client initialized at startup")
    except Exception as e:
        logger.warning(f"Failed to initialize memory service Chroma client at startup: {e}")
