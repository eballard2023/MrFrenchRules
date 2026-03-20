"""
AI Coach Interview System - FastAPI backend.
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from document_processor import get_document_processor
from interview_ai import extract_rules_from_conversation
from interview_graph import run_interview_turn
from memory.memory_service import save_memory, init_memory_client
from schemas import ChatMessage, StartInterviewRequest, UserRegisterRequest, UserLoginRequest
from supabase_client import supabase_client
from user_auth import user_auth

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}
MAX_FILE_SIZE_MB = 50
MAX_DOC_CHUNKS = int(os.getenv("MAX_DOC_CHUNKS", "8"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "23"))
DOC_QUERY_PATTERNS = [
    "see my doc", "uploaded file", "check my pdf", "check my ppt", "check my doc",
    "my document", "the file i uploaded", "uploaded document", "can you see",
    "do you have", "check the document", "look at my", "review my doc",
    "whats in", "what's in", "what is in", "in this doc", "in my doc", "contents of",
]

# -----------------------------------------------------------------------------
# App & Clients
# -----------------------------------------------------------------------------

app = FastAPI(
    title="AI Coach Interview Model",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_openai_key = os.getenv("OPENAI_API_KEY")
if not _openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = AsyncOpenAI(api_key=_openai_key, timeout=15.0)

app.mount("/static", StaticFiles(directory="."), name="static")
security = HTTPBearer()
session_counter = 0

# -----------------------------------------------------------------------------
# Auth Dependencies
# -----------------------------------------------------------------------------


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify user token (any valid role)."""
    user = user_auth.verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify admin token and role."""
    user = user_auth.verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if user.get("role") != "admin":
        logger.warning(f"Auth failed: {user.get('email')} is not admin (role={user.get('role')})")
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    global session_counter
    logger.info("Starting application...")
    try:
        if supabase_client.connect():
            logger.info("Connected to Supabase")
        else:
            logger.warning("Supabase connection failed")
    except Exception as e:
        logger.warning(f"Supabase error: {e}")
    if supabase_client.connected:
        try:
            max_id = await supabase_client.get_max_session_id()
            session_counter = max_id + 1
        except Exception as e:
            logger.warning(f"Could not get max session ID: {e}")
            session_counter = 1
    else:
        session_counter = 1

    # Eagerly initialize Chroma clients
    try:
        # Initialize memory-service Chroma client
        init_memory_client()
    except Exception as e:
        logger.warning(f"Memory service Chroma init failed: {e}")

    try:
        # Initialize document Chroma client (interview_documents)
        from chroma_client import get_chroma_client
        get_chroma_client()
        logger.info("Document processing: ChromaDB (chroma_client initialized)")
    except Exception as e:
        logger.warning(f"Document Chroma init failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(supabase_client, "close"):
        supabase_client.close()


# -----------------------------------------------------------------------------
# Document Helpers
# -----------------------------------------------------------------------------


def _is_document_query(message: str) -> bool:
    return any(p in (message or "").lower() for p in DOC_QUERY_PATTERNS)


async def _get_doc_context_for_session(session_id: str) -> str:
    """Fetch document context from ChromaDB for the interview prompt."""
    try:
        from chroma_client import get_chroma_client
        doc_processor = get_document_processor(client)
        session_docs = doc_processor.get_session_documents(session_id)
        if session_docs.get("total_chunks", 0) == 0:
            return ""
        chroma = get_chroma_client()
        result = chroma.collection.get(where={"session_id": session_id})
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        chunks = []
        for i, content in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            chunks.append((meta.get("chunk_index", i), content, meta.get("title", "Unknown")))
        chunks.sort(key=lambda x: x[0])
        return "\n\n".join(f"**{t}:**\n{c}" for _, c, t in chunks[:MAX_DOC_CHUNKS])
    except Exception as e:
        logger.warning(f"Could not fetch doc context: {e}")
        return ""


async def _build_document_context_for_extraction(session_id: str) -> str:
    """Build document context string for rule extraction (includes guidance)."""
    try:
        from chroma_client import get_chroma_client
        doc_processor = get_document_processor(client)
        session_docs = doc_processor.get_session_documents(session_id)
        if session_docs.get("total_chunks", 0) == 0:
            return ""
        chroma = get_chroma_client()
        result = chroma.collection.get(where={"session_id": session_id})
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        chunks = []
        titles = []
        for i, content in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            idx = meta.get("chunk_index", i)
            title = meta.get("title", "Unknown")
            chunks.append((idx, content, title))
            if title not in titles:
                titles.append(title)
        chunks.sort(key=lambda x: x[0])
        combined = "\n\n".join(c for _, c, _ in chunks[:50])
        return f"""

**UPLOADED DOCUMENTS:** {', '.join(titles)}

**DOCUMENT CONTENT:**
{combined}

**DOCUMENT EXTRACTION GUIDANCE:**
- Extract rules from BOTH conversation AND document content
- Convert document advice into "Jamie should..." format
- Focus on child behavior, family communication, task management
"""
    except Exception as e:
        logger.warning(f"Could not build doc context for extraction: {e}")
        return ""


async def _format_documents_for_user(session_id: str) -> tuple[str, bool]:
    """Fetch and format document content for user-facing response. Returns (message, had_docs)."""
    try:
        from chroma_client import get_chroma_client
        doc_processor = get_document_processor(client)
        session_docs = doc_processor.get_session_documents(session_id)
        if not session_docs.get("documents"):
            return "I don't see any uploaded documents for this session yet. You can upload PDF, DOCX, PPTX, or TXT files.", False
        chroma = get_chroma_client()
        result = chroma.collection.get(where={"session_id": session_id})
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        by_title: Dict[str, list] = {}
        for i, chunk in enumerate(docs):
            title = (metas[i] if i < len(metas) else {}).get("title", "Unknown Document")
            by_title.setdefault(title, []).append(chunk)
        sections = []
        for title, chunks in by_title.items():
            if len(chunks) <= 4:
                content = " ".join(chunks)
            else:
                mid = len(chunks) // 2
                sample = chunks[:2] + chunks[mid - 1 : mid + 1] + chunks[-2:]
                content = " ".join(sample) + f"\n\n[Note: {len(chunks)} sections total.]"
            sections.append(f"**{title}:**\n{content}")
        return f"Here's what I found in your documents:\n\n" + "\n\n".join(sections) + "\n\nWhat would you like to know?", True
    except Exception as e:
        logger.error(f"Document query error: {e}")
        return "I had trouble accessing document content. You can continue with the interview or try again.", False


# -----------------------------------------------------------------------------
# Session Helpers
# -----------------------------------------------------------------------------


async def _get_session(session_id: str) -> Dict[str, Any]:
    if not session_id or not supabase_client.connected:
        raise HTTPException(status_code=404, detail="Session not found")
    data = await supabase_client.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


def _owns_rule(rule: dict, user_id: int, user_email: str) -> bool:
    return rule.get("user_id") == user_id or rule.get("expert_email") == user_email


async def _find_rule(task_id: str) -> Optional[dict]:
    rules = await supabase_client.get_all_rules()
    return next((r for r in rules if str(r["id"]) == task_id), None)


async def _append_assistant_and_persist(
    session_id: str,
    session_data: Dict[str, Any],
    content: str,
    question_index: Optional[int] = None,
    complete: Optional[bool] = None,
) -> None:
    """Append assistant message to history, save to memory, and persist session."""
    session_data["conversation_history"].append({"role": "assistant", "content": content})
    try:
        await save_memory(session_id, "assistant", content)
    except Exception as e:
        logger.warning(f"Failed to save assistant message to memory: {e}")
    qi = question_index if question_index is not None else session_data["current_question_index"]
    ic = complete if complete is not None else session_data["is_complete"]
    await supabase_client.update_session(session_id, session_data["conversation_history"], qi, ic)


# -----------------------------------------------------------------------------
# Routes: Health & Upload
# -----------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": getattr(supabase_client, "connected", False),
    }


@app.post("/upload-doc")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    expert_name: str = Form(...),
    expert_email: str = Form(...),
):
    """Upload and process a document for interview context."""
    logger.info(f"Document upload: {file.filename} for session {session_id}")
    try:
        from chroma_client import get_chroma_client
        chroma = get_chroma_client()
        if not chroma.connected:
            raise HTTPException(status_code=503, detail="Document processing unavailable (ChromaDB)")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Document processing unavailable: {e}")
    ext = os.path.splitext((file.filename or "").lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB")
    await file.seek(0)
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            shutil.copyfileobj(file.file, tf)
            path = tf.name
        doc_processor = get_document_processor(client)
        result = await doc_processor.process_uploaded_file(
            file_path=path, filename=file.filename or "file",
            expert_email=expert_email, expert_name=expert_name, session_id=session_id,
        )
        if result.get("success"):
            return {
                "success": True,
                "message": f"Document '{file.filename}' processed and added to interview context.",
                "filename": result.get("filename", file.filename),
                "chunks_processed": result.get("chunks_processed", 0),
                "file_type": ext[1:].upper(),
            }
        raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if "does not exist" in str(e):
            raise HTTPException(status_code=503, detail="Document processing tables not found")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


# -----------------------------------------------------------------------------
# Routes: Chat & Interview (+ behavioral engine integration)
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
from behavior.sensors.extractor import SolidFeatureExtractor
from behavior.engine.statistics import BehavioralEngine
from behavior.initialise_baseline import initialize_user_baseline

MIN_WORDS_FOR_BASELINE = 250

# Global behavioral extractor + engine (initialized once baseline is built)
_behavior_extractor = SolidFeatureExtractor()
_behavior_engine: Optional[BehavioralEngine] = None
# Per-session last behavioral state for frontend visualization (session_id -> state dict)
_session_behavior_state: Dict[str, Dict[str, Any]] = {}


async def _ensure_behavior_baseline(session_data: Dict[str, Any]) -> None:
    """
    Ensure a behavioral baseline (truth_repo.json) exists.
    Uses all user messages in the conversation to build a baseline once
    there are at least MIN_WORDS_FOR_BASELINE words.
    """
    global _behavior_engine
    print("session_data", session_data)
    # Map expert_email -> user_id
    expert_email = session_data.get("expert_email")
    session_id = int(session_data.get("session_id"))
    if not expert_email:
        return

    user_row = supabase_client.get_user_by_email(expert_email)
    if not user_row:
        logger.info(f"Behavior baseline: no user row for email {expert_email}")
        return
    user_id = user_row["id"]

    # If DB says baseline exists, ensure engine is loaded and return
    has_baseline = await supabase_client.get_behavior_baseline_flag(user_id, session_id)  # type: ignore[arg-type]
    if has_baseline and _behavior_engine is not None and getattr(_behavior_engine, "baseline_mean", None) is not None:
        return

    # Collect all user messages so far
    conv = session_data.get("conversation_history") or []
    user_texts = [m.get("content", "") for m in conv if m.get("role") == "user"]
    long_text = "\n".join(user_texts).strip()
    if not long_text:
        return

    word_count = len(long_text.split())
    if word_count < MIN_WORDS_FOR_BASELINE:
        logger.info(
            f"Behavior baseline: only {word_count} words so far "
            f"(need {MIN_WORDS_FOR_BASELINE}); skipping baseline init."
        )
        return

    try:
        baseline_json = initialize_user_baseline(
            str(user_id),
            long_text,
            extractor=_behavior_extractor,
            save_to_file=False,
        )
        _behavior_engine = BehavioralEngine(baseline_data=baseline_json, threshold=1.5)
        # Store baseline JSON in DB for this user
        success = await supabase_client.set_behavior_baseline_flag(user_id, session_id, True, baseline_json)  # type: ignore[arg-type]
        if not success:
            logger.warning("Failed to set behavior baseline flag in DB")
        logger.info("Behavior baseline initialized from conversation history and stored in DB.")
    except Exception as e:
        logger.warning(f"Failed to initialize behavior baseline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize behavior baseline: {e}")

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    """Handle chat message and return AI reply."""
    session_id = chat_message.session_id
    session_data = await _get_session(session_id)
    if session_data.get("is_complete"):
        return {"message": "Interview has been completed.", "is_complete": True}

    # Append user message, save to memory, persist
    session_data["conversation_history"].append({"role": "user", "content": chat_message.message})
    try:
        await save_memory(session_id, "user", chat_message.message)
    except Exception as e:
        logger.warning(f"Failed to save user message to memory: {e}")
    await supabase_client.update_session(
        session_id,
        session_data["conversation_history"],
        session_data["current_question_index"],
        session_data["is_complete"],
    )

    # Ensure behavioral baseline exists (once enough words are available)
    await _ensure_behavior_baseline(session_data)

    # If behavioral baseline is not ready yet, stay in baseline collection phase
    if _behavior_engine is None or getattr(_behavior_engine, "baseline_mean", None) is None:
        conv = session_data.get("conversation_history") or []
        user_texts = [m.get("content", "") for m in conv if m.get("role") == "user"]
        long_text = "\n".join(user_texts).strip()
        word_count = len(long_text.split()) if long_text else 0

        if word_count < MIN_WORDS_FOR_BASELINE:
            baseline_msg = (
                "Before we start the structured interview, I'd like to get a solid baseline from you.\n"
                "Please share, in your own words, why you chose this profession and how you see your role. "
                f"Feel free to write a long, detailed answer over a few messages (we're aiming for at least {MIN_WORDS_FOR_BASELINE} words in total)."
            )
        else:
            baseline_msg = (
                "Thank you for sharing so much detail. I'm finalizing your baseline profile now; "
                "you can keep elaborating if you like, and we'll move into the structured interview next."
            )

        await _append_assistant_and_persist(session_id, session_data, baseline_msg)
        return {
            "message": baseline_msg,
            "question_number": session_data["current_question_index"] + 1,
            "is_complete": False,
            "auto_submitted": False,
            "final_note": None,
        }

    # Document-only query: return formatted docs and stop
    if _is_document_query(chat_message.message):
        msg, _ = await _format_documents_for_user(session_id)
        await _append_assistant_and_persist(session_id, session_data, msg)
        return {
            "message": msg,
            "question_number": session_data["current_question_index"] + 1,
            "is_complete": False,
            "auto_submitted": False,
            "final_note": None,
        }

    # Behavioral analysis (VAD + OCEAN drift monitoring)
    behavior_status = "OK"
    try:
        if _behavior_engine is not None and _behavior_engine.baseline_mean is not None:
            scores = _behavior_extractor.get_scores(chat_message.message)
            behavior_status, z_scores = _behavior_engine.update_and_check(scores)
            # Build latest behavioral state snapshot
            latest_state = {
                "status": behavior_status,
                "z_scores": z_scores.tolist() if hasattr(z_scores, "tolist") else list(z_scores),
                "cusum_pos": _behavior_engine.cusum_pos.tolist(),
                "cusum_neg": _behavior_engine.cusum_neg.tolist(),
                "threshold": 1.5,
                "labels": ["V", "A", "D", "O", "C", "E", "A", "N"],
            }
            # Store in-memory for fast access
            _session_behavior_state[session_id] = latest_state
            # Persist latest state in DB so it survives restarts
            try:
                expert_email = session_data.get("expert_email")
                if expert_email:
                    user_row = supabase_client.get_user_by_email(expert_email)
                    if user_row:
                        # behavior_baselines.session_id is stored as integer in DB
                        db_session_id: Optional[int] = None
                        try:
                            db_session_id = int(session_id)
                        except (TypeError, ValueError):
                            db_session_id = None
                        if db_session_id is not None:
                            await supabase_client.set_behavior_latest_state(  # type: ignore[arg-type]
                                user_row["id"],
                                db_session_id,
                                latest_state,
                            )
            except Exception as persist_err:
                logger.warning(f"Failed to persist behavioral latest state: {persist_err}")
    except Exception as e:
        logger.warning(f"Behavioral engine error (non-fatal): {e}")

    # Run interview turn and persist AI reply
    doc_context = await _get_doc_context_for_session(session_id)
    ai_message = await run_interview_turn(
        session_id=session_id,
        conversation_history=session_data["conversation_history"],
        doc_context=doc_context,
        expert_name=session_data.get("expert_name", "Expert"),
        expert_email=session_data.get("expert_email", ""),
        behavior_status=behavior_status,
    )
    q_idx = session_data["current_question_index"] + 1
    await _append_assistant_and_persist(session_id, session_data, ai_message, question_index=q_idx)

    is_complete = q_idx >= MAX_QUESTIONS
    auto_submitted = False
    final_note = None
    if is_complete:
        final_note = "Thank you for sharing your insights! The interview is now complete."
        await _append_assistant_and_persist(session_id, session_data, final_note, question_index=q_idx, complete=True)
        try:
            await _submit_interview_internal(session_id, None, None)
            auto_submitted = True
        except Exception as e:
            logger.error(f"Auto-submit failed: {e}")

    return {
        "message": ai_message,
        "question_number": q_idx + 1,
        "is_complete": is_complete,
        "auto_submitted": auto_submitted,
        "final_note": final_note,
    }


async def _submit_interview_internal(session_id: str, companion_id: Optional[int], user_id: Optional[int]):
    """Internal: extract rules and save to DB. Used by submit endpoint and auto-submit."""
    session_data = await supabase_client.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found")
    await supabase_client.update_session(
        session_id,
        session_data["conversation_history"],
        session_data["current_question_index"],
        True,
    )
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in session_data["conversation_history"])
    if len(conv_text) < 100:
        return {"message": "Interview too short for rule extraction.", "status": "completed", "tasks_extracted": 0, "tasks": []}
    doc_context = await _build_document_context_for_extraction(session_id)
    task_statements = await extract_rules_from_conversation(client, conv_text, doc_context)
    logger.info(f"Extracted {len(task_statements)} tasks for session {session_id}")
    final_companion = companion_id if companion_id is not None else session_data.get("companion_id")
    if supabase_client.connected and task_statements:
        for task in task_statements:
            await supabase_client.save_interview_rule(
                session_id=session_id,
                expert_name=session_data["expert_name"],
                expertise_area=session_data["expertise_area"],
                rule_text=task,
                expert_email=session_data.get("expert_email"),
                companion_id=final_companion,
                user_id=user_id,
            )
    return {
        "message": "Interview complete.",
        "status": "completed",
        "tasks_extracted": len(task_statements),
        "tasks": task_statements,
    }


@app.post("/submit_interview/{session_id}")
async def submit_interview(
    session_id: str,
    companion_id: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
):
    """Finalize interview and extract rules."""
    return await _submit_interview_internal(
        session_id,
        companion_id,
        current_user.get("id"),
    )


@app.get("/session/{session_id}/conversation")
async def get_conversation(session_id: str):
    if supabase_client.connected:
        try:
            s = await supabase_client.get_session(session_id)
            if s:
                return {
                    "session_id": session_id,
                    "conversation": s.get("conversation_history", []),
                    "is_complete": s.get("is_complete", False),
                    "companion_id": s.get("companion_id"),
                    "source": "database",
                }
        except Exception as e:
            logger.error(f"Error getting session: {e}")
    return {"session_id": session_id, "conversation": [], "is_complete": False, "source": "empty"}


# -----------------------------------------------------------------------------
# Routes: Auth
# -----------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
async def login(req: LoginRequest):
    result = user_auth.authenticate(req.email, req.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info(f"Login: {req.email} (role={result['user']['role']})")
    return result


@app.post("/users/register")
async def register_user(req: UserRegisterRequest):
    result = user_auth.register(req.email, req.password, req.name)
    if not result:
        raise HTTPException(status_code=400, detail="Unable to register user")
    return result


# -----------------------------------------------------------------------------
# Routes: User (authenticated)
# -----------------------------------------------------------------------------


@app.get("/users/me/sessions")
async def get_my_sessions(current_user: dict = Depends(get_current_user)):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    all_sessions = await supabase_client.get_all_sessions()
    email = current_user["email"]
    sessions = [
        {
            "session_id": s["session_id"],
            "expert_name": s.get("expert_name", "Expert"),
            "expertise_area": s.get("expertise_area", "General"),
            "completed": s.get("is_complete", False),
            "is_complete": s.get("is_complete", False),
            "created_at": s.get("created_at"),
            "messages": s.get("conversation_history", []),
            "companion_id": s.get("companion_id"),
        }
        for s in all_sessions if s.get("expert_email") == email and s.get("conversation_history")
    ]
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/users/me/stats")
async def get_my_stats(current_user: dict = Depends(get_current_user)):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    email, uid = current_user["email"], current_user["id"]
    all_sessions = await supabase_client.get_all_sessions()
    all_rules = await supabase_client.get_all_rules()
    user_rules = [r for r in all_rules if r.get("user_id") == uid or r.get("expert_email") == email]
    interviews = sum(1 for s in all_sessions if s.get("expert_email") == email)
    approved = sum(1 for r in user_rules if r.get("completed"))
    return {
        "total_interviews": interviews,
        "pending_tasks": len(user_rules) - approved,
        "approved_tasks": approved,
        "rejected_tasks": len(user_rules),
    }



@app.get("/users/me/tasks")
async def get_my_tasks(
    companion_id: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    email, uid = current_user["email"], current_user["id"]
    all_rules = await supabase_client.get_all_rules()
    user_rules = [r for r in all_rules if r.get("user_id") == uid or r.get("expert_email") == email]
    if companion_id is not None:
        user_rules = [r for r in user_rules if r.get("companion_id") == companion_id]
    companions = supabase_client.get_all_companions()
    comp_map = {c["id"]: c["name"] for c in companions}
    comp_map[None] = "Unassigned"
    tasks = [
        {
            "id": str(r["id"]),
            "session_id": str(r["session_id"]),
            "expert_name": str(r.get("expert_name", "Expert")),
            "task_text": str(r.get("rule_text", "")),
            "category": str(r.get("expertise_area", "General")),
            "status": "completed" if r.get("completed") else "pending",
            "companion_id": r.get("companion_id"),
            "companion_name": comp_map.get(r.get("companion_id"), "Unassigned"),
        }
        for r in user_rules
    ]
    return {"tasks": tasks, "companions": supabase_client.list_companions_for_user(uid)}


@app.post("/users/me/tasks/{task_id}/approve")
async def approve_user_task(task_id: str, current_user: dict = Depends(get_current_user)):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    rule = await _find_rule(task_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if not _owns_rule(rule, current_user["id"], current_user["email"]):
        raise HTTPException(status_code=403, detail="No permission to approve")
    await supabase_client.update_rule_completed(rule["id"], True)
    return {"success": True, "message": "Task approved"}


@app.post("/users/me/tasks/{task_id}/reject")
async def reject_user_task(task_id: str, current_user: dict = Depends(get_current_user)):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    rule = await _find_rule(task_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if not _owns_rule(rule, current_user["id"], current_user["email"]):
        raise HTTPException(status_code=403, detail="No permission to reject")
    await supabase_client.update_rule_completed(rule["id"], True)
    return {"success": True, "message": "Task rejected"}


@app.get("/users/me/behavior/{session_id}")
async def get_my_behavior(session_id: str, current_user: dict = Depends(get_current_user)):
    """
    Return the latest behavioral engine state (z-scores, CUSUM, status) for the session.

    Prefers in-memory cache (_session_behavior_state), but falls back to DB
    (behavior_baselines.latest_state) so charts keep working across restarts.
    """
    default = {
        "status": "NO_BASELINE",
        "z_scores": [],
        "cusum_pos": [],
        "cusum_neg": [],
        "threshold": 1.5,
        "labels": ["V", "A", "D", "O", "C", "E", "A", "N"],
    }

    # 1) Try in-memory state keyed by session_id string
    state = _session_behavior_state.get(session_id)

    # 2) If not present in memory, try DB (user_id + numeric session_id)
    if state is None:
        db_session_id: Optional[int] = None
        try:
            db_session_id = int(session_id)
        except (TypeError, ValueError):
            db_session_id = None

        if db_session_id is not None:
            try:
                latest = await supabase_client.get_behavior_latest_state(  # type: ignore[arg-type]
                    current_user["id"],
                    db_session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch behavioral latest state from DB: {e}")
                latest = None

            if latest:
                # Also warm the in-memory cache for this process
                state = {**default, **latest}
                _session_behavior_state[session_id] = state

    # 3) Fall back to default if still nothing
    if state is None:
        state = default

    # Ensure labels and threshold are always present
    if not state.get("labels"):
        state = {**default, **state, "labels": ["V", "A", "D", "O", "C", "E", "A", "N"]}
    if "threshold" not in state:
        state["threshold"] = 1.5

    return state


@app.post("/users/me/start_interview")
async def start_my_interview(body: StartInterviewRequest, current_user: dict = Depends(get_current_user)):
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    global session_counter
    session_id = str(session_counter)
    session_counter += 1
    c = supabase_client.get_companion_by_slug("jamie")
    companion_id = c["id"] if c else None
    expert_name = current_user.get("name") or "Expert"
    expert_email = current_user["email"]
    area = body.expertise_area or "General"
    await supabase_client.save_session(session_id, expert_name, expert_email, area, companion_id=companion_id)
    ai_message = (
        f"Hello {expert_name}! Thank you for sharing your expertise in {area}. "
        "I'm here to interview you. "
        "To start, could you describe why have you choose this profession ?🤔"
    )
    try:
        await supabase_client.update_session(session_id, [{"role": "assistant", "content": ai_message}], 0, False)
    except Exception as e:
        logger.error(f"Failed to save initial message: {e}")
    return {"session_id": session_id, "message": ai_message, "question_number": 0, "companion_id": companion_id}


# -----------------------------------------------------------------------------
# Routes: Admin
# -----------------------------------------------------------------------------


@app.get("/admin/conversations")
async def get_admin_conversations(current_admin: dict = Depends(get_current_admin)):
    if not supabase_client.connected:
        return {"conversations": []}
    try:
        all_sessions = await supabase_client.get_all_sessions()
        return {
            "conversations": [
                {
                    "session_id": s["session_id"],
                    "expert_name": s.get("expert_name", "Unknown"),
                    "expertise_area": s.get("expertise_area", "General"),
                    "completed": s.get("is_complete", False),
                    "is_complete": s.get("is_complete", False),
                    "created_at": s.get("created_at"),
                    "messages": s.get("conversation_history", []),
                }
                for s in all_sessions if s.get("conversation_history")
            ]
        }
    except Exception as e:
        logger.error(f"Admin conversations error: {e}")
        return {"conversations": []}


@app.get("/admin/tasks")
async def get_admin_tasks(
    companion_id: Optional[int] = None,
    current_admin: dict = Depends(get_current_admin),
):
    if not supabase_client.connected:
        return {"tasks": [], "companions": [], "tasks_by_companion": {}}
    try:
        companions = supabase_client.get_all_companions()
        comp_map = {c["id"]: c["name"] for c in companions}
        comp_map[None] = "Unassigned"
        rules = await supabase_client.get_all_rules()
        if companion_id is not None:
            rules = [r for r in rules if r.get("companion_id") == companion_id]
        tasks = [
            {
                "id": str(r["id"]),
                "session_id": str(r["session_id"]),
                "expert_name": str(r.get("expert_name", "Expert User")),
                "task_text": str(r.get("rule_text") or "No rule text"),
                "category": str(r.get("expertise_area", "General")),
                "priority": "medium",
                "status": "completed" if r.get("completed") else "pending",
                "companion_id": r.get("companion_id"),
                "companion_name": comp_map.get(r.get("companion_id"), "Unassigned"),
            }
            for r in rules
        ]
        tasks.sort(key=lambda x: (x["status"] == "completed", x["id"]))
        by_comp: Dict[str, list] = {}
        for t in tasks:
            by_comp.setdefault(t["companion_name"], []).append(t)
        return {"tasks": tasks, "companions": companions, "tasks_by_companion": by_comp}
    except Exception as e:
        logger.error(f"Admin tasks error: {e}")
        return {"tasks": [], "companions": [], "tasks_by_companion": {}}


@app.post("/admin/approve/{task_id}")
async def approve_task(task_id: str, current_admin: dict = Depends(get_current_admin)):
    if not supabase_client.connected:
        return {"success": False, "error": "Database not connected"}
    rule = await _find_rule(task_id)
    if not rule:
        return {"success": False, "error": "Rule not found"}
    await supabase_client.update_rule_completed(rule["id"], True)
    return {"success": True, "message": "Task approved"}


@app.post("/admin/reject/{task_id}")
async def reject_task(task_id: str, current_admin: dict = Depends(get_current_admin)):
    if not supabase_client.connected:
        return {"success": False, "error": "Database not connected"}
    rule = await _find_rule(task_id)
    if not rule:
        return {"success": False, "error": "Rule not found"}
    await supabase_client.update_rule_completed(rule["id"], True)
    return {"success": True, "message": "Task rejected"}


@app.get("/admin/stats")
async def get_admin_stats(current_admin: dict = Depends(get_current_admin)):
    if not supabase_client.connected:
        return {"total_interviews": 0, "pending_tasks": 0, "approved_tasks": 0, "rejected_tasks": 0}
    try:
        sessions = await supabase_client.get_all_sessions()
        rules = await supabase_client.get_all_rules()
        approved = sum(1 for r in rules if r.get("completed"))
        return {
            "total_interviews": len(sessions),
            "pending_tasks": len(rules) - approved,
            "approved_tasks": approved,
            "rejected_tasks": len(rules),
        }
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return {"total_interviews": 0, "pending_tasks": 0, "approved_tasks": 0, "rejected_tasks": 0}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("APP_PORT") or os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
