from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
# MongoDB removed - using Supabase + ChromaDB
from supabase_client import supabase_client
# ... imports ...
from user_auth import user_auth
from jira_client import jira_client
# admin_auth import removed

# ... (other imports) ...

from document_processor import get_document_processor
import logging
from schemas import ExpertInfo, ChatMessage, AdminLoginRequest, UserRegisterRequest, UserLoginRequest, StartInterviewRequest
from openai import AsyncOpenAI
from interview_ai import (
    SYSTEM_PROMPT,
    sanitize_question,
    clean_response,
    is_smalltalk_or_project,
    get_smalltalk_response,
    build_system_prompt_with_context,
    get_next_interview_reply,
    extract_rules_from_conversation,
)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Coach Interview Model",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Always provide friendly redirects for disabled docs endpoints
@app.get("/docs", include_in_schema=False)
async def _docs_redirect():
    return RedirectResponse(url="/")

@app.get("/redoc", include_in_schema=False)
async def _redoc_redirect():
    return RedirectResponse(url="/")

@app.get("/openapi.json", include_in_schema=False)
async def _openapi_blocked():
    return JSONResponse(status_code=404, content={"detail": "OpenAPI schema is disabled"})

# Redirect nested paths ending with /docs or /redoc back to home (e.g., /admin/dashboard/docs)
@app.get("/{_prefix:path}/docs", include_in_schema=False)
async def _nested_docs_redirect(_prefix: str):
    return RedirectResponse(url="/")

@app.get("/{_prefix:path}/redoc", include_in_schema=False)
async def _nested_redoc_redirect(_prefix: str):
    return RedirectResponse(url="/")

# Set up OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=15.0)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve MrFrench.png directly
@app.get("/MrFrench.png")
async def get_mr_french_image():
    from fastapi.responses import FileResponse
    return FileResponse("MrFrench.png")

templates = Jinja2Templates(directory="templates")


session_counter = 0  # Will be initialized from DB to avoid conflicts

security = HTTPBearer()

def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin token and role"""
    user = user_auth.verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    if user.get("role") != "admin":
        print(f"❌ AUTH FAILED: User {user.get('email')} is not an admin (role={user.get('role')})")
        raise HTTPException(status_code=403, detail="Admin privileges required")
        
    return user


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify user token (any valid role used for user access)"""
    user = user_auth.verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    global session_counter
    logger.info("Starting up the application...")
    
    # Connect to Supabase (optional)
    try:
        supabase_success = supabase_client.connect()
        if supabase_success:
            logger.info("✅ Successfully connected to Supabase")
        else:
            logger.warning("⚠️ Supabase connection failed - check .env file")
    except Exception as e:
        logger.warning(f"⚠️ Supabase connection error: {e}")
        logger.info("App will continue without Supabase - using fallback storage")
    
    # Vector store disabled; using Supabase only
    
    # Document processing now handled entirely by ChromaDB
    logger.info("📄 Document processing using ChromaDB only")

    # Initialize session counter from database to avoid conflicts
    if supabase_client.connected:
        try:
            max_session = await supabase_client.get_max_session_id()
            session_counter = max_session + 1
            logger.info(f"Session counter initialized to {session_counter}")
        except Exception as e:
            session_counter = 1
            logger.warning(f"Could not get max session ID, starting from 1: {e}")
    else:
        session_counter = 1
        logger.info("Session counter initialized to 1 (no database)")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    logger.info("Shutting down the application...")
    if hasattr(supabase_client, 'close'):
        supabase_client.close()


@app.get("/", response_class=HTMLResponse)
async def get_start_page(request: Request):
    """Serve the main user login/register page"""
    logger.info("🏠 Root page requested - serving user login")
    return templates.TemplateResponse("user_login.html", {"request": request})


@app.get("/start-interview", response_class=HTMLResponse)
async def get_start_interview_page(request: Request):
    """Serve the legacy expert info collection page"""
    logger.info("📄 Start interview page requested")
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/interview", response_class=HTMLResponse)
async def get_interview_page(request: Request):
    """Serve the interview page"""
    logger.info("💬 Interview page requested")
    return templates.TemplateResponse("interview.html", {"request": request})


@app.get("/user/dashboard", response_class=HTMLResponse)
async def get_user_dashboard_page(request: Request):
    """Serve the user interview dashboard (requires frontend token)"""
    logger.info("📊 User dashboard page requested")
    return templates.TemplateResponse("user_dashboard.html", {"request": request})

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    logger.info("🏥 Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": supabase_client.connected if supabase_client else False,
        "sessions_in_memory": 0
    }

@app.post("/start_interview_with_expert")
async def start_interview_with_expert(expert_info: ExpertInfo):
    """Start a new interview session with expert information"""
    logger.info(f"🚀 START_INTERVIEW_WITH_EXPERT: {expert_info.expert_name} ({expert_info.expert_email})")
    global session_counter
    session_id = str(session_counter)
    session_counter += 1  # Increment after using
    print(f"🔢 SESSION: Created session {session_id}, next will be {session_counter}")
    logger.info(f"Session counter: current={session_id}, next={session_counter}")
    
    # Save to database only
    if not supabase_client.connected:
        try:
            supabase_client.connect()
        except Exception as e:
            logger.error(f"❌ Supabase connection failed on session start: {e}")
        raise HTTPException(status_code=503, detail="Database not available")
    
    companion_id = None
    if expert_info.companion_id is not None:
        companion_id = expert_info.companion_id
    elif expert_info.companion_slug:
        c = supabase_client.get_companion_by_slug(expert_info.companion_slug)
        if c:
            companion_id = c["id"]
    if companion_id is None:
        jamie = supabase_client.get_companion_by_slug("jamie")
        companion_id = jamie["id"] if jamie else None

    await supabase_client.save_session(session_id, expert_info.expert_name, expert_info.expert_email, expert_info.expertise_area, companion_id=companion_id)
    
    # Using Supabase only for session storage
    print(f"💾 Using Supabase only for session {session_id}")
    
    # Greeting varies by target (Jamie vs user's persona)
    is_persona = companion_id and (supabase_client.get_companion_by_slug("jamie") or {}).get("id") != companion_id
    if is_persona:
        ai_message = (
            f"Hello {expert_info.expert_name}! Thank you for sharing your expertise in {expert_info.expertise_area}. "
            "I'm here to interview you so we can train your own expert persona—a model that will act like you. "
            "To start, could you describe your area of expertise and how you usually apply it?"
        )
    else:
        ai_message = (
            f"Hello {expert_info.expert_name}! Thank you for sharing your expertise in {expert_info.expertise_area}. "
            "I'm here to interview you for training Jamie, our conversational AI family assistant. "
            "Jamie helps families manage children's routines, tasks, and behavior. "
            "To start, could you describe your area of expertise and how it could help Jamie better support families?"
        )
    # Add AI message to database
    conversation_history = [{"role": "assistant", "content": ai_message}]
    try:
        await supabase_client.update_session(session_id, conversation_history, 0, False)
        logger.info(f"✅ Initial AI message saved for session {session_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save initial message for session {session_id}: {e}")
        # Continue anyway - the frontend can handle empty conversations
    
    return {
        "session_id": session_id,
        "message": ai_message,
        "question_number": 0
    }

@app.post("/start_interview")
async def start_interview():
    """Legacy endpoint - redirect to expert info collection"""
    raise HTTPException(status_code=400, detail="Please provide expert information first")
@app.get("/upload-doc/health")
def upload_doc_health():
    try:
        from chroma_client import get_chroma_client
        client = get_chroma_client()
        if not client.connected:
            # Check if credentials are present
            import os
            has_api_key = bool(os.getenv("CHROMA_API_KEY"))
            has_tenant = bool(os.getenv("CHROMA_TENANT"))
            has_database = bool(os.getenv("CHROMA_DATABASE"))
            return JSONResponse(
                status_code=503, 
                content={
                    "available": False,
                    "reason": "ChromaDB not connected",
                    "credentials_check": {
                        "has_api_key": has_api_key,
                        "has_tenant": has_tenant,
                        "has_database": has_database
                    },
                    "suggestion": "Check ChromaDB Cloud credentials in .env file and verify API key has access to the specified tenant/database"
                }
            )
        return {"available": True}
    except Exception as e:
        return JSONResponse(
            status_code=503, 
            content={
                "available": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

@app.post("/upload-doc")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    expert_name: str = Form(...),
    expert_email: str = Form(...)
):
    """Upload and process a document (PDF, DOCX, PPTX, TXT) for interview context"""
    logger.info(f"📄 Document upload requested: {file.filename} for session {session_id}")
    
    # Check if ChromaDB document processing is available
    try:
        from chroma_client import get_chroma_client
        chroma_client = get_chroma_client()
        if not chroma_client.connected:
            raise HTTPException(
                status_code=503, 
                detail="Document processing is not available. ChromaDB connection failed."
            )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Document processing is not available: {str(e)}"
        )
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.pptx', '.txt'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (max 50MB)
    max_file_size = 50 * 1024 * 1024  # 50MB
    file_content = await file.read()
    if len(file_content) > max_file_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")
    
    # Reset file pointer
    await file.seek(0)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        # Write uploaded content to temp file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Get document processor
        doc_processor = get_document_processor(client)
        
        # Process the document
        result = await doc_processor.process_uploaded_file(
            file_path=temp_file_path,
            filename=file.filename,
            expert_email=expert_email,
            expert_name=expert_name,
            session_id=session_id
        )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if result["success"]:
            logger.info(f"✅ Document processed successfully: {file.filename}")
            print(f"🎉 SUCCESS: {file.filename} - {result['chunks_processed']} chunks stored in ChromaDB")
            return {
                "success": True,
                "message": f"Document '{file.filename}' has been processed and added to your interview context. I can now discuss its contents with you.",
                "filename": result.get("filename", file.filename),
                "chunks_processed": result["chunks_processed"],
                "file_type": file_ext[1:].upper()
            }
        else:
            logger.error(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Document processing failed'))
            
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"❌ Error processing document upload: {e}")
        
        # Handle missing database tables gracefully
        if "does not exist" in str(e):
            raise HTTPException(
                status_code=503, 
                detail="Document processing tables not found. Please contact administrator to set up document processing."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/chat")
async def chat_with_interviewer(chat_message: ChatMessage):
    """Continue the interview conversation"""
    logger.info(f"💬 CHAT endpoint called for session: {chat_message.session_id}")
    logger.info(f"💬 User message: {chat_message.message[:100]}...")
    session_id = chat_message.session_id
    
    if not session_id or not supabase_client.connected:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get session from database
    session_data = await supabase_client.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_data['is_complete']:
        return {
            "message": "Interview has been completed.",
            "is_complete": True
        }
    
    # Add user's response to conversation history
    conversation_history = session_data['conversation_history']
    conversation_history.append({"role": "user", "content": chat_message.message})
    
    # ChromaDB disabled - using Supabase only
    print(f"💾 User message stored in Supabase for session {session_id}")
    
    logger.info(f"💬 Session {session_id} - User message added. Total messages: {len(conversation_history)}")

    # Check for document-related queries first
    doc_query_patterns = [
        "see my doc", "uploaded file", "check my pdf", "check my ppt", "check my doc",
        "my document", "the file i uploaded", "uploaded document", "can you see", 
        "do you have", "check the document", "look at my", "review my doc",
        "whats in", "what's in", "what is in", "in this doc", "in my doc", "contents of"
    ]
    
    is_doc_query = any(pattern in chat_message.message.lower() for pattern in doc_query_patterns)
    
    if is_doc_query:
        if os.getenv("ENV", "development") != "production":
            print(f"🔍 DOC QUERY DETECTED: '{chat_message.message}' in session {session_id}")
        # Get session documents from ChromaDB
        try:
            doc_processor = get_document_processor(client)
            session_docs_info = doc_processor.get_session_documents(session_id)
            
            if session_docs_info['documents']:
                doc_titles = [doc['title'] for doc in session_docs_info['documents']]
                if os.getenv("ENV", "development") != "production":
                    print(f"✅ FOUND DOCUMENTS: {doc_titles}")
                
                # Simple document retrieval - get all chunks for the session
                try:
                    from chroma_client import get_chroma_client
                    chroma_client = get_chroma_client()
                    
                    # Get ALL chunks for the session
                    all_chunks = chroma_client.collection.get(
                        where={"session_id": session_id}
                    )
                    if os.getenv("ENV", "development") != "production":
                        print(f"🔍 DEBUG: Found {len(all_chunks.get('documents', []))} chunks for session {session_id}")
                    
                    doc_content_preview = ""
                    if all_chunks['documents']:
                        if os.getenv("ENV", "development") != "production":
                            print(f"🔍 Retrieved {len(all_chunks['documents'])} total chunks from documents")
                        
                        # Limit to first 8 chunks for performance (about 2000 tokens max)
                        max_chunks_for_display = 8
                        chunks_to_show = min(len(all_chunks['documents']), max_chunks_for_display)
                        
                        # Organize content by document for better clarity
                        doc_content_map = {}
                        for i, doc_content in enumerate(all_chunks['documents']):
                            metadata = all_chunks['metadatas'][i] if all_chunks['metadatas'] else {}
                            doc_title = metadata.get('title', 'Unknown Document')
                            
                            if doc_title not in doc_content_map:
                                doc_content_map[doc_title] = []
                            doc_content_map[doc_title].append(doc_content)
                        
                        # Create organized response showing content from each document
                        doc_sections = []
                        for doc_title, chunks in doc_content_map.items():
                            # Smart sampling for document query response too
                            if len(chunks) <= 4:
                                # Small document - include all content
                                doc_content = " ".join(chunks)
                            else:
                                # Larger document - sample from beginning, middle, end
                                sample_chunks = chunks[:2] + chunks[len(chunks)//2-1:len(chunks)//2+1] + chunks[-2:]
                                doc_content = " ".join(sample_chunks)
                                if len(chunks) > 4:
                                    doc_content += f"\n\n[Note: This document has {len(chunks)} total sections. Showing key sections above.]"
                            
                            doc_sections.append(f"**{doc_title}:**\n{doc_content}")
                        
                        combined_content = "\n\n".join(doc_sections)
                        
                        ai_message = f"Here's what I found in your documents:\n\n{combined_content}\n\nThis is the actual content from your uploaded documents. What would you like to know about them?"
                    else:
                        ai_message = f"I can see your document '{', '.join(doc_titles)}' but couldn't retrieve its content. Would you like to try asking again or continue with the interview questions?"
                    
                except Exception as e:
                    print(f"Error processing document query: {e}")
                    ai_message = f"I can see your document '{', '.join(doc_titles)}' but had trouble accessing its content. Would you like to try asking again or continue with the interview questions?"
                
                conversation_history.append({"role": "assistant", "content": ai_message})
                await supabase_client.update_session(session_id, conversation_history, session_data['current_question_index'], session_data['is_complete'])
                return {
                    "message": ai_message,
                    "question_number": session_data['current_question_index'] + 1,
                    "is_complete": False,
                    "auto_submitted": False,
                    "final_note": None
                }
            else:
                ai_message = "I don't see any uploaded documents for this session yet. You can upload PDF, DOCX, PPTX, or TXT files using the attachment button, and I'll be able to reference their content in our discussion."
                conversation_history.append({"role": "assistant", "content": ai_message})
                await supabase_client.update_session(session_id, conversation_history, session_data['current_question_index'], session_data['is_complete'])
                return {
                    "message": ai_message,
                    "question_number": session_data['current_question_index'] + 1,
                    "is_complete": False,
                    "auto_submitted": False,
                    "final_note": None
                }
        except Exception as e:
            logger.error(f"Error checking documents: {e}")
            ai_message = "I'm having trouble accessing document information right now, but I can still help with the interview. Would you like to continue with the questions?"
            conversation_history.append({"role": "assistant", "content": ai_message})
            await supabase_client.update_session(session_id, conversation_history, session_data['current_question_index'], session_data['is_complete'])
            return {
                "message": ai_message,
                "question_number": session_data['current_question_index'] + 1,
                "is_complete": False,
                "auto_submitted": False,
                "final_note": None
            }

    # Guard: handle small-talk or project Qs without advancing the question index
    msg_type = is_smalltalk_or_project(chat_message.message)
    if session_data['current_question_index'] == 0 and (msg_type in ["greeting", "smalltalk", "who_are_you", "who_is_jamie", "who_is_timmy", "about_interview"]):
        ai_message = get_smalltalk_response(msg_type)
        if ai_message:
            conversation_history.append({"role": "assistant", "content": ai_message})
            await supabase_client.update_session(session_id, conversation_history, 1, False)
            return {
                "message": sanitize_question(ai_message),
                "question_number": 1,
                "is_complete": False,
                "auto_submitted": False,
                "final_note": None
            }
    
    try:
        # Build previous-questions context for the AI
        previous_questions = []
        for msg in conversation_history:
            if msg['role'] == 'assistant' and '?' in msg['content']:
                question_part = msg['content'].split('?')[0] + '?'
                previous_questions.append(question_part)
        
        # Search for relevant document context using ChromaDB
        doc_context = ""
        try:
            from chroma_client import get_chroma_client
            doc_processor = get_document_processor(client)
            
            # First, debug ChromaDB storage
            chroma_client = get_chroma_client()
            collection_info = chroma_client.get_collection_info()
            print(f"🔍 CHROMA DEBUG - Total chunks in DB: {collection_info.get('total_chunks', 0)}")
            
            # Check session documents
            session_docs = doc_processor.get_session_documents(session_id)
            print(f"🔍 SESSION DEBUG - Session {session_id} has {session_docs['total_chunks']} chunks, {len(session_docs['documents'])} documents")
            
            # Additional debug: show what documents are found
            if session_docs['documents']:
                for doc in session_docs['documents']:
                    print(f"  📄 Found document: {doc.get('title', 'Unknown')} (type: {doc.get('doc_type', 'Unknown')})")
            else:
                print(f"  ❌ No documents found for session {session_id}")
            
            if session_docs['total_chunks'] > 0:
                # Simple document context - get all chunks for the session
                try:
                    from chroma_client import get_chroma_client
                    chroma_client = get_chroma_client()
                    
                    print(f"🔍 DEBUG: Retrieving document context for session_id: {session_id}")
                    all_chunks = chroma_client.collection.get(
                        where={"session_id": session_id}
                    )
                    print(f"🔍 DEBUG: Found {len(all_chunks.get('documents', []))} chunks for context in session {session_id}")
                    
                    if all_chunks['documents']:
                        doc_context_parts = []
                        doc_titles = set()
                        
                        # Collect document titles and provide summary context
                        doc_titles = set()
                        for i, doc_content in enumerate(all_chunks['documents']):
                            metadata = all_chunks['metadatas'][i] if all_chunks['metadatas'] else {}
                            doc_title = metadata.get('title', 'Unknown Document')
                            doc_titles.add(doc_title)
                        
                        if doc_titles:
                            # Get actual document content for context - ensure we get content from ALL documents
                            doc_context_parts = []
                            
                            # Group chunks by document title to ensure we get content from each document
                            doc_content_map = {}
                            for i, doc_content in enumerate(all_chunks['documents']):
                                metadata = all_chunks['metadatas'][i] if all_chunks['metadatas'] else {}
                                doc_title = metadata.get('title', 'Unknown Document')
                                
                                if doc_title not in doc_content_map:
                                    doc_content_map[doc_title] = []
                                doc_content_map[doc_title].append(doc_content)
                            
                            # Get comprehensive content from each document - smart sampling to balance performance
                            for doc_title, chunks in doc_content_map.items():
                                # Smart sampling: take chunks from beginning, middle, and end
                                if len(chunks) <= 3:
                                    # Small document - include all chunks
                                    chunks_to_include = chunks
                                elif len(chunks) <= 6:
                                    # Medium document - take first 2, middle 2, last 2
                                    chunks_to_include = chunks[:2] + chunks[len(chunks)//2-1:len(chunks)//2+1] + chunks[-2:]
                                else:
                                    # Large document - take first 3, middle 2, last 3
                                    chunks_to_include = chunks[:3] + chunks[len(chunks)//2-1:len(chunks)//2+1] + chunks[-3:]
                                
                                for i, chunk_content in enumerate(chunks_to_include):
                                    # Moderate chunk size for good performance
                                    content_preview = chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content
                                    doc_context_parts.append(f"From {doc_title} (section {i+1}): {content_preview}")
                            
                            doc_header = f"**UPLOADED DOCUMENTS:** {', '.join(doc_titles)}\n\n**DOCUMENT CONTEXT:**\n"
                            doc_context = "\n\n" + doc_header + "\n\n".join(doc_context_parts)
                            if os.getenv("ENV", "development") != "production":
                                logger.info(f"📚 Added actual document context from {len(doc_titles)} documents")
                                print(f"🔍 FOUND DOCS: Actual content context from {list(doc_titles)}")
                        else:
                            if os.getenv("ENV", "development") != "production":
                                print(f"🔍 NO DOCS FOUND for session {session_id}")
                    else:
                        if os.getenv("ENV", "development") != "production":
                            print(f"🔍 NO DOCUMENTS in session {session_id} - skipping document search")
                        
                except Exception as e:
                    print(f"❌ DOCUMENT CONTEXT ERROR: {e}")
                    pass
            else:
                print(f"🔍 NO DOCUMENTS in session {session_id} - skipping document search")
                
        except Exception as e:
            # ChromaDB might not be available or connected - this is optional functionality
            if os.getenv("ENV", "development") != "production":
                logger.warning(f"⚠️ Document context search failed: {e}")
                print(f"❌ DOCUMENT SEARCH ERROR: {e}")
            pass
        
        final_system_prompt = build_system_prompt_with_context(SYSTEM_PROMPT, previous_questions, doc_context)
        if doc_context and os.getenv("ENV", "development") != "production":
            print(f"📋 DOCUMENT CONTEXT ADDED TO PROMPT: {len(doc_context)} characters")
        elif not doc_context and os.getenv("ENV", "development") != "production":
            print("📋 NO DOCUMENT CONTEXT - AI should not reference any documents")
        
        ai_message = await get_next_interview_reply(client, final_system_prompt, conversation_history)
        conversation_history.append({"role": "assistant", "content": ai_message})
        current_question_index = session_data['current_question_index'] + 1
        
        # ChromaDB disabled - using Supabase only
        print(f"💾 AI message stored in Supabase for session {session_id}")
        
        # Save to database
        await supabase_client.update_session(session_id, conversation_history, current_question_index, session_data['is_complete'])
        
        # Check if interview should be completed (basic heuristic)
        auto_submitted = False
        final_note = None
        if current_question_index >= 23 or "conclude" in ai_message.lower() or "summary" in ai_message.lower():
            is_complete = True
            # Add a closing message
            final_note = "Thank you for sharing your valuable expertise! The interview is now complete. Your insights will help improve Jamie's ability to support families."
            conversation_history.append({"role": "assistant", "content": final_note})
            await supabase_client.update_session(session_id, conversation_history, current_question_index, True)
            
            # Auto-submit the interview for rule extraction
            try:
                print(f"🎯 AUTO-SUBMIT: Starting rule extraction for completed session {session_id}")
                await submit_interview(session_id)
                auto_submitted = True
                print(f"✅ AUTO-SUBMIT: Rule extraction completed for session {session_id}")
            except Exception as e:
                logger.error(f"Auto-submit failed: {e}")
                print(f"❌ AUTO-SUBMIT ERROR: {e}")
        
        return {
            "message": ai_message,
            "question_number": current_question_index + 1,
            "is_complete": current_question_index >= 23 or "conclude" in ai_message.lower() or "summary" in ai_message.lower(),
            "auto_submitted": auto_submitted,
            "final_note": final_note
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in conversation: {str(e)}")

# Removed old JSON rule generation - now using simple task extraction in submit_interview

@app.post("/submit_interview/{session_id}")
async def submit_interview(session_id: str, companion_id: Optional[int] = None, current_user: dict = Depends(get_current_user)):
    """Finalize interview: store in ChromaDB, extract tasks, save to Supabase"""
    logger.info(f"🔍 Submit interview called for session_id: {session_id}")
    
    # Get session from database
    session_data = await supabase_client.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    # Mark session as complete
    await supabase_client.update_session(session_id, session_data['conversation_history'], session_data['current_question_index'], True)
    
    # Extract tasks from conversation
    try:
        # Create conversation text from session data
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in session_data['conversation_history']
        ])
        
        print(f"🔍 SUBMIT DEBUG: Session {session_id} has {len(session_data['conversation_history'])} messages")
        print(f"🔍 SUBMIT DEBUG: Conversation length: {len(conversation_text)} characters")
        
        if len(conversation_text) < 100:
            print(f"⚠️ SUBMIT WARNING: Very short conversation, may not extract meaningful rules")
            return {
                "message": "Interview too short for rule extraction.",
                "status": "completed",
                "tasks_extracted": 0,
                "tasks": []
            }
        
        # Get document summaries for this session if any exist
        document_context = ""
        try:
            from chroma_client import get_chroma_client
            from document_processor import get_document_processor
            
            doc_processor = get_document_processor(client)
            session_docs = doc_processor.get_session_documents(session_id)
            
            if session_docs['total_chunks'] > 0:
                print(f"📄 Including {session_docs['total_chunks']} document chunks in rule extraction")
                
                # Get all document chunks for this session
                chroma_client = get_chroma_client()
                all_chunks = chroma_client.collection.get(
                    where={"session_id": session_id}
                )
                
                if all_chunks['documents']:
                    # Combine all document content
                    doc_titles = []
                    combined_content = ""
                    
                    # Sort chunks by index for proper order
                    sorted_chunks = []
                    for i, doc_content in enumerate(all_chunks['documents']):
                        metadata = all_chunks['metadatas'][i] if all_chunks['metadatas'] else {}
                        chunk_index = metadata.get('chunk_index', i)
                        doc_title = metadata.get('title', 'Unknown')
                        sorted_chunks.append((chunk_index, doc_content, doc_title))
                        if doc_title not in doc_titles:
                            doc_titles.append(doc_title)
                    
                    sorted_chunks.sort(key=lambda x: x[0])
                    
                    # Combine content from all chunks - include more for comprehensive rule extraction
                    chunk_count = 0
                    for _, content, title in sorted_chunks:
                        if chunk_count >= 50:  # Increased limit for better rule extraction
                            combined_content += "\n[Additional content truncated for processing...]"
                            break
                        combined_content += f"{content}\n\n"
                        chunk_count += 1
                    
                    # Create document context for rule extraction
                    document_context = f"""

**UPLOADED DOCUMENTS:**
The expert also provided the following document(s): {', '.join(doc_titles)}

**DOCUMENT CONTENT:**
{combined_content.strip()}

**DOCUMENT EXTRACTION GUIDANCE:**
- CRITICAL: Extract rules from BOTH the conversation AND the document content above
- Look for specific strategies, techniques, or guidelines mentioned in the documents
- Convert document advice into "Jamie should..." format
- Only extract rules that apply to child behavior, family communication, or task management
- If the documents contain valuable expert knowledge that should be converted into actionable rules add that too
- If the conversation is short but documents contain rich content, extract rules primarily from the documents
- Each rule should start with "Jamie should..." and be actionable for the AI assistant
"""
                    
        except Exception as e:
            print(f"⚠️ Could not include documents in rule extraction: {e}")
            document_context = ""
        
        task_statements = await extract_rules_from_conversation(client, conversation_text, document_context)
        if os.getenv("ENV", "development") != "production":
            print(f"🤖 TASK EXTRACTION: Starting for session {session_id}")
            print(f"📝 CONVERSATION LENGTH: {len(conversation_text)} characters")
            print("🧠 GPT EXTRACTION: Calling GPT-4o-mini for task extraction")
            print(f"💬 CONVERSATION PREVIEW: {conversation_text[:200]}...")
        
        # Tasks will be stored in database only
        
        if os.getenv("ENV", "development") != "production":
            print(f"✅ GPT EXTRACTION SUCCESS: {len(task_statements)} tasks extracted")
            if len(task_statements) > 0:
                print(f"📝 SAMPLE TASK: {task_statements[0][:100]}...")
            else:
                print("⚠️ NO TASKS EXTRACTED - Check conversation content")
        
        # Save each task to Supabase (only if we have valid rules)
        if supabase_client.connected and task_statements:
            if os.getenv("ENV", "development") != "production":
                print(f"💾 SUPABASE SAVE: Saving {len(task_statements)} tasks to database")
            for i, task in enumerate(task_statements, 1):
                if os.getenv("ENV", "development") != "production":
                    print(f"💾 SAVING TASK {i}/{len(task_statements)}: {task[:50]}...")
                    print(f"💾 TASK META: email={session_data.get('expert_email')}, companion_id={session_data.get('companion_id')}")
                # Use passed companion_id OR the one from session_data
                final_companion_id = companion_id if companion_id is not None else session_data.get('companion_id')
                
                rule_id = await supabase_client.save_interview_rule(
                    session_id=session_id,
                    expert_name=session_data['expert_name'],
                    expertise_area=session_data['expertise_area'],
                    rule_text=task,
                    expert_email=session_data.get('expert_email'),
                    companion_id=final_companion_id,
                    user_id=current_user.get('id')
                )
                if os.getenv("ENV", "development") != "production":
                    print(f"💾 RULE SAVED: ID {rule_id} for session {session_id}")
        elif not task_statements:
            if os.getenv("ENV", "development") != "production":
                print("ℹ️ NO RULES TO SAVE: No actionable behavioral rules were extracted")
        else:
            if os.getenv("ENV", "development") != "production":
                print("⚠️ SUPABASE UNAVAILABLE: Tasks saved to memory only")
        
        logger.info(f"✅ Extracted {len(task_statements)} tasks for session {session_id}")
        
        if os.getenv("ENV", "development") != "production":
            print(f"🎉 INTERVIEW COMPLETE: Session {session_id} finished with {len(task_statements)} tasks")
        
    except Exception as e:
        logger.error(f"❌ Error processing interview submission: {e}")
        return {
            "message": "Interview saved, but there was an error processing tasks.",
            "status": "error"
        }


@app.get("/session/{session_id}/conversation")
async def get_conversation(session_id: str):
    """Get the full conversation history for a session"""
    logger.info(f"🔍 Getting conversation for session: {session_id}")
    
    # Check database first
    if supabase_client.connected:
        try:
            db_session = await supabase_client.get_session(session_id)
            if db_session:
                logger.info(f"✅ Found session {session_id} with {len(db_session.get('conversation_history', []))} messages")
                return {
                    "session_id": session_id,
                    "conversation": db_session.get('conversation_history', []),
                    "is_complete": db_session.get('is_complete', False),
                    "companion_id": db_session.get('companion_id'),
                    "source": "database"
                }
            else:
                logger.warning(f"⚠️ Session {session_id} not found in database")
        except Exception as e:
            logger.error(f"❌ Database error getting session {session_id}: {e}")
    else:
        logger.warning("⚠️ Database not connected")
    
    # Return empty conversation for new sessions instead of 404
    logger.info(f"📝 Returning empty conversation for session {session_id}")
    return {
        "session_id": session_id,
        "conversation": [],
        "is_complete": False,
        "source": "empty"
    }

# New database management endpoints
@app.get("/database/status")
async def get_database_status():
    """Get database connection status"""
    try:
        logger.info(f"Checking database status - supabase connected: {supabase_client.connected if supabase_client else False}")
        
        # Check ChromaDB status
        documents_ready = True
        documents_error = None
        
        try:
            from chroma_client import get_chroma_client
            chroma_client = get_chroma_client()
            chroma_info = chroma_client.get_collection_info()
            
            if not chroma_info.get("connected", False):
                documents_ready = False
                documents_error = chroma_info.get("error", "ChromaDB not connected")
        except Exception as e:
            documents_ready = False
            documents_error = f"ChromaDB error: {str(e)}"
        
        if supabase_client and supabase_client.connected:
            return {
                "connected": True,
                "database_type": "Supabase PostgreSQL",
                "message": "Database connection is healthy",
                "documents_ready": documents_ready,
                "documents_error": documents_error,
                "document_engine": "ChromaDB"
            }
        else:
            return {
                "connected": False,
                "message": "Database not connected - check startup logs",
                "documents_ready": False,
                "documents_error": "Main database not connected",
                "document_engine": "ChromaDB"
            }
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {
            "connected": False,
            "error": str(e),
            "message": "Database connection failed",
            "documents_ready": False,
            "documents_error": str(e),
            "document_engine": "ChromaDB"
        }

@app.get("/database/rules")
async def get_all_rules_from_db():
    """Get all extracted rules from database"""
    try:
        if not supabase_client or not supabase_client.connected:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        rules = await supabase_client.get_all_rules()
        return {
            "rules": rules,
            "total_count": len(rules)
        }
    except Exception as e:
        logger.error(f"Error getting rules from database: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rules: {str(e)}")

@app.get("/database/rules/session/{session_id}")
async def get_rules_by_session_from_db(session_id: str):
    """Get all rules for a specific session from database"""
    try:
        logger.info(f"🔍 Getting rules for session: {session_id}")
        if not supabase_client or not supabase_client.connected:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        rules = await supabase_client.get_rules_by_session(session_id)
        logger.info(f"📊 Found {len(rules)} rules for session {session_id}")
        
        return {
            "session_id": session_id,
            "rules": rules,
            "total_count": len(rules)
        }
    except Exception as e:
        logger.error(f"Error getting rules for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rules for session: {str(e)}")

@app.get("/database/rules/count")
async def get_rules_count():
    """Get total count of rules in database"""
    try:
        if not supabase_client or not supabase_client.connected:
            return {"count": 0, "connected": False}
        
        rules = await supabase_client.get_all_rules()
        return {"count": len(rules), "connected": True}
    except Exception as e:
        logger.error(f"Error getting rules count: {e}")
        return {"count": 0, "connected": False, "error": str(e)}

@app.get("/database/rules/collection/{session_id}")
async def get_rules_collection(session_id: str):
    """Get the rules collection for a session"""
    try:
        if not supabase_client or not supabase_client.connected:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        rules = await supabase_client.get_rules_by_session(session_id)
        if not rules:
            return {
                "session_id": session_id,
                "rules": [],
                "total_rules": 0,
                "status": "processing"
            }
        
        return {
            "session_id": session_id,
            "rules": rules,
            "total_rules": len(rules),
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error getting rules collection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rules collection: {str(e)}")

@app.get("/processing/status/{session_id}")
async def get_processing_status(session_id: str):
    """Check if rule processing is complete for a session"""
    try:
        if not supabase_client or not supabase_client.connected:
            return {"status": "database_unavailable"}
        
        rules = await supabase_client.get_rules_by_session(session_id)
        if rules:
            return {
                "status": "completed",
                "rules_count": len(rules)
            }
        else:
            return {"status": "processing"}
            
    except Exception as e:
        logger.error(f"Error checking processing status: {e}")
        return {"status": "error", "message": str(e)}


# Security dependency (unified)

# ----- Public user authentication routes (non-admin) -----

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
async def login(request: LoginRequest):
    """
    Unified login endpoint for Users and Admins.
    Returns token, user info, and role.
    """
    # Authenticate against unified users table
    result = user_auth.authenticate(request.email, request.password)
    if not result:
        # Avoid revealing if user exists or password is wrong
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    logger.info(f"✅ Login success: {request.email} (Role: {result['user']['role']})")
    return result

@app.post("/users/register")
async def register_user(request: UserRegisterRequest):
    """
    Register a new user (default role='user').
    """
    result = user_auth.register(request.email, request.password, request.name)
    if not result:
        raise HTTPException(status_code=400, detail="Unable to register user")
    return result

@app.post("/users/login")
async def login_user_legacy(request: UserLoginRequest):
    """Legacy user login -> Redirects logic to unified auth"""
    return await login(LoginRequest(email=request.email, password=request.password))


@app.get("/users/me/sessions")
async def get_my_sessions(current_user=Depends(get_current_user)):
    """
    Get all interview sessions that belong to the logged-in user.
    A session is considered owned by a user if its expert_email matches the user's email.
    """
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        all_sessions = await supabase_client.get_all_sessions()
        user_email = current_user["email"]
        
        conversations = []
        for session in all_sessions:
            if session.get("expert_email") == user_email:
                messages = session.get('conversation_history', [])
                if messages: # Only show sessions with messages
                    conversations.append({
                        "session_id": session['session_id'],
                        "expert_name": session.get('expert_name', 'Expert'),
                        "expertise_area": session.get('expertise_area', 'General'),
                        "completed": session.get('is_complete', False),
                        "messages": messages,
                        "companion_id": session.get("companion_id")
                    })
        
        return {
            "sessions": conversations,
            "total": len(conversations),
        }
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user sessions")


@app.get("/users/me/stats")
async def get_my_stats(current_user=Depends(get_current_user)):
    """Get statistics for the logged-in expert"""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        user_email = current_user["email"]
        user_id = current_user["id"]
        
        # Sessions count
        all_sessions = await supabase_client.get_all_sessions()
        total_interviews = sum(1 for s in all_sessions if s.get("expert_email") == user_email)
        
        # Rules stats
        db_rules = await supabase_client.get_all_rules()
        user_rules = [r for r in db_rules if r.get("user_id") == user_id or r.get("expert_email") == user_email]
        
        total_rules = len(user_rules)
        approved_rules = sum(1 for r in user_rules if r.get("completed", False))
        pending_rules = total_rules - approved_rules
        
        return {
            "total_interviews": total_interviews,
            "pending_tasks": pending_rules,
            "approved_tasks": approved_rules,
            "rejected_tasks": total_rules
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")


@app.get("/users/me/tasks")
async def get_my_tasks(companion_id: Optional[int] = None, current_user=Depends(get_current_user)):
    """Get tasks/rules belonging to the logged-in expert"""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        user_email = current_user["email"]
        user_id = current_user["id"]
        
        db_rules = await supabase_client.get_all_rules()
        # Filter by user_id or expert_email for safety/migration
        user_rules = [r for r in db_rules if r.get("user_id") == user_id or r.get("expert_email") == user_email]
        
        if companion_id is not None:
            user_rules = [r for r in user_rules if r.get("companion_id") == companion_id]
            
        all_companions = supabase_client.get_all_companions()
        companion_map = {c["id"]: c["name"] for c in all_companions}
        companion_map[None] = "Unassigned"
        
        user_companions = supabase_client.list_companions_for_user(user_id)
        
        tasks = []
        for rule in user_rules:
            completed = rule.get('completed', False)
            status = "completed" if completed else "pending"
            cid = rule.get('companion_id')
            cname = companion_map.get(cid, "Unassigned")
            
            tasks.append({
                "id": str(rule['id']),
                "session_id": str(rule['session_id']),
                "expert_name": str(rule.get('expert_name', 'Expert')),
                "task_text": str(rule.get('rule_text', '')),
                "category": str(rule.get('expertise_area', 'General')),
                "status": status,
                "companion_id": cid,
                "companion_name": cname
            })
            
        return {"tasks": tasks, "companions": user_companions}
    except Exception as e:
        logger.error(f"Error getting user tasks: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving tasks")


@app.post("/users/me/tasks/{task_id}/approve")
async def approve_user_task(task_id: str, current_user=Depends(get_current_user)):
    """Approve a task owned by the user and send to Jira"""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        user_email = current_user["email"]
        user_id = current_user["id"]
        
        db_rules = await supabase_client.get_all_rules()
        rule = next((r for r in db_rules if str(r['id']) == task_id), None)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
            
        # Verify ownership
        if rule.get("user_id") != user_id and rule.get("expert_email") != user_email:
            raise HTTPException(status_code=403, detail="You do not have permission to approve this rule")
            
        # Jira integration (same as admin)
        jira_key = jira_client.create_task(
            summary_text=f"AI Coach Rule: {rule['rule_text'][:100]}...",
            description=f"Expert: {rule.get('expert_name', 'Expert')}\nArea: {rule.get('expertise_area', 'General')}\nRule: {rule['rule_text']}"
        )
        
        if jira_key:
            await supabase_client.update_rule_completed(rule['id'], True)
            return {"success": True, "jira_key": jira_key, "message": f"Task approved and added to Jira: {jira_key}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create Jira task")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving user task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/me/tasks/{task_id}/reject")
async def reject_user_task(task_id: str, current_user=Depends(get_current_user)):
    """Reject a task owned by the user"""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
        
    try:
        user_email = current_user["email"]
        user_id = current_user["id"]
        
        db_rules = await supabase_client.get_all_rules()
        rule = next((r for r in db_rules if str(r['id']) == task_id), None)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
            
        # Verify ownership
        if rule.get("user_id") != user_id and rule.get("expert_email") != user_email:
            raise HTTPException(status_code=403, detail="You do not have permission to reject this rule")
            
        await supabase_client.update_rule_completed(rule['id'], True)
        return {"success": True, "message": "Task rejected"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting user task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/me/companions")
async def get_my_companions(current_user=Depends(get_current_user)):
    """List companions this user can train: Jamie (product) + My expert persona."""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    companions = supabase_client.list_companions_for_user(current_user["id"])
    # Ensure "My expert persona" is always an option (id=None until first use)
    has_persona = any(c.get("type") == "user_persona" for c in companions)
    if not has_persona:
        companions.append({"id": None, "name": "My expert persona", "slug": "my_persona", "type": "user_persona", "user_id": current_user["id"]})
    return {"companions": companions}


@app.post("/users/me/start_interview")
async def start_my_interview(body: StartInterviewRequest, current_user=Depends(get_current_user)):
    """Start an interview as the logged-in user. Uses user's name/email; target is Jamie or their persona."""
    if not supabase_client.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    global session_counter
    session_id = str(session_counter)
    session_counter += 1

    companion_id = None
    if body.companion_slug == "my_persona":
        persona = supabase_client.get_or_create_user_persona(
            current_user["id"],
            current_user.get("name") or "Expert",
            expertise_area=body.expertise_area or "General",
        )
        if not persona:
            raise HTTPException(status_code=500, detail="Could not create or load your expert persona")
        companion_id = persona["id"]
    else:
        c = supabase_client.get_companion_by_slug(body.companion_slug or "jamie")
        if c:
            companion_id = c["id"]
    if companion_id is None:
        jamie = supabase_client.get_companion_by_slug("jamie")
        companion_id = jamie["id"] if jamie else None

    expert_name = current_user.get("name") or "Expert"
    expert_email = current_user["email"]
    logger.info(f"💾 START INTERVIEW: expert={expert_name}, companion_id={companion_id}")
    await supabase_client.save_session(session_id, expert_name, expert_email, body.expertise_area or "General", companion_id=companion_id)

    is_persona = body.companion_slug == "my_persona"
    if is_persona:
        ai_message = (
            f"Hello {expert_name}! Thank you for sharing your expertise in {body.expertise_area or 'General'}. "
            "I'm here to interview you so we can train your own expert persona—a model that will act like you. "
            "To start, could you describe your area of expertise and how you usually apply it?"
        )
    else:
        ai_message = (
            f"Hello {expert_name}! Thank you for sharing your expertise in {body.expertise_area or 'General'}. "
            "I'm here to interview you for training Jamie, our conversational AI family assistant. "
            "Jamie helps families manage children's routines, tasks, and behavior. "
            "To start, could you describe your area of expertise and how it could help Jamie better support families?"
        )
    conversation_history = [{"role": "assistant", "content": ai_message}]
    try:
        await supabase_client.update_session(session_id, conversation_history, 0, False)
    except Exception as e:
        logger.error(f"Failed to save initial message: {e}")
    return {"session_id": session_id, "message": ai_message, "question_number": 0, "companion_id": companion_id}


# Admin Routes
@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Serve admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

# Removed insecure manual admin creation endpoint for production security

@app.post("/admin/login")
async def admin_login_legacy(login_request: AdminLoginRequest):
    """Legacy admin login -> Redirects logic to unified auth"""
    result = await login(LoginRequest(email=login_request.email, password=login_request.password))
    # Verify it is an admin
    if result["user"]["role"] != "admin":
         raise HTTPException(status_code=403, detail="Not an admin account")
    return result

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Serve admin dashboard - authentication handled in JavaScript"""
    response = templates.TemplateResponse("admin_dashboard.html", {"request": request})
    # Prevent caching to avoid back button access
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/admin/conversations")
async def get_admin_conversations(current_admin = Depends(get_current_admin)):
    """Get all conversations for admin panel from Supabase"""
    try:
        if not supabase_client.connected:
            return {"conversations": []}
        
        all_sessions = await supabase_client.get_all_sessions()
        conversations = []
        
        for session in all_sessions:
            conversation_history = session.get('conversation_history', [])
            if conversation_history:
                conversations.append({
                    "session_id": session['session_id'],
                    "expert_name": session.get('expert_name', 'Unknown Expert'),
                    "expertise_area": session.get('expertise_area', 'General'),
                    "completed": session.get('is_complete', False),
                    "messages": conversation_history
                })
        
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Error getting admin conversations: {e}")
        return {"conversations": []}

@app.get("/admin/companions")
async def get_admin_companions(current_admin=Depends(get_current_admin)):
    """List all companions (for admin filter/grouping)."""
    if not supabase_client.connected:
        return {"companions": []}
    companions = supabase_client.get_all_companions()
    return {"companions": companions}


@app.get("/admin/tasks")
async def get_admin_tasks(companion_id: Optional[int] = None, current_admin=Depends(get_current_admin)):
    """Get all tasks for admin panel. Optional companion_id to filter. Returns tasks, companions, and tasks grouped by companion."""
    try:
        if not supabase_client.connected:
            return {"tasks": [], "companions": [], "tasks_by_companion": {}}
        
        companions = supabase_client.get_all_companions()
        companion_map = {c["id"]: c["name"] for c in companions}
        companion_map[None] = "Unassigned"

        db_rules = await supabase_client.get_all_rules()
        if companion_id is not None:
            db_rules = [r for r in db_rules if r.get("companion_id") == companion_id]

        tasks = []
        for rule in db_rules:
            completed = rule.get('completed', False)
            status = "completed" if completed else "pending"
            cid = rule.get('companion_id')
            cname = companion_map.get(cid, "Unassigned")
            raw_rule_text = rule.get('rule_text')
            rule_text = str(raw_rule_text) if raw_rule_text else "No rule text available"
            tasks.append({
                "id": str(rule['id']),
                "session_id": str(rule['session_id']),
                "expert_name": str(rule.get('expert_name', 'Expert User')),
                "task_text": rule_text,
                "category": str(rule.get('expertise_area', 'General')),
                "priority": "medium",
                "status": status,
                "companion_id": cid,
                "companion_name": cname
            })

        tasks.sort(key=lambda x: (x['status'] == 'completed', x['id']))
        tasks_by_companion = {}
        for t in tasks:
            name = t["companion_name"]
            if name not in tasks_by_companion:
                tasks_by_companion[name] = []
            tasks_by_companion[name].append(t)
        return {"tasks": tasks, "companions": companions, "tasks_by_companion": tasks_by_companion}
    except Exception as e:
        logger.error(f"Error getting admin tasks: {e}")
        return {"tasks": [], "companions": [], "tasks_by_companion": {}}

@app.post("/admin/approve/{task_id}")
async def approve_task(task_id: str, current_admin = Depends(get_current_admin)):
    """Approve task and send to Jira"""
    try:
        print(f"🎯 APPROVE TASK: Starting approval for task {task_id}")
        
        if supabase_client.connected:
            # Get rule by ID
            db_rules = await supabase_client.get_all_rules()
            rule = next((r for r in db_rules if str(r['id']) == task_id), None)
            
            if rule:
                print(f"📋 RULE FOUND: {rule['rule_text'][:50]}...")
                # Create Jira task
                jira_key = jira_client.create_task(
                    summary_text=f"AI Coach Rule: {rule['rule_text'][:100]}...",
                    description=f"Expert: {rule.get('expert_name', 'Expert User')}\nArea: {rule.get('expertise_area', 'General')}\nRule: {rule['rule_text']}"
                )
                
                if jira_key:
                    print(f"✅ JIRA SUCCESS: Created task {jira_key}")
                    # Update rule status to completed
                    await supabase_client.update_rule_completed(rule['id'], True)
                    return {"success": True, "jira_key": jira_key, "message": f"Task approved and added to Jira: {jira_key}"}
                else:
                    print("❌ JIRA FAILED: Could not create task")
                    return {"success": False, "error": "Failed to create Jira task"}
            else:
                print(f"❌ NO RULE FOUND for task {task_id}")
                return {"success": False, "error": "Rule not found"}
        else:
            print("❌ DATABASE NOT CONNECTED")
            return {"success": False, "error": "Database not connected"}
        
    except Exception as e:
        print(f"❌ APPROVE ERROR: {e}")
        logger.error(f"Error approving task: {e}")
        return {"success": False, "error": str(e)}

@app.post("/admin/reject/{task_id}")
async def reject_task(task_id: str, current_admin = Depends(get_current_admin)):
    """Reject task without sending to Jira"""
    try:
        print(f"❌ REJECT TASK: Starting for task {task_id}")
        
        if supabase_client.connected:
            # Get rule by ID
            db_rules = await supabase_client.get_all_rules()
            rule = next((r for r in db_rules if str(r['id']) == task_id), None)
            
            if rule:
                # Mark as completed (rejected)
                await supabase_client.update_rule_completed(rule['id'], True)
                print(f"✅ REJECT SUCCESS: Task {task_id} rejected")
                return {"success": True, "message": "Task rejected"}
            else:
                return {"success": False, "error": "Rule not found"}
        else:
            print("❌ DATABASE NOT CONNECTED")
            return {"success": False, "error": "Database not connected"}
        
    except Exception as e:
        print(f"❌ REJECT ERROR: {e}")
        logger.error(f"Error rejecting task: {e}")
        return {"success": False, "error": str(e)}

@app.get("/admin/stats")
async def get_admin_stats(current_admin = Depends(get_current_admin)):
    """Get dashboard statistics"""
    try:
        if not supabase_client.connected:
            return {"total_interviews": 0, "pending_tasks": 0, "approved_tasks": 0, "rejected_tasks": 0}
        
        # Get all sessions (interviews) - this is the correct count
        all_sessions = await supabase_client.get_all_sessions()
        total_interviews = len(all_sessions)
        
        # Get task statistics from rules
        db_rules = await supabase_client.get_all_rules()
        total_rules = len(db_rules)
        approved_tasks = sum(1 for r in db_rules if r.get('completed', False))
        pending_tasks = total_rules - approved_tasks
        
        return {
            "total_interviews": total_interviews,
            "pending_tasks": pending_tasks,
            "approved_tasks": approved_tasks,
            "rejected_tasks": total_rules
        }
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return {"total_interviews": 0, "pending_tasks": 0, "approved_tasks": 0, "rejected_tasks": 0}

if __name__ == "__main__":
    import uvicorn
    # Use APP_PORT so it doesn't conflict with DB "port" (e.g. port=5432 in .env)
    port = int(os.getenv("APP_PORT") or os.getenv("PORT", "8000"))
    if os.getenv("ENV", "development") != "production":
        print("🚀 Starting AI Coach Interview System...")
        print(f"🌐 Binding to: 0.0.0.0:{port}")
        print(f"🔗 Interview Interface: http://localhost:{port}")
        print(f"🔗 Admin Panel: http://localhost:{port}/admin")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
