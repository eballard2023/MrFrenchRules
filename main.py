from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
import os
import json
import tempfile
import shutil
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List, Dict, Optional
import asyncio
# MongoDB removed - using Supabase + ChromaDB
from supabase_client import supabase_client
from admin_auth import admin_auth
from jira_client import jira_client
from document_processor import get_document_processor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure FastAPI docs exposure by environment
is_prod = os.getenv("ENV", "development") == "production"

app = FastAPI(
    title="AI Coach Interview Model",
    version="1.0.0",
    docs_url=None if is_prod else "/docs",
    redoc_url=None if is_prod else "/redoc",
    openapi_url=None if is_prod else "/openapi.json",
)

# In production, provide friendly redirects for disabled docs endpoints
if is_prod:
    @app.get("/docs", include_in_schema=False)
    async def _docs_redirect():
        return RedirectResponse(url="/")

    @app.get("/redoc", include_in_schema=False)
    async def _redoc_redirect():
        return RedirectResponse(url="/")

    @app.get("/openapi.json", include_in_schema=False)
    async def _openapi_blocked():
        return JSONResponse(status_code=404, content={"detail": "OpenAPI schema is disabled in production"})

# Set up OpenAI client
from openai import AsyncOpenAI
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

# Helper: filter to keep only child/family behavior rules, drop meta/interview/project rules
def is_behavior_rule(rule: Dict) -> bool:
    try:
        # Gather text from rule fields
        text_parts: List[str] = []
        if isinstance(rule, dict):
            if_part = rule.get("if", {})
            then_part = rule.get("then", {})
            for part in (if_part, then_part, rule):
                if isinstance(part, dict):
                    for v in part.values():
                        if isinstance(v, str):
                            text_parts.append(v)
                elif isinstance(part, str):
                    text_parts.append(part)
        full_text = " \n ".join(text_parts).lower()

        # Hard filters for meta/interview/project chatter
        meta_terms = [
            "interview", "question", "script", "facilitate", "introduce yourself", "who is mr. french",
            "who is timmy", "what is mr. french", "project context", "characters", "definition",
            "i'm here to help", "let's dive right in", "area of expertise", "describe your expertise"
        ]
        if any(term in full_text for term in meta_terms):
            return False

        # Require presence of behavior/parenting/child-context cues
        behavior_terms = [
            "child", "kid", "parent", "family", "behavior", "behaviour", "routine", "task",
            "reward", "consequence", "reinforcement", "positive", "timeout", "break",
            "homework", "bedtime", "screen", "calm", "de-escalation", "encourage", "motivate",
            "red zone", "green zone", "blue zone", "emotion", "frustrated", "angry", "upset",
            "praise", "token", "sticker", "chore", "schedule", "reminder"
        ]
        return any(term in full_text for term in behavior_terms)
    except Exception:
        return False

# Helper: sanitize AI question to remove numbering/bullets
import re
def sanitize_question(text: str) -> str:
    try:
        # Remove leading numbering or bullet patterns at the start of lines
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(r"^\s*(?:\(?\d+\)?[\).:-]\s+|[-*•]\s+)", "", line)
            cleaned_lines.append(cleaned)
        cleaned_text = "\n".join(cleaned_lines).strip()
        return cleaned_text
    except Exception:
        return text

# Helper: remove praise/evaluative phrases for neutral tone
def neutralize_praise(text: str) -> str:
    try:
        phrases = [
            r"\bthat's\s+great\b", r"\bgreat\b", r"\bexcellent\b", r"\blove\s+that\b",
            r"\bawesome\b", r"\bperfect\b", r"\bwonderful\b", r"\bbrilliant\b",
            r"\bthat's\s+exactly\s+right\b", r"\bwell\s+done\b", r"\bgood\s+job\b",
            r"\bamazing\b", r"\bfantastic\b", r"\bimpressive\b", r"\bnice\b",
            r"\bthank\s+you\s+for\s+sharing\b", r"\bappreciate\b"
        ]
        neutral = text
        for p in phrases:
            neutral = re.sub(p, "", neutral, flags=re.IGNORECASE)
        # Collapse extra spaces created by removals
        neutral = re.sub(r"\s{2,}", " ", neutral).strip()
        return neutral if neutral else text
    except Exception:
        return text

# Helper: detect small-talk or project questions
def _matches_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None

def _any_phrase(text: str, phrases: list[str]) -> bool:
    return any(_matches_phrase(text, p) for p in phrases)

def is_smalltalk_or_project(message: str) -> str:
    m = (message or "").strip().lower()
    if not m:
        return "none"
    greetings = ["hello", "hi", "hey"]
    smalltalk = ["how are you", "how r u", "how are u", "how's it going"]
    who_are_you = ["who are you", "who r u", "what are you"]
    who_is_mrfrench = ["who is mr french", "what is mr french"]
    who_is_timmy = ["who is timmy", "what is timmy"]
    about_interview = [
        "what is this about", "what is this interview about", "what is this interview", "what's this about",
        "why am i here", "what will you ask", "purpose of this interview", "what is this for"
    ]
    if _any_phrase(m, greetings):
        return "greeting"
    if _any_phrase(m, smalltalk):
        return "smalltalk"
    if _any_phrase(m, who_are_you):
        return "who_are_you"
    if _any_phrase(m, who_is_mrfrench):
        return "who_is_mrfrench"
    if _any_phrase(m, who_is_timmy):
        return "who_is_timmy"
    if _any_phrase(m, about_interview):
        return "about_interview"
    return "none"

# Helper: clean up any leading punctuation artifacts
def clean_response(text: str) -> str:
    try:
        content = (text or "").strip()
        # Remove leading punctuation marks that shouldn't be there
        cleaned = re.sub(r"^[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/\s]+", "", content)
        # Remove quotes around the entire text (straight or smart quotes)
        cleaned = re.sub(r'^["“”\'](.+)["“”\']$', r"\1", cleaned)
        return cleaned.strip() if cleaned else text
    except Exception:
        return text

# Helper: extract only the first question from text
def keep_only_first_question(text: str) -> str:
    try:
        content = (text or "").strip()
        if not content:
            return content
        q_index = content.find("?")
        if q_index == -1:
            return content
        # Walk backwards to previous sentence boundary
        start = max(content.rfind(".", 0, q_index), content.rfind("!", 0, q_index), content.rfind("?", 0, q_index))
        start = 0 if start == -1 else start + 1
        question = content[start:q_index + 1].strip()
        # Strip wrapping quotes
        if (question.startswith('"') and question.endswith('"')) or (question.startswith("'") and question.endswith("'")):
            question = question[1:-1].strip()
        return question
    except Exception:
        return text

# Helper: ensure acknowledgment for AI responses
def ensure_acknowledgment(text: str, ack: str = "Understood.") -> str:
    """Ensures the text ends with an acknowledgment if it doesn't already."""
    if text.strip().endswith(ack):
        return text
    return f"{text} {ack}"


FIRST_QUESTION_TEXT = "To start, could you describe your area of expertise and how you usually apply it?"

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
    
    # Attempt document schema setup; continue if it fails, but record status
    try:
        supabase_client.ensure_document_schema()
        supabase_client.documents_ready = True
        supabase_client.documents_error = None
        logger.info("✅ Document schema verified")
    except Exception as e:
        supabase_client.documents_ready = False
        supabase_client.documents_error = str(e)
        logger.warning(f"⚠️ Document schema not ready: {e}")

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

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class InterviewSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = []
        self.extracted_rules = []
        self.current_question_index = 0
        self.is_complete = False
        self.created_at = datetime.now()
        self.expert_name = "Unknown Expert"
        self.expert_email = "unknown@example.com"
        self.expertise_area = "General"
        self.asked_questions = set()  # Track asked questions to prevent repetition

# System prompt for the AI interviewer
SYSTEM_PROMPT = """You are an AI interviewer designed to extract behavioral rules and best practices from subject matter experts (SMEs). These rules will be fed into Mr. French AI to make it behave like an expert.

**MR. FRENCH PROJECT CONTEXT:**
Mr. French is a conversational AI that helps families manage children's routines, tasks, and behavior through three connected chat experiences:
- Parent ↔ Mr. French (task management, progress reports, zone updates, context discussions)
- Timmy (child) ↔ Mr. French (reminders, encouragement, task completion, "what's due" queries)
- Parent ↔ Timmy (capturing real family instructions like "Timmy, do the dishes" into actionable tasks)

Current Implementation - The Three Chats
Mr. French ties together three distinct but connected conversation types. Collections are stored in ChromaDB with metadata (role, sender, timestamp, group_id for family isolation).

### 4.1 Parent ↔ Mr. French (`parent-mrfrench`)
- **Audience**: The Parent talking directly to Mr. French
- **Purpose**: Ask for summaries, add/update/delete tasks, change Timmy's zone when explicitly requested, receive progress reports, and discuss context
- **Flow**: Parent sends a message → Mr. French analyzes intent → Mr. French replies to the Parent and may update tasks/zone and notify Timmy
- **Storage**: Chroma collection `parent-mrfrench` with family isolation
- **Authentication**: Required via session token

### 4.2 Timmy ↔ Mr. French (`timmy-mrfrench`)
- **Audience**: Timmy talking directly to Mr. French
- **Purpose**: Ask "what's due," declare task completion, receive reminders, get encouragement, and learn about rewards
- **Flow**: Timmy sends a message → Mr. French analyzes intent (e.g., update completion) → Mr. French replies and may notify the Parent
- **Storage**: Chroma collection `timmy-mrfrench` with family isolation
- **Authentication**: Required via session token

### 4.3 Parent ↔ Timmy (`parent-timmy`)
- **Audience**: Parent and Timmy talking to each other (human conversation), optionally AI-simulated
- **Purpose**: Capture real family instructions like "Timmy, do the dishes," which should become actionable tasks. Optionally generate full AI conversations for testing or demos
- **Flow**: Message is saved to `parent-timmy-realtime`, then Mr. French's analyzer parses the latest turn and updates tasks when appropriate (e.g., direct commands to Timmy)
- **Storage**: Chroma collection `parent-timmy-realtime` with family isolation. For AI-simulated exchanges, `ai-parent-timmy` is used
- **Authentication**: Required via session token

**CORE FUNCTIONALITIES:**
- Converts everyday language into structured, trackable tasks with due dates/times and rewards
- Maintains memory and context across multiple conversation threads using vector storage
- Triggers reminders and updates automatically through scheduling
- Keeps Parent informed and gently guides Timmy
- Supports multiple families with secure authentication and data isolation
- Collects comprehensive child information through guided onboarding

**TIMMY ZONE SYSTEM:**
- Red Zone: High stress, frustration, or emotional distress - requires calm, supportive responses
- Green Zone: Normal, engaged state - can handle routine tasks and learning
- Blue Zone: Low energy, tired, or disengaged - needs gentle encouragement and simple tasks

**AREAS OF EXPERTISE NEEDED:**
- Parental Communication Strategies
- Child Task Management and Motivation
- Behavioral Analysis and Response Patterns
- Age-Appropriate Reward Systems
- Routine Establishment and Maintenance
- Crisis Management and De-escalation
- Progress Measurement and Feedback
- Family Dynamic Understanding
- Task Intent Recognition from Natural Language
- Cross-Chat Context Management

**YOUR ROLE:**
- You are an INTERVIEWER, not a general assistant
- If the user greets (e.g., "how are you?"), reply briefly and warmly, then pivot to the interview
- If asked "who are you?", reply: you are an AI interviewer to extract expert rules for Mr. French, then ask if they’re ready to continue
- If asked about the Mr. French project or Timmy, answer concisely from context, then ask if they’re ready to continue
- For unrelated general-knowledge/trivia (e.g., celebrities), politely say it’s out of scope and steer back to the interview
- Dont introduce yourself unless asked; keep responses concise and conversational
- ALWAYS conduct the interview using the script below, one question at a time, listening to their views

**INTERVIEW SCRIPT - Ask ONE question at a time, framed around Mr. French:**

**KICKOFF PHASE:**
1. "To start, could you describe your area of expertise and how it could help Mr. French better support families?"
2. "What guiding principles or philosophies shape your approach to working with children and families?"
3. "What outcomes do you try to help families achieve through your methods?"
4. "How do you usually measure progress or success in family and child development?"

**PROCESSES & METHODS:**
5. "Can you walk me through the main steps or stages of your approach that Mr. French could implement?"
6. "Are there specific frameworks, routines, or tools you rely on that could help Mr. French create better family routines?"
7. "What common challenges do families face with children, and how do you recommend handling them?"
8. "How do you adapt your methods for different ages, personalities, or family contexts?"

**GUARDRAILS & BOUNDARIES:**
9. "What should Mr. French never do or say when supporting families?"
10. "Are there disclaimers or boundaries that Mr. French must always respect when helping with children?"
11. "When should Mr. French step back and suggest human involvement instead?"

**TONE & STYLE:**
12. "How should Mr. French 'sound' when talking to children — more like a coach, a teacher, a friend, or something else?"
13. "Are there certain words, metaphors, or examples you often use that Mr. French could adopt?"
14. "How should Mr. French adjust its style for different ages, cultures, or learning levels?"

**HANDLING VARIABILITY & EXCEPTIONS:**
15. "What are the most frequent mistakes families make with children, and how should Mr. French respond?"
16. "If a child misunderstands or resists, how should Mr. French handle it?"
17. "When Mr. French reaches its limit in helping a family, what's the right next step?"

**KNOWLEDGE DEPTH & UPDATING:**
18. "Which parts of your knowledge about child development are timeless, and which may change as research evolves?"
19. "How should Mr. French keep its knowledge about child development current over time?"
20. "Are there sources or references you trust that Mr. French should prioritize for family guidance?"

**OPTIONAL DEEP DIVES:**
21. "Could you share a typical family scenario that illustrates your approach?"
22. "If Mr. French could only carry one principle from your expertise, what should it be?"
23. "What red flags should Mr. French watch for that suggest a family situation needs immediate attention?"

**INTERVIEW RULES:**
- Ask ONLY ONE question at a time
- Wait for their response before asking the next
- Be conversational and natural
- Start EVERY interview with Mr. French introduction and purpose explanation
- Ask if they want to know about current implementation and how they can help
- Do NOT number or list questions; phrase naturally
- Do NOT wrap questions in quotation marks; write conversationally without quotes
- After small-talk or project questions (who are you / Mr. French / Timmy), answer briefly and ask if they’re ready to continue the interview
- For unrelated trivia, decline and return to the interview
- **CRITICAL**: On greeting ("hello", "hi"), reply with greeting and continue the interview dont give intro of mr french again and again tell him if he asks otherwise continue the interview
- **NEVER** respond with "I'm here to help" or similar general assistant language
- **RESPONSE STYLE**: Keep responses brief and neutral. Avoid praise or evaluative language (e.g., "great", "excellent", "love that", "that's exactly right"). After receiving an answer, give a short neutral acknowledgment (e.g., "Noted." or "Understood."). If the user asks a question at the end of their response (indicated by a question mark), acknowledge it briefly (e.g., "That dashboard concept could be valuable for families.") then proceed with the next interview question. Don't elaborate on their previous response unless they specifically ask for clarification.
- **DOCUMENT AWARENESS**: ONLY reference documents if explicit document context is provided in this prompt. If no document context appears below, DO NOT make up or invent document content. Simply say you cannot access the document content and focus on the interview questions instead. When document context IS provided, reference it confidently with the document title.
- **CRITICAL**: NEVER explain, analyze, judge, compliment, congratulate, or praise their previous answer. Just acknowledge briefly and ask the next question. If they ask a question, give a brief 1-sentence response then ask your next question. Keep total responses under 3 sentences.
- **SCRIPT ADHERENCE**: While being responsive to their answers, ensure you cover the key areas from the interview script above. You can ask follow-up questions based on their responses, but make sure to eventually cover all the main topics: expertise/principles, outcomes/measurement, processes/methods, guardrails/boundaries, tone/style, handling variability, and knowledge depth.
- **CRITICAL**: NEVER repeat questions that have already been asked in this session. Keep track of what has been covered and move to new topics. If a similar area needs exploration, ask from a different angle or focus on a different aspect.
- **QUESTION TRACKING**: Before asking any question, consider what has already been discussed. Avoid asking about the same topic twice, even if phrased differently."""

@app.get("/", response_class=HTMLResponse)
async def get_start_page(request: Request):
    """Serve the expert info collection page"""
    logger.info("🏠 Start page requested")
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/interview", response_class=HTMLResponse)
async def get_interview_page(request: Request):
    """Serve the interview page"""
    logger.info("💬 Interview page requested")
    return templates.TemplateResponse("interview.html", {"request": request})

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

class ExpertInfo(BaseModel):
    expert_name: str
    expert_email: str
    expertise_area: str = "General"

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
        raise HTTPException(status_code=503, detail="Database not available")
    
    await supabase_client.save_session(session_id, expert_info.expert_name, expert_info.expert_email, expert_info.expertise_area)
    
    # Using Supabase only for session storage
    print(f"💾 Using Supabase only for session {session_id}")
    
    # Start with personalized greeting
    ai_message = (
        f"Hello {expert_info.expert_name}! Thank you for sharing your expertise in {expert_info.expertise_area}. "
        "I'm here to interview you for training Mr. French, our conversational AI family assistant. "
        "Mr. French helps families manage children's routines, tasks, and behavior. "
        "To start, could you describe your area of expertise and how it could help Mr. French better support families?"
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

@app.post("/upload-doc")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    expert_name: str = Form(...)
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
        "do you have", "check the document", "look at my", "review my doc"
    ]
    
    is_doc_query = any(pattern in chat_message.message.lower() for pattern in doc_query_patterns)
    
    if is_doc_query:
        print(f"🔍 DOC QUERY DETECTED: '{chat_message.message}' in session {session_id}")
        # Get session documents from ChromaDB
        try:
            doc_processor = get_document_processor(client)
            session_docs_info = doc_processor.get_session_documents(session_id)
            print(f"📄 SESSION DOCS: {session_docs_info}")
            
            if session_docs_info['documents']:
                doc_titles = [doc['title'] for doc in session_docs_info['documents']]
                print(f"✅ FOUND DOCUMENTS: {doc_titles}")
                ai_message = f"Yes, I can see your uploaded document(s): {', '.join(doc_titles)}. I have access to their content and can reference them in our discussion. Would you like me to incorporate insights from these documents into the next question, or would you prefer to continue with the standard interview questions?"
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
    if session_data['current_question_index'] == 0 and (msg_type in ["greeting", "smalltalk", "who_are_you", "who_is_mrfrench", "who_is_timmy", "about_interview"]):
        if msg_type == "greeting":
            # Start with Mr. French introduction and ask about current implementation knowledge
            ai_message = (
                "Hi! I'm here to interview you about improving Mr. French, our conversational AI family assistant. "
                "Mr. French helps families manage children's routines, tasks, and behavior through connected chats between parents and children. "
                "This interview is to extract expert rules to make Mr. French better at supporting families. "
                "Would you like to know about our current implementation details and how you can help?"
            )
        elif msg_type == "smalltalk":
            ai_message = (
                "I'm good, thanks for asking! I'm here to interview you about improving Mr. French, our conversational AI family assistant. "
                "Mr. French helps families manage children's routines, tasks, and behavior through connected chats between parents and children. "
                "This interview is to extract expert rules to make Mr. French better at supporting families. "
                "Would you like to know about our current implementation details and how you can help?"
            )
        elif msg_type == "who_are_you":
            ai_message = (
                "I'm an AI interviewer to capture your expertise for improving Mr. French, our conversational AI family assistant. "
                "Mr. French helps families manage children's routines, tasks, and behavior through connected chats between parents and children. "
                "This interview is to extract expert rules to make Mr. French better at supporting families. "
                "Would you like to know about our current implementation details and how you can help?"
            )
        elif msg_type == "who_is_mrfrench":
            ai_message = (
                "Mr. French is a conversational AI family assistant. It turns everyday parent instructions into structured tasks with due dates, reminders, and rewards. "
                "There are three connected chats: Parent ↔ Mr. French (to create/manage tasks and get progress), Timmy (child) ↔ Mr. French (to receive reminders, encouragement, and complete tasks), and Parent ↔ Timmy (to capture real instructions). "
                "It keeps context over time and uses a simple Red/Green/Blue 'Timmy Zone' to guide tone and responses. Would you like to know more about the current implementation details?"
            )
        elif msg_type == "who_is_timmy":
            ai_message = (
                "Timmy is the child persona that Mr. French supports. Timmy receives friendly reminders, step-by-step help, encouragement, and simple rewards for completing tasks like homework, chores, and bedtime routines. "
                "Mr. French adjusts its tone using the Red/Green/Blue 'Timmy Zone' (e.g., calm guidance if Timmy is frustrated). Would you like to know more about the current implementation details?"
            )
        else:  # about_interview
            ai_message = (
                "In this interview, we'll discuss your expertise, guiding principles, outcomes you aim for, how you measure progress, methods you use, challenges you face, and more related to your area of expertise. "
                "We capture your expertise so Mr. French behaves like an expert in real family conversations. Would you like to know about our current implementation details first?"
            )
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
        # Create context about previously asked questions to prevent repetition
        previous_questions = []
        for msg in conversation_history:
            if msg['role'] == 'assistant' and '?' in msg['content']:
                # Extract the question part
                question_part = msg['content'].split('?')[0] + '?'
                previous_questions.append(question_part)
        
        # Add context to system prompt about what's been asked
        enhanced_system_prompt = SYSTEM_PROMPT
        if previous_questions:
            questions_context = "\n\nPREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):\n" + "\n".join([f"- {q}" for q in previous_questions[-5:]])  # Last 5 questions
            enhanced_system_prompt += questions_context
        
        # Search for relevant document context using ChromaDB
        doc_context = ""
        try:
            doc_processor = get_document_processor(client)
            similar_chunks = await doc_processor.search_similar_chunks(
                query=chat_message.message,
                session_id=session_id,
                limit=5
            )
            
            if similar_chunks:
                doc_context_parts = []
                print(f"🔍 SIMILARITY SCORES:")
                for i, chunk in enumerate(similar_chunks):
                    sim_score = chunk.get('similarity', 'None')
                    print(f"  Chunk {i+1}: {sim_score:.3f} from {chunk.get('document_title', 'Unknown')}")
                    
                    # Lower threshold to 0.3 for better recall
                    if chunk.get('similarity', 0) >= 0.3:
                        source_info = f"From {chunk['document_title']}"
                        if chunk.get('page_number'):
                            source_info += f" (Page {chunk['page_number']})"
                        elif chunk.get('slide_number'):
                            source_info += f" (Slide {chunk['slide_number']})"
                        
                        doc_context_parts.append(f"{source_info}:\n{chunk['content']}")
                
                if doc_context_parts:
                    doc_titles = list(set(chunk['document_title'] for chunk in similar_chunks if chunk.get('similarity', 0) >= 0.3))
                    doc_header = f"**UPLOADED DOCUMENTS:** {', '.join(doc_titles)}\n\n**DOCUMENT CONTEXT:**\n"
                    doc_context = "\n\n" + doc_header + "\n\n---\n\n".join(doc_context_parts)
                    logger.info(f"📚 Added document context: {len(doc_context_parts)} relevant chunks from {len(doc_titles)} documents")
                    print(f"🔍 FOUND DOCS: {len(doc_context_parts)} chunks from {doc_titles}")
                else:
                    print(f"🔍 NO DOCS FOUND - all similarities below 0.3 threshold for query: '{chat_message.message[:50]}...' in session {session_id}")
        except Exception as e:
            # ChromaDB might not be available or connected - this is optional functionality
            logger.warning(f"⚠️ Document context search failed: {e}")
            pass
        
        # Enhanced system prompt with document context
        final_system_prompt = enhanced_system_prompt
        if doc_context:
            final_system_prompt += f"\n\n{doc_context}\n\nYou can reference and discuss this document content naturally during the interview. If the expert asks about the document or mentions something related to it, feel free to engage with that context."
            print(f"📋 DOCUMENT CONTEXT ADDED TO PROMPT: {len(doc_context)} characters")
        else:
            print("📋 NO DOCUMENT CONTEXT - AI should not reference any documents")
        
        # Get AI's next question/response
        messages = [{"role": "system", "content": final_system_prompt}] + conversation_history
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster for question generation
            messages=messages,
            max_tokens=1200,  # Increased for better comprehension and response quality
            temperature=0.8,  # Slightly higher for more natural conversation
            timeout=30.0  # Increased timeout for longer responses
        )
        
        ai_message = sanitize_question(response.choices[0].message.content)
        ai_message = clean_response(ai_message)
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
            final_note = "Thank you for sharing your valuable expertise! The interview is now complete. Your insights will help improve Mr. French's ability to support families."
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
async def submit_interview(session_id: str):
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
        
        # Extract behavioral rules for Mr. French
        extraction_prompt = f"""You are analyzing an interview with a behavioral expert to extract specific rules for Mr. French AI.

**ABOUT MR. FRENCH:**
Mr. French is a conversational AI family assistant that helps manage children's routines, tasks, and behavior. It has three chat modes:
1. Parent ↔ Mr. French (task management, progress reports)
2. Child ↔ Mr. French (reminders, encouragement, task completion)  
3. Parent ↔ Child (capturing family instructions)

Mr. French uses a zone system: Red (frustrated/stressed), Green (normal), Blue (tired/low energy).

**EXTRACTION RULES:**
- ONLY extract rules if the expert provided specific behavioral advice or recommendations
- If the expert gave no meaningful advice or just answered basic questions, return "NONE"
- Ignore general interview questions and AI interviewer responses
- Extract actionable rules Mr. French can implement
- Each rule should start with "Mr. French should..."
- Focus on child behavior management, communication strategies, and family dynamics
- Ignore meta-conversation about the interview itself
- DO NOT generate rules from your own knowledge - only from what the expert explicitly stated

**EXAMPLES:**
- "Mr. French should use calm, reassuring language when a child is in the red zone"
- "Mr. French should break complex tasks into 2-3 smaller steps for better completion"
- "Mr. French should offer specific praise for effort rather than general compliments"

**CONVERSATION:**
{conversation_text}

**EXTRACTED RULES:"""
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract simple task statements from behavioral expert interviews. Return only clear, actionable statements."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse task statements
        tasks_text = response.choices[0].message.content.strip()
        if tasks_text.upper() == 'NONE' or not tasks_text:
            task_statements = []
        else:
            task_statements = [task.strip() for task in tasks_text.split('\n') if task.strip() and not task.strip().upper().startswith('NONE')]
        
        print(f"🤖 TASK EXTRACTION: Starting for session {session_id}")
        print(f"📝 CONVERSATION LENGTH: {len(conversation_text)} characters")
        print("🧠 GPT EXTRACTION: Calling GPT-4o-mini for task extraction")
        print(f"💬 CONVERSATION PREVIEW: {conversation_text[:200]}...")
        
        # Tasks will be stored in database only
        
        print(f"✅ GPT EXTRACTION SUCCESS: {len(task_statements)} tasks extracted")
        if len(task_statements) > 0:
            print(f"📝 SAMPLE TASK: {task_statements[0][:100]}...")
        else:
            print("⚠️ NO TASKS EXTRACTED - Check conversation content")
        
        # Save each task to Supabase
        if supabase_client.connected:
            print(f"💾 SUPABASE SAVE: Saving {len(task_statements)} tasks to database")
            for i, task in enumerate(task_statements, 1):
                print(f"💾 SAVING TASK {i}/{len(task_statements)}: {task[:50]}...")
                rule_id = await supabase_client.save_interview_rule(
                    session_id=session_id,
                    expert_name=session_data['expert_name'],
                    expertise_area=session_data['expertise_area'],
                    rule_text=task
                )
                print(f"💾 RULE SAVED: ID {rule_id} for session {session_id}")
        else:
            print("⚠️ SUPABASE UNAVAILABLE: Tasks saved to memory only")
        
        logger.info(f"✅ Extracted {len(task_statements)} tasks for session {session_id}")
        
        print(f"🎉 INTERVIEW COMPLETE: Session {session_id} finished with {len(task_statements)} tasks")
        return {
            "message": f"Thank you for the interview. {len(task_statements)} tasks have been extracted.",
            "status": "completed",
            "tasks_extracted": len(task_statements),
            "tasks": task_statements
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing interview submission: {e}")
        return {
            "message": "Interview saved, but there was an error processing tasks.",
            "status": "error"
        }

# Removed auto_submit_interview - now using submit_interview directly

# Removed background processing - now using direct task extraction in submit_interview

@app.get("/sessions")
async def get_sessions():
    """Get all interview sessions from memory"""
    try:
        return {
            "sessions": [
                {
                    "session_id": session_id,
                    "created_at": session.created_at.isoformat(),
                    "questions_asked": session.current_question_index,
                    "is_complete": session.is_complete,
                    "status": "in_progress" if not session.is_complete else "completed",
                    "completed_at": None,
                    "source": "memory"
                }
                for session_id, session in interview_sessions.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")

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

# Admin Authentication Models
class AdminLoginRequest(BaseModel):
    email: str
    password: str

class TaskActionRequest(BaseModel):
    task_id: str
    action: str  # "approve" or "reject"

# Security dependency
security = HTTPBearer()

def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin token"""
    print(f"🔐 AUTH CHECK: Received token: {credentials.credentials[:20]}...")
    user = admin_auth.verify_token(credentials.credentials)
    if not user:
        print("❌ AUTH FAILED: Invalid or expired token")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    print(f"✅ AUTH SUCCESS: User {user['email']}")
    return user

# Admin Routes
@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Serve admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/admin/create-user")
async def create_admin_user():
    """Manually create admin user - for deployment troubleshooting"""
    try:
        if not supabase_client.connected:
            supabase_client.connect()
        
        if not supabase_client.connection:
            return {"success": False, "error": "Database not connected"}
        
        import bcrypt
        password_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        with supabase_client.connection.cursor() as cur:
            cur.execute("""
                INSERT INTO admin_users (email, password_hash, name)
                VALUES (%s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                name = EXCLUDED.name
            """, ("admin@coachai.com", password_hash, "Admin User"))
            supabase_client.connection.commit()
        
        return {"success": True, "message": "Admin user created: admin@coachai.com / admin123"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/admin/login")
async def admin_login(login_request: AdminLoginRequest):
    """Admin login endpoint"""
    logger.info(f"Admin login attempt for: {login_request.email}")
    
    auth_result = admin_auth.authenticate(login_request.email, login_request.password)
    if not auth_result:
        logger.warning(f"Failed login attempt for: {login_request.email}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    logger.info(f"Successful login for: {login_request.email}")
    return auth_result

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

@app.get("/admin/tasks")
async def get_admin_tasks(current_admin = Depends(get_current_admin)):
    """Get all tasks for admin panel"""
    try:
        if not supabase_client.connected:
            return {"tasks": []}
        
        db_rules = await supabase_client.get_all_rules()
        tasks = []
        
        for rule in db_rules:
            completed = rule.get('completed', False)
            status = "completed" if completed else "pending"
            
            raw_rule_text = rule.get('rule_text')
            rule_text = str(raw_rule_text) if raw_rule_text else "No rule text available"
            
            tasks.append({
                "id": str(rule['id']),
                "session_id": str(rule['session_id']),
                "expert_name": str(rule.get('expert_name', 'Expert User')),
                "task_text": rule_text,
                "category": str(rule.get('expertise_area', 'General')),
                "priority": "medium",
                "status": status
            })
        
        tasks.sort(key=lambda x: (x['status'] == 'completed', x['id']))
        return {"tasks": tasks}
        
    except Exception as e:
        logger.error(f"Error getting admin tasks: {e}")
        return {"tasks": []}

@app.post("/admin/tasks/action")
async def admin_task_action(action_request: TaskActionRequest, current_admin = Depends(get_current_admin)):
    """Approve or reject a task"""
    try:
        task_id = action_request.task_id
        action = action_request.action
        
        if action == "approve":
            # Get rule from Supabase
            if supabase_client.connected:
                rules = await supabase_client.get_all_rules()
                rule = next((r for r in rules if str(r['id']) == task_id), None)
                
                if rule:
                    task_data = {
                        "task_text": rule['rule_text'],
                        "expert_name": rule.get('expert_name', 'Expert User'),
                        "category": rule.get('expertise_area', 'General'),
                        "priority": "medium"
                    }
                    
                    # Send to Jira
                    jira_issue_key = jira_client.create_task(
                        summary_text=f"AI Coach Rule: {task_data['task_text'][:100]}...",
                        description=f"Expert: {task_data['expert_name']}\nCategory: {task_data['category']}\nRule: {task_data['task_text']}"
                    )
                    
                    # Mark rule as completed in Supabase
                    async with supabase_client.pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE interview_rules SET completed = TRUE WHERE id = $1",
                            rule['id']
                        )
                    
                    return {
                        "success": True,
                        "message": "Rule approved and sent to Jira",
                        "jira_issue_key": jira_issue_key
                    }
                else:
                    raise HTTPException(status_code=404, detail="Rule not found")
            else:
                raise HTTPException(status_code=503, detail="Database not available")
            
        elif action == "reject":
            # For now, just return success (no status update needed in simplified schema)
            return {
                "success": True,
                "message": "Rule rejected"
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        logger.error(f"Error processing task action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    print("🚀 Starting AI Coach Interview System...")
    port = int(os.getenv("PORT", 8003))
    print(f"🌐 Binding to: 0.0.0.0:{port}")
    print(f"🔗 Interview Interface: http://localhost:{port}")
    print(f"🔗 Admin Panel: http://localhost:{port}/admin")
    print("🔑 Admin Credentials: admin@coachai.com / admin123")
    print("🔧 Manual admin creation: POST /admin/create-user")
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
