from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List, Dict, Optional
import asyncio
# MongoDB removed - using Supabase + ChromaDB
from supabase_client import supabase_client
try:
    from chromadb_client import chromadb_client
except (ImportError, AttributeError) as e:
    chromadb_client = None
    print(f"ChromaDB not available: {e}")
    print("Continuing without ChromaDB - conversations won't have embeddings")
from admin_auth import admin_auth
from jira_client import jira_client
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Coach Interview Model", version="1.0.0")

# Set up OpenAI client
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=15.0)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")


templates = Jinja2Templates(directory="templates")


interview_sessions = {}
session_counter = 0  # Simple counter for session IDs - will be initialized from DB

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
    
    # ChromaDB is optional and handled in client init
    if chromadb_client and chromadb_client.client:
        logger.info("✅ ChromaDB available")
    else:
        logger.info("ℹ️ ChromaDB disabled - continuing without embeddings")
    
    # Initialize session counter
    session_counter = 0
    logger.info("Session counter initialized to 0")

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
- **RESPONSE STYLE**: Keep responses brief and neutral. Avoid praise or evaluative language (e.g., "great", "excellent", "love that", "that's exactly right"). After receiving an answer, give a short neutral acknowledgment (e.g., "Noted." or "Understood.") and then ask the next question directly. Don't elaborate on their previous response.
- **CRITICAL**: NEVER explain, analyze, judge, compliment, congratulate, or praise their previous answer. Just acknowledge briefly and ask the next question. Keep responses under 2 sentences.is 
- Dont repeat questions if once answered in the same session."""

@app.get("/", response_class=HTMLResponse)
async def get_interview_page(request: Request):
    """Serve the main interview page"""
    logger.info("🏠 Main page requested")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    logger.info("🏥 Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": supabase_client.connected if supabase_client else False,
        "sessions_in_memory": len(interview_sessions)
    }

@app.post("/start_interview")
async def start_interview():
    """Start a new interview session"""
    logger.info("🚀 START_INTERVIEW endpoint called!")
    global session_counter
    session_counter += 1
    session_id = str(session_counter)  # Simple: 1, 2, 3, etc.
    logger.info(f"📝 Creating new session: {session_id}")
    session = InterviewSession(session_id)
    interview_sessions[session_id] = session
    
    # Interview stored in memory and ChromaDB
    
    # Start with Mr. French introduction and first question
    ai_message = (
        "Hi! I'm here to interview you for training Mr. French, our conversational AI family assistant. "
        "Mr. French helps families manage children's routines, tasks, and behavior through connected chats between parents and children. "
        "This interview is to extract expert rules to train Mr. French for supporting families. "
        "Could you please tell your name and how you could help train Mr. French to better support families?"
    )
    session.conversation_history.append({"role": "assistant", "content": ai_message})
    
    return {
        "session_id": session_id,
        "message": ai_message,
        "question_number": 0
    }

@app.post("/chat")
async def chat_with_interviewer(chat_message: ChatMessage):
    """Continue the interview conversation"""
    logger.info(f"💬 CHAT endpoint called for session: {chat_message.session_id}")
    logger.info(f"💬 User message: {chat_message.message[:100]}...")
    session_id = chat_message.session_id
    
    if not session_id:
        raise HTTPException(status_code=404, detail="Interview session not found")

    # Session stored in memory only

    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session = interview_sessions[session_id]
    
    if session.is_complete:
        return {
            "message": "Interview has been completed. Please generate the rules using the /generate_rules endpoint.",
            "is_complete": True
        }
    
    # Add user's response to conversation history
    session.conversation_history.append({"role": "user", "content": chat_message.message})
    
    # ChromaDB storage removed
    
    logger.info(f"💬 Session {session_id} - User message added. Total messages: {len(session.conversation_history)}")

    # Guard: handle small-talk or project Qs without advancing the question index
    msg_type = is_smalltalk_or_project(chat_message.message)
    if session.current_question_index == 0 and (msg_type in ["greeting", "smalltalk", "who_are_you", "who_is_mrfrench", "who_is_timmy", "about_interview"]):
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
        session.conversation_history.append({"role": "assistant", "content": ai_message})
        return {
            "message": sanitize_question(ai_message),
            "question_number": 1,
            "is_complete": False,
            "auto_submitted": False,
            "final_note": None
        }
    
    try:
        # Get AI's next question/response
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.conversation_history
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster for question generation
            messages=messages,
            max_tokens=1200,  # Increased for better comprehension and response quality
            temperature=0.8,  # Slightly higher for more natural conversation
            timeout=30.0  # Increased timeout for longer responses
        )
        
        ai_message = sanitize_question(response.choices[0].message.content)
        ai_message = clean_response(ai_message)
        session.conversation_history.append({"role": "assistant", "content": ai_message})
        session.current_question_index += 1
        
        # ChromaDB storage removed
        
        # Check if interview should be completed (basic heuristic)
        auto_submitted = False
        final_note = None
        if session.current_question_index >= 23 or "conclude" in ai_message.lower() or "summary" in ai_message.lower():
            session.is_complete = True
            # Add a closing message
            final_note = "Thank you. The interview is complete. You can now click Submit to save your responses."
            session.conversation_history.append({"role": "assistant", "content": final_note})
        
        return {
            "message": ai_message,
            "question_number": session.current_question_index + 1,
            "is_complete": session.is_complete,
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
    
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session = interview_sessions[session_id]
    session.is_complete = True
    
    # Extract tasks from conversation
    try:
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in session.conversation_history
        ])
        
        # Extract simple task statements
        extraction_prompt = f"""Extract simple, actionable task statements from this interview conversation with a behavioral expert.

Each task should be a clear statement like:
- "Mr. French should use calm language when child is frustrated"
- "Mr. French should break tasks into smaller steps when child feels overwhelmed"
- "Mr. French should suggest breaks when child shows signs of stress"

Return ONLY the task statements, one per line. No explanations or formatting.

CONVERSATION:
{conversation_text}

TASK STATEMENTS:"""
        
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
        task_statements = [task.strip() for task in tasks_text.split('\n') if task.strip()]
        
        print(f"🤖 TASK EXTRACTION: Starting for session {session_id}")
        print(f"📝 CONVERSATION LENGTH: {len(conversation_text)} characters")
        print("🧠 GPT EXTRACTION: Calling GPT-4o-mini for task extraction")
        
        # Store tasks in memory for admin panel
        if not hasattr(session, 'extracted_tasks'):
            session.extracted_tasks = []
        session.extracted_tasks.extend(task_statements)
        
        print(f"✅ GPT EXTRACTION SUCCESS: {len(task_statements)} tasks extracted")
        
        # Save each task to Supabase
        if supabase_client.connected:
            print(f"💾 SUPABASE SAVE: Saving {len(task_statements)} tasks to database")
            for i, task in enumerate(task_statements, 1):
                print(f"💾 SAVING TASK {i}/{len(task_statements)}: {task[:50]}...")
                await supabase_client.save_interview_rule(
                    session_id=session_id,
                    expert_name="Expert User",
                    expertise_area="General",
                    rule_text=task
                )
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
    
    if session_id in interview_sessions:
        session = interview_sessions[session_id]
        logger.info(f"📝 Session {session_id} found in memory - {len(session.conversation_history)} messages")
        return {
            "session_id": session_id,
            "conversation": session.conversation_history,
            "is_complete": session.is_complete,
            "source": "memory"
        }
    
    raise HTTPException(status_code=404, detail="Session not found")

# New database management endpoints
@app.get("/database/status")
async def get_database_status():
    """Get database connection status"""
    try:
        logger.info(f"Checking database status - supabase connected: {supabase_client.connected if supabase_client else False}")
        
        if supabase_client and supabase_client.connected:
            return {
                "connected": True,
                "database_type": "Supabase PostgreSQL",
                "message": "Database connection is healthy"
            }
        else:
            return {
                "connected": False,
                "message": "Database not connected - check startup logs"
            }
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {
            "connected": False,
            "error": str(e),
            "message": "Database connection failed"
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
    user = admin_auth.verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

# Admin Routes
@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Serve admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

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
    """Serve admin dashboard"""
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/admin/conversations")
async def get_admin_conversations(current_admin = Depends(get_current_admin)):
    """Get all conversations for admin panel"""
    try:
        conversations = []
        
        # Get from in-memory sessions only
        for session_id, session in interview_sessions.items():
            conversations.append({
                "session_id": session_id,
                "expert_name": "Expert User",
                "expertise_area": "General",
                "completed": session.is_complete,
                "messages": session.conversation_history
            })
        
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Error getting admin conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/tasks")
async def get_admin_tasks(current_admin = Depends(get_current_admin)):
    """Get all tasks for admin panel"""
    try:
        tasks = []
        task_id = 1
        
        # Get tasks from completed interview sessions
        for session_id, session in interview_sessions.items():
            if hasattr(session, 'extracted_tasks') and session.extracted_tasks:
                for task_text in session.extracted_tasks:
                    tasks.append({
                        "id": str(task_id),
                        "session_id": session_id,
                        "expert_name": "Expert User",
                        "task_text": task_text,
                        "category": "General",
                        "priority": "medium",
                        "status": "pending"
                    })
                    task_id += 1
        
        # Add sample task if no real tasks exist
        if not tasks:
            tasks = [{
                "id": "1",
                "session_id": "sample",
                "expert_name": "Sample Expert",
                "task_text": "Mr. French should use calm and reassuring language when a child expresses frustration",
                "category": "Communication",
                "priority": "high",
                "status": "pending"
            }]
        
        return {"tasks": tasks}
        
    except Exception as e:
        logger.error(f"Error getting admin tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/admin/approve/{session_id}")
async def approve_task(session_id: str, current_admin = Depends(get_current_admin)):
    """Approve task and send to Jira"""
    try:
        print(f"🎯 APPROVE TASK: Starting approval for session {session_id}")
        
        if supabase_client.connected:
            rule = supabase_client.get_rule_by_session_id(session_id)
            if rule:
                print(f"📋 RULE FOUND: {rule}")
                # Create Jira task
                jira_key = jira_client.create_task(
                    summary_text=f"AI Coach Rule: {rule[4][:100]}...",
                    description=f"Expert: {rule[2]}\nArea: {rule[3]}\nRule: {rule[4]}"
                )
                
                if jira_key:
                    print(f"✅ JIRA SUCCESS: Created task {jira_key}")
                    # Update rule status to completed
                    supabase_client.update_rule_status(session_id, True)
                    return {"success": True, "jira_key": jira_key}
                else:
                    print("❌ JIRA FAILED: Could not create task")
                    return {"success": False, "error": "Failed to create Jira task"}
            else:
                print(f"❌ NO RULE FOUND for session {session_id}")
                return {"success": False, "error": "Rule not found"}
        else:
            print("❌ DATABASE NOT CONNECTED")
            return {"success": False, "error": "Database not connected"}
        
    except Exception as e:
        print(f"❌ APPROVE ERROR: {e}")
        logger.error(f"Error approving task: {e}")
        return {"success": False, "error": str(e)}

@app.post("/admin/disapprove/{session_id}")
async def disapprove_task(session_id: str, current_admin = Depends(get_current_admin)):
    """Disapprove task without sending to Jira"""
    try:
        print(f"❌ DISAPPROVE TASK: Starting for session {session_id}")
        
        if supabase_client.connected:
            success = supabase_client.update_rule_status(session_id, True)
            print(f"✅ DISAPPROVE SUCCESS: {success}")
            return {"success": success}
        else:
            print("❌ DATABASE NOT CONNECTED")
            return {"success": False, "error": "Database not connected"}
        
    except Exception as e:
        print(f"❌ DISAPPROVE ERROR: {e}")
        logger.error(f"Error disapproving task: {e}")
        return {"success": False, "error": str(e)}

@app.get("/admin/stats")
async def get_admin_stats(current_admin = Depends(get_current_admin)):
    """Get dashboard statistics"""
    try:
        # Get from Supabase
        if supabase_client.connected:
            stats = await supabase_client.get_stats()
        else:
            # Fallback mock data
            stats = {
                "total_interviews": 5,
                "pending_tasks": 3,
                "approved_tasks": 8,
                "rejected_tasks": 2
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Coach Interview System...")
    print("🌐 Binding to: 0.0.0.0:8001")
    print("🔗 Interview Interface: http://localhost:8001")
    print("🔗 Admin Panel: http://localhost:8001/admin")
    print("🔑 Admin Credentials: admin@aicoach.com / admin123")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
