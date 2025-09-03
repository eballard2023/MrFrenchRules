from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import openai
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List, Dict, Optional
import asyncio
from database import db_manager, InterviewModel, RulesCollectionModel
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

def is_affirmative(message: str) -> bool:
    m = (message or "").strip().lower()
    affirm = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "ready", "let's start", "lets start",
        "begin", "start", "go ahead", "yup"
    ]
    return any(_matches_phrase(m, a) for a in affirm)

FIRST_QUESTION_TEXT = "To start, could you describe your area of expertise and how you usually apply it?"

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    global session_counter
    logger.info("Starting up the application...")
    success = await db_manager.connect()
    if not success:
        logger.error("Failed to connect to MongoDB. Application will continue with in-memory storage.")
    else:
        logger.info("Successfully connected to MongoDB")
        # Initialize session counter from database
        try:
            interviews = await db_manager.get_all_interviews()
            if interviews:
                # Find the highest session ID number
                max_session_id = 0
                for interview in interviews:
                    try:
                        session_num = int(interview.session_id)
                        max_session_id = max(max_session_id, session_num)
                    except ValueError:
                        # Skip non-numeric session IDs (old format)
                        continue
                session_counter = max_session_id
                logger.info(f"Initialized session counter to {session_counter} from database")
            else:
                logger.info("No existing interviews found, starting session counter at 0")
        except Exception as e:
            logger.error(f"Failed to initialize session counter from database: {e}")
            session_counter = 0

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    logger.info("Shutting down the application...")
    await db_manager.disconnect()

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
- After small-talk or project questions (who are you / Mr. French / Timmy), answer briefly and ask if they’re ready to continue the interview
- For unrelated trivia, decline and return to the interview
- **CRITICAL**: On greeting ("hello", "hi"), start with Mr. French introduction, then ask about current implementation knowledge
- **NEVER** respond with "I'm here to help" or similar general assistant language
- **RESPONSE STYLE**: Keep responses brief. After receiving an answer, give a short acknowledgment (like "Interesting" or "That's great") and then ask the next question directly. Don't elaborate on their previous response.
- **CRITICAL**: NEVER explain, analyze, or comment on their previous answer. Just acknowledge briefly and ask the next question. Keep responses under 2 sentences."""

@app.get("/", response_class=HTMLResponse)
async def get_interview_page(request: Request):
    """Serve the main interview page"""
    return templates.TemplateResponse("interview.html", {"request": request})

@app.post("/start_interview")
async def start_interview():
    """Start a new interview session"""
    global session_counter
    session_counter += 1
    session_id = str(session_counter)  # Simple: 1, 2, 3, etc.
    session = InterviewSession(session_id)
    interview_sessions[session_id] = session
    
    # Save to database 
    try:
        if db_manager.database is not None:
            interview_model = InterviewModel(
                session_id=session_id,
                started_at=datetime.now(timezone.utc),
                conversation_history=[],
                questions_asked=0,
                is_complete=False,
                status="in_progress"
            )
            await db_manager.save_interview(interview_model)
            logger.info(f"Interview session {session_id} saved to database")
    except Exception as e:
        logger.error(f"Failed to save interview to database: {e}")
        # Continue with in-memory storage
    
    # Start with Mr. French introduction and ask about implementation details
    ai_message = (
        "Hi! I'm here to interview you about improving Mr. French, our conversational AI family assistant. "
        "Mr. French helps families manage children's routines, tasks, and behavior through connected chats between parents and children. "
        "This interview is to extract expert rules to make Mr. French better at supporting families. "
        "Would you like to know about Mr.French or how you can help?"
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
    session_id = chat_message.session_id
    
    if not session_id:
        raise HTTPException(status_code=404, detail="Interview session not found")

    # Restore from DB if session not present in memory (e.g., after reload)
    if session_id not in interview_sessions:
        try:
            if db_manager.database is not None:
                interview = await db_manager.get_interview(session_id)
                if interview:
                    restored = InterviewSession(session_id)
                    restored.conversation_history = interview.conversation_history or []
                    restored.current_question_index = interview.questions_asked or 0
                    restored.is_complete = interview.is_complete
                    interview_sessions[session_id] = restored
        except Exception:
            pass

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
            max_tokens=200,  # Shorter responses for speed
            temperature=0.7,
            timeout=15.0  # Faster timeout for questions
        )
        
        ai_message = sanitize_question(response.choices[0].message.content)
        session.conversation_history.append({"role": "assistant", "content": ai_message})
        # Only advance beyond the first question after readiness/answer
        if session.current_question_index == 0:
            if is_affirmative(chat_message.message):
                # User wants to know about implementation, so explain CURRENT IMPLEMENTATION details from system prompt
                ai_message = (
                    "Great! Here's a brief overview of our current implementation:\n\n"
                    "Mr. French is a conversational AI that helps families manage children's routines, tasks, and behavior through three connected chat experiences:\n"
                    "• Parent ↔ Mr. French (task management, progress reports, context discussions)\n"
                    "• Timmy (child) ↔ Mr. French (reminders, encouragement, task completion)\n"
                    "• Parent ↔ Timmy (capturing real family instructions into actionable tasks)\n\n"
                    "It converts everyday language into structured tasks, maintains context across conversations, and provides automated reminders.\n\n"
                    "Now let's delve into how your expertise can enhance Mr. French's capabilities. "
                    "To start, could you describe your area of expertise and how it could help Mr. French better support families?"
                )
                session.conversation_history.append({"role": "assistant", "content": ai_message})
                session.current_question_index = 1
                return {
                    "message": ai_message,
                    "question_number": 2,
                    "is_complete": False,
                    "auto_submitted": False,
                    "final_note": None
                }
            else:
                session.current_question_index = 0
        else:
            session.current_question_index += 1
        
        # Update database (persist history and counters)
        try:
            if db_manager.database is not None:
                ok = await db_manager.update_interview(session_id, {
                    "conversation_history": session.conversation_history,
                    "questions_asked": session.current_question_index
                })
                if not ok:
                    logger.warning(f"DB update returned false for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to update interview in database: {e}")
        
        # Check if interview should be completed (basic heuristic)
        auto_submitted = False
        final_note = None
        if session.current_question_index >= 23 or "conclude" in ai_message.lower() or "summary" in ai_message.lower():
            session.is_complete = True
            # Update completion status in database
            try:
                if db_manager.database is not None:
                    ok = await db_manager.update_interview(session_id, {
                        "is_complete": True,
                        "completed_at": datetime.now(timezone.utc),
                        "status": "completed"
                    })
                    if not ok:
                        logger.warning(f"DB completion update returned false for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to update completion status in database: {e}")
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

@app.post("/generate_rules/{session_id}")
async def generate_rules(session_id: str):
    """Generate structured JSON rules from the interview conversation"""
    logger.info(f"🔍 Generate rules called for session_id: {session_id}")
    logger.info(f"📝 Available sessions in memory: {list(interview_sessions.keys())}")
    
    if session_id not in interview_sessions:
        logger.info(f"⚠️ Session {session_id} not in memory, trying to restore from DB...")
        # Attempt to restore from DB for post-reload continuity
        try:
            if db_manager.database is not None:
                interview = await db_manager.get_interview(session_id)
                if interview:
                    logger.info(f"✅ Found interview in DB, restoring to memory")
                    restored = InterviewSession(session_id)
                    restored.conversation_history = interview.conversation_history or []
                    restored.current_question_index = interview.questions_asked or 0
                    restored.is_complete = interview.is_complete
                    interview_sessions[session_id] = restored
                else:
                    logger.warning(f"❌ Interview {session_id} not found in database")
        except Exception as e:
            logger.error(f"❌ Error restoring session from DB: {e}")
            pass
        if session_id not in interview_sessions:
            logger.error(f"❌ Session {session_id} not found in memory or DB")
            raise HTTPException(status_code=404, detail="Interview session not found")
    
    session = interview_sessions[session_id]
    
    # Prepare conversation for rule extraction (limit to last 10 messages to avoid timeout)
    recent_messages = session.conversation_history[-10:] if len(session.conversation_history) > 10 else session.conversation_history
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in recent_messages
    ])
    
    # Log the conversation for debugging
    logger.info(f"Extracting rules from conversation for session {session_id}:")
    logger.info(f"Conversation length: {len(session.conversation_history)} messages")
    logger.info(f"Conversation content: {conversation_text[:500]}...")  # Log first 500 chars
    
    extraction_prompt = f"""Based on the following interview conversation with a subject matter expert (SME), extract actionable CHILD/FAMILY BEHAVIOR rules and convert them into a structured JSON format for Mr. French AI.
    STRICTLY EXCLUDE meta-interview or project chatter (e.g., who is Mr. French, who is Timmy, introductions, interview logistics, character definitions, facilitator phrases).
    Only extract rules that Mr. French can APPLY to help with children's behavior, routines, motivation, communication, de-escalation, rewards/consequences, or parent guidance.
    If the conversation does NOT contain actionable behavior guidance, return an empty JSON array [].

Extract ALL behavioral rules, best practices, and guidance that can be inferred from the expert's responses. Look for:
- How they handle specific situations
- What they recommend in different contexts
- Their approach to problems
- Their communication style
- Their methods and techniques

Each rule should follow this EXACT format:
{{
    "if": {{
        "event": "specific situation or trigger condition",
        "context": "additional context if needed",
        "user_type": "target audience (e.g., 'child', 'student', 'client', 'general')"
    }},
    "then": {{
        "action": "specific action the AI should take",
        "response": "exact words or approach the AI should use",
        "duration": "time duration if applicable (e.g., '5_minutes', 'until_calm')",
        "tone": "how the AI should sound (e.g., 'calm', 'encouraging', 'firm', 'supportive')"
    }},
    "priority": "high/medium/low",
    "category": "rule category (e.g., 'crisis_management', 'motivation', 'discipline', 'communication')"
}}

CONVERSATION TO ANALYZE:
{conversation_text}

Extract all applicable behavior rules and guidance that meet the criteria above. If none exist, return []."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured behavioral rules from conversational data. Always return valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=1400,
            temperature=0.3,
            timeout=30.0  # Increase timeout to 30 seconds
        )
        
        rules_content = response.choices[0].message.content
        
        # Try to parse as JSON to validate
        try:
            rules_json = json.loads(rules_content)
            session.extracted_rules = rules_json
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', rules_content, re.DOTALL)
            if json_match:
                rules_json = json.loads(json_match.group())
                session.extracted_rules = rules_json
            else:
                raise ValueError("Could not extract valid JSON from response")

        # Filter to keep only behavior-applicable rules
        if isinstance(rules_json, dict):
            rules_json = [rules_json]
        if isinstance(rules_json, list):
            rules_json = [r for r in rules_json if is_behavior_rule(r)]
        
        # Save rules to file
        filename = f"extracted_rules_{session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rules_json, f, indent=2, ensure_ascii=False)
        
        # Save to database
        try:
            if db_manager.database is not None:
                logger.info(f"Starting to save rules for session {session_id} to database")
                # Save rules collection
                rules_collection_model = RulesCollectionModel(
                    interview_session_id=session_id,
                    rules=rules_json,
                    total_rules=len(rules_json),
                    extracted_at=datetime.now(timezone.utc),
                    filename=filename
                )
                logger.info(f"Saving rules collection with {len(rules_json)} rules")
                collection_id = await db_manager.save_rules_collection(rules_collection_model)
                
                # Save individual rules
                logger.info(f"Saving individual rules to database")
                rule_ids = await db_manager.save_individual_rules(session_id, rules_json)
                
                logger.info(f"✅ Rules saved to database - Collection ID: {collection_id}, Rule IDs: {len(rule_ids)}")
                
                return {
                    "message": "Rules successfully extracted and saved to database",
                    "filename": filename,
                    "rules_count": len(rules_json),
                    "rules": rules_json,
                    "database_saved": True,
                    "collection_id": collection_id
                }
            else:
                logger.warning("Database not connected - skipping database save")
        except Exception as e:
            logger.error(f"❌ Failed to save rules to database: {e}")
            import traceback
            logger.error(f"Full error trace: {traceback.format_exc()}")
            
        return {
            "message": "Rules successfully extracted and saved to file",
            "filename": filename,
            "rules_count": len(rules_json),
            "rules": rules_json,
            "database_saved": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating rules: {str(e)}")

@app.post("/submit_interview/{session_id}")
async def submit_interview(session_id: str):
    """Finalize interview: mark as complete immediately, process rules in background"""
    logger.info(f"🔍 Submit interview called for session_id: {session_id}")
    
    # Quick validation and session restoration
    if session_id not in interview_sessions:
        try:
            if db_manager.database is not None:
                interview = await db_manager.get_interview(session_id)
                if interview:
                    restored = InterviewSession(session_id)
                    restored.conversation_history = interview.conversation_history or []
                    restored.current_question_index = interview.questions_asked or 0
                    restored.is_complete = interview.is_complete
                    interview_sessions[session_id] = restored
        except Exception as e:
            logger.error(f"Error restoring session from DB: {e}")
            
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session = interview_sessions[session_id]
    
    # Mark interview as completed immediately
    session.is_complete = True
    try:
        if db_manager.database is not None:
            await db_manager.update_interview(session_id, {
                "is_complete": True,
                "completed_at": datetime.now(timezone.utc),
                "status": "completed"
            })
            logger.info(f"✅ Interview {session_id} marked as completed")
    except Exception as e:
        logger.error(f"Error updating completion status: {e}")
    
    # Start background processing
    asyncio.create_task(process_rules_background(session_id))
    
    return {
        "message": "Thank you for the interview. Your responses have been saved and are being processed.",
        "status": "processing"
    }

async def process_rules_background(session_id: str):
    """Process rules in background without blocking the response"""
    try:
        logger.info(f"🔄 Starting background rule processing for session {session_id}")
        
        if session_id not in interview_sessions:
            logger.error(f"Session {session_id} not found for background processing")
            return
            
        session = interview_sessions[session_id]
        
        # Process ALL messages (not just last 10)
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in session.conversation_history
        ])
        
        logger.info(f"Processing {len(session.conversation_history)} messages for session {session_id}")
        
        # Use the best model for perfect rule analysis
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Best quality for rule extraction
            messages=[
                {"role": "system", "content": "You extract structured CHILD/FAMILY BEHAVIOR rules strictly from the given conversation. Exclude meta-interview/logistics/project-definition content. Only return valid JSON."},
                {"role": "user", "content": f"""From the following interview, extract ONLY actionable CHILD/FAMILY BEHAVIOR rules in JSON.

STRICTLY EXCLUDE: interview logistics, facilitator phrases, character definitions (Mr. French, Timmy), or general chit-chat. Include only rules Mr. French can apply for children's behavior, routines, motivation, de-escalation, rewards/consequences, communication, or parent guidance.

Each rule format:
{{\n  \"if\": {{\n    \"event\": \"specific trigger or situation\",\n    \"context\": \"additional context (only if needed)\",\n    \"user_type\": \"target audience\"\n  }},\n  \"then\": {{\n    \"action\": \"specific action to take\",\n    \"response\": \"exact words or approach\",\n    \"duration\": \"time duration (if applicable)\",\n    \"tone\": \"communication style\"\n  }},\n  \"priority\": \"high/medium/low\",\n  \"category\": \"rule category\"\n}}

CONVERSATION:\n{conversation_text}

If no applicable behavior rules exist, return []."""}
            ],
            max_tokens=3000,  # More tokens for comprehensive analysis
            temperature=0.2,  # Lower temperature for more consistent, precise output
            timeout=120.0  # 2 minutes for thorough analysis
        )
        
        rules_content = response.choices[0].message.content.strip()
        
        # Log the response for debugging
        logger.info(f"Raw AI response for session {session_id}: {rules_content[:200]}...")
        
        # Check if response is empty or invalid
        if not rules_content or rules_content.isspace():
            logger.error(f"Empty response from AI for session {session_id}")
            rules_content = "[]"
        
        # Parse and save rules with quality validation
        try:
            # Try to extract JSON if it's wrapped in other text
            import re
            json_match = re.search(r'\[.*\]', rules_content, re.DOTALL)
            if json_match:
                rules_content = json_match.group()
            
            rules_json = json.loads(rules_content)
            if not isinstance(rules_json, list):
                rules_json = [rules_json]
            
            # Validate and clean rules with strict formatting
            validated_rules = []
            for i, rule in enumerate(rules_json):
                if isinstance(rule, dict) and "if" in rule and "then" in rule:
                    # Clean and validate the "if" section
                    if_section = rule.get("if", {})
                    if isinstance(if_section, dict):
                        clean_if = {
                            "event": str(if_section.get("event", "")).strip(),
                            "context": str(if_section.get("context", "")).strip(),
                            "user_type": str(if_section.get("user_type", "general")).strip()
                        }
                        # Remove empty fields
                        clean_if = {k: v for k, v in clean_if.items() if v}
                    else:
                        clean_if = {"event": str(if_section).strip()}
                    
                    # Clean and validate the "then" section
                    then_section = rule.get("then", {})
                    if isinstance(then_section, dict):
                        clean_then = {
                            "action": str(then_section.get("action", "")).strip(),
                            "response": str(then_section.get("response", "")).strip(),
                            "duration": str(then_section.get("duration", "")).strip(),
                            "tone": str(then_section.get("tone", "")).strip()
                        }
                        # Remove empty fields
                        clean_then = {k: v for k, v in clean_then.items() if v}
                    else:
                        clean_then = {"action": str(then_section).strip()}
                    
                    # Only add rule if it has meaningful content
                    if (clean_if.get("event") or clean_if.get("context")) and (clean_then.get("action") or clean_then.get("response")):
                        validated_rule = {
                            "if": clean_if,
                            "then": clean_then,
                            "priority": str(rule.get("priority", "medium")).lower().strip(),
                            "category": str(rule.get("category", "general")).lower().strip()
                        }
                        
                        # Ensure priority and category are valid
                        if validated_rule["priority"] not in ["high", "medium", "low"]:
                            validated_rule["priority"] = "medium"
                        
                        validated_rules.append(validated_rule)
                    else:
                        logger.warning(f"Skipping rule {i} - insufficient content")
                else:
                    logger.warning(f"Skipping invalid rule {i}: {rule}")
            
            # Keep only behavior-applicable rules
            validated_rules = [r for r in validated_rules if is_behavior_rule(r)]

            # Remove duplicate rules based on event and action
            unique_rules = []
            seen_combinations = set()
            for rule in validated_rules:
                event = rule.get("if", {}).get("event", "")
                action = rule.get("then", {}).get("action", "")
                combination = f"{event}|{action}"
                
                if combination not in seen_combinations and combination != "|":
                    unique_rules.append(rule)
                    seen_combinations.add(combination)
                else:
                    logger.info(f"Removing duplicate rule: {event} -> {action}")
            
            if not unique_rules:
                logger.info(f"No applicable behavior rules found for session {session_id}. Saving empty set.")
            else:
                logger.info(f"Cleaned {len(validated_rules)} rules down to {len(unique_rules)} unique rules")
                
            # Save to file with metadata
            filename = f"extracted_rules_{session_id}.json"
            output_data = {
                "session_id": session_id,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "total_messages_processed": len(session.conversation_history),
                "rules_count": len(unique_rules),
                "rules": unique_rules
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Save to database
            if db_manager.database is not None:
                rules_collection_model = RulesCollectionModel(
                    interview_session_id=session_id,
                    rules=unique_rules,
                    total_rules=len(unique_rules),
                    extracted_at=datetime.now(timezone.utc),
                    filename=filename
                )
                await db_manager.save_rules_collection(rules_collection_model)
                await db_manager.save_individual_rules(session_id, unique_rules)
                
            logger.info(f"✅ Background processing completed for session {session_id}: {len(unique_rules)} clean, unique rules saved")
            
        except Exception as e:
            logger.error(f"❌ Error in background rule processing: {e}")
            import traceback
            logger.error(f"Full error trace: {traceback.format_exc()}")
            
    except Exception as e:
        logger.error(f"❌ Background processing failed for session {session_id}: {e}")

@app.get("/sessions")
async def get_sessions():
    """Get all interview sessions"""
    try:
        # Try to get from database first
        if db_manager.database is not None:
            db_interviews = await db_manager.get_all_interviews()
            sessions_from_db = [
                {
                    "session_id": interview.session_id,
                    "created_at": interview.started_at.isoformat(),
                    "questions_asked": interview.questions_asked,
                    "is_complete": interview.is_complete,
                    "status": interview.status,
                    "completed_at": interview.completed_at.isoformat() if interview.completed_at else None,
                    "source": "database"
                }
                for interview in db_interviews
            ]
            
            # Also include in-memory sessions
            sessions_from_memory = [
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
            
            return {"sessions": sessions_from_db + sessions_from_memory}
        
        # Fallback to in-memory sessions
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
    if session_id not in interview_sessions:
        # Try DB
        try:
            if db_manager.database is not None:
                interview = await db_manager.get_interview(session_id)
                if interview:
                    return {
                        "session_id": session_id,
                        "conversation": interview.conversation_history or [],
                        "is_complete": interview.is_complete
                    }
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interview_sessions[session_id]
    return {
        "session_id": session_id,
        "conversation": session.conversation_history,
        "is_complete": session.is_complete
    }

# New database management endpoints
@app.get("/database/status")
async def get_database_status():
    """Get database connection status"""
    try:
        logger.info(f"Checking database status - db_manager.database: {db_manager.database}")
        if db_manager.database is not None:
            # Test connection
            logger.info("Testing database ping...")
            await db_manager.database.command("ping")
            logger.info("Database ping successful")
            return {
                "connected": True,
                "database_name": db_manager.database.name,
                "message": "Database connection is healthy"
            }
        else:
            logger.warning("Database is None")
            return {
                "connected": False,
                "message": "Database not connected"
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
        if db_manager.database is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        rules = await db_manager.get_all_rules()
        return {
            "rules": [
                {
                    "interview_session_id": rule.interview_session_id,
                    "rule_id": rule.rule_id,
                    "trigger": rule.trigger,
                    "action": rule.action,
                    "priority": rule.priority,
                    "category": rule.category,
                    "extracted_at": (rule.extracted_at.isoformat() if hasattr(rule.extracted_at, "isoformat") else rule.extracted_at)
                }
                for rule in rules
            ],
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
        if db_manager.database is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        rules = await db_manager.get_rules_by_session(session_id)
        logger.info(f"📊 Found {len(rules)} rules for session {session_id}")
        
        # Debug: Check what session IDs exist in the rules collection
        all_rules = await db_manager.get_all_rules()
        session_ids_in_rules = list(set([rule.interview_session_id for rule in all_rules]))
        logger.info(f"🔍 All session IDs in rules collection: {session_ids_in_rules}")
        
        return {
            "session_id": session_id,
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "trigger": rule.trigger,
                    "action": rule.action,
                    "priority": rule.priority,
                    "category": rule.category,
                    "extracted_at": (rule.extracted_at.isoformat() if hasattr(rule.extracted_at, "isoformat") else rule.extracted_at)
                }
                for rule in rules
            ],
            "total_count": len(rules),
            "debug_info": {
                "all_session_ids_in_rules": session_ids_in_rules,
                "requested_session_id": session_id
            }
        }
    except Exception as e:
        logger.error(f"Error getting rules for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rules for session: {str(e)}")

@app.get("/database/rules/count")
async def get_rules_count():
    """Get total count of rules in database"""
    try:
        if db_manager.database is None:
            return {"count": 0, "connected": False}
        
        count = await db_manager.rules_collection.count_documents({})
        return {"count": count, "connected": True}
    except Exception as e:
        logger.error(f"Error getting rules count: {e}")
        return {"count": 0, "connected": False, "error": str(e)}

@app.get("/database/rules/collection/{session_id}")
async def get_rules_collection(session_id: str):
    """Get the rules collection for a session"""
    try:
        if db_manager.database is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        collection = await db_manager.get_rules_collection(session_id)
        if not collection:
            return {
                "session_id": session_id,
                "rules": [],
                "total_rules": 0,
                "status": "processing"  # Still processing
            }
        
        return {
            "session_id": session_id,
            "rules": collection.rules,
            "total_rules": collection.total_rules,
            "extracted_at": collection.extracted_at.isoformat(),
            "status": "completed"  # Processing complete
        }
    except Exception as e:
        logger.error(f"Error getting rules collection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rules collection: {str(e)}")

@app.get("/processing/status/{session_id}")
async def get_processing_status(session_id: str):
    """Check if rule processing is complete for a session"""
    try:
        if db_manager.database is None:
            return {"status": "database_unavailable"}
        
        # Check if rules exist in database
        collection = await db_manager.get_rules_collection(session_id)
        if collection and collection.rules:
            return {
                "status": "completed",
                "rules_count": len(collection.rules),
                "extracted_at": collection.extracted_at.isoformat()
            }
        else:
            return {"status": "processing"}
            
    except Exception as e:
        logger.error(f"Error checking processing status: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
