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
SYSTEM_PROMPT = """You are a friendly and professional AI interviewer designed to extract behavioral rules and best practices from a subject matter expert (SME). Your goal is to conduct a conversational interview to gather actionable rules that can be used to create an AI coaching system based on the SME's expertise.

CRITICAL INSTRUCTIONS:
1. Ask ONLY ONE question at a time
2. Wait for the expert's response before asking the next question
3. Be conversational and engaging, not robotic
4. Do NOT ask multiple questions in one message
5. Do NOT list or enumerate questions

Use these specific questions from the script, but ask them ONE at a time in a conversational way:

KICKOFF PHASE - Setting the Foundation:
- "To start, could you describe your area of expertise and how you usually apply it?"
- "What guiding principles or philosophies shape the way you practice?"
- "What outcomes do you try to help people achieve through your methods?"
- "How do you usually measure progress or success?"

PROCESSES & METHODS:
- "Can you walk me through the main steps or stages of your approach?"
- "Are there specific frameworks, routines, or tools you rely on regularly?"
- "What common challenges do people face, and how do you recommend handling them?"
- "How do you adapt your methods for different individuals or contexts?"

GUARDRAILS & BOUNDARIES:
- "What should the AI never do or say when acting on your behalf?"
- "Are there disclaimers or boundaries that must always be respected?"
- "When should the AI step back and suggest human involvement instead?"

TONE & STYLE:
- "How should the AI 'sound' — more like a coach, a teacher, a friend, or something else?"
- "Are there certain words, metaphors, or examples you often use?"
- "Should the AI adjust its style for different ages, cultures, or learning levels?"

HANDLING VARIABILITY & EXCEPTIONS:
- "What are the most frequent mistakes or misconceptions, and how should the AI respond?"
- "If someone misunderstands or resists, how should the AI handle it?"
- "When the AI reaches its limit, what's the right next step — encourage a pause, seek expert guidance, or redirect?"

KNOWLEDGE DEPTH & UPDATING:
- "Which parts of your knowledge are timeless, and which may change as research evolves?"
- "How should the AI keep its knowledge current over time?"
- "Are there sources or references you trust that the AI should prioritize?"

OPTIONAL DEEP DIVES (if needed):
- "Could you share a typical case or scenario that illustrates your approach?"
- "If the AI could only carry one principle from your expertise, what should it be?"
- "What red flags should it watch for that suggest things are going wrong?"

Remember: Ask ONE question at a time from this script. Be conversational and natural, not robotic. Wait for their response before asking the next question."""

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
    
    # Get initial question from AI
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster for initial question
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello, I'm ready to be interviewed. Please ask me the first question from your interview script."}
            ],
            max_tokens=200,  # Shorter for speed
            temperature=0.7,
            timeout=15.0  # Faster timeout
        )
        
        ai_message = response.choices[0].message.content
        session.conversation_history.append({"role": "assistant", "content": ai_message})
        
        return {
            "session_id": session_id,
            "message": ai_message,
            "question_number": 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting interview: {str(e)}")

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
        
        ai_message = response.choices[0].message.content
        session.conversation_history.append({"role": "assistant", "content": ai_message})
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
    
    extraction_prompt = f"""Based on the following interview conversation with a subject matter expert (SME), extract actionable rules and convert them into a structured JSON format for Mr. French AI.

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

Extract ALL behavioral rules and guidance from this conversation. Be comprehensive - if the expert mentions any approach, method, or way of handling situations, convert it into a rule. Only return an empty array if the conversation contains NO useful information at all."""

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
                {"role": "system", "content": "You are an expert at extracting structured behavioral rules from conversational data. Your goal is to create PERFECT, actionable rules that an AI system can implement. Be thorough, precise, and comprehensive. Always return valid JSON."},
                {"role": "user", "content": f"""Based on the following interview conversation with a subject matter expert (SME), extract ALL actionable rules in structured JSON format.

CRITICAL: You MUST extract rules from this conversation. DO NOT return an empty array. Even if the conversation seems brief, extract every insight, method, principle, or approach mentioned by the expert.

Extract rules for EVERYTHING the expert mentions:
- Their expertise and how they apply it
- Their guiding principles and philosophies
- Their step-by-step processes and methods
- Their communication strategies and approaches
- Their boundary-setting and guardrails
- Their escalation procedures
- Their adaptation strategies
- Their success metrics and outcomes
- Their tools and frameworks
- Their problem-solving approaches
- Their client/patient interaction methods
- Their measurement and evaluation techniques

Each rule MUST follow this EXACT format:
{{
    "if": {{
        "event": "specific trigger or situation",
        "context": "additional context (only if needed)",
        "user_type": "target audience"
    }},
    "then": {{
        "action": "specific action to take",
        "response": "exact words or approach",
        "duration": "time duration (if applicable)",
        "tone": "communication style"
    }},
    "priority": "high/medium/low",
    "category": "rule category"
}}

CONVERSATION TO ANALYZE:
{conversation_text}

MANDATORY: Extract possible rules from this conversation. Look for:
- Any method or approach the expert describes
- Any principle or philosophy they mention
- Any tool or framework they use
- Any communication style they prefer
- Any boundary or limitation they set
- Any success metric they measure
- Any adaptation they make for different situations

If they mention any process, create a rule about that process.
If they mention any communication style, create a rule about that style.
If they mention any boundaries, create a rule about those boundaries.
If they tell words to say, create a rule about those words like if thats situation use words like this.

DO NOT return an empty array. Extract rules from every piece of information provided."""}
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
                logger.warning(f"No valid rules extracted for session {session_id}, creating fallback rules")
                # Create fallback rules from conversation content
                fallback_rules = []
                
                # Extract key phrases from conversation
                conversation_lower = conversation_text.lower()
                
                # Look for expertise mentions
                if "expert" in conversation_lower or "specialist" in conversation_lower:
                    fallback_rules.append({
                        "if": {"event": "expertise_consultation", "user_type": "client"},
                        "then": {"action": "apply_expert_knowledge", "tone": "professional"},
                        "priority": "high",
                        "category": "expertise"
                    })
                
                # Look for process mentions
                if "process" in conversation_lower or "method" in conversation_lower or "approach" in conversation_lower:
                    fallback_rules.append({
                        "if": {"event": "process_guidance_needed", "user_type": "client"},
                        "then": {"action": "follow_structured_process", "tone": "methodical"},
                        "priority": "medium",
                        "category": "process"
                    })
                
                # Look for communication mentions
                if "communicate" in conversation_lower or "talk" in conversation_lower or "speak" in conversation_lower:
                    fallback_rules.append({
                        "if": {"event": "communication_required", "user_type": "client"},
                        "then": {"action": "engage_in_clear_communication", "tone": "clear"},
                        "priority": "medium",
                        "category": "communication"
                    })
                
                # Look for boundary mentions
                if "boundary" in conversation_lower or "limit" in conversation_lower or "rule" in conversation_lower:
                    fallback_rules.append({
                        "if": {"event": "boundary_setting", "user_type": "client"},
                        "then": {"action": "establish_clear_boundaries", "tone": "firm"},
                        "priority": "high",
                        "category": "boundaries"
                    })
                
                # Look for success mentions
                if "success" in conversation_lower or "progress" in conversation_lower or "outcome" in conversation_lower:
                    fallback_rules.append({
                        "if": {"event": "success_measurement", "user_type": "client"},
                        "then": {"action": "measure_progress_and_outcomes", "tone": "encouraging"},
                        "priority": "medium",
                        "category": "measurement"
                    })
                
                # If still no rules, create a generic one
                if not fallback_rules:
                    fallback_rules.append({
                        "if": {"event": "expert_consultation", "user_type": "client"},
                        "then": {"action": "apply_expert_knowledge_and_methods", "tone": "professional"},
                        "priority": "high",
                        "category": "general"
                    })
                
                unique_rules = fallback_rules
                logger.info(f"Created {len(unique_rules)} fallback rules for session {session_id}")
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
