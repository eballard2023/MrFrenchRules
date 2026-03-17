"""
LangGraph-based interview flow.
Replaces the previous OpenAI-direct chat logic.
"""

import os
import re
from typing import List, TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from prompts import (
    INTERVIEW_SYSTEM_PROMPT,
    DOCUMENT_CONTEXT_SUFFIX,
    PREVIOUS_QUESTIONS_SUFFIX,
    RECALL_MEMORIES_PREFIX,
)
from behavior.brain.prompts import get_system_prompt as get_behavior_prompt
from utils import get_thread_id, get_expert_name, get_sender_person, get_current_time
from memory.memory_service import get_recent_dialogues, search_memories


# --- State ---

class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    recall_memories: List[str]
    doc_context: str
    previous_questions: List[str]


# --- Model ---

def _get_model():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.8,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _sanitize_question(text: str) -> str:
    """Remove leading numbering/bullets from AI response."""
    lines = (text or "").splitlines()
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r"^\s*(?:\(?\d+\)?[\).:-]\s+|[-*•]\s+)", "", line))
    return "\n".join(cleaned).strip()


def _clean_response(text: str) -> str:
    """Remove leading punctuation and wrapping quotes."""
    content = (text or "").strip()
    content = re.sub(r"^[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/\s]+", "", content)
    content = re.sub(r'^["\u201c\u201d\']+(.+?)["\u201c\u201d\']+$', r"\1", content)
    return content.strip() or text


# --- Nodes ---

async def load_memories(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """
    Load recalled memories for the thread.
    Boilerplate: uses memory_service.search_memories; implement your own.
    """
    if isinstance(state, dict):
        messages = state.get("messages", [])
        recall_memories_state = state.get("recall_memories", [])
        doc_context = state.get("doc_context", "")
        previous_questions = state.get("previous_questions", [])
    else:
        messages = state["messages"]
        recall_memories_state = state.get("recall_memories", [])
        doc_context = state.get("doc_context", "")
        previous_questions = state.get("previous_questions", [])

    thread_id = get_thread_id(config)
    sender = get_sender_person(config)

    convo_str = get_buffer_string(messages)
    current_msg = messages[-1].content if messages else ""
    query = f"{current_msg} {convo_str[-500:]}".strip()

    docs = await search_memories(user_id=thread_id, role=sender, query=query, k=30)
    recall_memories = []
    if docs:
        for doc in docs:
            meta = doc.get("metadata") or {}
            ts = meta.get("timestamp", "unknown")
            content = doc.get("page_content", str(doc))
            recall_memories.append(f"[{ts}] {content}")
    return {
        "messages": messages,
        "recall_memories": recall_memories,
        "doc_context": doc_context,
        "previous_questions": previous_questions,
    }


async def agent(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Generate the next interview response."""
    messages = state["messages"]
    recall_memories = state.get("recall_memories", [])
    doc_context = state.get("doc_context", "")
    previous_questions = state.get("previous_questions", [])

    configurable = config.get("configurable") or {}
    behavior_status = configurable.get("behavior_status", "OK")

    # Build system prompt from behavioral brain + interview instructions
    system = get_behavior_prompt(behavior_status) + "\n\n"
    if recall_memories:
        system += RECALL_MEMORIES_PREFIX.format(
            recall_memories="\n".join(recall_memories)
        )
    if previous_questions:
        system += PREVIOUS_QUESTIONS_SUFFIX.format(
            previous_questions="\n".join(f"- {q}" for q in previous_questions[-5:])
        )
    if doc_context:
        system += DOCUMENT_CONTEXT_SUFFIX.format(document_content=doc_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="messages"),
    ])
    model = _get_model()
    chain = prompt | model

    result = await chain.ainvoke({"messages": messages})
    content = result.content if hasattr(result, "content") else str(result)
    content = _clean_response(_sanitize_question(content))

    return {
        "messages": [AIMessage(content=content)],
        "recall_memories": recall_memories,
        "doc_context": doc_context,
        "previous_questions": previous_questions,
    }


# --- Graph ---

_checkpointer = MemorySaver()

builder = StateGraph(InterviewState)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_edge("agent", END)

graph = builder.compile(checkpointer=_checkpointer)


async def run_interview_turn(
    session_id: str,
    conversation_history: List[dict],
    doc_context: str = "",
    expert_name: str = "Expert",
    expert_email: str = "",
    behavior_status: str = "OK",
) -> str:
    """
    Run one interview turn and return the AI's next message.
    """
    messages: List[BaseMessage] = []
    previous_questions: List[str] = []
    for m in conversation_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
            if "?" in content:
                previous_questions.append(content.split("?")[0] + "?")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": session_id,
            "expert_name": expert_name,
            "expert_email": expert_email,
            "sender_person": expert_name,
            "behavior_status": behavior_status,
        }
    }

    initial_state: InterviewState = {
        "messages": messages,
        "recall_memories": [],
        "doc_context": doc_context,
        "previous_questions": previous_questions,
    }

    result = await graph.ainvoke(
        initial_state,
        config=config,
    )

    out_messages = result.get("messages", [])
    for m in reversed(out_messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""
