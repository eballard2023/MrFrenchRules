"""
Rule extraction from conversation (simple LLM call).
Interview flow is handled by interview_graph.py (LangGraph).
"""

from typing import List
from openai import AsyncOpenAI

from prompts import RULE_EXTRACTION_SYSTEM, RULE_EXTRACTION_PROMPT


def _filter_task_statements(raw_tasks: List[str]) -> List[str]:
    """Filter and validate extracted rule lines."""
    task_statements = []
    for task in raw_tasks:
        if task.upper() in ["NONE", "NO RULES", "NO BEHAVIORAL RULES", "N/A"]:
            continue
        if len(task) < 20:
            continue
        if not any(
            term in task.lower()
            for term in ["child", "parent", "family", "behavior", "task", "routine", "jamie"]
        ):
            continue
        task_statements.append(task)
    return task_statements


async def extract_rules_from_conversation(
    client: AsyncOpenAI,
    conversation_text: str,
    document_context: str,
) -> List[str]:
    """Call OpenAI to extract Jamie rules from conversation and optional document context."""
    prompt = RULE_EXTRACTION_PROMPT.format(
        conversation_text=conversation_text,
        document_context=document_context or "",
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RULE_EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.3,
    )
    tasks_text = (response.choices[0].message.content or "").strip()
    if tasks_text.upper() == "NONE" or not tasks_text or len(tasks_text.strip()) < 10:
        return []
    raw_tasks = [t.strip() for t in tasks_text.split("\n") if t.strip()]
    return _filter_task_statements(raw_tasks)
