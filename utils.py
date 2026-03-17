"""
Utilities for LangGraph RunnableConfig.
Extract session_id, expert name, etc. from configurable dict.
"""

from datetime import datetime
from typing import Any


def get_thread_id(config: dict) -> str:
    """Thread/session ID for checkpointing and memory."""
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if not thread_id:
        raise ValueError("thread_id must be provided in config.configurable")
    return str(thread_id)


def get_expert_name(config: dict) -> str:
    """Expert name for personalization."""
    configurable = config.get("configurable") or {}
    return configurable.get("expert_name", "Expert")


def get_expert_email(config: dict) -> str:
    """Expert email for session context."""
    configurable = config.get("configurable") or {}
    return configurable.get("expert_email", "")


def get_sender_person(config: dict) -> str:
    """Sender/speaker identifier (e.g. for memory roles)."""
    configurable = config.get("configurable") or {}
    return configurable.get("sender_person", get_expert_name(config))


def get_current_time(config: dict) -> str:
    """Current time string for context."""
    configurable = config.get("configurable") or {}
    if "current_time" in configurable:
        return str(configurable["current_time"])
    return datetime.utcnow().isoformat()
