"""
Configuration for the AI Coach Interview System.
Load from environment via dotenv.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    openai_api_key: str
    env: str
    max_questions: int
    max_doc_chunks: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            env=os.getenv("ENV", "development"),
            max_questions=int(os.getenv("MAX_QUESTIONS", "23")),
            max_doc_chunks=int(os.getenv("MAX_DOC_CHUNKS", "8")),
        )
