from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ExpertInfo(BaseModel):
    expert_name: str
    expert_email: str
    expertise_area: str = "General"
    companion_id: Optional[int] = None
    companion_slug: Optional[str] = None  # e.g. 'jamie' or 'my_persona'; resolved to companion_id if needed

class AdminLoginRequest(BaseModel):
    email: str
    password: str


class UserRegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class StartInterviewRequest(BaseModel):
    """Used when starting an interview from the user dashboard (authenticated)."""
    expertise_area: str = "General"
    companion_slug: str = "jamie"  # 'jamie' or 'my_persona'
