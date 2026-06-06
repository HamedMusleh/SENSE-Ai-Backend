"""
Session-related schemas.

A "session" represents one conversation between a child and Teta AI.
The backend owns session lifecycle (create / track / close) but never
interprets the conversation content — that is the AI pipeline's job.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.schemas.common import RiskLevel


class ConversationTurn(BaseModel):
    """A single exchange in the conversation."""
    turn: int = Field(..., ge=1, description="1-based turn index")
    child_text: str = Field(..., description="Transcribed child speech")
    teta_reply: str = Field(..., description="Teta AI's reply text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionStartResponse(BaseModel):
    """Returned when a new session is created."""
    session_id: str = Field(..., description="Unique session identifier (UUID)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message: str = Field(default="Session started successfully")


class SessionState(BaseModel):
    """
    Full snapshot of a session. Returned by GET /api/session/{id}.
    """
    session_id: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = Field(default=0, ge=0)
    rich_history: List[dict] = Field(
        default_factory=list,
        description="Full pipeline turn data (triage+emotion) for final analysis",
    )
    conversation: List[ConversationTurn] = Field(default_factory=list)
    final_risk_level: Optional[RiskLevel] = Field(
        default=None,
        description="Set only after final analysis runs; None during conversation",
    )
    is_closed: bool = Field(default=False)
