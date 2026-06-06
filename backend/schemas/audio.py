"""
Audio + analysis response schemas.

These describe what the server returns to the mobile app after processing
an audio turn or running the final session analysis.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

from backend.schemas.common import RiskLevel


class UploadAudioResponse(BaseModel):
    """
    Server -> Mobile, after one audio turn is processed through the pipeline.

    This is the core contract from the role description:
        {
          "reply_text": "...",
          "audio_response": "..."
        }
    extended with session + transcription context.
    """
    session_id: str
    turn: int = Field(..., ge=1)
    transcribed_text: str = Field(
        ..., description="What the child said (STT output)"
    )
    reply_text: str = Field(..., description="Teta AI's text reply")
    audio_response: Optional[str] = Field(
        default=None,
        description="URL or base64 of TTS audio. None until TTS is wired in.",
    )


class MessageReport(BaseModel):
    """Per-message analysis item produced by the AI pipeline."""
    turn: int
    child_text: str
    message_summary: str
    emotional_signals: List[str] = Field(default_factory=list)
    message_risk_level: RiskLevel = RiskLevel.UNKNOWN


class AnalyzeResponse(BaseModel):
    """
    Server -> Mobile, after final whole-conversation analysis.

    The backend passes through the AI pipeline's structured output without
    modifying its meaning. `raw` keeps the original payload for the specialist
    report layer downstream.
    """
    session_id: str
    overall_summary: str = Field(default="")
    final_risk_level: RiskLevel = RiskLevel.UNKNOWN
    message_reports: List[MessageReport] = Field(default_factory=list)
    raw: Optional[Any] = Field(
        default=None,
        description="Original unmodified pipeline output (for audit / export)",
    )
