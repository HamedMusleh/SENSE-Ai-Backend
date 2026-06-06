"""
SENSE Backend — API Schemas (Contracts)
========================================
Owner: Student 2 (Backend / Server Engineer)

This module defines the request/response contracts between the mobile app
and the server. These schemas are the single source of truth for the API.

RULE: Do NOT put AI logic here. Schemas only describe the SHAPE of data.
"""

from backend.schemas.session import (
    SessionStartResponse,
    SessionState,
    ConversationTurn,
)
from backend.schemas.audio import (
    UploadAudioResponse,
    AnalyzeResponse,
)
from backend.schemas.common import (
    RiskLevel,
    ErrorResponse,
    HealthResponse,
)

__all__ = [
    "SessionStartResponse",
    "SessionState",
    "ConversationTurn",
    "UploadAudioResponse",
    "AnalyzeResponse",
    "RiskLevel",
    "ErrorResponse",
    "HealthResponse",
]
