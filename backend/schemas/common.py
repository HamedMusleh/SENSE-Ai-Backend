"""
Common shared schemas and enums.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """
    Triage risk levels used across the system.
    Mirrors the AI pipeline's Green / Yellow / Red triage labels.

    NOTE: The backend does NOT decide these values. It only transports
    whatever the AI pipeline returns. This enum exists for validation
    and documentation only.
    """
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"
    UNKNOWN = "Unknown"


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""
    status: str = Field(..., examples=["ok"])
    service: str = Field(default="sense-backend")
    version: str = Field(default="0.1.0")
    pipeline_mode: str = Field(
        ...,
        description="Whether the AI pipeline is 'real' or 'mock'",
        examples=["real", "mock"],
    )


class ErrorResponse(BaseModel):
    """Standard error envelope returned by all endpoints on failure."""
    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")
    detail: Optional[Any] = Field(
        default=None, description="Optional extra context for debugging"
    )
