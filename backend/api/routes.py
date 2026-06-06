"""
REST API Routes
===============
Owner: Student 2 (Backend)

Thin HTTP layer. Each endpoint:
  1. validates the request
  2. delegates to a service (orchestrator / session manager)
  3. returns a schema-typed response

NO AI logic here. Routes are wiring only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.schemas.session import SessionStartResponse, SessionState
from backend.schemas.audio import UploadAudioResponse, AnalyzeResponse
from backend.schemas.common import HealthResponse
from backend.services.session_manager import get_session_manager
from backend.services.orchestrator import get_orchestrator
from backend.utils.config import get_settings
from backend.utils.logger import get_logger
from backend.utils.errors import InvalidAudioError

logger = get_logger("api")
router = APIRouter(prefix="/api", tags=["sense"])


# --------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse)
def health():
    orch = get_orchestrator()
    return HealthResponse(
        status="ok",
        pipeline_mode=orch.adapter.effective_mode,
    )


# --------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------- #
@router.post("/session/start", response_model=SessionStartResponse)
def start_session():
    state = get_session_manager().create_session()
    return SessionStartResponse(
        session_id=state.session_id,
        created_at=state.created_at,
    )


@router.get("/session/{session_id}", response_model=SessionState)
def get_session(session_id: str):
    return get_session_manager().get_session(session_id)


# --------------------------------------------------------------------- #
# Audio turn
# --------------------------------------------------------------------- #
@router.post("/upload_audio", response_model=UploadAudioResponse)
async def upload_audio(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Mobile -> Server: one audio turn.
    Saves the file, runs it through the orchestrator, returns Teta's reply.
    """
    settings = get_settings()

    # Validate extension
    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio type '{suffix}'",
            detail=sorted(settings.ALLOWED_AUDIO_EXTENSIONS),
        )

    # Ensure session exists (raises SessionNotFoundError if not)
    get_session_manager().get_session(session_id)

    # Persist upload to disk
    dest = settings.UPLOAD_DIR / f"{session_id}_{audio_file.filename}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(audio_file.file, out)
    finally:
        await audio_file.close()

    # Size guard
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        dest.unlink(missing_ok=True)
        raise InvalidAudioError(
            f"Audio too large ({size_mb:.1f} MB > {settings.MAX_UPLOAD_MB} MB)"
        )

    logger.info("Audio saved for session %s (%.2f MB)", session_id, size_mb)

    # Orchestrate the full turn
    return get_orchestrator().process_audio_turn(session_id, str(dest))


# --------------------------------------------------------------------- #
# Final analysis
# --------------------------------------------------------------------- #
@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(session_id: str = Form(...)):
    """Run final whole-conversation analysis and close the session."""
    get_session_manager().get_session(session_id)  # validate existence
    return get_orchestrator().analyze_session(session_id)
