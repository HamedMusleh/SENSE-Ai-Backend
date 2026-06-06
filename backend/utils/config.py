"""
Centralized configuration for the SENSE backend.

All environment-dependent values live here so deployment (local / docker /
cloud) only changes env vars, never code.
"""

import os
from functools import lru_cache
from pathlib import Path


# Project root = .../sense-ai-demo
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings:
    """Runtime settings, populated from environment variables with defaults."""

    # --- Server ---
    HOST: str = os.getenv("SENSE_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("SENSE_PORT", "8000"))
    DEBUG: bool = os.getenv("SENSE_DEBUG", "true").lower() == "true"

    # --- Pipeline mode ---
    # "real"  -> call the actual ai_pipeline functions
    # "mock"  -> return canned responses (no AI deps needed)
    # "hybrid"-> try real, fall back to mock on import/runtime failure
    PIPELINE_MODE: str = os.getenv("SENSE_PIPELINE_MODE", "hybrid").lower()

    # --- Uploads ---
    UPLOAD_DIR: Path = Path(
        os.getenv("SENSE_UPLOAD_DIR", str(BACKEND_DIR / "uploads"))
    )
    MAX_UPLOAD_MB: int = int(os.getenv("SENSE_MAX_UPLOAD_MB", "25"))
    ALLOWED_AUDIO_EXTENSIONS: set = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}

    # --- Timeouts (seconds) ---
    PIPELINE_TIMEOUT: int = int(os.getenv("SENSE_PIPELINE_TIMEOUT", "120"))

    # --- Sessions ---
    SESSION_TTL_MINUTES: int = int(os.getenv("SENSE_SESSION_TTL_MIN", "60"))

    # --- CORS ---
    # Comma-separated list of allowed origins; "*" allows all (dev only).
    CORS_ORIGINS: list = os.getenv("SENSE_CORS_ORIGINS", "*").split(",")

    def ensure_dirs(self) -> None:
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
