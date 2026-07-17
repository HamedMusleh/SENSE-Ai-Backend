"""
STT engine router: Cohere (cloud) or OpenAI, selected via STT_ENGINE env var.

    STT_ENGINE=cohere   -> cohere-transcribe-03-2026 (default; faster, lower WER)
    STT_ENGINE=openai   -> gpt-4o-mini-transcribe (legacy fallback)

Both paths return the SAME contract as the legacy engine:
    (raw_text: str, processed_text: str)

Cohere accepts mp3/wav/flac/ogg but NOT webm/mp4. Browser recordings in those
containers are transcoded to 16kHz mono wav via an in-memory ffmpeg pipe
(no temp file on disk -> faster on Windows).
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from ai_pipeline.stt.preprocessing import nlp_preprocess_pipeline

# Ensure COHERE_API_KEY is available even when called outside the backend.
load_dotenv()

# Cohere accepts these container formats directly; anything else is transcoded.
_COHERE_OK = {".mp3", ".wav", ".flac", ".mpeg", ".mpga", ".ogg"}


def _read_audio_bytes(audio_path: str) -> tuple[bytes, str]:
    """
    Return (audio_bytes, filename_for_upload). If the container is unsupported
    (webm/mp4/...), transcode to 16kHz mono wav in memory via ffmpeg pipe.
    """
    suffix = Path(audio_path).suffix.lower()
    if suffix in _COHERE_OK:
        with open(audio_path, "rb") as f:
            return f.read(), Path(audio_path).name

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Audio format {suffix} needs transcoding but ffmpeg is not on PATH."
        )

    # Transcode via stdin/stdout pipe (no temp file).
    with open(audio_path, "rb") as f:
        raw = f.read()

    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
         "-f", "wav", "pipe:1"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return proc.stdout, "recording.wav"


def transcribe_with_cohere(audio_path: str) -> tuple[str, str]:
    """Transcribe one file via Cohere cloud. Returns (raw_text, processed_text)."""
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is missing from environment / .env")

    client = cohere.ClientV2(api_key=api_key)
    model = os.getenv("COHERE_STT_MODEL", "cohere-transcribe-03-2026")

    audio_bytes, upload_name = _read_audio_bytes(audio_path)

    buf = io.BytesIO(audio_bytes)
    buf.name = upload_name  # cohere SDK uses .name to infer the format
    resp = client.audio.transcriptions.create(
        model=model, file=buf, language="ar", temperature=0
    )
    raw_text = resp.text.strip()

    processed_text = nlp_preprocess_pipeline(raw_text)
    return raw_text, processed_text


def transcribe(audio_path: str) -> tuple[str, str]:
    """
    Router. Reads STT_ENGINE (default 'cohere'). Falls back to the legacy
    OpenAI engine only when explicitly requested.
    """
    engine = os.getenv("STT_ENGINE", "cohere").lower()
    if engine == "openai":
        from ai_pipeline.stt.openai_audio_engine import transcribe_with_openai
        return transcribe_with_openai(audio_path)
    return transcribe_with_cohere(audio_path)
