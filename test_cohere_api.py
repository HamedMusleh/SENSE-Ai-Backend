import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "raw_audio"
    / "child_voice.mp3"
)

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise RuntimeError(
        "COHERE_API_KEY is missing from .env"
    )

if not AUDIO_FILE.exists():
    raise FileNotFoundError(AUDIO_FILE)

url = (
    "https://api.cohere.com"
    "/v2/audio/transcriptions"
)

headers = {
    "Authorization": f"Bearer {api_key}",
}

data = {
    "model": (
        "cohere-transcribe-03-2026"
    ),
    "language": "ar",
    "temperature": "0",
}

with AUDIO_FILE.open("rb") as audio_file:
    files = {
        "file": (
            AUDIO_FILE.name,
            audio_file,
            "audio/wav",
        )
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
        files=files,
        timeout=120,
    )

print("Status:", response.status_code)

print(
    "Debug Trace ID:",
    response.headers.get(
        "x-debug-trace-id"
    ),
)

print("Response:")
print(response.text)