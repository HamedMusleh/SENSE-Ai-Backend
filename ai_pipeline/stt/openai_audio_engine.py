from __future__ import annotations
import os
import re
from pathlib import Path
from ai_pipeline.core.openai_client import client

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_prompt(relative_path):
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text


def remove_common_noise(text):
    noise = ["اشترك بالقناة", "شكرا للمشاهدة", "ترجمة", "موسيقى", "تصفيق"]
    for p in noise:
        text = text.replace(p, "")
    return re.sub(r"\s+", " ", text).strip()


def nlp_preprocess_pipeline(raw_text):
    text = remove_common_noise(raw_text)
    text = text.strip()
    return text


def transcribe_with_openai(audio_path):
    prompt = load_prompt("prompts/transcription_prompt.txt")
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="ar",
            response_format="text",
            prompt=prompt if prompt else None,
        )
    raw_text = transcription.strip()
    processed_text = nlp_preprocess_pipeline(raw_text)
    return raw_text, processed_text
