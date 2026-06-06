from ai_pipeline.core.openai_client import client

# =========================
# OpenAI Audio Transcription
# =========================
from ai_pipeline.core.openai_client import client
from ai_pipeline.llm.prompt_loader import load_prompt
from ai_pipeline.stt.preprocessing import nlp_preprocess_pipeline


def transcribe_with_openai_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="ar",
            response_format="text",
            prompt=load_prompt("prompts/transcription_prompt.txt")
        )

    raw_text = transcription.strip()
    processed_text = nlp_preprocess_pipeline(raw_text)

    return raw_text, processed_text