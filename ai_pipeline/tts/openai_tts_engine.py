# from __future__ import annotations

# from pathlib import Path
# from uuid import uuid4

# from ai_pipeline.core.openai_client import client


# def synthesize_with_openai_tts(
#     text: str,
#     output_dir: str | Path = "backend/uploads",
#     voice: str = "shimmer",
# ) -> str:
#     """
#     Convert assistant reply text to speech using OpenAI TTS.

#     Returns:
#         Path to generated mp3 file.
#     """
#     if not text or not text.strip():
#         raise ValueError("Cannot synthesize empty text")

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     output_path = output_dir / f"teta_reply_{uuid4().hex}.mp3"

#     response = client.audio.speech.create(
#         model="gpt-4o-mini-tts",
#         voice=voice,
#         input=text.strip(),
#     )

#     response.write_to_file(output_path)

#     return str(output_path)


#====================================================

from __future__ import annotations

import base64

from ai_pipeline.core.openai_client import client


def synthesize_openai_tts_to_base64(
    text: str,
    voice: str = "coral",
) -> str:
    """
    Convert reply text to speech using OpenAI TTS.

    Returns:
        base64 encoded mp3 audio string.
    """
    if not text or not text.strip():
        return ""

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text.strip(),
        instructions=(
            "Speak in Arabic with a warm, gentle, soft grandmother-like tone. "
            "The voice should feel safe, kind, and comforting for a child. "
            "Use a natural Palestinian-friendly style. "
            "Do not sound like a formal news announcer. "
            "Speak clearly with moderate speed, not too slowly."
        ),
        response_format="mp3",
        speed=1.12,
    )

    audio_bytes = response.read()
    return base64.b64encode(audio_bytes).decode("utf-8")