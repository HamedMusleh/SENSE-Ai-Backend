import base64

from ai_pipeline.tts.openai_tts_engine import synthesize_openai_tts_to_base64

text = "مرحبا يا روحي، أنا تيتا هون معك. احكيلي شو مضايقك بهدوء."

for voice in ["coral", "shimmer", "nova"]:
    audio_b64 = synthesize_openai_tts_to_base64(text, voice=voice)

    with open(f"test_{voice}.mp3", "wb") as f:
        f.write(base64.b64decode(audio_b64))

    print(f"Saved test_{voice}.mp3")