import whisper

print("⏳ Loading Whisper model (medium)...")
model = whisper.load_model("medium")
print("✅ Whisper ready")

# Initial prompt helps Whisper recognize Palestinian Arabic dialect
# patterns and common child-speech phrases. Whisper uses this as a
# context hint, not as text to transcribe.
PALESTINIAN_INITIAL_PROMPT = (
    "حوار باللهجة الفلسطينية مع طفل. "
    "كلمات شائعة: بدي، بصحى، بحس، ماما، بابا، تيتا، "
    "كوابيس، خايف، مبسوط، اليوم، أصحابي، صاحبي، "
    "ما عرفت، مش عارف، بدي أنام، نفسي."
)


def transcribe_with_whisper(audio_path):
    print("🎤 Starting Whisper transcription...")
    print(f"📁 Audio path: {audio_path}")
    result = model.transcribe(
        audio_path,
        language="ar",                              # ISO code is more reliable
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
        initial_prompt=PALESTINIAN_INITIAL_PROMPT,  # 👈 الإضافة الجديدة
        beam_size=5,                                # 👈 search أعمق = دقة أفضل
        best_of=5,                                  # 👈 يجرّب 5 candidates
        no_speech_threshold=0.6,                    # 👈 يقلل hallucinations
    )
    print("✅ Whisper transcription finished")
    return result["text"].strip()