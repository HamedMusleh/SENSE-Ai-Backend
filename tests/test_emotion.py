import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_pipeline.audio_emotion.emotion_service import analyze_audio_emotion

AUDIO_FILES = [
    ROOT / "datasets/raw_audio/safe_test.mp3",
    ROOT / "datasets/raw_audio/distressed_test.mp3",
    ROOT / "datasets/raw_audio/high_risk_test.mp3",
    ROOT / "datasets/raw_audio/unclear_test.mp3",
]

for audio_path in AUDIO_FILES:
    print("\n==============================")
    print("Audio:", audio_path)

    result = analyze_audio_emotion(str(audio_path))

    print("Emotion:", result["audio_emotion"])
    print("Arousal:", result["arousal"])
    print("Valence:", result["valence"])
    print("Engagement:", result["engagement"])
    print("Embedding size:", result["embedding_size"])