"""
Emotion Diagnostic — compares neutral vs emotional recordings.
Run from project root:  py diagnose_emotion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_pipeline.audio_emotion.wav2vec_emotion_engine import analyze_audio_emotion

PAIRS = [
    ("Safe",      "datasets/raw_audio/safe_test.mp3",      "datasets/raw_audio/safe_emotion.mp3"),
    ("Distressed","datasets/raw_audio/distressed_test.mp3","datasets/raw_audio/distressed_emotion.mp3"),
    ("High Risk", "datasets/raw_audio/high_risk_test.mp3", "datasets/raw_audio/high_risk_emotion.mp3"),
    ("Unclear",   "datasets/raw_audio/unclear_test.mp3",   "datasets/raw_audio/unclear_emotion.mp3"),
]


def show(label, path):
    if not os.path.exists(path):
        print(f"   ❌ NOT FOUND: {path}")
        return
    r = analyze_audio_emotion(path)
    p = r["prosody"]
    print(f"   {label:8} | pitch={p['pitch_mean']:6.1f}Hz "
          f"energy={p['energy']:.4f} "
          f"rate={p['speaking_rate']:.2f} "
          f"VA={p['voice_activity']:.2f} "
          f"| arousal={r['arousal']:.3f} valence={r['valence']:.3f} "
          f"eng={r['engagement']:.3f} -> {r['audio_emotion']}")


print("=" * 100)
print("EMOTION DIAGNOSTIC — neutral (test) vs emotional (emotion)")
print("=" * 100)

for name, neutral_path, emotion_path in PAIRS:
    print(f"\n🎯 {name}")
    show("NEUTRAL", neutral_path)
    show("EMOTION", emotion_path)

print("\n" + "=" * 100)
print("✅ Done. Compare NEUTRAL vs EMOTION rows — do pitch/energy/arousal differ?")
print("=" * 100)