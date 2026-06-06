import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_pipeline.orchestrator.ai_orchestrator import (
    process_child_audio,
    end_session
)

# =========================================================
# Test dataset
# =========================================================

AUDIO_FILES = [
    ("datasets/raw_audio/safe_test.mp3", "Safe"),
    ("datasets/raw_audio/distressed_test.mp3", "Distressed"),
    ("datasets/raw_audio/high_risk_test.mp3", "High Risk"),
    ("datasets/raw_audio/unclear_test.mp3", "Unclear"),
]

# =========================================================
# Single conversation session test
# =========================================================

def run_conversation_test():
    print()
    print("=" * 70)
    print("🧠 Running Conversation Session Test")
    print("=" * 70)

    conversation_history = []

    try:
        result = process_child_audio(
            "datasets/raw_audio/child_voice.mp3",
            conversation_history
        )

        print()
        print("📝 Processed Text:")
        print(result["processed_text"])

        print()
        print("🎯 Predicted Label:")
        print(result["triage_result"]["predicted_label"])

        print()
        print("🤖 Teta Reply:")
        print(result["reply_text"])

        report = end_session(conversation_history)

        print()
        print("📄 Specialist Report:")
        print(report)

    except Exception as e:
        print(f"❌ Conversation test failed: {e}")


# =========================================================
# Batch classification tests
# =========================================================

def run_batch_tests():
    print()
    print("=" * 70)
    print("🧪 Running Batch Audio Tests")
    print("=" * 70)

    for audio_path, expected_label in AUDIO_FILES:

        print()
        print("-" * 70)
        print(f"🎤 Testing: {audio_path}")
        print(f"📌 Expected: {expected_label}")
        print("-" * 70)

        try:
            result = process_child_audio(audio_path)

            triage = result["triage_result"]

            actual_label = triage["predicted_label"]

            print()
            print("📝 Whisper:")
            print(f"Raw:       {result['raw_text']}")
            print(f"Processed: {result['processed_text']}")

            print()
            print("🎯 Triage:")
            print(f"Label:      {actual_label}")
            print(f"Signal:     {triage['risk_signal']}")
            print(f"Confidence: {triage['confidence']}")
            print(f"Review?:    {triage['needs_review']}")

            print()
            print("🤖 Teta:")
            print(f"Source:     {result['response_source']}")
            print(f"Strategy:   {result['response_strategy_label']}")
            print(f"Reply:      {result['reply_text']}")
            emotion = result["emotion_result"]
            weighting = result["weighted_result"]

            print()
            print("🎭 Audio Emotion:")
            print(f"Emotion:    {emotion['emotion']}")
            print(f"Arousal:    {emotion['arousal']}")
            print(f"Valence:    {emotion['valence']}")
            print(f"Engagement: {emotion['engagement']}")

            print()
            print("⚖️ Multimodal Weighting:")
            print(f"Agreement:      {weighting['agreement']}")
            print(f"Audio Distress: {weighting['audio_shows_distress']}")
            print(f"Final Conf:     {weighting['final_confidence']}")
            print(f"Note:           {weighting['weighting_note']}")
            match = (
                "✅"
                if expected_label.lower() in actual_label.lower()
                else "⚠️"
            )

            print()
            print(
                f"{match} Expected: {expected_label} | Got: {actual_label}"
            )

        except FileNotFoundError:
            print(f"❌ File not found: {audio_path}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print()
    print("=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    run_conversation_test()

    run_batch_tests()