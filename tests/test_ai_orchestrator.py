import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.orchestrator.ai_orchestrator import process_child_audio

audio_path = PROJECT_ROOT / "datasets" / "raw_audio" / "child_voice.mp3"

result = process_child_audio(str(audio_path))

print("\nRAW TEXT:")
print(result["raw_text"])

print("\nPROCESSED TEXT:")
print(result["processed_text"])

print("\nTETA REPLY:")
print(result["reply_text"])
print("\nEMOTION RESULT:")
print(result["emotion_result"])

print("\nWEIGHTED RESULT:")
print(result["weighted_result"])

print("\nTRIAGE RESULT:")
print(result["triage_result"])