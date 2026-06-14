from __future__ import annotations
from typing import Any
from ai_pipeline.orchestrator.ai_orchestrator import (
    process_child_audio,
    end_session,
)
from ai_pipeline.tts.coqui_tts_engine import synthesize_teta_voice_to_base64


def process_turn(
    audio_path: str,
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    if conversation_history is None:
        conversation_history = []

    result = process_child_audio(audio_path, conversation_history)

    triage   = result.get("triage_result", {})
    emotion  = result.get("emotion_result", {})
    weighting = result.get("weighted_result", {})
    reply_text = result.get("reply_text", "")

    # TTS
    try:
        audio_base64 = synthesize_teta_voice_to_base64(reply_text)
    except Exception as e:
        print(f"TTS error: {e}", flush=True)
        audio_base64 = ""

    return {
        "transcribed_text": result.get("processed_text", ""),
        "raw_text": result.get("raw_text", ""),
        "reply_text": reply_text,
        "audio_response": audio_base64,
        "response_source": result.get("response_source", ""),
        "triage_label": triage.get("predicted_label", "Unclear / Need More Context"),
        "triage_signal": triage.get("risk_signal", ""),
        "triage_confidence": triage.get("confidence", 0.5),
        "needs_review": triage.get("needs_review", True),
        "emotion": emotion.get("emotion", "unknown"),
        "weighting_agreement": weighting.get("agreement", ""),
        "conversation_history": result.get("conversation_history", conversation_history),
    }


def analyze_session(conversation_history: list[dict]) -> dict[str, Any]:
    report = end_session(
        conversation_history,
        print_report=False,
    )
    gpt = report.get("gpt_analysis", {}) or {}
    if "message_reports" not in report:
        report["message_reports"] = gpt.get("message_reports", [])
    return report
