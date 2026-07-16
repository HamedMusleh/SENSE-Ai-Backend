# from __future__ import annotations
# from typing import Any
# from ai_pipeline.orchestrator.ai_orchestrator import (
#     process_child_audio,
#     end_session,
# )
# #from ai_pipeline.tts.coqui_tts_engine import synthesize_teta_voice_to_base64


# def process_turn(
#     audio_path: str,
#     conversation_history: list[dict] | None = None,
# ) -> dict[str, Any]:
#     if conversation_history is None:
#         conversation_history = []

#     result = process_child_audio(audio_path, conversation_history)

#     triage   = result.get("triage_result", {})
#     emotion  = result.get("emotion_result", {})
#     weighting = result.get("weighted_result", {})
#     reply_text = result.get("reply_text", "")

#     # TTS
#     try:
#         audio_base64 = synthesize_teta_voice_to_base64(reply_text)
#     except Exception as e:
#         print(f"TTS error: {e}", flush=True)
#         audio_base64 = ""

#     return {
#         "transcribed_text": result.get("processed_text", ""),
#         "raw_text": result.get("raw_text", ""),
#         "reply_text": reply_text,
#         "audio_response": audio_base64,
#         "response_source": result.get("response_source", ""),
#         "triage_label": triage.get("predicted_label", "Unclear / Need More Context"),
#         "triage_signal": triage.get("risk_signal", ""),
#         "triage_confidence": triage.get("confidence", 0.5),
#         "needs_review": triage.get("needs_review", True),
#         "emotion": emotion.get("emotion", "unknown"),
#         "weighting_agreement": weighting.get("agreement", ""),
#         "conversation_history": result.get("conversation_history", conversation_history),
#     }


# def analyze_session(conversation_history: list[dict]) -> dict[str, Any]:
#     report = end_session(
#         conversation_history,
#         print_report=False,
#     )
#     gpt = report.get("gpt_analysis", {}) or {}
#     if "message_reports" not in report:
#         report["message_reports"] = gpt.get("message_reports", [])
#     return report

#====================================================
from __future__ import annotations

import time
from typing import Any

from ai_pipeline.orchestrator.ai_orchestrator import (
    process_child_audio,
    end_session,
)
from ai_pipeline.tts.openai_tts_engine import synthesize_openai_tts_to_base64


def _print_time(label: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[TIME] {label}: {elapsed:.2f}s", flush=True)


def process_turn(
    audio_path: str,
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    if conversation_history is None:
        conversation_history = []

    # =========================
    # AI Pipeline:
    # OpenAI STT + text triage + Teta reply
    # =========================
    pipeline_start = time.perf_counter()
    result = process_child_audio(audio_path, conversation_history)
    _print_time("PIPELINE WITHOUT TTS", pipeline_start)

    triage = result.get("triage_result", {}) or {}
    emotion = result.get("emotion_result", {}) or {}
    weighting = result.get("weighted_result", {}) or {}

    reply_text = result.get("reply_text", "") or ""

    # =========================
    # OpenAI TTS
    # =========================
    audio_base64 = ""

    if reply_text.strip():
        try:
            tts_start = time.perf_counter()
            audio_base64 = synthesize_openai_tts_to_base64(reply_text)
            _print_time("OPENAI TTS", tts_start)

        except Exception as e:
            print(f"OpenAI TTS error: {e}", flush=True)
            audio_base64 = ""
    else:
        print("[WARN] Empty reply_text, skipping TTS.", flush=True)

    response = {
        "transcribed_text": result.get("processed_text", ""),
        "raw_text": result.get("raw_text", ""),

        "reply_text": reply_text,
        "audio_response": audio_base64,

        "response_source": result.get("response_source", "openai"),

        "triage_label": triage.get(
            "predicted_label",
            "Unclear / Need More Context",
        ),
        "triage_signal": triage.get("risk_signal", ""),
        "triage_confidence": triage.get("confidence", 0.5),
        "needs_review": triage.get("needs_review", True),

        "emotion": emotion.get("emotion", "disabled_for_demo"),
        "weighting_agreement": weighting.get("agreement", "disabled_for_demo"),

        "conversation_history": result.get(
            "conversation_history",
            conversation_history,
        ),
    }

    _print_time("TOTAL PROCESS TURN", total_start)

    return response


def analyze_session(conversation_history: list[dict]) -> dict[str, Any]:
    analysis_start = time.perf_counter()

    report = end_session(
        conversation_history,
        print_report=False,
    )

    gpt = report.get("gpt_analysis", {}) or {}

    if "message_reports" not in report:
        report["message_reports"] = gpt.get("message_reports", [])

    _print_time("ANALYZE SESSION", analysis_start)

    return report