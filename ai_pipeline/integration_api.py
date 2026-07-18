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
from collections.abc import Iterator
from typing import Any

from ai_pipeline.orchestrator.ai_orchestrator import (
    process_child_audio,
    end_session,
)
from ai_pipeline.llm.teta_engine import ask_teta_reply_stream
from ai_pipeline.stt.cohere_stt_engine import transcribe as transcribe_stt
from ai_pipeline.triage.triage_classifier import classify_triage
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


def process_turn_stream(
    audio_path: str,
    conversation_history: list[dict] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one audio turn and stream sentence-level TTS events.

    STT and triage complete before the first event. The safety strategy then
    decides whether to use the streaming LLM path or emit one vetted High Risk
    response. The existing ``process_turn`` entrypoint remains unchanged for
    REST callers.
    """
    total_start = time.perf_counter()

    if conversation_history is None:
        conversation_history = []

    stt_start = time.perf_counter()
    raw_text, processed_text = transcribe_stt(audio_path)
    _print_time("STREAM STT", stt_start)

    triage_history: list[dict[str, str]] = []
    for turn in conversation_history:
        child_text = turn.get("child_text")
        if child_text:
            triage_history.append({"role": "user", "content": child_text})
        assistant_reply = turn.get("assistant_reply")
        if assistant_reply:
            triage_history.append(
                {"role": "assistant", "content": assistant_reply}
            )

    triage_start = time.perf_counter()
    triage_result = classify_triage(
        processed_text,
        conversation_history=triage_history,
    )
    _print_time("STREAM TRIAGE", triage_start)

    triage_label = triage_result.get(
        "predicted_label",
        "Unclear / Need More Context",
    )
    yield {
        "type": "transcript",
        "raw_text": raw_text,
        "processed_text": processed_text,
        "transcribed_text": processed_text,
        "triage_label": triage_label,
    }

    emotion_result = {
        "source": "disabled_for_demo",
        "reason": "Local wav2vec audio-emotion model disabled due to CPU latency.",
    }
    weighted_result = {
        "source": "disabled_for_demo",
        "reason": "Multimodal fusion disabled because audio emotion is disabled.",
    }
    conversation_history.append(
        {
            "turn": len(conversation_history) + 1,
            "raw_transcription": raw_text,
            "child_text": processed_text,
            "triage_result": triage_result,
            "emotion_result": emotion_result,
            "weighted_result": weighted_result,
        }
    )

    response_source = "openai"
    strategy_label = None
    strategy_signal = None
    full_text = ""
    sentence_texts: list[str] = []

    reply_start = time.perf_counter()
    for event in ask_teta_reply_stream(
        processed_text,
        conversation_history,
        triage_result=triage_result,
    ):
        event_type = event.get("type")

        if event_type == "meta":
            response_source = event.get("source", response_source)
            strategy_label = event.get("strategy_label")
            strategy_signal = event.get("strategy_risk_signal")
            continue

        if event_type == "sentence":
            sentence = event.get("text", "").strip()
            if not sentence:
                continue

            sentence_texts.append(sentence)
            audio_base64 = ""
            try:
                tts_start = time.perf_counter()
                audio_base64 = synthesize_openai_tts_to_base64(sentence)
                _print_time("STREAM TTS SENTENCE", tts_start)
            except Exception as exc:  # noqa: BLE001
                print(f"OpenAI streaming TTS error: {exc}", flush=True)

            yield {
                "type": "audio_chunk",
                "text": sentence,
                "audio_base64": audio_base64,
            }
            continue

        if event_type == "done":
            full_text = event.get("full_text", "").strip()

    _print_time("STREAM LLM + TTS", reply_start)

    if not full_text:
        full_text = " ".join(sentence_texts).strip()

    conversation_history[-1]["assistant_reply"] = full_text
    conversation_history[-1]["response_source"] = response_source

    yield {
        "type": "turn_complete",
        "transcribed_text": processed_text,
        "raw_text": raw_text,
        "reply_text": full_text,
        "audio_response": "",
        "response_source": response_source,
        "response_strategy_label": strategy_label,
        "response_strategy_signal": strategy_signal,
        "triage_label": triage_label,
        "triage_signal": triage_result.get("risk_signal", ""),
        "triage_confidence": triage_result.get("confidence", 0.5),
        "needs_review": triage_result.get("needs_review", True),
        "emotion": emotion_result.get("emotion", "disabled_for_demo"),
        "weighting_agreement": weighted_result.get(
            "agreement",
            "disabled_for_demo",
        ),
        "conversation_history": conversation_history,
    }
    _print_time("TOTAL STREAM PROCESS TURN", total_start)


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
