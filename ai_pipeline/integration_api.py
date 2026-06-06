"""
SENSE AI Pipeline — Public Integration API
==========================================
Owner: Student 1 (AI / Pipeline Lead)

Stable public interface the backend integrates against. The backend calls
ONLY these functions and passes back the `turn_data` we return, so the full
triage + emotion context survives into the final session analysis.

Contract:
    process_turn(audio_path, history)  -> dict   (one full turn; includes turn_data)
    analyze_session(history)           -> dict   (final specialist report)
"""

from __future__ import annotations

from typing import Any

from ai_pipeline.orchestrator.ai_orchestrator import (
    process_child_audio,
    end_session,
)


def process_turn(
    audio_path: str,
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    """Run ONE full turn through the entire AI pipeline.

    Runs: STT -> preprocessing -> emotion -> triage -> weighting -> Teta.

    Args:
        audio_path: Path to the child's audio file.
        conversation_history: The RICH history list (the same list returned
            in previous turns' `conversation_history`). Pass it back each turn
            so triage + emotion context accumulates for the final analysis.

    Returns:
        {
            # --- flat fields for the backend's immediate response ---
            "transcribed_text": str,
            "raw_text": str,
            "reply_text": str,
            "response_source": str,
            "triage_label": str,
            "triage_signal": str,
            "triage_confidence": float,
            "needs_review": bool,
            "emotion": str,
            "weighting_agreement": str,

            # --- the rich, accumulated history (give this back next turn) ---
            "conversation_history": list[dict],
        }
    """
    if conversation_history is None:
        conversation_history = []

    result = process_child_audio(audio_path, conversation_history)

    triage = result.get("triage_result", {})
    emotion = result.get("emotion_result", {})
    weighting = result.get("weighted_result", {})

    return {
        "transcribed_text": result.get("processed_text", ""),
        "raw_text": result.get("raw_text", ""),
        "reply_text": result.get("reply_text", ""),
        "response_source": result.get("response_source", ""),
        "triage_label": triage.get("predicted_label", "Unclear / Need More Context"),
        "triage_signal": triage.get("risk_signal", ""),
        "triage_confidence": triage.get("confidence", 0.5),
        "needs_review": triage.get("needs_review", True),
        "emotion": emotion.get("emotion", "unknown"),
        "weighting_agreement": weighting.get("agreement", ""),
        # The orchestrator mutated this list in place to include the full
        # rich turn (triage_result, emotion_result, etc.). Return it so the
        # backend can store and replay it into analyze_session().
        "conversation_history": result.get("conversation_history", conversation_history),
    }


def analyze_session(conversation_history: list[dict]) -> dict[str, Any]:
    """Run the final whole-conversation specialist analysis.

    Args:
        conversation_history: The RICH history accumulated across turns
            (the `conversation_history` returned by process_turn). It must
            contain the per-turn triage_result / emotion_result so the
            turn-by-turn summary can be built.

    Returns:
        The specialist report dict, with `message_reports` lifted to the top
        level for the backend's convenience (in addition to its original
        place inside `gpt_analysis`).
    """
    report = end_session(
        conversation_history,
        print_report=False,
    )

    # Lift message_reports to the top level so the backend orchestrator can
    # read raw["message_reports"] directly (it currently looks there).
    gpt = report.get("gpt_analysis", {}) or {}
    if "message_reports" not in report:
        report["message_reports"] = gpt.get("message_reports", [])

    return report