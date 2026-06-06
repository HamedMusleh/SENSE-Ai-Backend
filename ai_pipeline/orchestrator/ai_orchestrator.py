from ai_pipeline.stt.whisper_engine import transcribe_with_whisper
from ai_pipeline.triage.triage_classifier import classify_triage
from ai_pipeline.stt.preprocessing import nlp_preprocess_pipeline
from ai_pipeline.llm.teta_engine import ask_teta_reply
from ai_pipeline.llm.session_analysis import generate_session_report, print_specialist_report
from ai_pipeline.audio_emotion.wav2vec_emotion_engine import detect_emotion
from ai_pipeline.audio_emotion.comparison_weights import compare_weights

def process_child_audio(audio_path, conversation_history=None):

    if conversation_history is None:
        conversation_history = []

    # =========================
    # Whisper STT
    # =========================
    raw_text = transcribe_with_whisper(audio_path)

    # =========================
    # NLP preprocessing
    # =========================
    processed_text = nlp_preprocess_pipeline(raw_text)

    # =========================
    # Emotion Detection (audio)
    # =========================
    emotion_result = detect_emotion(audio_path)

    # =========================
    # Triage Classification (text)  -- runs BEFORE weighting now
    # =========================
    triage_history = _build_triage_history(conversation_history)
    triage_result = classify_triage(
        processed_text,
        conversation_history=triage_history,
    )

    # =========================
    # Weighting Layer (multimodal fusion: text + audio)
    # =========================
    weighted_result = compare_weights(triage_result, emotion_result)

    # =========================
    # Append turn to history
    # =========================
    turn_number = len(conversation_history) + 1
    conversation_history.append({
        "turn": turn_number,
        "raw_transcription": raw_text,
        "child_text": processed_text,
        "emotion_result": emotion_result,
        "weighted_result": weighted_result,
        "triage_result": triage_result,
    })

    # =========================
    # Teta AI
    # =========================
    teta_output = ask_teta_reply(
        processed_text,
        conversation_history,
        triage_result=triage_result,
    )

    reply_text = teta_output["reply_text"]

    conversation_history[-1]["assistant_reply"] = reply_text
    conversation_history[-1]["response_source"] = teta_output["source"]

    return {
        "raw_text": raw_text,
        "processed_text": processed_text,
        "emotion_result": emotion_result,
        "weighted_result": weighted_result,
        "triage_result": triage_result,
        "reply_text": reply_text,
        "response_source": teta_output["source"],
        "response_strategy_label": teta_output["strategy_label"],
        "response_strategy_signal": teta_output["strategy_risk_signal"],
        "conversation_history": conversation_history,
    }


def end_session(conversation_history, session_id=None, print_report=True):
    """Call this when the session ends to generate the specialist report.

    Args:
        conversation_history: The full list of turns from process_child_audio.
        session_id: Optional. Auto-generated from timestamp if None.
        print_report: If True, prints the formatted report to console.

    Returns:
        The full session report dict.

    Usage:
        report = end_session(conversation_history)

        # Or save it:
        import json
        with open("session_report.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    """
    report = generate_session_report(
        conversation_history,
        session_id=session_id,
    )
    if print_report:
        print_specialist_report(report)
    return report


def _build_triage_history(conversation_history):
    """Convert orchestrator history to OpenAI-style format for triage classifier."""
    triage_history = []
    for turn in conversation_history:
        child_text = turn.get("child_text")
        if child_text:
            triage_history.append({"role": "user", "content": child_text})
        assistant_reply = turn.get("assistant_reply")
        if assistant_reply:
            triage_history.append({"role": "assistant", "content": assistant_reply})
    return triage_history