"""
Multimodal Weighting Layer (X1 — smart fusion)
==============================================

Combines the TEXT triage result with the AUDIO emotion signal.

Core safety principle (binding):
  - High Risk from TEXT is NEVER downgraded by audio. Safety over accuracy.

Fusion logic:
  1. If text says High Risk            -> keep High Risk, note audio agreement.
  2. If text & audio AGREE on distress -> boost confidence (more certain).
  3. If text says Safe but audio shows  -> raise a review flag; the audio
     strong distress (high arousal,        adds doubt the text missed.
     negative valence, low engagement)
  4. Otherwise                         -> text leads, audio annotates.

This layer does NOT change the predicted_label except to RAISE concern.
It can never make the system *less* cautious than the text classifier.
"""


# Audio emotions that indicate the child may be in distress.
_DISTRESS_AUDIO = {"fearful_or_agitated", "sad_or_low", "withdrawn"}
_CALM_AUDIO = {"calm_or_neutral", "excited_or_alert"}


def _audio_shows_distress(emotion_result):
    """Heuristic: does the audio signal suggest distress?"""
    emotion = emotion_result.get("emotion", "unknown")
    arousal = emotion_result.get("arousal", 0.0)
    valence = emotion_result.get("valence", 0.5)
    engagement = emotion_result.get("engagement", 1.0)

    if emotion in _DISTRESS_AUDIO:
        return True
    # high arousal + negative valence = agitation/fear
    if arousal >= 0.60 and valence < 0.45:
        return True
    # very low engagement = withdrawal
    if engagement < 0.30:
        return True
    return False


def compare_weights(triage_result, emotion_result):
    """Fuse text triage + audio emotion.

    Args:
        triage_result: dict from classify_triage() — must contain
            'predicted_label', 'confidence', 'risk_signal', 'needs_review'.
        emotion_result: dict from detect_emotion() — contains
            'emotion', 'arousal', 'valence', 'engagement', 'confidence'.

    Returns:
        dict describing the fused decision and any escalation.
    """
    print("⚖️ Running multimodal weighting layer...")

    text_label = triage_result.get("predicted_label", "Unclear / Need More Context")
    text_conf = float(triage_result.get("confidence", 0.5))
    text_needs_review = bool(triage_result.get("needs_review", False))

    audio_emotion = emotion_result.get("emotion", "unknown")
    audio_distress = _audio_shows_distress(emotion_result)

    # Defaults — text leads.
    final_label = text_label
    final_confidence = text_conf
    needs_review = text_needs_review
    agreement = "text_leads"
    weighting_note = "Audio used as annotation; text triage leads."

    is_high_risk = text_label.startswith("High Risk")
    is_distressed = text_label.startswith("Distressed")
    is_safe = text_label.startswith("Safe")

    # --- Rule 1: High Risk is sacred. Never downgrade. ---
    if is_high_risk:
        final_label = text_label
        if audio_distress:
            agreement = "text_audio_agree_high_risk"
            final_confidence = min(text_conf + 0.05, 0.95)
            weighting_note = "Audio distress corroborates text High Risk."
        else:
            agreement = "audio_neutral_text_high_risk"
            weighting_note = "Text High Risk retained despite neutral audio (safety)."
        needs_review = True  # High Risk always reviewed

    # --- Rule 2: Distressed + audio agrees -> more confident ---
    elif is_distressed:
        if audio_distress:
            agreement = "text_audio_agree_distress"
            final_confidence = min(text_conf + 0.10, 0.85)
            weighting_note = "Audio distress corroborates text Distressed."
        else:
            agreement = "audio_neutral_text_distress"
            weighting_note = "Text Distressed retained; audio neutral."

    # --- Rule 3: text Safe but audio distressed -> flag doubt ---
    elif is_safe:
        if audio_distress:
            agreement = "conflict_text_safe_audio_distress"
            needs_review = True
            # Lower our confidence in 'Safe' — the audio disagrees.
            final_confidence = min(text_conf, 0.50)
            weighting_note = (
                "Audio shows distress while text reads Safe. "
                "Flagged for review — possible masked distress."
            )
        else:
            agreement = "text_audio_agree_safe"
            weighting_note = "Audio neutral; text Safe corroborated."

    # --- Rule 4: Unclear -> audio can nudge attention ---
    else:  # Unclear / Need More Context
        if audio_distress:
            agreement = "unclear_text_audio_distress"
            needs_review = True
            weighting_note = (
                "Text Unclear; audio shows distress. "
                "Flagged for review."
            )
        else:
            agreement = "unclear_audio_neutral"
            weighting_note = "Text Unclear; audio neutral. More context needed."

    return {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 2),
        "needs_review": needs_review,
        "agreement": agreement,
        "text_label": text_label,
        "text_confidence": round(text_conf, 2),
        "audio_emotion": audio_emotion,
        "audio_shows_distress": audio_distress,
        "audio_arousal": emotion_result.get("arousal", 0.0),
        "audio_valence": emotion_result.get("valence", 0.5),
        "audio_engagement": emotion_result.get("engagement", 0.0),
        "weighting_note": weighting_note,
        "method": "multimodal_fusion_v1",
    }