"""
SENSE Rule-Based Triage Classifier
===================================

Replaces the placeholder in ai_pipeline/triage/triage_classifier.py.

Maps preprocessed Palestinian Arabic text into the four-label triage schema
defined in resources/annotation_guide.md:

    - Safe / Regulated
    - Distressed / Needs Support
    - High Risk / Urgent
    - Unclear / Need More Context

Design principles (binding, from annotation_guide §3):
    P1. Safety over accuracy — escalate under genuine doubt.
    P2. Conservative escalation under ambiguity — Unclear is a valid label.
    P3. Triage, not diagnosis — urgency only, no condition naming.
    P4. Cultural and developmental authenticity — Palestinian child voice.
    P5. The text speaks; the annotator does not invent.

Output contract (unchanged from placeholder so the orchestrator stays
identical):

    {
        "predicted_label":   str,
        "predicted_emotion": str,
        "risk_signal":       str,
        "needs_review":      bool,
        "confidence":        float,
        "source":            "rule_based_triage_v1",
        # additional fields below are additive — orchestrator can ignore:
        "matched_signals":   List[Dict],   # all rule matches with weights
        "review_notes":      str,          # short English reasoning for reviewer
        "hedge_detected":    bool,
        "retraction_detected": bool,
    }

Confidence ceilings (per design decision; rule-based confidence is bounded
so the future ML classifier has headroom to claim higher confidence):

    High Risk (explicit, e.g. active_death_wish)        → 0.90
    High Risk (implicit, e.g. passive_death_wish)       → 0.85
    High Risk (sensory/dissociative, single signal)     → 0.80
    Distressed                                          → 0.70
    Safe                                                → 0.70
    Unclear                                             → 0.50

Conversation handling (pre-alpha scope):
    The classifier accepts an optional `conversation_history` argument.
    Core classification still runs on the single current utterance, but
    two simple cumulative rules from annotation_guide §6 are honored:

      6.2 Escalation across turns — if any prior child turn was High Risk
          and the current is lower, set needs_review=True and note it.
      6.3 Retraction does not de-escalate — if the current utterance is
          a retraction of a prior High Risk turn, the label inherits
          High Risk (still per Principle 1) with needs_review=True.

    Full cumulative ambiguity and session-level reporting are deferred.
"""

from __future__ import annotations
import codecs
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_LEXICON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources",
    "triage_lexicons.json",
)

LABEL_SAFE = "Safe / Regulated"
LABEL_DISTRESSED = "Distressed / Needs Support"
LABEL_HIGH_RISK = "High Risk / Urgent"
LABEL_UNCLEAR = "Unclear / Need More Context"

# Confidence ceilings — see module docstring.
_CONFIDENCE_CEILINGS = {
    "high_risk_explicit": 0.90,
    "high_risk_implicit": 0.85,
    "high_risk_sensory": 0.80,
    "distressed": 0.70,
    "safe": 0.70,
    "unclear": 0.50,
}

# Signals classified as "explicit" High Risk (highest confidence tier).
_EXPLICIT_HIGH_RISK_SIGNALS = {
    "active_death_wish",
    "death_wish_via_physical_event",
}

# Signals classified as "sensory/dissociative" High Risk (slightly lower tier).
_SENSORY_HIGH_RISK_SIGNALS = {
    "depersonalization",
    "derealization",
    "perceptual_disturbance",
}

# Emotion mapping per risk_signal — uses annotation_guide §8.1 vocabulary.
_SIGNAL_TO_EMOTION = {
    # High Risk
    "passive_death_wish": "hopelessness",
    "active_death_wish": "hopelessness",
    "death_idealization": "hopelessness",
    "perceived_burdensomeness_with_leaving": "hopelessness",
    "existential_indifference": "hopelessness",
    "future_collapse": "hopelessness",
    "survivor_guilt_with_self_death_framing": "guilt",
    "death_wish_via_physical_event": "hopelessness",
    "depersonalization": "dissociation",
    "derealization": "dissociation",
    "perceptual_disturbance": "dissociation",
    # Distressed
    "hypervigilance": "fear",
    "intrusive_thoughts": "anxiety",
    "unresolved_grief": "grief",
    "avoidance": "fear",
    "somatic_response": "anxiety",
    "somatic_dissociation": "numbing",
    "emotional_suppression": "suppressed_grief",
    "rumination": "anxiety",
    "school_avoidance": "fear",
    "loss_of_interest": "anhedonia",
    "social_isolation": "withdrawal",
    "behavioral_change": "withdrawal",
    "secrecy_seeking": "anxiety",
    "family_tension": "fear",
    "displacement_shame": "shame",
    "secondary_stress_about_other": "helplessness",
    "environmental_helplessness": "collective_fear",
    # Unclear
    "minimal_response": "unknown",
    "undisclosed_event": "unknown",
    "caregiver_imposed_silence": "unknown",
    "vague_somatic": "unknown",
    "retrieval_block": "unknown",
    "deflection_to_caregiver": "unknown",
    # Safe
    "positive_affect": "joy",
    "future_orientation": "anticipation",
    "daily_life_engagement": "contentment",
    "emotional_resolution": "resolved_conflict",
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Match:
    """A single rule match found in the text."""

    category: str          # "high_risk" | "distressed" | "unclear" | "safe"
    signal: str            # e.g. "passive_death_wish"
    pattern: str           # the matched pattern (debug only)
    weight: float          # from lexicons
    co_signal_note: str = ""


@dataclass
class _MatchSet:
    """All matches found in a single utterance, organized by category."""

    high_risk: list[_Match] = field(default_factory=list)
    distressed: list[_Match] = field(default_factory=list)
    unclear: list[_Match] = field(default_factory=list)
    safe: list[_Match] = field(default_factory=list)
    hedge_detected: bool = False
    retraction_detected: bool = False

    def any_matches(self) -> bool:
        return bool(self.high_risk or self.distressed or self.unclear or self.safe)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Matches Arabic diacritics (tashkeel).
_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

# Replacement table for normalization (Alef variants → bare Alef; Ya/Alef Maqsura; Ta Marbuta → Ha; remove tatweel).
_NORMALIZATION_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    "ـ": "",
})


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic for robust matching against lexicon patterns.

    Strips diacritics, unifies Alef/Ya/Ta-Marbuta variants, removes tatweel
    and Arabic punctuation, collapses whitespace, lowercases Latin chars.
    """
    if not text:
        return ""
    text = _DIACRITICS_RE.sub("", text)
    text = text.translate(_NORMALIZATION_MAP)
    # Arabic + ASCII punctuation → whitespace so word-boundary matches work.
    text = re.sub(r"[،؟!؛.,?!;:\-—_(){}\[\]\"'`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split(" ") if t]


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

_LEXICON_CACHE: dict[str, Any] | None = None


def _load_lexicons(path: str = _DEFAULT_LEXICON_PATH) -> dict[str, Any]:
    """Load and cache the triage lexicons JSON."""
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    with open(path, "r", encoding="utf-8") as f:
        _LEXICON_CACHE = json.loads(f.read().lstrip('\ufeff'))
    return _LEXICON_CACHE


def reload_lexicons(path: str = _DEFAULT_LEXICON_PATH) -> None:
    """Force-reload the lexicons (useful for tests and live edits)."""
    global _LEXICON_CACHE
    _LEXICON_CACHE = None
    _load_lexicons(path)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _pattern_matches(pattern: str, normalized_text: str, tokens: list[str]) -> bool:
    """Match a single lexicon pattern against normalized text.

    Patterns are normalized the same way as input text before matching.
    Supports three prefixes:
      EXACT:   — full-token match (utterance is exactly the token, or token
                 appears as a standalone word AND text is very short)
      PHRASE:  — contiguous phrase substring match in normalized text
      (none)   — default substring match
    """
    if pattern.startswith("EXACT:"):
        bare = _normalize_arabic(pattern[len("EXACT:"):])
        # EXACT means the whole utterance equals the pattern (after normalization),
        # OR the utterance is very short (≤ 3 tokens) and contains the pattern as
        # a standalone token. This is what minimal_response really means.
        if normalized_text == bare:
            return True
        if len(tokens) <= 3 and bare in tokens:
            return True
        return False

    if pattern.startswith("PHRASE:"):
        bare = _normalize_arabic(pattern[len("PHRASE:"):])
        return bare in normalized_text

    bare = _normalize_arabic(pattern)
    return bare in normalized_text


def _find_matches(text: str, lexicons: dict[str, Any]) -> _MatchSet:
    """Scan the utterance and return all matched signals across categories."""
    normalized = _normalize_arabic(text)
    tokens = _tokenize(normalized)
    matches = _MatchSet()

    if not normalized:
        return matches

    # Iterate categories in priority order (high_risk first).
    for category in ("high_risk", "distressed", "unclear", "safe"):
        category_block = lexicons.get(category, {})
        bucket = getattr(matches, category)
        for signal_name, signal_def in category_block.items():
            if signal_name.startswith("_"):
                continue
            patterns = signal_def.get("patterns", [])
            weight = signal_def.get("weight", 0.5)
            co_note = signal_def.get("co_signal_note", "")
            for pattern in patterns:
                if _pattern_matches(pattern, normalized, tokens):
                    bucket.append(_Match(
                        category=category,
                        signal=signal_name,
                        pattern=pattern,
                        weight=weight,
                        co_signal_note=co_note,
                    ))
                    break  # one hit per signal is enough

    # Hedges & retraction (recorded only — do NOT change label per §6.3).
    for hedge_block_name, attr in (("hedges", "hedge_detected"),
                                   ("retraction", "retraction_detected")):
        block = lexicons.get(hedge_block_name, {})
        for pattern in block.get("patterns", []):
            if _pattern_matches(pattern, normalized, tokens):
                setattr(matches, attr, True)
                break

    return matches


# ---------------------------------------------------------------------------
# Co-signal resolution (specific exclusions from annotation_guide §5)
# ---------------------------------------------------------------------------

def _resolve_co_signals(matches: _MatchSet, normalized_text: str) -> _MatchSet:
    """Apply discriminative-pair rules from annotation_guide §5.

    These are the cases where a naive "any High Risk keyword → High Risk"
    rule would over-fire. We selectively remove High Risk matches that
    fail their co-signal requirement, while staying conservative (per P1).
    """
    # Pair D — emotional_suppression vs depersonalization.
    # If both fire together, depersonalization wins (already High Risk).
    # If only emotional_suppression fires, it stays Distressed (no change needed).

    # Burden language without explicit "leaving/death for self" reference.
    # The lexicon already includes co_signal_note; here we enforce it.
    # Specifically: if perceived_burdensomeness_with_leaving fires but NONE of
    # {leaving language, death language, sleep-no-wake} is present, demote.
    filtered_high_risk = []
    leaving_self_patterns = [
        _normalize_arabic(p) for p in [
            "اروح", "ما اصحى", "ما افيق", "اختفي", "ما اكون موجود",
            "اموت", "احسن اموت",
        ]
    ]
    for m in matches.high_risk:
        if m.signal == "perceived_burdensomeness_with_leaving":
            has_leaving = any(p in normalized_text for p in leaving_self_patterns)
            if not has_leaving:
                # Demote: this is just burden talk → distressed signal.
                matches.distressed.append(_Match(
                    category="distressed",
                    signal="emotional_suppression",  # closest distressed slot
                    pattern=m.pattern,
                    weight=0.50,
                    co_signal_note="Demoted from perceived_burdensomeness — no self-leaving co-signal.",
                ))
                continue
        if m.signal == "perceptual_disturbance" and "زي مش حصلت معي" in normalized_text:
            # Event-distance phrasing is memory/emotional numbing unless the
            # child also describes self/body/reality disownership.
            matches.distressed.append(_Match(
                category="distressed",
                signal="emotional_suppression",
                pattern=m.pattern,
                weight=0.55,
                co_signal_note="Demoted: event felt distant, not self/reality disowned.",
            ))
            continue
        if m.signal == "depersonalization" and any(
            p in normalized_text for p in ["بالقصه", "بكتب", "باسم بنت"]
        ):
            # Narrative distancing in writing/story play is Distressed unless
            # it is reported outside the story context as waking self-fragmentation.
            matches.distressed.append(_Match(
                category="distressed",
                signal="emotional_suppression",
                pattern=m.pattern,
                weight=0.55,
                co_signal_note="Demoted: dissociative metaphor limited to writing/story context.",
            ))
            continue
        filtered_high_risk.append(m)
    matches.high_risk = filtered_high_risk

    return matches


def _apply_safe_guards(matches: _MatchSet, normalized_text: str) -> _MatchSet:
    """Remove broad Safe hits when the child is negating enjoyment/attachment."""
    negative_safe_patterns = [
        "ما بحبها",
        "ما بحب اشوف",
        "بطلت احب",
        "ما عاد بحب",
        "ما احب",
    ]
    if any(p in normalized_text for p in negative_safe_patterns):
        matches.safe = [
            m for m in matches.safe
            if m.signal != "positive_affect" or m.pattern not in {"PHRASE:بحب", "PHRASE:أحب"}
        ]
    return matches


def _has_contained_grief_safe_signal(normalized_text: str) -> bool:
    """Grief can be Safe when the utterance contains caregiver/spiritual containment."""
    grief_or_longing = ["اشتقتلها", "اشتقتلو", "اشتقت لها", "اشتقت له"]
    containment = ["امي قالتلي", "بتشوفنا من فوق"]
    return (
        any(p in normalized_text for p in grief_or_longing)
        and any(p in normalized_text for p in containment)
    )


def _apply_high_risk_heuristics(matches: _MatchSet, normalized_text: str) -> _MatchSet:
    """Add narrow High Risk matches for metaphorical child-language cases.

    These heuristics cover Palestinian Arabic expressions that are too
    indirect for a literal lexicon phrase alone: forced self-nonexistence,
    body/reality unreality, identity fragmentation, and grief phrasing where
    the child explicitly questions their own continued existence.
    """
    if not normalized_text:
        return matches

    def add(signal: str, note: str, weight: float = 0.90) -> None:
        if any(m.signal == signal and m.pattern == "HEURISTIC" for m in matches.high_risk):
            return
        matches.high_risk.append(_Match(
            category="high_risk",
            signal=signal,
            pattern="HEURISTIC",
            weight=weight,
            co_signal_note=note,
        ))

    # Direct force-escalators: these are self-erasure, body-disownership, or
    # death-process monitoring, so Safe/Distressed co-signals must not dilute them.
    forced_patterns = {
        "ليش انا لساتي هون": "existential_indifference",
        "اتخلص منو": "active_death_wish",
        "اوقف اكون موجود": "passive_death_wish",
        "عشان اعرف لما يوقف": "death_wish_via_physical_event",
        "لو ما كان عندها ولاد": "perceived_burdensomeness_with_leaving",
    }
    for phrase, signal in forced_patterns.items():
        if phrase in normalized_text:
            add(signal, f"Forced High Risk phrase: {phrase}")

    # Body unreality, sensory absence, and identity-fragmentation metaphors are
    # treated as urgent because the child describes self/body function as absent
    # or not owned, not merely sadness or fear.
    body_unreality_patterns = [
        "ما بحس بيدي",
        "جسمي صار مش حاسس",
        "صرت رمادي",
        "ما في لون فيي",
        "حدا تاني داخلي",
        "ما في صوت لما احكي",
        "حلقي مش بشتغل",
    ]
    if any(p in normalized_text for p in body_unreality_patterns):
        add("depersonalization", "Body/self unreality heuristic", 0.85)

    sensory_absence_patterns = [
        "وما بوجعني",
        "ما بحس بالحر",
        "ولا البرد",
        "ما عاد في طعم",
    ]
    if any(p in normalized_text for p in sensory_absence_patterns) and any(
        p in normalized_text for p in ["جسمي", "بيدي", "بضربها", "بضربه", "طعم"]
    ):
        add("perceptual_disturbance", "Sensory absence heuristic", 0.80)

    # Derealization of an attachment figure is escalated only when a caregiver
    # or close relation is named, because bare "مش هو" can be ordinary contrast.
    caregiver_terms = ["ابوي", "ابي", "بابا", "امي", "اخوي", "اختي", "ستي", "جدي"]
    if "مش هو" in normalized_text and any(term in normalized_text for term in caregiver_terms):
        add("derealization", "Caregiver/attachment derealization heuristic", 0.85)

    # Grief/loss + self-reference + existential wording is not ordinary grief:
    # the child is placing their own existence inside the loss frame.
    grief_terms = ["راحوا", "راح", "مات", "ماتوا", "فقدت", "اشتقت", "ما رجع"]
    self_terms = ["انا", "اني", "لساتي", "هون", "موجود"]
    existential_terms = ["ليش انا لساتي هون", "لساتي هون", "ما عاد فيني فرق", "اوقف اكون موجود"]
    if (
        any(g in normalized_text for g in grief_terms)
        and any(s in normalized_text for s in self_terms)
        and any(e in normalized_text for e in existential_terms)
    ):
        add("existential_indifference", "Grief plus self-existence heuristic", 0.88)

    return matches


# ---------------------------------------------------------------------------
# Label decision
# ---------------------------------------------------------------------------

def _decide_label(matches: _MatchSet, normalized_text: str) -> tuple[str, str, float, str, str]:
    """Apply the four-label decision tree.

    Returns: (label, primary_risk_signal, confidence, review_notes, emotion_signal)

    `emotion_signal` is used internally to pick the emotion vocabulary token.
    For Safe utterances it carries the safe-bucket signal (e.g. "future_orientation")
    so the emotion mapping can return "anticipation"; for all other labels it
    equals `primary_risk_signal`. The `risk_signal` field in the public output
    stays "none" for Safe per annotation_guide §8.2.
    """
    tokens = _tokenize(normalized_text)
    short_utterance = len(tokens) <= 3

    # ---- High Risk path ----
    if matches.high_risk:
        # Pick the highest-weight High Risk match as primary.
        primary = max(matches.high_risk, key=lambda m: m.weight)
        if primary.signal in _EXPLICIT_HIGH_RISK_SIGNALS:
            conf = _CONFIDENCE_CEILINGS["high_risk_explicit"]
            tier_note = "explicit"
        elif primary.signal in _SENSORY_HIGH_RISK_SIGNALS:
            conf = _CONFIDENCE_CEILINGS["high_risk_sensory"]
            tier_note = "sensory/dissociative"
        else:
            conf = _CONFIDENCE_CEILINGS["high_risk_implicit"]
            tier_note = "implicit"

        notes = f"High Risk ({tier_note}): {primary.signal}"
        if len(matches.high_risk) > 1:
            others = ", ".join(sorted({m.signal for m in matches.high_risk if m.signal != primary.signal}))
            notes += f"; co-signals: {others}"
        if matches.hedge_detected:
            notes += "; HEDGE present (does not de-escalate per P1)"
        if matches.retraction_detected:
            notes += "; retraction present (does not de-escalate per §6.3)"
        if matches.distressed:
            notes += "; distressed context co-present"
        return LABEL_HIGH_RISK, primary.signal, conf, notes, primary.signal

    # ---- Safe override: contained grief ----
    if _has_contained_grief_safe_signal(normalized_text):
        notes = "Safe: grief/longing contained by caregiver or spiritual reassurance"
        return LABEL_SAFE, "none", _CONFIDENCE_CEILINGS["safe"], notes, "emotional_resolution"

    # ---- Distressed path ----
    if matches.distressed:
        primary = max(matches.distressed, key=lambda m: m.weight)
        conf = _CONFIDENCE_CEILINGS["distressed"]
        # If many distressed signals fire, slight bump (still bounded).
        if len(matches.distressed) >= 2:
            conf = min(conf, 0.70)
        notes = f"Distressed: {primary.signal}"
        if len(matches.distressed) > 1:
            others = ", ".join(sorted({m.signal for m in matches.distressed if m.signal != primary.signal}))
            notes += f"; co-signals: {others}"
        if matches.safe:
            notes += "; some safe markers also present"
        return LABEL_DISTRESSED, primary.signal, conf, notes, primary.signal

    # ---- Safe path ----
    if matches.safe:
        primary = max(matches.safe, key=lambda m: m.weight)
        conf = _CONFIDENCE_CEILINGS["safe"]
        notes = f"Safe: {primary.signal}"
        if len(matches.safe) > 1:
            others = ", ".join(sorted({m.signal for m in matches.safe if m.signal != primary.signal}))
            notes += f"; co-signals: {others}"
        # risk_signal is "none" per §8.2; emotion_signal carries the safe slot.
        return LABEL_SAFE, "none", conf, notes, primary.signal

    # ---- Unclear path (default) ----
    if matches.unclear:
        primary = max(matches.unclear, key=lambda m: m.weight)
        notes = f"Unclear: {primary.signal}"
        if primary.signal == "retrieval_block":
            notes += "; possible dissociative signal — flag for specialist"
        return LABEL_UNCLEAR, primary.signal, _CONFIDENCE_CEILINGS["unclear"], notes, primary.signal

    # No matches at all — either empty text or fully unrecognized.
    if not normalized_text:
        return LABEL_UNCLEAR, "minimal_response", _CONFIDENCE_CEILINGS["unclear"], "Empty utterance.", "minimal_response"
    if short_utterance:
        return LABEL_UNCLEAR, "minimal_response", _CONFIDENCE_CEILINGS["unclear"], "Very short utterance, no signals matched.", "minimal_response"
    # Longer text with no matches → still Unclear per P2.
    return LABEL_UNCLEAR, "minimal_response", _CONFIDENCE_CEILINGS["unclear"], "No lexicon signals matched.", "minimal_response"


# ---------------------------------------------------------------------------
# Cumulative rules (conversation-level, pre-alpha scope)
# ---------------------------------------------------------------------------

def _apply_cumulative_rules(
    current_label: str,
    current_signal: str,
    current_confidence: float,
    current_notes: str,
    current_match_set: _MatchSet,
    conversation_history: list[dict] | None,
) -> tuple[str, str, float, str, bool]:
    """Apply two simple cumulative rules from annotation_guide §6.

    Returns: (label, signal, confidence, notes, needs_review_override)

    Pre-alpha scope (deferred for later):
      - Full cumulative ambiguity (6.4)
      - Per-turn escalation_pattern computation
      - Session-level final risk
    """
    if not conversation_history:
        return current_label, current_signal, current_confidence, current_notes, False

    # Look only at prior CHILD turns (assistant turns shouldn't affect risk).
    prior_child_turns = [
        t for t in conversation_history
        if isinstance(t, dict) and t.get("role") == "user" and t.get("content")
    ]
    if not prior_child_turns:
        return current_label, current_signal, current_confidence, current_notes, False

    prior_had_high_risk = False
    for turn in prior_child_turns:
        prior_matches = _find_matches(turn["content"], _load_lexicons())
        # We don't re-apply co-signal resolution on history (kept simple in pre-alpha).
        if prior_matches.high_risk:
            prior_had_high_risk = True
            break

    if not prior_had_high_risk:
        return current_label, current_signal, current_confidence, current_notes, False

    # §6.3 — Retraction does not de-escalate.
    # If current utterance is a retraction and prior turn was High Risk,
    # the conversation-level label remains High Risk.
    if current_match_set.retraction_detected and current_label != LABEL_HIGH_RISK:
        new_notes = current_notes + " | §6.3: prior High Risk + current retraction → conversation remains High Risk"
        return (
            LABEL_HIGH_RISK,
            "active_death_wish",  # we can't know which; default to the safest catch
            _CONFIDENCE_CEILINGS["high_risk_implicit"],
            new_notes,
            True,
        )

    # §6.2 — Escalation across turns: prior High Risk + current lower-severity
    # should at minimum flag needs_review (Principle 1).
    if current_label != LABEL_HIGH_RISK:
        new_notes = current_notes + " | §6.2: prior turn was High Risk — flagging for review"
        return current_label, current_signal, current_confidence, new_notes, True

    return current_label, current_signal, current_confidence, current_notes, False


# ---------------------------------------------------------------------------
# needs_review policy
# ---------------------------------------------------------------------------

def _decide_needs_review(
    label: str,
    matches: _MatchSet,
    confidence: float,
    cumulative_override: bool,
) -> bool:
    """All High Risk requires review (per safety_rules.md). Other triggers:
       - hedges or retractions present
       - cumulative override fired
       - High Risk + Distressed signals co-occurring (mixed picture)
       - Unclear with retrieval_block signal
       - low confidence (≤ 0.5)
    """
    if label == LABEL_HIGH_RISK:
        return True
    if cumulative_override:
        return True
    if matches.hedge_detected or matches.retraction_detected:
        return True
    if confidence <= 0.50:
        return True
    if label == LABEL_UNCLEAR and any(m.signal == "retrieval_block" for m in matches.unclear):
        return True
    if matches.high_risk and matches.distressed:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    text: str,
    conversation_history: list[dict] | None = None,
    lexicon_path: str | None = None,
) -> dict[str, Any]:
    """Classify a single child utterance into the SENSE triage schema.

    Args:
        text: Preprocessed Palestinian Arabic utterance.
        conversation_history: Optional list of prior turns in the format
            [{"role": "user"|"assistant", "content": "..."}, ...].
            Used for the two cumulative rules described in the module docstring.
        lexicon_path: Optional override for the lexicon JSON path (for tests).

    Returns:
        Dict matching the placeholder contract, with additional fields for
        review traceability (matched_signals, review_notes, hedge_detected,
        retraction_detected). Orchestrator can ignore unknown fields.
    """
    if lexicon_path:
        reload_lexicons(lexicon_path)
    lexicons = _load_lexicons()

    normalized = _normalize_arabic(text or "")
    matches = _find_matches(text or "", lexicons)
    matches = _resolve_co_signals(matches, normalized)
    matches = _apply_high_risk_heuristics(matches, normalized)
    matches = _apply_safe_guards(matches, normalized)

    label, signal, confidence, notes, emotion_signal = _decide_label(matches, normalized)

    label, signal, confidence, notes, cumulative_override = _apply_cumulative_rules(
        label, signal, confidence, notes, matches, conversation_history
    )

    # Use emotion_signal for emotion lookup (for Safe utterances this differs
    # from risk_signal, which stays "none" per §8.2).
    emotion = _SIGNAL_TO_EMOTION.get(emotion_signal, "unknown")
    needs_review = _decide_needs_review(label, matches, confidence, cumulative_override)

    matched_signals_serialized = [
        {"category": m.category, "signal": m.signal, "weight": m.weight}
        for bucket in (matches.high_risk, matches.distressed, matches.unclear, matches.safe)
        for m in bucket
    ]

    return {
        "predicted_label": label,
        "predicted_emotion": emotion,
        "risk_signal": signal,
        "needs_review": needs_review,
        "confidence": round(confidence, 2),
        "source": "rule_based_triage_v1",
        "matched_signals": matched_signals_serialized,
        "review_notes": notes,
        "hedge_detected": matches.hedge_detected,
        "retraction_detected": matches.retraction_detected,
    }


# Backward-compatibility alias matching the placeholder's likely entry point.
def run_triage(text: str, conversation_history: list[dict] | None = None) -> dict[str, Any]:
    """Alias for `classify`. Use whichever name the orchestrator expects."""
    return classify(text, conversation_history=conversation_history)
  
def classify_triage(text, conversation_history=None):
    return classify(text, conversation_history=conversation_history)
