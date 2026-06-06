"""
SENSE Response Strategy Builder
================================

Maps a triage_result (from triage_classifier) into a response strategy
that tells the Teta engine HOW to respond.

This module enforces the safety policy from resources/safety_rules.md:

    - High Risk → hard-coded safe response from a vetted JSON list.
                  The LLM does NOT generate anything in this case.
                  This protects against LLM improvisation in the most
                  sensitive cases.

    - Distressed / Safe / Unclear → LLM generates a response, but the
                  prompt is augmented with a strategy-specific instruction
                  block loaded from prompts/strategy_*.txt.

Public API:

    build_strategy(triage_result) -> Strategy

Returns a Strategy object the orchestrator and teta_engine can use:

    Strategy(
        use_llm: bool,                # False for High Risk
        hard_coded_response: str,     # filled if use_llm=False
        strategy_prompt: str,         # filled if use_llm=True (preformatted)
        label: str,                   # for logging
        risk_signal: str,             # for logging
        follow_up_type: str | None,   # for hard-coded responses, otherwise None
        source: str,                  # "hard_coded" | "llm_with_strategy"
    )
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DEFAULT_HIGH_RISK_RESPONSES_PATH = os.path.join(
    _PROJECT_ROOT, "resources", "high_risk_safe_responses.json"
)

_DEFAULT_PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")

# Labels (must match triage_classifier exactly).
LABEL_SAFE = "Safe / Regulated"
LABEL_DISTRESSED = "Distressed / Needs Support"
LABEL_HIGH_RISK = "High Risk / Urgent"
LABEL_UNCLEAR = "Unclear / Need More Context"

# Strategy prompt files per label.
_STRATEGY_PROMPT_FILES = {
    LABEL_SAFE: "strategy_safe.txt",
    LABEL_DISTRESSED: "strategy_distressed.txt",
    LABEL_UNCLEAR: "strategy_unclear.txt",
}

# Ultimate fallback if everything fails — short, safe, non-clinical.
_HARDCODED_EMERGENCY_FALLBACK = (
    "أنا هون معك. اللي بتحس فيه مهم. خليك مع حدا كبير بتحبه هلأ."
)


# ---------------------------------------------------------------------------
# Strategy data class
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """The response strategy for the current turn."""

    use_llm: bool
    label: str
    risk_signal: str
    source: str  # "hard_coded" | "llm_with_strategy" | "hard_coded_fallback"

    # Only one of these will be populated, depending on `use_llm`:
    hard_coded_response: str = ""
    strategy_prompt: str = ""

    # Optional metadata
    follow_up_type: str | None = None
    review_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_llm": self.use_llm,
            "label": self.label,
            "risk_signal": self.risk_signal,
            "source": self.source,
            "hard_coded_response": self.hard_coded_response,
            "strategy_prompt_present": bool(self.strategy_prompt),
            "follow_up_type": self.follow_up_type,
            "review_notes": self.review_notes,
        }


# ---------------------------------------------------------------------------
# Caches (loaded once per process)
# ---------------------------------------------------------------------------

_HIGH_RISK_CACHE: dict[str, Any] | None = None
_PROMPT_CACHE: dict[str, str] = {}


def _load_high_risk_responses(path: str = _DEFAULT_HIGH_RISK_RESPONSES_PATH) -> dict[str, Any]:
    global _HIGH_RISK_CACHE
    if _HIGH_RISK_CACHE is not None:
        return _HIGH_RISK_CACHE
    with open(path, "r", encoding="utf-8") as f:
        _HIGH_RISK_CACHE = json.load(f)
    return _HIGH_RISK_CACHE


def _load_strategy_prompt(filename: str, prompts_dir: str = _DEFAULT_PROMPTS_DIR) -> str:
    cache_key = os.path.join(prompts_dir, filename)
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    with open(cache_key, "r", encoding="utf-8") as f:
        content = f.read()
    _PROMPT_CACHE[cache_key] = content
    return content


def reload_resources(
    high_risk_path: str = _DEFAULT_HIGH_RISK_RESPONSES_PATH,
    prompts_dir: str = _DEFAULT_PROMPTS_DIR,
) -> None:
    """Force-reload all cached resources (for tests and live edits)."""
    global _HIGH_RISK_CACHE, _PROMPT_CACHE
    _HIGH_RISK_CACHE = None
    _PROMPT_CACHE = {}
    _load_high_risk_responses(high_risk_path)
    for fname in _STRATEGY_PROMPT_FILES.values():
        _load_strategy_prompt(fname, prompts_dir)


# ---------------------------------------------------------------------------
# High Risk response selection
# ---------------------------------------------------------------------------

def _select_high_risk_response(risk_signal: str) -> tuple[str, str | None]:
    """Pick a hard-coded safe response for the given risk_signal.

    Returns: (response_text, follow_up_type)

    Selection order:
      1. Exact match on risk_signal.
      2. "generic_high_risk" fallback.
      3. "_fallback_on_error" if even the JSON is malformed.

    Variants for the same signal are chosen at random to avoid repetition
    when the same signal fires across multiple turns.
    """
    try:
        responses = _load_high_risk_responses()
    except (FileNotFoundError, json.JSONDecodeError):
        return _HARDCODED_EMERGENCY_FALLBACK, None

    # Try exact signal match.
    signal_block = responses.get(risk_signal)
    if signal_block and "responses" in signal_block:
        variants = signal_block["responses"]
        if variants:
            chosen = random.choice(variants)
            return chosen["text"], chosen.get("follow_up_type")

    # Fall back to generic_high_risk.
    generic = responses.get("generic_high_risk")
    if generic and "responses" in generic:
        variants = generic["responses"]
        if variants:
            chosen = random.choice(variants)
            return chosen["text"], chosen.get("follow_up_type")

    # Last-resort fallback from the JSON itself.
    fallback = responses.get("_fallback_on_error", {})
    return fallback.get("text", _HARDCODED_EMERGENCY_FALLBACK), None


# ---------------------------------------------------------------------------
# LLM strategy prompt builder
# ---------------------------------------------------------------------------

def _build_llm_strategy_prompt(label: str, risk_signal: str, review_notes: str) -> str:
    """Load the strategy template for the label and inject signal context."""
    filename = _STRATEGY_PROMPT_FILES.get(label)
    if not filename:
        # Shouldn't happen — caller only passes Distressed/Safe/Unclear here.
        # Fall back to Distressed (most cautious of the three).
        filename = _STRATEGY_PROMPT_FILES[LABEL_DISTRESSED]

    template = _load_strategy_prompt(filename)

    # Build a brief, non-clinical context note for Teta's awareness only.
    # The strategy prompt explicitly forbids relaying this to the child.
    context_line = f"- Signal detected: {risk_signal}"
    if review_notes:
        context_line += f"\n- Reviewer note: {review_notes}"

    return template.replace("{risk_signal_context}", context_line)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_strategy(triage_result: dict[str, Any]) -> Strategy:
    """Build the response strategy from a triage_result.

    Args:
        triage_result: Output of triage_classifier.classify(), containing at
            minimum `predicted_label` and `risk_signal`. Optional fields like
            `review_notes` are passed through to the LLM context.

    Returns:
        A Strategy object. For High Risk, `use_llm=False` and the response
        is already chosen. For all other labels, `use_llm=True` and the
        strategy_prompt is preformatted and ready to feed to Teta.
    """
    if not isinstance(triage_result, dict):
        # Defensive: malformed input → emergency fallback.
        return Strategy(
            use_llm=False,
            label="UNKNOWN",
            risk_signal="UNKNOWN",
            source="hard_coded_fallback",
            hard_coded_response=_HARDCODED_EMERGENCY_FALLBACK,
            review_notes="triage_result was not a dict — emergency fallback used.",
        )

    label = triage_result.get("predicted_label", LABEL_UNCLEAR)
    risk_signal = triage_result.get("risk_signal", "minimal_response")
    review_notes = triage_result.get("review_notes", "")

    # ----- High Risk: hard-coded path, NEVER touches LLM -----
    if label == LABEL_HIGH_RISK:
        response_text, follow_up_type = _select_high_risk_response(risk_signal)
        return Strategy(
            use_llm=False,
            label=label,
            risk_signal=risk_signal,
            source="hard_coded",
            hard_coded_response=response_text,
            follow_up_type=follow_up_type,
            review_notes=review_notes,
        )

    # ----- Safe / Distressed / Unclear: LLM with strategy prompt -----
    if label not in _STRATEGY_PROMPT_FILES:
        # Unknown label — treat conservatively as Unclear.
        label = LABEL_UNCLEAR

    try:
        strategy_prompt = _build_llm_strategy_prompt(label, risk_signal, review_notes)
        return Strategy(
            use_llm=True,
            label=label,
            risk_signal=risk_signal,
            source="llm_with_strategy",
            strategy_prompt=strategy_prompt,
            review_notes=review_notes,
        )
    except (FileNotFoundError, IOError):
        # Prompt file missing — fall back to a safe hard-coded message
        # rather than letting the LLM run with no strategy guidance.
        return Strategy(
            use_llm=False,
            label=label,
            risk_signal=risk_signal,
            source="hard_coded_fallback",
            hard_coded_response=_HARDCODED_EMERGENCY_FALLBACK,
            review_notes="Strategy prompt file missing — emergency fallback used.",
        )