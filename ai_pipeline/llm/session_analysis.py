"""
SENSE Final Session Analysis
==============================

Analyzes the full conversation_history after the child-facing session ends
and generates a structured specialist-facing report.

This module is called AFTER the session — not during it.
The output is for a human specialist or reviewer, NOT for the child.

Entry points:
    generate_session_report(conversation_history) -> dict
    print_specialist_report(report)               -> None (console)

Output schema:
    See prompts/final_analysis_prompt.txt for the full JSON schema
    that GPT returns. This module wraps it with session-level metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from ai_pipeline.llm.prompt_loader import load_prompt
from ai_pipeline.core.openai_client import client, SENSE_VECTOR_STORE_ID

# ---------------------------------------------------------------------------
# Risk level ordering (for trajectory computation)
# ---------------------------------------------------------------------------

_RISK_ORDER = {
    "Green": 0,
    "Safe / Regulated": 0,
    "Unclear": 1,
    "Unclear / Need More Context": 1,
    "Yellow": 2,
    "Distressed / Needs Support": 2,
    "Red": 3,
    "High Risk / Urgent": 3,
}

_TRAJECTORY_LABELS = {
    "stable_safe": "stable — no significant distress",
    "stable_distressed": "stable — persistent distress",
    "escalating": "escalating ⚠️",
    "de_escalating": "de-escalating",
    "mixed": "mixed — fluctuating signals",
    "single_turn": "single turn — insufficient for trajectory",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_child_turns(conversation_history: list[dict]) -> list[dict]:
    """Pull the child turns out of the orchestrator's rich history format."""
    turns = []
    for turn in conversation_history:
        child_text = turn.get("child_text") or turn.get("content", "")
        if not child_text:
            continue
        turns.append({
            "turn": turn.get("turn", len(turns) + 1),
            "child_text": child_text,
            "triage_result": turn.get("triage_result", {}),
            "assistant_reply": turn.get("assistant_reply", ""),
            "response_source": turn.get("response_source", ""),
        })
    return turns


def _compute_trajectory(child_turns: list[dict]) -> str:
    """Compute the risk trajectory across turns."""
    if len(child_turns) <= 1:
        return "single_turn"

    levels = []
    for t in child_turns:
        triage = t.get("triage_result", {})
        label = triage.get("predicted_label", "Unclear / Need More Context")
        levels.append(_RISK_ORDER.get(label, 1))

    if all(l == levels[0] for l in levels):
        if levels[0] <= 1:
            return "stable_safe"
        return "stable_distressed"

    # Check if generally increasing or decreasing
    increases = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i - 1])
    decreases = sum(1 for i in range(1, len(levels)) if levels[i] < levels[i - 1])

    if increases > 0 and decreases == 0:
        return "escalating"
    if decreases > 0 and increases == 0:
        return "de_escalating"
    return "mixed"


def _find_peak_risk(child_turns: list[dict]) -> dict:
    """Find the turn with the highest risk level."""
    peak = {"turn": None, "label": "Unclear / Need More Context",
            "level": 0, "text": "", "signal": ""}
    for t in child_turns:
        triage = t.get("triage_result", {})
        label = triage.get("predicted_label", "Unclear / Need More Context")
        level = _RISK_ORDER.get(label, 1)
        if level >= peak["level"]:
            peak = {
                "turn": t["turn"],
                "label": label,
                "level": level,
                "text": t["child_text"],
                "signal": triage.get("risk_signal", ""),
            }
    return peak


def _collect_signals(child_turns: list[dict]) -> list[str]:
    """Collect all unique risk signals detected across the session."""
    signals = []
    for t in child_turns:
        sig = t.get("triage_result", {}).get("risk_signal", "")
        if sig and sig not in ("minimal_response", "none", "") and sig not in signals:
            signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# GPT call
# ---------------------------------------------------------------------------

def _call_gpt_analysis(conversation_history: list[dict]) -> dict[str, Any]:
    """Send conversation_history to GPT-5 for structured analysis."""
    system_prompt = load_prompt("prompts/final_analysis_prompt.txt")

    user_content = (
        "Use the SENSE annotation guide, safety rules, and triage labels "
        "from the uploaded resources.\n"
        "Apply the rules strictly.\n"
        "Return ONLY valid JSON — no markdown, no explanation outside the JSON.\n\n"
        "Conversation history:\n"
        + json.dumps(conversation_history, ensure_ascii=False, indent=2)
    )

    response = client.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [SENSE_VECTOR_STORE_ID],
                "max_num_results": 5,
            }
        ],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.output_text.strip()

    # Strip markdown fences if GPT added them despite instructions.
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return a degraded report so the pipeline doesn't crash.
        return {
            "overall_summary": raw,
            "key_patterns_across_conversation": [],
            "concerning_phrases": [],
            "final_label": "Unclear / Need More Context",
            "final_risk_level": "Unclear",
            "final_reason": "GPT returned non-JSON output.",
            "recommendation": "Manual review required.",
            "specialist_notes": raw,
            "safety_disclaimer": "This is triage screening only, not a clinical diagnosis.",
            "message_reports": [],
            "_parse_error": True,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_session_report(
    conversation_history: list[dict],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate a structured specialist report from the full session.

    Args:
        conversation_history: The orchestrator's conversation_history list.
            Each turn should have at minimum:
                - child_text (str)
                - triage_result (dict from triage_classifier)
                - assistant_reply (str)
        session_id: Optional identifier. Auto-generated from timestamp if None.

    Returns:
        A dict combining:
            - session metadata (id, timestamp, turns, trajectory, peak risk)
            - GPT-generated per-turn analysis and overall summary
            - specialist recommendation and disclaimer
    """
    if session_id is None:
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    child_turns = _extract_child_turns(conversation_history)
    trajectory = _compute_trajectory(child_turns)
    peak = _find_peak_risk(child_turns)
    signals = _collect_signals(child_turns)

    # Build a clean history for GPT (only what it needs).
    gpt_history = [
        {
            "turn": t["turn"],
            "child_text": t["child_text"],
            "triage_label": t["triage_result"].get("predicted_label", ""),
            "risk_signal": t["triage_result"].get("risk_signal", ""),
            "confidence": t["triage_result"].get("confidence", 0),
            "assistant_reply": t["assistant_reply"],
            "response_source": t["response_source"],
        }
        for t in child_turns
    ]

    gpt_report = _call_gpt_analysis(gpt_history)

    needs_immediate = (
        peak["label"] == "High Risk / Urgent"
        or any(
            t.get("triage_result", {}).get("needs_review", False)
            for t in child_turns
        )
    )

    report = {
        # Session metadata
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "total_turns": len(child_turns),

        # Rule-based analysis (from triage_classifier)
        "risk_trajectory": trajectory,
        "risk_trajectory_label": _TRAJECTORY_LABELS.get(trajectory, trajectory),
        "peak_risk_label": peak["label"],
        "peak_risk_turn": peak["turn"],
        "peak_risk_text": peak["text"],
        "peak_risk_signal": peak["signal"],
        "signals_detected": signals,
        "needs_immediate_review": needs_immediate,

        # Turn-level summary from rule-based classifier
        "turn_summary": [
            {
                "turn": t["turn"],
                "child_text": t["child_text"],
                "label": t["triage_result"].get("predicted_label", ""),
                "signal": t["triage_result"].get("risk_signal", ""),
                "confidence": t["triage_result"].get("confidence", 0),
                "needs_review": t["triage_result"].get("needs_review", False),
                "assistant_reply": t["assistant_reply"],
                "response_source": t["response_source"],
            }
            for t in child_turns
        ],

        # GPT-generated specialist content
        "gpt_analysis": gpt_report,

        # Top-level convenience fields from GPT output
        "final_label": gpt_report.get("final_label", "Unclear / Need More Context"),
        "final_risk_level": gpt_report.get("final_risk_level", "Unclear"),
        "overall_summary": gpt_report.get("overall_summary", ""),
        "recommendation": gpt_report.get("recommendation", ""),
        "specialist_notes": gpt_report.get("specialist_notes", ""),
        "concerning_phrases": gpt_report.get("concerning_phrases", []),
        "key_patterns": gpt_report.get("key_patterns_across_conversation", []),
        "safety_disclaimer": gpt_report.get(
            "safety_disclaimer",
            "This is triage screening only, not a clinical diagnosis."
        ),
    }

    return report


# ---------------------------------------------------------------------------
# Specialist report printer
# ---------------------------------------------------------------------------

_RISK_EMOJI = {
    "Green": "🟢",
    "Yellow": "🟡",
    "Red": "🔴",
    "Unclear": "⚪",
    "Safe / Regulated": "🟢",
    "Distressed / Needs Support": "🟡",
    "High Risk / Urgent": "🔴",
    "Unclear / Need More Context": "⚪",
}

_TRAJ_EMOJI = {
    "escalating": "📈 ⚠️",
    "de_escalating": "📉",
    "stable_safe": "➡️",
    "stable_distressed": "➡️ ⚠️",
    "mixed": "↕️",
    "single_turn": "•",
}


def print_specialist_report(report: dict[str, Any]) -> None:
    """Print a formatted specialist report to the console."""
    sep = "═" * 70
    thin = "─" * 70

    print()
    print(sep)
    print("📊  SENSE SESSION REPORT — SPECIALIST VIEW")
    print(sep)
    print(f"  Session ID   : {report.get('session_id', 'N/A')}")
    print(f"  Generated    : {report.get('timestamp', 'N/A')}")
    print(f"  Total Turns  : {report.get('total_turns', 0)}")
    print(thin)

    # Risk trajectory
    traj = report.get("risk_trajectory", "")
    traj_label = report.get("risk_trajectory_label", traj)
    traj_emoji = _TRAJ_EMOJI.get(traj, "•")
    peak_label = report.get("peak_risk_label", "")
    peak_emoji = _RISK_EMOJI.get(peak_label, "⚪")

    print(f"  Risk Trajectory : {traj_emoji}  {traj_label}")
    print(f"  Peak Risk       : {peak_emoji}  {peak_label}"
          f"  (Turn {report.get('peak_risk_turn', '?')})")
    print(f"  Peak Text       : \"{report.get('peak_risk_text', '')}\"")

    signals = report.get("signals_detected", [])
    if signals:
        print(f"  Signals         : {', '.join(signals)}")

    if report.get("needs_immediate_review"):
        print()
        print("  ⚠️  IMMEDIATE SPECIALIST REVIEW REQUIRED")

    print(thin)

    # Turn-by-turn
    print("  TURN-BY-TURN SUMMARY:")
    print()
    for t in report.get("turn_summary", []):
        label = t.get("label", "")
        emoji = _RISK_EMOJI.get(label, "⚪")
        short_label = label.split(" /")[0] if "/" in label else label
        conf = t.get("confidence", 0)
        sig = t.get("signal", "")
        review = " 👁️" if t.get("needs_review") else ""
        src = " [hard-coded]" if t.get("response_source") == "hard_coded" else ""
        print(f"  Turn {t.get('turn', '?'):>2}  {emoji}  {short_label:<22}"
              f"  {sig:<30}  conf={conf:.2f}{review}{src}")
        print(f"          Child: \"{t.get('child_text', '')}\"")
        print(f"          Teta : \"{t.get('assistant_reply', '')}\"")
        print()

    print(thin)

    # GPT analysis
    print("  OVERALL SUMMARY (GPT):")
    print(f"  {report.get('overall_summary', '')}")
    print()

    patterns = report.get("key_patterns", [])
    if patterns:
        print("  KEY PATTERNS:")
        for p in patterns:
            print(f"  • {p}")
        print()

    phrases = report.get("concerning_phrases", [])
    if phrases:
        print("  CONCERNING PHRASES:")
        for p in phrases:
            print(f"  ⚠️  {p}")
        print()

    final_label = report.get("final_label", "")
    final_emoji = _RISK_EMOJI.get(final_label, "⚪")
    print(thin)
    print(f"  FINAL LABEL  : {final_emoji}  {final_label}")
    print(f"  FINAL REASON : {report.get('final_reason', '')}")
    print()
    print(f"  RECOMMENDATION:")
    print(f"  {report.get('recommendation', '')}")
    print()
    print(f"  SPECIALIST NOTES:")
    print(f"  {report.get('specialist_notes', '')}")
    print()
    print(thin)
    print(f"  ⚠️  {report.get('safety_disclaimer', '')}")
    print(sep)
    print()