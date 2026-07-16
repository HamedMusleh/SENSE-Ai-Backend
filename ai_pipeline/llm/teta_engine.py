"""
SENSE Teta AI Reply Engine
===========================

Generates the safe response shown to the child.

Flow (per the orchestrator design):

    processed_text + triage_result + conversation_history
            ↓
    response_strategy.build_strategy(triage_result)
            ↓
    ┌─────────────────────────────────┐
    │ if strategy.use_llm is False:   │
    │   return strategy.hard_coded     │  ← LLM is NEVER called for High Risk
    │ else:                            │
    │   call GPT-5 with with         │
    │   strategy_prompt injected       │
    └─────────────────────────────────┘

The hard-coded path is mandatory for High Risk per resources/safety_rules.md
("High Risk outputs require human review", and per project decision to
prevent LLM improvisation in the most sensitive cases).
"""

from __future__ import annotations

import json
from typing import Any

from ai_pipeline.llm.prompt_loader import load_prompt
from ai_pipeline.llm.response_strategy import build_strategy, Strategy
from ai_pipeline.core.openai_client import client, SENSE_VECTOR_STORE_ID


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_teta_reply(
    child_text: str,
    conversation_history: list[dict],
    triage_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate Teta's reply, routed by the triage result.

    Args:
        child_text: Preprocessed child utterance (Palestinian Arabic).
        conversation_history: Prior turns in OpenAI message format.
        triage_result: Output of triage_classifier. If None, treated as Unclear
            (cautious default — full LLM path with Unclear strategy).

    Returns:
        {
            "reply_text": str,
            "source": "hard_coded" | "llm_with_strategy" | "hard_coded_fallback",
            "strategy_label": str,
            "strategy_risk_signal": str,
            "follow_up_type": str | None,
        }

    Backward compatibility:
        The previous version returned a bare string. Callers that expected a
        string can read `result["reply_text"]`. A thin wrapper
        `ask_teta_reply_text()` is provided for legacy callers.
    """
    # If no triage result was supplied (e.g., during transition or testing),
    # synthesize a cautious Unclear placeholder so the strategy builder works.
    if triage_result is None:
        triage_result = {
            "predicted_label": "Unclear / Need More Context",
            "risk_signal": "minimal_response",
            "review_notes": "No triage_result supplied — defaulting to Unclear.",
        }

    strategy = build_strategy(triage_result)

    # --------------------- Hard-coded path (High Risk + fallback) ----------
    if not strategy.use_llm:
        return {
            "reply_text": strategy.hard_coded_response,
            "source": strategy.source,
            "strategy_label": strategy.label,
            "strategy_risk_signal": strategy.risk_signal,
            "follow_up_type": strategy.follow_up_type,
        }

    # --------------------- LLM path (Safe / Distressed / Unclear) ----------
    reply_text = _call_teta_llm(
        child_text=child_text,
        conversation_history=conversation_history,
        strategy=strategy,
    )

    return {
        "reply_text": reply_text,
        "source": strategy.source,
        "strategy_label": strategy.label,
        "strategy_risk_signal": strategy.risk_signal,
        "follow_up_type": None,
    }


def ask_teta_reply_text(
    child_text: str,
    conversation_history: list[dict],
    triage_result: dict[str, Any] | None = None,
) -> str:
    """Legacy-compatible wrapper that returns just the reply string."""
    return ask_teta_reply(child_text, conversation_history, triage_result)["reply_text"]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_teta_llm(
    child_text: str,
    conversation_history: list[dict],
    strategy: Strategy,
) -> str:
    """Call GPT-5 with the base system prompt + the strategy prompt.

    The strategy prompt is appended to the system prompt, NOT to the user
    message, so it's clearly framed as authoritative instruction rather than
    something the child said.
    """
    base_system_prompt = load_prompt("prompts/teta_system_prompt.txt")

    # Strategy prompt is appended with a clear separator. The classifier
    # context inside it is for Teta's awareness only — strategy_prompt.txt
    # explicitly tells Teta not to relay it to the child.
    combined_system_prompt = (
        f"{base_system_prompt}\n\n"
        f"---\n"
        f"{strategy.strategy_prompt}"
    )

    user_message = (
        f"Use the SENSE resources for mental-health triage, safety rules, "
        f"risk classification, and supportive response behavior.\n"
        f"For harmless general questions outside the resources, answer briefly "
        f"using general knowledge.\n"
        f"If any risk signal appears, prioritize SENSE safety rules.\n\n"
        f"Conversation so far:\n"
        f"{json.dumps(conversation_history, ensure_ascii=False)}\n\n"
        f"Latest child message:\n"
        f"{child_text}"
    )

    response = client.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [SENSE_VECTOR_STORE_ID],
                "max_num_results": 3,
            }
        ],
        input=[
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.output_text.strip()