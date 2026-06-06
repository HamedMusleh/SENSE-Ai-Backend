"""
Tests for the SENSE response strategy builder.

Coverage:
    - High Risk → hard-coded path (LLM NOT called)
    - Distressed / Safe / Unclear → LLM with strategy prompt
    - Risk-signal-specific response selection
    - Fallback behavior on malformed input
    - Strategy prompt contains the right instructions per label

Run from project root:
    python -m pytest tests/test_response_strategy.py -v
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import pytest  # noqa: E402

from ai_pipeline.llm.response_strategy import (  # noqa: E402
    build_strategy,
    Strategy,
    LABEL_SAFE,
    LABEL_DISTRESSED,
    LABEL_HIGH_RISK,
    LABEL_UNCLEAR,
)


# ===========================================================================
# HIGH RISK — hard-coded responses, LLM is NEVER called
# ===========================================================================

class TestHighRiskHardCoded:

    def test_active_death_wish_uses_hard_coded(self):
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "active_death_wish",
            "review_notes": "test",
        }
        s = build_strategy(triage)
        assert s.use_llm is False, "LLM must NOT be called for High Risk"
        assert s.source == "hard_coded"
        assert s.hard_coded_response, "Hard-coded response must be populated"
        assert s.strategy_prompt == "", "No strategy prompt should be set for hard-coded path"
        assert s.label == LABEL_HIGH_RISK
        assert s.risk_signal == "active_death_wish"

    def test_passive_death_wish_uses_hard_coded(self):
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "passive_death_wish",
        }
        s = build_strategy(triage)
        assert s.use_llm is False
        assert s.hard_coded_response

    def test_depersonalization_uses_hard_coded(self):
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "depersonalization",
        }
        s = build_strategy(triage)
        assert s.use_llm is False
        assert s.hard_coded_response

    def test_unknown_high_risk_signal_falls_back_to_generic(self):
        # If the classifier returns a High Risk label with an unrecognized
        # signal (shouldn't happen, but defensive), we still get a response.
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "some_made_up_signal_xyz",
        }
        s = build_strategy(triage)
        assert s.use_llm is False
        assert s.hard_coded_response  # generic_high_risk fallback fires

    def test_high_risk_response_has_no_clinical_terms(self):
        # Spot-check the active_death_wish response.
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "active_death_wish",
        }
        s = build_strategy(triage)
        text = s.hard_coded_response
        forbidden_terms = ["PTSD", "depression", "anxiety disorder", "trauma diagnosis",
                           "اكتئاب", "قلق نفسي", "صدمة نفسية"]
        for term in forbidden_terms:
            assert term not in text, f"Forbidden clinical term '{term}' found in High Risk response"

    def test_high_risk_response_includes_no_confidentiality_promise(self):
        # safety_rules.md forbids promising confidentiality.
        triage = {
            "predicted_label": LABEL_HIGH_RISK,
            "risk_signal": "active_death_wish",
        }
        s = build_strategy(triage)
        text = s.hard_coded_response
        forbidden_promises = ["سر بيني وبينك", "ما حدا رح يعرف", "بضل سر"]
        for phrase in forbidden_promises:
            assert phrase not in text, f"Forbidden confidentiality promise '{phrase}' found"


# ===========================================================================
# DISTRESSED — LLM with strategy prompt
# ===========================================================================

class TestDistressedStrategy:

    def test_distressed_uses_llm(self):
        triage = {
            "predicted_label": LABEL_DISTRESSED,
            "risk_signal": "hypervigilance",
            "review_notes": "Distressed: hypervigilance",
        }
        s = build_strategy(triage)
        assert s.use_llm is True
        assert s.source == "llm_with_strategy"
        assert s.hard_coded_response == ""
        assert s.strategy_prompt, "Strategy prompt must be loaded"

    def test_distressed_prompt_includes_validation_instruction(self):
        triage = {"predicted_label": LABEL_DISTRESSED, "risk_signal": "hypervigilance"}
        s = build_strategy(triage)
        # The strategy prompt should mention validation as the first action.
        assert "Validate" in s.strategy_prompt or "validation" in s.strategy_prompt.lower()

    def test_distressed_prompt_forbids_direct_trauma_questions(self):
        triage = {"predicted_label": LABEL_DISTRESSED, "risk_signal": "intrusive_thoughts"}
        s = build_strategy(triage)
        assert "FORBIDDEN" in s.strategy_prompt or "forbidden" in s.strategy_prompt.lower()

    def test_distressed_prompt_injects_signal_context(self):
        triage = {
            "predicted_label": LABEL_DISTRESSED,
            "risk_signal": "intrusive_thoughts",
            "review_notes": "Distressed: intrusive_thoughts",
        }
        s = build_strategy(triage)
        assert "intrusive_thoughts" in s.strategy_prompt
        # The {placeholder} should have been replaced.
        assert "{risk_signal_context}" not in s.strategy_prompt


# ===========================================================================
# SAFE — LLM with engagement strategy
# ===========================================================================

class TestSafeStrategy:

    def test_safe_uses_llm(self):
        triage = {
            "predicted_label": LABEL_SAFE,
            "risk_signal": "none",
            "review_notes": "Safe: positive_affect",
        }
        s = build_strategy(triage)
        assert s.use_llm is True
        assert s.strategy_prompt

    def test_safe_prompt_does_not_project_distress(self):
        triage = {"predicted_label": LABEL_SAFE, "risk_signal": "none"}
        s = build_strategy(triage)
        # The Safe strategy prompt should warn against projecting distress.
        assert "project distress" in s.strategy_prompt.lower() or \
               "do not project" in s.strategy_prompt.lower()


# ===========================================================================
# UNCLEAR — LLM with cautious follow-up strategy
# ===========================================================================

class TestUnclearStrategy:

    def test_unclear_uses_llm(self):
        triage = {
            "predicted_label": LABEL_UNCLEAR,
            "risk_signal": "minimal_response",
        }
        s = build_strategy(triage)
        assert s.use_llm is True
        assert s.strategy_prompt

    def test_unclear_prompt_warns_against_pushing(self):
        triage = {"predicted_label": LABEL_UNCLEAR, "risk_signal": "undisclosed_event"}
        s = build_strategy(triage)
        prompt_lower = s.strategy_prompt.lower()
        # Should warn against pressuring the child.
        assert "push" in prompt_lower or "pry" in prompt_lower or "pressure" in prompt_lower

    def test_unclear_retrieval_block_special_case_documented(self):
        # The retrieval_block signal has a special handling note in the prompt.
        triage = {"predicted_label": LABEL_UNCLEAR, "risk_signal": "retrieval_block"}
        s = build_strategy(triage)
        # The prompt template mentions retrieval_block by name as a special case.
        assert "retrieval_block" in s.strategy_prompt


# ===========================================================================
# DEFENSIVE BEHAVIOR
# ===========================================================================

class TestDefensiveBehavior:

    def test_none_triage_result_falls_back_safely(self):
        # When the orchestrator hasn't filled triage_result yet, we should
        # still get a usable strategy (cautious default).
        # The contract: non-dict input → hard_coded_fallback.
        s = build_strategy(None)  # type: ignore[arg-type]
        assert s.use_llm is False
        assert s.source == "hard_coded_fallback"
        assert s.hard_coded_response, "Emergency fallback must be present"

    def test_malformed_triage_result_falls_back(self):
        s = build_strategy("not a dict")  # type: ignore[arg-type]
        assert s.use_llm is False
        assert s.source == "hard_coded_fallback"

    def test_unknown_label_treated_as_unclear(self):
        # If the classifier somehow returns a label we don't recognize,
        # we treat it conservatively as Unclear (LLM with Unclear prompt).
        triage = {
            "predicted_label": "Some Future Label",
            "risk_signal": "minimal_response",
        }
        s = build_strategy(triage)
        assert s.use_llm is True
        assert s.label == LABEL_UNCLEAR

    def test_missing_risk_signal_defaults_safely(self):
        # If risk_signal key is missing entirely.
        triage = {"predicted_label": LABEL_DISTRESSED}
        s = build_strategy(triage)
        assert s.use_llm is True
        assert s.risk_signal == "minimal_response"  # documented default

    def test_high_risk_with_missing_risk_signal_uses_generic(self):
        triage = {"predicted_label": LABEL_HIGH_RISK}
        s = build_strategy(triage)
        assert s.use_llm is False
        assert s.hard_coded_response  # generic_high_risk fires


# ===========================================================================
# SAFETY INVARIANTS — must hold for every High Risk case
# ===========================================================================

class TestSafetyInvariants:
    """These invariants are the core safety guarantee of the system.
    If any of these break, the safety contract is violated."""

    HIGH_RISK_SIGNALS = [
        "active_death_wish",
        "passive_death_wish",
        "death_idealization",
        "perceived_burdensomeness_with_leaving",
        "existential_indifference",
        "future_collapse",
        "survivor_guilt_with_self_death_framing",
        "death_wish_via_physical_event",
        "depersonalization",
        "derealization",
        "perceptual_disturbance",
    ]

    @pytest.mark.parametrize("signal", HIGH_RISK_SIGNALS)
    def test_every_high_risk_signal_skips_llm(self, signal):
        """No High Risk signal should ever invoke the LLM."""
        triage = {"predicted_label": LABEL_HIGH_RISK, "risk_signal": signal}
        s = build_strategy(triage)
        assert s.use_llm is False, f"LLM was invoked for High Risk signal '{signal}'"

    @pytest.mark.parametrize("signal", HIGH_RISK_SIGNALS)
    def test_every_high_risk_signal_returns_nonempty_response(self, signal):
        """Every High Risk signal must produce a non-empty safe response."""
        triage = {"predicted_label": LABEL_HIGH_RISK, "risk_signal": signal}
        s = build_strategy(triage)
        assert len(s.hard_coded_response) > 10, (
            f"High Risk signal '{signal}' produced empty/too-short response"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))