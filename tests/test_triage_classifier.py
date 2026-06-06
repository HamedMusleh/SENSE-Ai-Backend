"""
Tests for the SENSE rule-based triage classifier.

Coverage:
    - All four labels (Safe, Distressed, High Risk, Unclear)
    - All seven discriminative pairs from annotation_guide §5
    - Hedge / retraction handling (P1, §6.3)
    - Cumulative rules (§6.2, §6.3) for pre-alpha conversation support
    - Confidence ceilings
    - Schema completeness (output contract matches orchestrator expectations)

Run from project root:
    python -m pytest tests/test_triage_classifier.py -v
"""

from __future__ import annotations

import os
import sys

# Allow running directly without installing the package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import pytest  # noqa: E402

from ai_pipeline.triage.triage_classifier import (  # noqa: E402
    classify,
    LABEL_SAFE,
    LABEL_DISTRESSED,
    LABEL_HIGH_RISK,
    LABEL_UNCLEAR,
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "predicted_label",
    "predicted_emotion",
    "risk_signal",
    "needs_review",
    "confidence",
    "source",
}


def _assert_schema(result: dict) -> None:
    """Ensure the output contains everything the orchestrator expects."""
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing required keys: {missing}"
    assert isinstance(result["needs_review"], bool)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["source"] == "rule_based_triage_v1"


# ===========================================================================
# SAFE / REGULATED
# ===========================================================================

class TestSafeLabel:
    """Reference examples from annotation_guide §4.1."""

    def test_food_and_home_routine(self):
        # gold_001 style
        result = classify("اليوم أمي طبخت مقلوبة وأكلت كتير")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_SAFE
        assert result["needs_review"] is False
        assert result["confidence"] <= 0.70

    def test_fear_that_resolves(self):
        # gold_004 — fear with resolution
        result = classify("خايف شوي من الظلام لما الكهربا بتروح بس بعدين بنام")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_SAFE, (
            f"Expected Safe (resolution marker present), got {result['predicted_label']}"
        )

    def test_future_orientation_positive(self):
        # gold_007 — parental connection and anticipation
        result = classify("بابا قاللي بكرا رح يجيب لي دفتر تلوين")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_SAFE
        assert result["predicted_emotion"] == "anticipation"


# ===========================================================================
# DISTRESSED / NEEDS SUPPORT
# ===========================================================================

class TestDistressedLabel:
    """Reference examples from annotation_guide §4.2."""

    def test_hypervigilance_startle(self):
        # gold_013 — uncontrollable startle response
        result = classify("كل ما اسمع صوت عالي بركض عند امي مش قادر اوقف حالي")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_DISTRESSED
        assert result["risk_signal"] == "hypervigilance"

    def test_secondary_stress_about_sibling(self):
        # gold_025 — helplessness about other, not self
        result = classify("أخوي الصغير بعيط كل الليل ما عرفت كيف أساعده")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_DISTRESSED

    def test_trauma_linked_nightmares(self):
        result = classify("بصحى من كوابيس عن القصف كل ليلة")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_DISTRESSED

    def test_external_threat_fear_not_high_risk(self):
        # §5 Pair E — fear of dying from external threat is Distressed, NOT High Risk
        result = classify("بخاف نموت من القصف")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_DISTRESSED, (
            f"External-threat fear should be Distressed per §5 Pair E, "
            f"got {result['predicted_label']}"
        )


# ===========================================================================
# HIGH RISK / URGENT
# ===========================================================================

class TestHighRiskLabel:
    """Reference examples from annotation_guide §4.3."""

    def test_active_death_wish_explicit(self):
        result = classify("بدي أموت")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] == "active_death_wish"
        assert result["needs_review"] is True
        assert result["confidence"] == 0.90  # explicit ceiling

    def test_passive_death_wish(self):
        # gold_039 — "لو نمت وما صحيت كان أحسن"
        result = classify("لو نمت وما صحيت كان احسن")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] == "passive_death_wish"
        assert result["confidence"] == 0.85  # implicit ceiling

    def test_depersonalization(self):
        # gold_034 — body disownership
        result = classify("حاسس انو جسمي مش جسمي زي ما انا شايف حالي من برا")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] in ("depersonalization", "derealization")
        assert result["confidence"] == 0.80  # sensory ceiling

    def test_burdensomeness_with_leaving(self):
        # gold_037 — implicit death wish + perceived burdensomeness
        result = classify("رح يكون احسن لما اروح امي رح ترتاح")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] == "perceived_burdensomeness_with_leaving"

    def test_future_collapse(self):
        # gold_035 — finality marker
        result = classify("ما في اشي استناه خلص")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] == "future_collapse"

    def test_death_idealization(self):
        # gold_033 — "the dead are better off"
        result = classify("اللي ماتوا احسن منا هم ارتاحوا")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["risk_signal"] == "death_idealization"

    def test_death_wish_via_physical_event(self):
        # gold_040 — wish heart had stopped
        result = classify("حسيت انو قلبي وقف وتمنيت ما يرجع يشتغل")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK


# ===========================================================================
# UNCLEAR / NEED MORE CONTEXT
# ===========================================================================

class TestUnclearLabel:
    """Reference examples from annotation_guide §4.4."""

    def test_minimal_response(self):
        result = classify("تمام")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR

    def test_refusal_to_disclose(self):
        # gold_043
        result = classify("صار اشي بس ما بدي احكي عنو هون")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR
        assert result["risk_signal"] == "undisclosed_event"

    def test_caregiver_imposed_silence(self):
        # gold_046
        result = classify("امي قالتلي ما احكي عن هاد الموضوع")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR
        assert result["risk_signal"] == "caregiver_imposed_silence"

    def test_retrieval_block_flagged_for_review(self):
        # §4.4 — retrieval block may itself be dissociative; flag for review
        result = classify("لما تسالني بنسى كل اشي")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR
        assert result["needs_review"] is True, (
            "retrieval_block must be flagged for review per §4.4"
        )


# ===========================================================================
# DISCRIMINATIVE PAIRS — annotation_guide §5
# ===========================================================================

class TestDiscriminativePairs:

    # Pair A — grief about another vs death wish for self
    def test_pair_a_grief_about_other_is_distressed(self):
        result = classify("اشتقتلها كتير قلبي وجعني عليها")
        assert result["predicted_label"] == LABEL_DISTRESSED

    def test_pair_a_death_wish_for_self_is_high_risk(self):
        result = classify("بدي انام وما اصحى")
        assert result["predicted_label"] == LABEL_HIGH_RISK

    # Pair B — sleep difficulty vs death-in-sleep framing
    def test_pair_b_sleep_difficulty_is_distressed(self):
        result = classify("ما بقدر انام بفكر كتير")
        assert result["predicted_label"] == LABEL_DISTRESSED

    def test_pair_b_no_wake_marker_escalates(self):
        result = classify("بدي انام وما اصحى")
        assert result["predicted_label"] == LABEL_HIGH_RISK

    # Pair E — realistic fear vs death wish
    def test_pair_e_realistic_external_fear_not_high_risk(self):
        result = classify("بخاف يصير اشي ونموت من القصف")
        assert result["predicted_label"] == LABEL_DISTRESSED
        assert result["predicted_label"] != LABEL_HIGH_RISK


# ===========================================================================
# HEDGES & RETRACTIONS — Principle 1 / §6.3
# ===========================================================================

class TestHedgesAndRetractions:

    def test_hedge_does_not_de_escalate_high_risk(self):
        # The conflict case from our design discussion.
        result = classify("بدي اموت بس مش جدياً بس تعبت من القصف")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK, (
            "Per Principle 1, hedges do NOT de-escalate explicit death wishes."
        )
        assert result["hedge_detected"] is True
        assert "HEDGE" in result["review_notes"]
        assert result["needs_review"] is True

    def test_retraction_within_same_utterance_does_not_de_escalate(self):
        result = classify("بدي اموت لا مش هيك انا بس قلت هيك")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert result["retraction_detected"] is True


# ===========================================================================
# CUMULATIVE CONVERSATION RULES — §6.2 / §6.3 (pre-alpha scope)
# ===========================================================================

class TestCumulativeRules:

    def test_no_history_no_change(self):
        result = classify("اليوم لعبت مع صاحبي")
        assert result["predicted_label"] == LABEL_SAFE
        assert result["needs_review"] is False

    def test_prior_high_risk_flags_current_safe_for_review(self):
        # §6.2 — earlier safety does not cancel later disclosures, and vice versa:
        # if an earlier turn was High Risk, any later turn should at minimum be
        # flagged for review.
        history = [
            {"role": "user", "content": "بدي اموت"},
            {"role": "assistant", "content": "..."},
        ]
        result = classify("اليوم اكلت مع امي", conversation_history=history)
        _assert_schema(result)
        assert result["needs_review"] is True
        assert "§6.2" in result["review_notes"]

    def test_retraction_after_high_risk_turn_stays_high_risk(self):
        # §6.3 — retraction of a prior High Risk statement does NOT de-escalate.
        history = [
            {"role": "user", "content": "بدي اموت"},
            {"role": "assistant", "content": "..."},
        ]
        result = classify("لا مش هيك انا بس قلت هيك", conversation_history=history)
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_HIGH_RISK
        assert "§6.3" in result["review_notes"]


# ===========================================================================
# CONFIDENCE CEILINGS — design contract
# ===========================================================================

class TestConfidenceCeilings:

    def test_high_risk_explicit_is_0_90(self):
        result = classify("بدي اموت")
        assert result["confidence"] == 0.90

    def test_high_risk_implicit_is_0_85(self):
        result = classify("لو نمت وما صحيت كان احسن")
        assert result["confidence"] == 0.85

    def test_high_risk_sensory_is_0_80(self):
        result = classify("جسمي مش جسمي شايف حالي من برا")
        assert result["confidence"] == 0.80

    def test_distressed_capped_at_0_70(self):
        result = classify("كل ما اسمع صوت بركض عند امي مش قادر اوقف حالي")
        assert result["confidence"] <= 0.70

    def test_safe_capped_at_0_70(self):
        result = classify("اليوم لعبت مع صاحبي وكنت مبسوط")
        assert result["confidence"] <= 0.70

    def test_unclear_capped_at_0_50(self):
        result = classify("تمام")
        assert result["confidence"] <= 0.50


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self):
        result = classify("")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR

    def test_whitespace_only(self):
        result = classify("   ")
        _assert_schema(result)
        assert result["predicted_label"] == LABEL_UNCLEAR

    def test_diacritics_do_not_break_matching(self):
        # Same content with full tashkeel — should still match.
        result = classify("بَدِّي أَمُوت")
        assert result["predicted_label"] == LABEL_HIGH_RISK

    def test_alef_normalization(self):
        # Alef with hamza variants should normalize.
        result = classify("إلي ماتوا أحسن منا")
        # Should match death_idealization despite alef variants
        assert result["predicted_label"] == LABEL_HIGH_RISK


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))