"""
AI Pipeline Adapter
===================
Owner: Student 2 (Backend)

This is the ONLY place in the backend that talks to ai_pipeline/.
It calls the AI team's STABLE PUBLIC INTERFACE (ai_pipeline.integration_api)
and normalizes the shape of what comes back.

IMPORTANT (architecture rule):
    The backend does NOT implement AI logic. This adapter only ROUTES calls
    to the pipeline's public API and maps its triage labels to the backend's
    RiskLevel enum.
"""

from __future__ import annotations

import json
from typing import Any

from backend.utils.config import get_settings
from backend.utils.logger import get_logger
from backend.utils.errors import PipelineError

logger = get_logger("ai_adapter")


# --------------------------------------------------------------------- #
# Triage label  ->  backend RiskLevel mapping
# The AI pipeline returns descriptive labels; the backend transports
# Green/Yellow/Red/Unknown. This is the agreed contract with Student 1.
# --------------------------------------------------------------------- #
_TRIAGE_TO_RISK = {
    "Safe / Regulated": "Green",
    "Distressed / Needs Support": "Yellow",
    "High Risk / Urgent": "Red",
    "Unclear / Need More Context": "Unknown",
}


def map_triage_to_risk(triage_label: str) -> str:
    """Map a pipeline triage label to the backend RiskLevel value."""
    return _TRIAGE_TO_RISK.get(triage_label, "Unknown")


class AIPipelineAdapter:
    """
    Unified interface to the AI pipeline.

    Public methods (the contract the orchestrator relies on):
        process_turn(audio_path, history)   -> dict   (full audio turn)
        generate_reply(text, history)       -> str    (text-only turn, WS)
        analyze_session(history)            -> dict
    """

    def __init__(self, mode: str | None = None):
        settings = get_settings()
        self.mode = (mode or settings.PIPELINE_MODE).lower()
        self._api = None  # lazy-loaded integration_api module handle

        if self.mode in ("real", "hybrid"):
            self._try_load_real()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _try_load_real(self) -> None:
        """
        Import the AI pipeline's PUBLIC integration API only.
        In hybrid mode a failure falls back to mock; in real mode it's fatal.
        """
        try:
            from ai_pipeline import integration_api  # type: ignore
            self._api = integration_api
            logger.info("Real AI pipeline loaded (integration_api).")
        except Exception as exc:  # noqa: BLE001
            if self.mode == "real":
                raise PipelineError(
                    "Failed to load real AI pipeline", detail=str(exc)
                ) from exc
            logger.warning(
                "AI pipeline unavailable, falling back to mock. Reason: %s", exc
            )
            self.mode = "mock"

    @property
    def effective_mode(self) -> str:
        return "mock" if self._api is None else "real"

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def process_turn(self, audio_path: str, history: list[dict]) -> dict:
        """Run ONE full turn (STT + emotion + triage + weighting + reply)."""
        if self._api is None:
            return self._mock_turn(audio_path, history)
        try:
            return self._api.process_turn(audio_path, history)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("Pipeline turn failed", detail=str(exc)) from exc

    def generate_reply(self, text: str, history: list[dict]) -> str:
        """
        Run ONE text-only turn (no STT / no audio). Used by the WebSocket
        text protocol. Returns the reply string only.
        """
        if self._api is None:
            return self._mock_reply(text, history)

        try:
            # Preferred: a dedicated text entrypoint in the pipeline API.
            if hasattr(self._api, "process_text_turn"):
                result = self._api.process_text_turn(text, history)
                if isinstance(result, dict):
                    return result.get("reply_text", "")
                return str(result)

            # Fallback: a raw generate_reply if the pipeline exposes one.
            if hasattr(self._api, "generate_reply"):
                return str(self._api.generate_reply(text, history))

            # Pipeline has no text path yet -> explicit, visible error.
            raise PipelineError(
                "Pipeline text turn failed",
                detail="integration_api has no text entrypoint "
                       "(expected process_text_turn or generate_reply)",
            )
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("Pipeline text turn failed", detail=str(exc)) from exc

    def analyze_session(self, history: list[dict]) -> dict:
        if self._api is None:
            return self._mock_analysis(history)
        try:
            result = self._api.analyze_session(history)
            return self._coerce_to_dict(result)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("Session analysis failed", detail=str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_to_dict(result: Any) -> dict:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"overall_summary": result, "final_risk_level": "Unknown"}
        return {"overall_summary": str(result), "final_risk_level": "Unknown"}

    # ------------------------------------------------------------------ #
    # Mock implementations (used when AI pipeline is not available)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mock_reply(text: str, history: list[dict]) -> str:
        logger.info("[MOCK] generate_reply(%r)", text[:40])
        return "أنا سامعتك يا حبيبي 🌸 احكيلي أكثر، أنا هون معك."

    @staticmethod
    def _mock_turn(audio_path: str, history: list[dict]) -> dict:
        logger.info("[MOCK] process_turn(%s)", audio_path)
        return {
            "transcribed_text": "مرحبا، أنا حاسس بخوف شوي اليوم.",
            "raw_text": "مرحبا انا حاسس بخوف شوي اليوم",
            "reply_text": "أنا سامعك 🌸 منيح إنك حكيت اللي جواك. أنا هون معك.",
            "response_source": "mock",
            "triage_label": "Distressed / Needs Support",
            "triage_signal": "mock_signal",
            "triage_confidence": 0.7,
            "needs_review": True,
            "emotion": "calm_or_neutral",
            "weighting_agreement": "mock",
            "conversation_history": history,
        }

    @staticmethod
    def _mock_analysis(history: list[dict]) -> dict:
        logger.info("[MOCK] analyze_session (%d turns)", len(history))
        return {
            "overall_summary": "Mock analysis — child expressed mild fear.",
            "final_label": "Distressed / Needs Support",
            "final_risk_level": "Yellow",
            "message_reports": [
                {
                    "turn": i + 1,
                    "child_text": turn.get("child_text", ""),
                    "message_summary": "mock summary",
                    "emotional_signals": ["fear"],
                    "message_risk_level": "Yellow",
                }
                for i, turn in enumerate(history)
            ],
        }