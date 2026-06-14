"""
Orchestrator
============
Owner: Student 2 (Backend)

The orchestration center. It wires together session state + the AI pipeline
adapter, defining the END-TO-END FLOW for a turn and for final analysis.

It contains NO AI logic — only routing and sequencing:

    audio_path
        -> AIAdapter.transcribe            (STT)
        -> AIAdapter.generate_reply        (Teta AI)
        -> SessionManager.add_turn         (state)
        -> [TTS later]
        -> response to mobile

This matches the whiteboard flow:
    Whisper -> (Emotion/Weighting) -> LLM (Teta) -> TTS -> response
Emotion + weighting + TTS are pipeline concerns; we call them through the
adapter as they become available, without changing this flow's shape.
"""

from __future__ import annotations

from backend.services.ai_adapter import AIPipelineAdapter
from backend.services.session_manager import get_session_manager
from backend.schemas.audio import UploadAudioResponse, AnalyzeResponse, MessageReport
from backend.schemas.common import RiskLevel
from backend.utils.logger import get_logger

logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, adapter: AIPipelineAdapter | None = None):
        self.adapter = adapter or AIPipelineAdapter()
        self.sessions = get_session_manager()
        logger.info(
            "Orchestrator ready (pipeline mode: %s)", self.adapter.effective_mode
        )

    # ------------------------------------------------------------------ #
    # One audio turn, end to end
    # ------------------------------------------------------------------ #
    def process_audio_turn(
            self, session_id: str, audio_path: str
        ) -> UploadAudioResponse:
            logger.info("Processing audio turn for session %s", session_id)

            # Pass the accumulated RICH history so context survives across turns
            rich_history = self.sessions.get_rich_history(session_id)
            result = self.adapter.process_turn(audio_path, rich_history)

            transcribed = result.get("transcribed_text", "")
            reply = result.get("reply_text", "")

            # Persist: lightweight turn (for session state) + rich history (for analysis)
            turn = self.sessions.add_turn(session_id, transcribed, reply)
            self.sessions.set_rich_history(
                session_id, result.get("conversation_history", rich_history)
            )

            return UploadAudioResponse(
              session_id=session_id,
              turn=turn.turn,
              transcribed_text=transcribed,
              reply_text=reply,
              audio_response=result.get("audio_response") or None,
)
              
    
    def analyze_session(self, session_id: str) -> AnalyzeResponse:
        logger.info("Running final analysis for session %s", session_id)

        rich_history = self.sessions.get_rich_history(session_id)
        raw = self.adapter.analyze_session(rich_history)

        # The pipeline returns a descriptive label; map it to RiskLevel.
        from backend.services.ai_adapter import map_triage_to_risk
        final_label = raw.get("final_label", "Unclear / Need More Context")
        final_risk = self._safe_risk(map_triage_to_risk(final_label))

        reports: list[MessageReport] = []
        for item in raw.get("message_reports", []) or []:
            reports.append(
                MessageReport(
                    turn=item.get("turn", 0),
                    child_text=item.get("child_text", ""),
                    message_summary=item.get("message_summary", ""),
                    emotional_signals=item.get("emotional_signals", []) or [],
                    message_risk_level=self._safe_risk(
                        item.get("message_risk_level")
                    ),
                )
            )

        # Record the outcome on the session, then close it.
        self.sessions.set_final_analysis(session_id, final_risk.value)
        self.sessions.close_session(session_id)

        return AnalyzeResponse(
            session_id=session_id,
            overall_summary=raw.get("overall_summary", ""),
            final_risk_level=final_risk,
            message_reports=reports,
            raw=raw,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_risk(value) -> RiskLevel:
        """Validate a risk string from the pipeline; default to Unknown."""
        try:
            return RiskLevel(value)
        except (ValueError, TypeError):
            return RiskLevel.UNKNOWN


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
