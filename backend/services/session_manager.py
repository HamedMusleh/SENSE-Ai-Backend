"""
Session Manager
===============
Owner: Student 2 (Backend)

Owns session lifecycle: create, fetch, append turns, close, expire.
Storage is in-memory for now (a dict). The interface is written so it can
be swapped for Redis / a database later without touching the API layer.

The backend does NOT interpret conversation content here — it only stores
and retrieves it.
"""

from __future__ import annotations

import uuid
import threading
from datetime import datetime, timedelta

from backend.schemas.session import SessionState, ConversationTurn
from backend.utils.config import get_settings
from backend.utils.logger import get_logger
from backend.utils.errors import SessionNotFoundError

logger = get_logger("session_manager")


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=get_settings().SESSION_TTL_MINUTES)

    # ------------------------------------------------------------------ #
    def create_session(self) -> SessionState:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        state = SessionState(
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._sessions[session_id] = state
        logger.info("Session created: %s", session_id)
        return state

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        return state

    def add_turn(
        self, session_id: str, child_text: str, teta_reply: str
    ) -> ConversationTurn:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(f"Session '{session_id}' not found")

            turn = ConversationTurn(
                turn=state.turn_count + 1,
                child_text=child_text,
                teta_reply=teta_reply,
            )
            state.conversation.append(turn)
            state.turn_count += 1
            state.updated_at = datetime.utcnow()
        logger.info("Session %s: added turn %d", session_id, turn.turn)
        return turn

    def set_final_analysis(self, session_id: str, risk_level: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            state.final_risk_level = risk_level
            state.updated_at = datetime.utcnow()

    def close_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            state.is_closed = True
            state.updated_at = datetime.utcnow()
        logger.info("Session closed: %s", session_id)

    def history_as_dicts(self, session_id: str) -> list[dict]:
        """Return conversation in the plain-dict shape the AI pipeline expects."""
        state = self.get_session(session_id)
        return [
            {"turn": t.turn, "child_text": t.child_text, "teta_reply": t.teta_reply}
            for t in state.conversation
        ]
    
    def set_rich_history(self, session_id: str, rich_history: list[dict]) -> None:
            """Store the full pipeline history (with triage + emotion per turn).

            The AI pipeline returns an enriched history that the final analysis
            needs. We keep it verbatim so analyze_session() gets full context.
            """
            with self._lock:
                state = self._sessions.get(session_id)
                if state is None:
                    raise SessionNotFoundError(f"Session '{session_id}' not found")
                state.rich_history = rich_history
                state.updated_at = datetime.utcnow()

    def get_rich_history(self, session_id: str) -> list[dict]:
        """Return the stored rich history (empty list if none yet)."""
        state = self.get_session(session_id)
        return getattr(state, "rich_history", []) or []
    
    
    def cleanup_expired(self) -> int:
        """Remove sessions older than the TTL. Returns count removed."""
        cutoff = datetime.utcnow() - self._ttl
        removed = 0
        with self._lock:
            for sid in list(self._sessions.keys()):
                if self._sessions[sid].updated_at < cutoff:
                    del self._sessions[sid]
                    removed += 1
        if removed:
            logger.info("Cleaned up %d expired sessions", removed)
        return removed


# Single shared instance for the app
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
