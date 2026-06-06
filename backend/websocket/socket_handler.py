"""
WebSocket Handler
=================
Owner: Student 2 (Backend)

Real-time channel for low-latency conversation. For the pre-alpha we support
a simple message protocol; binary audio streaming chunks can be added later
on the same socket.

Protocol (JSON text frames):

  client -> server:
    {"type": "start"}                         # open a session
    {"type": "text", "session_id": "...", "text": "..."}   # a turn by text
    {"type": "analyze", "session_id": "..."}  # final analysis
    {"type": "close", "session_id": "..."}

  server -> client:
    {"type": "session_started", "session_id": "..."}
    {"type": "reply", "session_id": "...", "turn": N, "reply_text": "...",
     "transcribed_text": "..."}
    {"type": "analysis", ...}
    {"type": "error", "error": "...", "message": "..."}

NOTE: text turns reuse the SAME orchestrator path as REST, keeping one source
of truth for the flow. Binary audio frames -> save -> orchestrator can be
layered on top without changing the orchestrator.
"""

from __future__ import annotations

from fastapi import WebSocket, WebSocketDisconnect

from backend.services.session_manager import get_session_manager
from backend.services.orchestrator import get_orchestrator
from backend.schemas.audio import UploadAudioResponse
from backend.utils.logger import get_logger
from backend.utils.errors import SenseBackendError

logger = get_logger("websocket")


class ConnectionManager:
    """Tracks active websocket connections."""

    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info("WS connected (%d active)", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info("WS disconnected (%d active)", len(self.active))


manager = ConnectionManager()


async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    sessions = get_session_manager()
    orch = get_orchestrator()

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            try:
                if mtype == "start":
                    state = sessions.create_session()
                    await ws.send_json(
                        {"type": "session_started", "session_id": state.session_id}
                    )

                elif mtype == "text":
                    session_id = msg["session_id"]
                    text = msg["text"]
                    # Reuse orchestrator's reply path directly for text turns
                    history = sessions.history_as_dicts(session_id)
                    reply = orch.adapter.generate_reply(text, history)
                    turn = sessions.add_turn(session_id, text, reply)
                    payload = UploadAudioResponse(
                        session_id=session_id,
                        turn=turn.turn,
                        transcribed_text=text,
                        reply_text=reply,
                        audio_response=None,
                    )
                    await ws.send_json({"type": "reply", **payload.model_dump()})

                elif mtype == "analyze":
                    result = orch.analyze_session(msg["session_id"])
                    await ws.send_json({"type": "analysis", **result.model_dump()})

                elif mtype == "close":
                    sessions.close_session(msg["session_id"])
                    await ws.send_json(
                        {"type": "closed", "session_id": msg["session_id"]}
                    )

                else:
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": "unknown_type",
                            "message": f"Unknown message type: {mtype}",
                        }
                    )

            except SenseBackendError as exc:
                await ws.send_json(
                    {"type": "error", "error": exc.error_code, "message": exc.message}
                )

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as exc:  # noqa: BLE001
        logger.exception("WS fatal error: %s", exc)
        manager.disconnect(ws)
