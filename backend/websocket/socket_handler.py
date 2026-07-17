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
    {"type": "audio_turn", "session_id": "...",
     "audio_base64": "...", "filename": "recording.webm"}
    {"type": "analyze", "session_id": "..."}  # final analysis
    {"type": "close", "session_id": "..."}

  server -> client:
    {"type": "session_started", "session_id": "..."}
    {"type": "reply", "session_id": "...", "turn": N, "reply_text": "...",
     "transcribed_text": "..."}
    {"type": "transcript", ...}
    {"type": "audio_chunk", "text": "...", "audio_base64": "...", "index": N}
    {"type": "turn_complete", "turn": N, "reply_text": "..."}
    {"type": "analysis", ...}
    {"type": "error", "error": "...", "message": "..."}

NOTE: text turns reuse the SAME orchestrator path as REST, keeping one source
of truth for the flow. Audio turns are saved with the REST upload rules, then
streamed through the pipeline adapter without blocking the event loop.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from backend.services.session_manager import get_session_manager
from backend.services.orchestrator import get_orchestrator
from backend.schemas.audio import UploadAudioResponse
from backend.utils.config import get_settings
from backend.utils.logger import get_logger
from backend.utils.errors import InvalidAudioError, SenseBackendError

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


def _save_audio_turn(session_id: str, filename: str, audio_base64: str) -> str:
    """Validate and persist one base64 WebSocket upload like the REST route."""
    settings = get_settings()
    safe_filename = Path(filename or "recording.webm").name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio type '{suffix}'",
            detail=sorted(settings.ALLOWED_AUDIO_EXTENSIONS),
        )

    encoded = audio_base64.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidAudioError("Invalid base64 audio payload") from exc

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise InvalidAudioError(
            f"Audio too large ({size_mb:.1f} MB > {settings.MAX_UPLOAD_MB} MB)"
        )

    destination = settings.UPLOAD_DIR / f"{session_id}_{safe_filename}"
    destination.write_bytes(audio_bytes)
    logger.info("WS audio saved for session %s (%.2f MB)", session_id, size_mb)
    return str(destination)


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
                    # Reuse orchestrator's reply path directly for text turns.
                    # run_in_threadpool: the adapter call is synchronous and may
                    # take seconds in real mode -> don't block the event loop.
                    history = sessions.history_as_dicts(session_id)
                    reply = await run_in_threadpool(
                        orch.adapter.generate_reply, text, history
                    )
                    turn = sessions.add_turn(session_id, text, reply)
                    payload = UploadAudioResponse(
                        session_id=session_id,
                        turn=turn.turn,
                        transcribed_text=text,
                        reply_text=reply,
                        audio_response=None,
                    )
                    await ws.send_json({"type": "reply", **payload.model_dump()})

                elif mtype == "audio_turn":
                    session_id = msg["session_id"]
                    sessions.get_session(session_id)
                    audio_path = await run_in_threadpool(
                        _save_audio_turn,
                        session_id,
                        msg.get("filename", "recording.webm"),
                        msg.get("audio_base64", ""),
                    )

                    rich_history = sessions.get_rich_history(session_id)
                    stream = orch.adapter.process_turn_stream(
                        audio_path,
                        rich_history,
                    )
                    transcript_text = ""
                    audio_index = 0

                    async for event in iterate_in_threadpool(iter(stream)):
                        event_type = event.get("type")

                        if event_type == "transcript":
                            transcript_text = (
                                event.get("processed_text")
                                or event.get("transcribed_text")
                                or ""
                            )
                            await ws.send_json(
                                {**event, "session_id": session_id}
                            )
                            continue

                        if event_type == "audio_chunk":
                            await ws.send_json(
                                {
                                    **event,
                                    "session_id": session_id,
                                    "index": audio_index,
                                }
                            )
                            audio_index += 1
                            continue

                        if event_type == "turn_complete":
                            transcribed = (
                                event.get("transcribed_text")
                                or transcript_text
                            )
                            reply = event.get("reply_text", "")
                            turn = sessions.add_turn(
                                session_id,
                                transcribed,
                                reply,
                            )
                            sessions.set_rich_history(
                                session_id,
                                event.get(
                                    "conversation_history",
                                    rich_history,
                                ),
                            )
                            await ws.send_json(
                                {
                                    "type": "turn_complete",
                                    "session_id": session_id,
                                    "turn": turn.turn,
                                    "transcribed_text": transcribed,
                                    "reply_text": reply,
                                    "triage_label": event.get("triage_label"),
                                }
                            )

                elif mtype == "analyze":
                    result = await run_in_threadpool(
                        orch.analyze_session, msg["session_id"]
                    )
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
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "error": "internal_error",
                    "message": "Unexpected server error",
                }
            )
        except Exception:  # noqa: BLE001
            pass
        manager.disconnect(ws)
