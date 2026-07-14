# 03 — API Reference

Base URL (local): `http://127.0.0.1:8000`
All endpoints perform validation and delegation only — business logic lives in the
orchestrator and the AI pipeline.

---

## GET `/api/health`

Server liveness + active pipeline mode.

**Response 200**
```json
{ "status": "ok", "pipeline_mode": "real" }
```

---

## POST `/api/session/start`

Creates a new conversation session.

**Response 200**
```json
{ "session_id": "550e8400-e29b-41d4-a716-446655440000" }
```

---

## POST `/api/upload_audio`

Uploads one conversational turn (multipart form). The backend validates extension and size,
persists the file, delegates to the orchestrator (STT → triage → reply → TTS), and returns
the result.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | From `/api/session/start` |
| `file` | audio file | WAV (Flutter) or webm/opus (web client) |

**Response 200 (typical)**
```json
{
  "session_id": "550e8400-...",
  "transcript": "...نص كلام الطفل...",
  "triage_label": "distressed",
  "reply_text": "...رد تيتا...",
  "reply_audio_url": "..."
}
```

**Behavior notes**
- If the turn is triaged **High Risk**, `reply_text` comes from the pre-vetted hard-coded
  set — the LLM is never called for that turn.
- Errors return a unified JSON error envelope (custom `SenseBackendError` handling).

---

## GET `/api/session/{session_id}`

Full session snapshot: all turns with transcripts, per-turn triage labels, and replies.

---

## POST `/api/analyze`

Triggers the final whole-conversation analysis and **closes the session**. The final label
follows the no-downgrade rule: any Red turn in the session ⇒ final label Red.

**Response 200 (typical)**
```json
{
  "session_id": "550e8400-...",
  "final_label": "high_risk",
  "summary": "...",
  "turn_count": 7
}
```

---

## WebSocket

Real-time channel (`backend/websocket/socket_handler.py`) for streaming session events to
connected clients — e.g., stage progress (transcribing / thinking / speaking) and triage
updates during a turn.

---

> ✅ **Verification note:** exact response field names should be confirmed against
> `backend/schemas/` (Pydantic models are the single source of truth for the API contract).
> The Swagger UI at `/docs` always reflects the deployed contract.
