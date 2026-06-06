# SENSE Backend — API Contract

**Owner:** Student 2 (Backend / Server Engineer)
**Base URL (local):** `http://127.0.0.1:8000`
**Interactive docs:** `/docs` (Swagger)

This is the single source of truth for how the **mobile app** talks to the
**server**. The backend orchestrates; it does not implement AI logic.

---

## Conversation lifecycle

```
1. POST /api/session/start         -> get session_id
2. POST /api/upload_audio (xN)     -> one turn per audio file
3. POST /api/analyze               -> final risk + report, session closes
```

---

## REST Endpoints

### `GET /api/health`
Returns server + pipeline status.
```json
{ "status": "ok", "service": "sense-backend", "version": "0.1.0", "pipeline_mode": "real" }
```

### `POST /api/session/start`
Response:
```json
{ "session_id": "uuid", "created_at": "2026-05-30T07:00:00Z", "message": "Session started successfully" }
```

### `POST /api/upload_audio`  (multipart/form-data)
Fields: `session_id` (form), `audio_file` (file: .wav/.mp3/.m4a/.ogg/.webm)
Response:
```json
{
  "session_id": "uuid",
  "turn": 1,
  "transcribed_text": "...",
  "reply_text": "...",
  "audio_response": null
}
```
> `audio_response` is `null` until TTS is wired into the pipeline.

### `GET /api/session/{session_id}`
Returns the full session snapshot (conversation turns, turn_count, final_risk_level, is_closed).

### `POST /api/analyze`  (form field: `session_id`)
Response:
```json
{
  "session_id": "uuid",
  "overall_summary": "...",
  "final_risk_level": "Green | Yellow | Red | Unknown",
  "message_reports": [
    { "turn": 1, "child_text": "...", "message_summary": "...",
      "emotional_signals": ["fear"], "message_risk_level": "Yellow" }
  ],
  "raw": { }
}
```

---

## WebSocket  `/ws`  (JSON text frames)

| Client sends | Server replies |
|---|---|
| `{"type":"start"}` | `{"type":"session_started","session_id":"..."}` |
| `{"type":"text","session_id":"...","text":"..."}` | `{"type":"reply","turn":N,"reply_text":"...","transcribed_text":"..."}` |
| `{"type":"analyze","session_id":"..."}` | `{"type":"analysis", ...}` |
| `{"type":"close","session_id":"..."}` | `{"type":"closed","session_id":"..."}` |

---

## Error envelope (all endpoints)
```json
{ "error": "session_not_found", "message": "Session 'x' not found", "detail": null }
```
| HTTP | error code |
|---|---|
| 400 | `invalid_audio` |
| 404 | `session_not_found` |
| 502 | `pipeline_error` |
| 504 | `pipeline_timeout` |
