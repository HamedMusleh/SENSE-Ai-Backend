# SENSE Backend

**Role:** Student 2 — Backend / Server Engineer
**Stack:** FastAPI · Uvicorn · Pydantic · WebSockets

The backend is the **orchestration center**: it receives audio from the
mobile app, routes it through the AI pipeline, and returns Teta AI's reply.
It contains **no AI logic** — all AI processing goes through `ai_pipeline/`
via a single adapter.

## Architecture

```
Mobile App
   │  REST / WebSocket
   ▼
backend/api ─ backend/websocket      (HTTP + realtime wiring)
   │
   ▼
backend/services/orchestrator.py     (sequences the flow)
   │
   ├─ session_manager.py             (session lifecycle)
   └─ ai_adapter.py  ───────────────▶ ai_pipeline/   (STT, Teta, analysis)
```

The orchestrator flow per turn:
`audio → transcribe → generate_reply → store turn → (TTS later) → response`

## Folder map
| Path | Purpose |
|---|---|
| `api/routes.py` | REST endpoints |
| `websocket/socket_handler.py` | Real-time channel |
| `services/orchestrator.py` | Flow sequencing |
| `services/session_manager.py` | Session state |
| `services/ai_adapter.py` | **Only** bridge to `ai_pipeline/` |
| `schemas/` | Request/response contracts |
| `utils/` | config, logging, errors |

## Run locally
```bash
pip install -r backend/requirements-backend.txt
bash deployment/local/run_local.sh
# or:
uvicorn backend.main:app --reload
```
Docs: http://127.0.0.1:8000/docs

## Pipeline modes (env `SENSE_PIPELINE_MODE`)
- `mock`   — canned responses, zero AI deps (great for backend dev/tests)
- `real`   — calls the actual `ai_pipeline` functions
- `hybrid` — tries real, falls back to mock if the pipeline can't load

## Test
```bash
SENSE_PIPELINE_MODE=mock pytest tests/backend -v
```

## Docker
```bash
docker build -f deployment/docker/Dockerfile -t sense-backend .
docker run -p 8000:8000 -e SENSE_PIPELINE_MODE=mock sense-backend
```

## Boundaries (teamwork rules)
- Does **not** modify `ai_pipeline/`, `prompts/`, `datasets/`, `resources/`,
  `evaluation/`, `mobile_app/`.
- The API contract in `docs/api/API_CONTRACT.md` is the agreement with the
  mobile team — change it only after discussion.
