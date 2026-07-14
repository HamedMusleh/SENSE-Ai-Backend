# 02 — Setup & Running

## Requirements

- Python 3.11+
- An OpenAI API key with access to `gpt-5`, `gpt-4o-transcribe`, `gpt-4o-mini-tts`
- (Optional) Docker
- (Mobile) Flutter SDK + Android toolchain

## Installation

```bash
git clone https://github.com/HamedMusleh/SENSE-Ai-Backend
cd SENSE-Ai-Backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Values | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Required for `real` / `hybrid` modes |
| `PIPELINE_MODE` | `real` \| `mock` \| `hybrid` | Selects the AI adapter mode (see docs/06) |

`mock` mode runs the full backend with canned AI responses — no API key or network needed.
Use it for frontend development and integration tests.

## Run locally

```bash
uvicorn backend.main:app --reload
```

- Interactive API docs (Swagger): http://127.0.0.1:8000/docs
- Health check: `GET http://127.0.0.1:8000/api/health` → returns status + pipeline mode

## Run with Docker

```bash
docker build -t sense-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY -e PIPELINE_MODE=real sense-backend
```

## Clients

**Web (recommended for demos):** open `sense_web.html` in Chrome/Edge. Uses the
MediaRecorder API (webm/opus) and a Web Audio `AnalyserNode` for the live level meter.
Grant microphone permission when prompted.

**Flutter (Android):** standard `flutter run` from the mobile project.
⚠️ **The Android emulator's virtual microphone records silence** — this is a known platform
limitation, not an app bug. Test audio on a physical device or the web client.

## Tests

```bash
pytest --import-mode=importlib
```

Six backend integration tests cover the REST endpoints and orchestrator flow. Run them in
`mock` mode so they don't consume API credits.

## Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: backend` | Run from repo root; ensure `backend/__init__.py` exists |
| pytest can't resolve imports | Use `--import-mode=importlib` |
| Emulator records silence | Platform limitation — use web client or real device |
