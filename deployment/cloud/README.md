# Cloud Deployment Notes (SENSE Backend)

This folder holds cloud configuration. The backend is stateless except for
in-memory sessions, so horizontal scaling requires moving sessions to a shared
store (Redis) first — see `backend/services/session_manager.py` (swappable).

## Generic container platform (Render / Railway / Fly / Cloud Run)
- Build with `deployment/docker/Dockerfile`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Required env:
  - `SENSE_PIPELINE_MODE=real` (or `hybrid`)
  - `SENSE_CORS_ORIGINS=https://your-mobile-or-web-origin`
  - `OPENAI_API_KEY` (if the real engine uses OpenAI)

## Before production
- [ ] Move sessions to Redis (replace SessionManager storage)
- [ ] Put uploads on object storage (S3/GCS) instead of local disk
- [ ] Set explicit CORS origins (no `*`)
- [ ] Add auth between mobile app and server
