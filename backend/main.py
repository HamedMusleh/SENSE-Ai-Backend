"""
SENSE Backend — Application Entry Point
=======================================
Owner: Student 2 (Backend / Server Engineer)

Assembles the FastAPI app: routes, websocket, CORS, error handling, lifecycle.

Run:
    uvicorn backend.main:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router as api_router
from backend.websocket.socket_handler import websocket_endpoint
from backend.utils.config import get_settings
from backend.utils.logger import get_logger, setup_logging
from backend.utils.errors import SenseBackendError

setup_logging()
logger = get_logger("main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SENSE backend starting (pipeline mode=%s)", settings.PIPELINE_MODE)
    # Warm the orchestrator (loads the AI adapter once at startup)
    from backend.services.orchestrator import get_orchestrator
    get_orchestrator()
    yield
    logger.info("SENSE backend shutting down")


app = FastAPI(
    title="SENSE Backend",
    description="Orchestration server for the SENSE emotional-screening system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Register the WebSocket endpoint explicitly at the path used by
# sense_web.html (ws://<backend-host>/ws).
app.add_api_websocket_route(
    "/ws",
    websocket_endpoint,
    name="sense-websocket",
)

#===================================
from fastapi.staticfiles import StaticFiles
import os

# Serve the web demo page
_web_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/web", StaticFiles(directory=_web_dir, html=True), name="web")
#===================================


# --------------------------------------------------------------------- #
# Unified error handling -> clean JSON, no stack traces leaked to client
# --------------------------------------------------------------------- #
@app.exception_handler(SenseBackendError)
async def backend_error_handler(_: Request, exc: SenseBackendError):
    logger.warning("%s: %s", exc.error_code, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "sense-backend",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }
