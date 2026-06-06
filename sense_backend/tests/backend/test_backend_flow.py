"""
Backend integration tests (mock pipeline mode).
Owner: Student 2.

Run:
    SENSE_PIPELINE_MODE=mock pytest tests/backend -v
"""

import io
import os

os.environ.setdefault("SENSE_PIPELINE_MODE", "mock")

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_start_session_returns_id():
    r = client.post("/api/session/start")
    assert r.status_code == 200
    assert "session_id" in r.json()


def test_get_unknown_session_404():
    r = client.get("/api/session/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"] == "session_not_found"


def test_full_turn_and_analysis():
    sid = client.post("/api/session/start").json()["session_id"]

    audio = io.BytesIO(b"RIFFfake")
    up = client.post(
        "/api/upload_audio",
        data={"session_id": sid},
        files={"audio_file": ("a.wav", audio, "audio/wav")},
    )
    assert up.status_code == 200
    body = up.json()
    assert body["turn"] == 1
    assert body["reply_text"]

    a = client.post("/api/analyze", data={"session_id": sid})
    assert a.status_code == 200
    assert a.json()["final_risk_level"] in {"Green", "Yellow", "Red", "Unknown"}


def test_reject_bad_extension():
    sid = client.post("/api/session/start").json()["session_id"]
    bad = io.BytesIO(b"not audio")
    r = client.post(
        "/api/upload_audio",
        data={"session_id": sid},
        files={"audio_file": ("note.txt", bad, "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_audio"


def test_websocket_text_flow():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        started = ws.receive_json()
        assert started["type"] == "session_started"
        sid = started["session_id"]

        ws.send_json({"type": "text", "session_id": sid, "text": "مرحبا"})
        reply = ws.receive_json()
        assert reply["type"] == "reply"
        assert reply["reply_text"]
