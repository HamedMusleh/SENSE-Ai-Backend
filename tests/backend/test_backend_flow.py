"""
Backend integration tests (mock pipeline mode).
Owner: Student 2.

Run:
    SENSE_PIPELINE_MODE=mock pytest tests/backend -v
"""

import base64
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


def test_websocket_audio_turn_streaming_flow():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        started = ws.receive_json()
        sid = started["session_id"]

        ws.send_json(
            {
                "type": "audio_turn",
                "session_id": sid,
                "filename": "recording.webm",
                "audio_base64": base64.b64encode(b"fake audio").decode("ascii"),
            }
        )

        events = []
        while not events or events[-1]["type"] != "turn_complete":
            events.append(ws.receive_json())

        event_types = [event["type"] for event in events]
        assert event_types[0] == "transcript"
        assert event_types[-1] == "turn_complete"
        assert event_types[1:-1]
        assert set(event_types[1:-1]) == {"audio_chunk"}

        chunks = events[1:-1]
        assert [chunk["index"] for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk["text"] for chunk in chunks)
        assert events[-1]["turn"] == 1
        assert events[-1]["reply_text"]

        session = client.get(f"/api/session/{sid}").json()
        assert session["turn_count"] == 1
        assert session["conversation"][0]["teta_reply"] == events[-1]["reply_text"]
