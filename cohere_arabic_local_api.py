"""
Cohere Transcribe Arabic — local OpenAI-compatible transcription server.

Loads CohereLabs/cohere-transcribe-arabic-07-2026 ONCE at startup (CPU/float32,
since the local MX450 2GB GPU is too small for a 2B model), then exposes:

    POST /v1/audio/transcriptions   (multipart: file=<audio>, language=ar)

Response shape mirrors OpenAI's transcription endpoint: {"text": "..."}.
This lets compare_stt.py treat OpenAI, Cohere-cloud, and Cohere-Arabic-local
through one uniform HTTP interface.

Run:
    python -m uvicorn cohere_arabic_local_api:app --host 127.0.0.1 --port 8001 --workers 1
"""
from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
DEVICE = "cpu"
DTYPE = torch.float32

app = FastAPI(title="Cohere Transcribe Arabic (local)")

# Loaded once at startup, reused for every request.
_processor = None
_model = None


@app.on_event("startup")
def _load_model() -> None:
    global _processor, _model
    print(f"[startup] Loading {MODEL_ID} on {DEVICE} (first run downloads ~5GB)...")
    t0 = time.perf_counter()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True
    )
    _model.to(DEVICE)
    _model.eval()
    print(f"[startup] Model ready in {time.perf_counter() - t0:.2f}s")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "ready": _model is not None}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    model: str = Form(MODEL_ID),
    max_new_tokens: int = Form(256),
) -> dict:
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await file.read()

    # load_audio needs a path; write the upload to a temp file, keeping the
    # original suffix so ffmpeg can decode mp3/wav/webm correctly.
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path, sampling_rate=16000)
        inputs = _processor(
            audio, sampling_rate=16000, return_tensors="pt", language=language
        )
        inputs = inputs.to(_model.device, dtype=_model.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        infer_s = time.perf_counter() - t0

        decoded = _processor.decode(outputs, skip_special_tokens=True)
        text = (
            " ".join(map(str, decoded)) if isinstance(decoded, list) else str(decoded)
        ).strip()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    return {"text": text, "language": language, "inference_seconds": round(infer_s, 4)}
