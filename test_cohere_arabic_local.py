"""Run Cohere Transcribe Arabic locally on one SENSE audio file (CPU/float32)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
DEFAULT_AUDIO = PROJECT_ROOT / "datasets" / "raw_audio" / "high_risk_test.mp3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Cohere Transcribe Arabic locally.")
    p.add_argument("audio", nargs="?", type=Path, default=DEFAULT_AUDIO)
    p.add_argument("--language", choices=("ar", "en"), default="ar")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = "cpu"
    dtype = torch.float32

    print(f"Model:  {MODEL_ID}")
    print(f"Audio:  {audio_path}")
    print(f"Device: {device}")
    print("Warning: CPU inference is slow and needs several GB of RAM.")

    print("\nLoading processor and model (first run downloads ~5GB)...")
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start
    print(f"Model ready in {load_seconds:.2f}s")

    print("Loading and preprocessing audio...")
    audio = load_audio(str(audio_path), sampling_rate=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=args.language)
    inputs = inputs.to(model.device, dtype=model.dtype)

    print("Transcribing...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    infer_seconds = time.perf_counter() - t0

    decoded = processor.decode(outputs, skip_special_tokens=True)
    transcript = (" ".join(map(str, decoded)) if isinstance(decoded, list) else str(decoded)).strip()

    print("\n" + "=" * 72)
    print("COHERE TRANSCRIBE ARABIC - LOCAL RESULT")
    print("=" * 72)
    print(f"Transcript: {transcript}")
    print(f"Inference:  {infer_seconds:.4f}s")
    print(f"Model load: {load_seconds:.2f}s (exclude from per-turn latency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
