import csv
import json
import os
import re
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cohere
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ========================= Configuration =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "datasets" / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "stt_comparison"
REFERENCES_FILE = AUDIO_DIR / "references.json"

OPENAI_MODEL = "gpt-4o-transcribe"
COHERE_MODEL = "cohere-transcribe-03-2026"
COHERE_LOCAL_MODEL = "cohere-transcribe-arabic-07-2026"
COHERE_LOCAL_URL = "http://127.0.0.1:8001/v1/audio/transcriptions"

LANGUAGE = "ar"
N_RUNS = 3
SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".mpeg", ".mpga", ".ogg"}

PROVIDER_MODEL = {
    "OpenAI": OPENAI_MODEL,
    "Cohere": COHERE_MODEL,
    "CohereArabicLocal": COHERE_LOCAL_MODEL,
}

# ========================= API keys =========================
load_dotenv(PROJECT_ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY is missing from .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# ========================= Arabic normalization =========================
# Unicode escapes only (never literal Arabic) to stay encoding-safe.
ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
ALEF_VARIANTS = re.compile("[\u0623\u0625\u0622\u0671]")  # أ إ آ ٱ
TATWEEL = "\u0640"
ALEF_MAQSURA = "\u0649"  # ى
YEH = "\u064A"           # ي
TEH_MARBUTA = "\u0629"   # ة
HEH = "\u0647"           # ه
ARABIC_RANGE = "\u0600-\u06FF"


def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("\u0627", text)       # -> ا
    text = text.replace(ALEF_MAQSURA, YEH)         # ى -> ي
    text = text.replace(TEH_MARBUTA, HEH)          # ة -> ه
    text = re.sub("[^\\w\\s" + ARABIC_RANGE + "]", " ", text)
    return " ".join(text.split())


# ========================= WER / CER =========================
def edit_distance(reference, hypothesis):
    previous_row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current_row = [i]
        for j, h in enumerate(hypothesis, start=1):
            cost = 0 if r == h else 1
            current_row.append(
                min(current_row[j - 1] + 1, previous_row[j] + 1, previous_row[j - 1] + cost)
            )
        previous_row = current_row
    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    r = normalize_arabic(reference).split()
    h = normalize_arabic(hypothesis).split()
    return edit_distance(r, h) / max(1, len(r))


def calculate_cer(reference, hypothesis):
    r = normalize_arabic(reference).replace(" ", "")
    h = normalize_arabic(hypothesis).replace(" ", "")
    return edit_distance(list(r), list(h)) / max(1, len(r))


# ========================= References =========================
def load_references():
    if not REFERENCES_FILE.exists():
        return {}
    with REFERENCES_FILE.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("references.json must contain a JSON object.")
    return data


# ========================= Providers =========================
def transcribe_openai(audio_path):
    t0 = time.perf_counter()
    with audio_path.open("rb") as f:
        resp = openai_client.audio.transcriptions.create(
            model=OPENAI_MODEL, file=f, language=LANGUAGE, temperature=0
        )
    return resp.text.strip(), round(time.perf_counter() - t0, 4)


def transcribe_cohere(audio_path):
    t0 = time.perf_counter()
    with audio_path.open("rb") as f:
        resp = cohere_client.audio.transcriptions.create(
            model=COHERE_MODEL, file=f, language=LANGUAGE, temperature=0
        )
    return resp.text.strip(), round(time.perf_counter() - t0, 4)


def transcribe_cohere_local(audio_path):
    t0 = time.perf_counter()
    with audio_path.open("rb") as f:
        resp = requests.post(
            COHERE_LOCAL_URL,
            files={"file": (audio_path.name, f)},
            data={"language": LANGUAGE},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()["text"].strip(), round(time.perf_counter() - t0, 4)


PROVIDERS = {
    "OpenAI": transcribe_openai,
    "Cohere": transcribe_cohere,
    "CohereArabicLocal": transcribe_cohere_local,
}


def run_provider_n_times(provider_name, fn, audio_path, reference):
    """Run a provider N_RUNS times; report per-run + averaged metrics."""
    latencies, transcripts, errors = [], [], []
    for _ in range(N_RUNS):
        try:
            transcript, latency = fn(audio_path)
            transcripts.append(transcript)
            latencies.append(latency)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    if not transcripts:  # all runs failed
        return {
            "file": audio_path.name,
            "provider": provider_name,
            "model": PROVIDER_MODEL[provider_name],
            "status": "error",
            "runs": 0,
            "latency_seconds": None,
            "reference": reference,
            "transcript": "",
            "wer": None,
            "cer": None,
            "error": errors[0] if errors else "unknown",
        }

    # Transcript is deterministic (temp=0); take the last successful one.
    transcript = transcripts[-1]
    wer = cer = None
    if reference:
        wer = round(calculate_wer(reference, transcript), 4)
        cer = round(calculate_cer(reference, transcript), 4)

    return {
        "file": audio_path.name,
        "provider": provider_name,
        "model": PROVIDER_MODEL[provider_name],
        "status": "ok",
        "runs": len(latencies),
        "latency_seconds": round(statistics.mean(latencies), 4),
        "latency_stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "reference": reference,
        "transcript": transcript,
        "wer": wer,
        "cer": cer,
        "error": errors[0] if errors else None,
    }


# ========================= Summary + save =========================
def create_summary(results):
    summary = {}
    for provider in PROVIDERS:
        rows = [r for r in results if r["provider"] == provider and r["status"] == "ok"]
        lat = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None]
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        cers = [r["cer"] for r in rows if r["cer"] is not None]
        summary[provider] = {
            "successful_files": len(rows),
            "average_latency_seconds": round(statistics.mean(lat), 4) if lat else None,
            "average_wer": round(statistics.mean(wers), 4) if wers else None,
            "average_cer": round(statistics.mean(cers), 4) if cers else None,
        }
    return summary


def save_results(results, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"stt_comparison_{ts}.json"
    csv_path = OUTPUT_DIR / f"stt_comparison_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "models": PROVIDER_MODEL,
                "language": LANGUAGE,
                "n_runs": N_RUNS,
                "summary": summary,
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )

    fields = ["file", "provider", "model", "status", "runs", "latency_seconds",
              "latency_stdev", "reference", "transcript", "wer", "cer", "error"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\nResults saved:")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


# ========================= Main =========================
def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    audio_files = sorted(
        p for p in AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO
    )
    if not audio_files:
        raise RuntimeError(f"No audio files in {AUDIO_DIR}")

    references = load_references()
    results = []

    print(f"Found {len(audio_files)} audio files. N_RUNS={N_RUNS}")
    if not references:
        print("WARNING: no references.json - WER/CER will be null.")

    for idx, audio_path in enumerate(audio_files, start=1):
        print("\n" + "=" * 70)
        print(f"{idx}/{len(audio_files)}  {audio_path.name}")
        print("=" * 70)
        reference = references.get(audio_path.name)

        for provider_name, fn in PROVIDERS.items():
            r = run_provider_n_times(provider_name, fn, audio_path, reference)
            results.append(r)
            print(f"\n[{provider_name}]")
            if r["status"] == "error":
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  Latency: {r['latency_seconds']}s (avg of {r['runs']})")
                if r["wer"] is not None:
                    print(f"  WER: {r['wer']:.2%}   CER: {r['cer']:.2%}")
                print(f"  Transcript: {r['transcript']}")

    summary = create_summary(results)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    save_results(results, summary)


if __name__ == "__main__":
    main()
