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
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "datasets" / "raw_audio"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "stt_comparison"
)

# Optional reference transcripts:
REFERENCES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "stt_references.json"
)

OPENAI_MODEL = "gpt-4o-transcribe"
COHERE_MODEL = "cohere-transcribe-03-2026"

LANGUAGE = "ar"

SUPPORTED_AUDIO = {
    ".mp3",
    ".wav",
    ".flac",
    ".mpeg",
    ".mpga",
    ".ogg",
}


# =========================================================
# Load API keys
# =========================================================

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing from .env"
    )

if not COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY is missing from .env"
    )


openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=120.0,
)

cohere_client = cohere.ClientV2(
    api_key=COHERE_API_KEY,
)


# =========================================================
# Arabic text normalization
# =========================================================

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F"
    r"\u0670\u06D6-\u06ED]"
)


def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub("", text)

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Normalize Alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)

    # Normalize Alef Maqsura
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text,
    )

    # Remove repeated spaces
    return " ".join(text.split())


# =========================================================
# WER and CER calculation
# =========================================================

def edit_distance(reference, hypothesis):
    previous_row = list(
        range(len(hypothesis) + 1)
    )

    for i, reference_item in enumerate(
        reference,
        start=1,
    ):
        current_row = [i]

        for j, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            substitution_cost = (
                0
                if reference_item == hypothesis_item
                else 1
            )

            current_row.append(
                min(
                    current_row[j - 1] + 1,
                    previous_row[j] + 1,
                    previous_row[j - 1]
                    + substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    reference = normalize_arabic(reference)
    hypothesis = normalize_arabic(hypothesis)

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    errors = edit_distance(
        reference_words,
        hypothesis_words,
    )

    return errors / max(
        1,
        len(reference_words),
    )


def calculate_cer(reference, hypothesis):
    reference = normalize_arabic(
        reference
    ).replace(" ", "")

    hypothesis = normalize_arabic(
        hypothesis
    ).replace(" ", "")

    errors = edit_distance(
        list(reference),
        list(hypothesis),
    )

    return errors / max(
        1,
        len(reference),
    )


# =========================================================
# Load reference transcripts
# =========================================================

def load_references():
    if not REFERENCES_FILE.exists():
        return {}

    with REFERENCES_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "stt_references.json must contain "
            "a JSON object."
        )

    return data


# =========================================================
# OpenAI transcription
# =========================================================

def transcribe_openai(audio_path):
    start_time = time.perf_counter()

    with audio_path.open("rb") as audio_file:
        response = (
            openai_client
            .audio
            .transcriptions
            .create(
                model=OPENAI_MODEL,
                file=audio_file,
                language=LANGUAGE,
                temperature=0,
            )
        )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    return {
        "provider": "OpenAI",
        "model": OPENAI_MODEL,
        "transcript": response.text.strip(),
        "latency_seconds": round(
            elapsed_time,
            4,
        ),
    }


# =========================================================
# Cohere transcription
# =========================================================

def transcribe_cohere(audio_path):
    start_time = time.perf_counter()

    with audio_path.open("rb") as audio_file:
        response = (
            cohere_client
            .audio
            .transcriptions
            .create(
                model=COHERE_MODEL,
                file=audio_file,
                language=LANGUAGE,
                temperature=0,
            )
        )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    return {
        "provider": "Cohere",
        "model": COHERE_MODEL,
        "transcript": response.text.strip(),
        "latency_seconds": round(
            elapsed_time,
            4,
        ),
    }


# =========================================================
# Run one provider safely
# =========================================================

def run_provider(
    provider_name,
    transcription_function,
    audio_path,
):
    try:
        return transcription_function(
            audio_path
        )

    except Exception as error:
        return {
            "provider": provider_name,
            "model": (
                OPENAI_MODEL
                if provider_name == "OpenAI"
                else COHERE_MODEL
            ),
            "transcript": "",
            "latency_seconds": None,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


# =========================================================
# Save results
# =========================================================

def save_results(results, summary):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        OUTPUT_DIR
        / f"stt_comparison_{timestamp}.json"
    )

    csv_path = (
        OUTPUT_DIR
        / f"stt_comparison_{timestamp}.csv"
    )

    json_data = {
        "created_at": datetime.now().isoformat(),
        "models": {
            "openai": OPENAI_MODEL,
            "cohere": COHERE_MODEL,
        },
        "language": LANGUAGE,
        "summary": summary,
        "results": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "file",
        "provider",
        "model",
        "status",
        "latency_seconds",
        "reference",
        "transcript",
        "wer",
        "cer",
        "error",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print("\nResults saved:")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


# =========================================================
# Summary
# =========================================================

def create_summary(results):
    summary = {}

    for provider in ["OpenAI", "Cohere"]:
        provider_results = [
            result
            for result in results
            if result["provider"] == provider
            and result["status"] == "ok"
        ]

        latencies = [
            result["latency_seconds"]
            for result in provider_results
            if result["latency_seconds"]
            is not None
        ]

        wers = [
            result["wer"]
            for result in provider_results
            if result["wer"] is not None
        ]

        cers = [
            result["cer"]
            for result in provider_results
            if result["cer"] is not None
        ]

        summary[provider] = {
            "successful_files": len(
                provider_results
            ),
            "average_latency_seconds": (
                round(
                    statistics.mean(
                        latencies
                    ),
                    4,
                )
                if latencies
                else None
            ),
            "average_wer": (
                round(
                    statistics.mean(wers),
                    4,
                )
                if wers
                else None
            ),
            "average_cer": (
                round(
                    statistics.mean(cers),
                    4,
                )
                if cers
                else None
            ),
        }

    return summary


# =========================================================
# Main comparison
# =========================================================

def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: "
            f"{AUDIO_DIR}"
        )

    audio_files = sorted(
        audio_path
        for audio_path in AUDIO_DIR.iterdir()
        if (
            audio_path.is_file()
            and audio_path.suffix.lower()
            in SUPPORTED_AUDIO
        )
    )

    if not audio_files:
        raise RuntimeError(
            f"No audio files found in "
            f"{AUDIO_DIR}"
        )

    references = load_references()
    results = []

    print(
        f"Found {len(audio_files)} "
        f"audio files."
    )

    if not references:
        print(
            "No reference transcripts found. "
            "WER and CER will not be calculated."
        )

    for index, audio_path in enumerate(
        audio_files,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(
            f"{index}/{len(audio_files)} "
            f"{audio_path.name}"
        )
        print("=" * 70)

        reference = references.get(
            audio_path.name
        )

        providers = [
            (
                "OpenAI",
                transcribe_openai,
            ),
            (
                "Cohere",
                transcribe_cohere,
            ),
        ]

        # Alternate request order to reduce
        # systematic network-order bias.
        if index % 2 == 0:
            providers.reverse()

        for (
            provider_name,
            transcription_function,
        ) in providers:
            response = run_provider(
                provider_name,
                transcription_function,
                audio_path,
            )

            transcript = response.get(
                "transcript",
                "",
            )

            error = response.get("error")

            wer = None
            cer = None

            if reference and not error:
                wer = calculate_wer(
                    reference,
                    transcript,
                )

                cer = calculate_cer(
                    reference,
                    transcript,
                )

            result = {
                "file": audio_path.name,
                "provider": response[
                    "provider"
                ],
                "model": response["model"],
                "status": (
                    "error"
                    if error
                    else "ok"
                ),
                "latency_seconds": response[
                    "latency_seconds"
                ],
                "reference": reference,
                "transcript": transcript,
                "wer": (
                    round(wer, 4)
                    if wer is not None
                    else None
                ),
                "cer": (
                    round(cer, 4)
                    if cer is not None
                    else None
                ),
                "error": error,
            }

            results.append(result)

            print(
                f"\n[{result['provider']}]"
            )

            if error:
                print(f"ERROR: {error}")

            else:
                print(
                    "Latency: "
                    f"{result['latency_seconds']}"
                    " seconds"
                )

                if result["wer"] is not None:
                    print(
                        f"WER: "
                        f"{result['wer']:.2%}"
                    )
                    print(
                        f"CER: "
                        f"{result['cer']:.2%}"
                    )

                print(
                    f"Transcript: "
                    f"{result['transcript']}"
                )

    summary = create_summary(results)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    save_results(
        results,
        summary,
    )


if __name__ == "__main__":
    main()