"""
SENSE data integrity checker.

Checks:
1. Exact-duplicate detection WITHIN each dataset file (catches bugs like a
   file being accidentally concatenated with itself).
2. Exact-text leakage BETWEEN training_dataset.jsonl and the gold/unseen
   evaluation sets (a training example whose text exactly matches an eval
   example would inflate reported accuracy).
3. Near-duplicate leakage (normalized, whitespace/punctuation-insensitive)
   between training and eval sets, to catch trivial rephrasings.
4. Per-file label distribution and a SHA-256 hash of each file's content,
   for the "gold set integrity" record mentioned in project documentation.

Run before every evaluation cycle:
    python check_data_leakage.py

Exit code is non-zero if exact leakage or in-file duplicates are found, so
this can be wired into CI or a pre-eval hook later.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = {
    "training": PROJECT_ROOT / "datasets" / "training" / "training_dataset.jsonl",
    "gold": PROJECT_ROOT / "datasets" / "gold" / "gold_test_set.jsonl",
    "unseen": PROJECT_ROOT / "datasets" / "gold" / "unseen_validation_set.jsonl",
}

ARABIC_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ALEF_VARIANTS = re.compile("[\u0623\u0625\u0622\u0671]")
TATWEEL = "\u0640"
ALEF_MAQSURA = "\u0649"
YEH = "\u064A"
TEH_MARBUTA = "\u0629"
HEH = "\u0647"
ARABIC_RANGE = "\u0600-\u06FF"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("\u0627", text)
    text = text.replace(ALEF_MAQSURA, YEH)
    text = text.replace(TEH_MARBUTA, HEH)
    text = re.sub("[^\\w\\s" + ARABIC_RANGE + "]", " ", text)
    return " ".join(text.split())


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [MISSING] {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [BAD JSON] {path} line {line_no}: {exc}")
    return rows


def file_hash(path: Path) -> str:
    if not path.exists():
        return "N/A"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_in_file_duplicates(name: str, rows: list[dict]) -> int:
    seen_ids = Counter(r.get("id") for r in rows)
    seen_texts = Counter(normalize(r.get("text", "")) for r in rows)

    dup_ids = {k: v for k, v in seen_ids.items() if v > 1}
    dup_texts = {k: v for k, v in seen_texts.items() if v > 1}

    problems = 0
    if dup_ids:
        problems += len(dup_ids)
        print(f"  [DUPLICATE IDS] {name}: {len(dup_ids)} id(s) repeated, e.g. "
              f"{list(dup_ids.items())[:5]}")
    if dup_texts:
        problems += len(dup_texts)
        print(f"  [DUPLICATE TEXT] {name}: {len(dup_texts)} normalized text(s) "
              f"repeated, e.g. {list(dup_texts.items())[:3]}")
    if not dup_ids and not dup_texts:
        print(f"  [OK] {name}: no in-file duplicates")
    return problems


def check_cross_leakage(train_rows: list[dict], eval_name: str,
                         eval_rows: list[dict]) -> int:
    train_norm = {normalize(r.get("text", "")): r.get("id") for r in train_rows}
    leaks = 0
    for r in eval_rows:
        norm = normalize(r.get("text", ""))
        if norm in train_norm:
            leaks += 1
            print(f"  [LEAKAGE] {eval_name} id={r.get('id')} matches "
                  f"training id={train_norm[norm]}: {r.get('text')!r}")
    if leaks == 0:
        print(f"  [OK] training vs {eval_name}: no exact/normalized-text leakage")
    return leaks


def label_distribution(name: str, rows: list[dict]) -> None:
    labels = Counter(r.get("label") for r in rows)
    print(f"  {name}: {len(rows)} rows")
    for label, count in labels.most_common():
        print(f"    {label!r}: {count}")


def main() -> int:
    print("=" * 70)
    print("SENSE data integrity check")
    print("=" * 70)

    data = {}
    for name, path in DATASETS.items():
        print(f"\nLoading {name}: {path}")
        data[name] = load(path)

    print("\n" + "-" * 70)
    print("SHA-256 hashes")
    print("-" * 70)
    for name, path in DATASETS.items():
        print(f"  {name}: {file_hash(path)}")

    print("\n" + "-" * 70)
    print("Label distributions")
    print("-" * 70)
    for name, rows in data.items():
        label_distribution(name, rows)

    print("\n" + "-" * 70)
    print("In-file duplicate check")
    print("-" * 70)
    total_problems = 0
    for name, rows in data.items():
        total_problems += check_in_file_duplicates(name, rows)

    print("\n" + "-" * 70)
    print("Cross-dataset leakage check (training vs eval sets)")
    print("-" * 70)
    if data.get("training"):
        for eval_name in ("gold", "unseen"):
            if data.get(eval_name):
                total_problems += check_cross_leakage(
                    data["training"], eval_name, data[eval_name]
                )
    else:
        print("  [SKIP] training set not loaded")

    print("\n" + "=" * 70)
    if total_problems:
        print(f"RESULT: {total_problems} issue(s) found. Fix before training/eval.")
    else:
        print("RESULT: clean. Safe to proceed.")
    print("=" * 70)

    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
