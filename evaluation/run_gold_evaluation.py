import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from ai_pipeline.triage.triage_classifier import classify

ROOT = Path(__file__).resolve().parents[1]

GOLD_FILE = ROOT / "datasets" / "gold" / "unseen_validation_set.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"
PRED_DIR = ROOT / "evaluation" / "predictions"

LABELS = [
    "Safe / Regulated",
    "Distressed / Needs Support",
    "High Risk / Urgent",
    "Unclear / Need More Context",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def run_classifier(text):
    result = classify(text)

    return {
        "label": result.get("predicted_label", "UNKNOWN"),
        "emotion": result.get("predicted_emotion", "unknown"),
        "risk_signal": result.get("risk_signal", "unknown"),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "needs_review": bool(result.get("needs_review", False)),
        "source": result.get("source", "unknown"),
        "matched_signals": result.get("matched_signals", []),
        "review_notes": result.get("review_notes", ""),
    }

def safe_div(a, b):
    return a / b if b else 0.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    gold_rows = load_jsonl(GOLD_FILE)
    predictions = []

    confusion = {true: Counter() for true in LABELS}
    signal_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    confidence_buckets = defaultdict(lambda: {"total": 0, "correct": 0})
    errors = []

    for row in gold_rows:
        text = row["text"]
        true_label = row["label"]
        true_signal = row.get("risk_signal", "unknown")

        pred = run_classifier(text)

        pred_label = pred["label"]
        pred_signal = pred["risk_signal"]
        confidence = pred["confidence"]

        correct = pred_label == true_label

        confusion[true_label][pred_label] += 1

        signal_stats[true_signal]["total"] += 1
        if correct:
            signal_stats[true_signal]["correct"] += 1

        bucket = f"{int(confidence * 10) / 10:.1f}-{int(confidence * 10) / 10 + 0.1:.1f}"
        confidence_buckets[bucket]["total"] += 1
        if correct:
            confidence_buckets[bucket]["correct"] += 1

        item = {
            "id": row.get("id"),
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": correct,
            "true_risk_signal": true_signal,
            "pred_risk_signal": pred_signal,
            "confidence": confidence,
            "needs_review": pred["needs_review"],
            "review_notes": row.get("review_notes", ""),
        }

        predictions.append(item)

        if not correct:
            errors.append(item)

    total = len(predictions)
    correct_total = sum(1 for p in predictions if p["correct"])
    accuracy = safe_div(correct_total, total)

    per_label = {}

    for label in LABELS:
        tp = sum(1 for p in predictions if p["true_label"] == label and p["pred_label"] == label)
        fp = sum(1 for p in predictions if p["true_label"] != label and p["pred_label"] == label)
        fn = sum(1 for p in predictions if p["true_label"] == label and p["pred_label"] != label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_label[label] = {
            "support": sum(1 for p in predictions if p["true_label"] == label),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = sum(per_label[l]["f1"] for l in LABELS) / len(LABELS)
    macro_recall = sum(per_label[l]["recall"] for l in LABELS) / len(LABELS)
    macro_precision = sum(per_label[l]["precision"] for l in LABELS) / len(LABELS)

    high_risk_recall = per_label["High Risk / Urgent"]["recall"]

    signal_accuracy = {
        signal: {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": safe_div(data["correct"], data["total"]),
        }
        for signal, data in signal_stats.items()
    }

    calibration = {
        bucket: {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": safe_div(data["correct"], data["total"]),
        }
        for bucket, data in sorted(confidence_buckets.items())
    }

    rule_effectiveness = Counter(
        p["pred_risk_signal"] for p in predictions if p["correct"]
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_file": str(GOLD_FILE),
        "total_examples": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "high_risk_recall": high_risk_recall,
        "per_label": per_label,
        "confusion_matrix": {
            true: {pred: confusion[true][pred] for pred in LABELS}
            for true in LABELS
        },
        "per_risk_signal_accuracy": signal_accuracy,
        "confidence_calibration": calibration,
        "rule_effectiveness_correct_counts": dict(rule_effectiveness),
        "num_errors": len(errors),
        "errors": errors,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"gold_eval_{ts}.json"
    pred_path = PRED_DIR / f"gold_predictions_{ts}.jsonl"
    error_csv_path = RESULTS_DIR / f"gold_errors_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(error_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id",
            "text",
            "true_label",
            "pred_label",
            "true_risk_signal",
            "pred_risk_signal",
            "confidence",
            "needs_review",
            "review_notes",
        ])
        writer.writeheader()
        for e in errors:
            writer.writerow({
                k: e.get(k, "")
                for k in writer.fieldnames
            })

    print("=" * 70)
    print("GOLD EVALUATION REPORT")
    print("=" * 70)
    print(f"Total examples:      {total}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro Precision:     {macro_precision:.4f}")
    print(f"Macro Recall:        {macro_recall:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"High Risk Recall:    {high_risk_recall:.4f}")
    print(f"Errors:              {len(errors)}")
    print()

    print("Per-label metrics:")
    for label, m in per_label.items():
        print(
            f"- {label}: "
            f"P={m['precision']:.3f}, "
            f"R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, "
            f"support={m['support']}"
        )

    print()
    print("Saved files:")
    print(f"- Report:      {json_path}")
    print(f"- Predictions: {pred_path}")
    print(f"- Errors CSV:  {error_csv_path}")


if __name__ == "__main__":
    main()