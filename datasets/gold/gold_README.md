# Gold Test Set — SENSE Triage System

**Project:** SENSE (Soul of Mediterranean Resilience) — Teta AI
**Dataset version:** 1.0
**Created:** 2026-05-08
**Status:** ⚠️ Draft — pending expert clinical review. **Not yet a validated gold standard.**

---

## What this is

200 Palestinian Arabic child utterances annotated with triage labels for the SENSE pediatric mental health triage system. Each example has a label, an emotion tag, a risk signal, and reviewer notes documenting the annotation reasoning.

The dataset is designed for **evaluation only** — measuring rule-based and ML triage models. It must never be used as training data.

---

## What this is NOT

- **Not validated.** No licensed Palestinian child mental health specialist has reviewed these examples yet. All entries carry `label_verified: false`.
- **Not real children's speech.** All utterances were synthesized to match patterns described in the project's annotation guide. They reflect AI-generated approximations of how children might speak in crisis contexts, not transcribed real speech.
- **Not a diagnostic instrument.** Labels reflect triage urgency (does this need human follow-up, and how fast?) — not clinical diagnoses.
- **Not balanced** in the statistical sense. The set is intentionally weighted toward borderline and high-risk cases, per project decision.

---

## File structure

```
data/gold/
├── gold_test_set.jsonl       # The full 200-example dataset (use this)
├── batch_original.jsonl      # Examples 1–50 (initial set)
├── batch_safe.jsonl          # Examples 51–80 (Safe additions)
├── batch_distressed.jsonl    # Examples 81–130 (Distressed additions)
├── batch_high_risk.jsonl     # Examples 131–175 (High Risk additions)
├── batch_unclear.jsonl       # Examples 176–200 (Unclear additions)
└── README.md                 # This file
```

The batch files exist for traceability — to know which examples came from which annotation pass. For evaluation, use only `gold_test_set.jsonl`.

---

## Label distribution

| Label | Count | Percentage |
|---|---|---|
| Distressed / Needs Support | 70 | 35.0% |
| High Risk / Urgent | 55 | 27.5% |
| Safe / Regulated | 42 | 21.0% |
| Unclear / Need More Context | 33 | 16.5% |
| **Total** | **200** | **100%** |

This distribution is intentional. Per project decision, the set focuses on borderline cases. High Risk is over-represented (vs base rate in any real population) so that **recall on High Risk** can be measured with adequate statistical confidence — this is the metric SENSE prioritizes.

---

## Schema

Each line is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier (`gold_NNN`) |
| `text` | string | The Palestinian Arabic utterance |
| `label` | string | One of four triage labels (see annotation guide §4) |
| `emotion` | string | Primary emotion (controlled vocabulary §8.1) |
| `risk_signal` | string | Specific signal detected (controlled vocabulary §8.2) |
| `source` | string | Always `"gold_test"` |
| `review_status` | string | `"needs_expert_review"` for all current entries |
| `reviewer_type` | string | `"none"` until specialist review |
| `label_verified` | boolean | `false` for all entries |
| `review_notes` | string | English reasoning, ≤200 chars |

---

## How this set was constructed

**Examples 1–50** were created in an earlier annotation pass using emerging conventions.

**Examples 51–200** were created using `docs/annotation-guide.md` v1.0 as the binding rulebook. Each new example was annotated using:

1. The four-label definitions in guide §4.
2. The seven discriminative pairs in guide §5 to handle borderline cases.
3. The controlled vocabularies in guide §8.

Borderline cases were intentionally over-represented because per project decision: "easy cases don't reveal weaknesses; borderline cases separate safe systems from dangerous ones."

For every borderline example, `review_notes` documents:
- The two labels under consideration.
- Which guide rule was applied to choose.
- What specifically the specialist reviewer should examine.

---

## Critical limitations

### 1. No expert clinical review yet
Every example needs to be confirmed or relabeled by a licensed child mental health specialist (psychologist, child psychiatrist, or trauma-trained social worker) familiar with Palestinian children. Until that review happens, **this set is a draft, not a benchmark**.

### 2. Single-annotator pass
Examples were produced by one annotator (AI-assisted). For a publishable benchmark, the annotation guide §11 requires:
- Two independent annotators per example.
- Cohen's kappa ≥ 0.80 overall, ≥ 0.85 for High Risk.
- Specialist arbitration on disagreements.

This work has not been done.

### 3. Synthesized text
The utterances are AI-generated approximations. Real children's speech in crisis contexts may include:
- Non-standard grammar and code-switching the synthesizer underrepresents.
- Disfluencies, repetitions, and self-corrections.
- Religious framings, idioms, and cultural-specific expressions in proportions different from those generated here.
- Indirect/metaphorical content the synthesizer may have flattened.

### 4. No real children were involved
This dataset was built without IRB approval, parental consent, or real child speakers. It reflects what the annotation guide *says* children-in-crisis speech looks like — which may diverge from what children actually say.

### 5. Cultural-religious framing of death
Palestinian Arabic includes religious/cultural references to death and afterlife that may carry different weight than Western diagnostic frameworks assume. The annotation guide §13 flags this as an open question. Specialist review must address this directly — particularly for examples like `gold_154` and `gold_160` which use religious framing.

### 6. Borderline overweight
The intentional focus on borderline cases means the distribution does NOT reflect the base rate of a real screening population. This is appropriate for evaluating model robustness, but means absolute precision/recall numbers from this set should not be quoted as "expected real-world performance."

---

## How to use this set

### For evaluation

```python
import json

with open('data/gold/gold_test_set.jsonl', 'r', encoding='utf-8') as f:
    gold = [json.loads(line) for line in f if line.strip()]

# Run your triage system on each example.text
# Compare predictions to example.label
# Compute: accuracy, macro F1, per-class recall, confusion matrix
# Pay primary attention to High Risk recall
```

### For NOT using this set
- ❌ Do not include in training data, even by mistake. Hash-check before any train/test split.
- ❌ Do not show the labels to the system being evaluated.
- ❌ Do not quote performance numbers from this set as if it were validated.
- ❌ Do not present this set publicly or in publications until expert review is complete.

---

## What needs to happen next

1. **Specialist review.** Submit all 200 examples to a licensed Palestinian child mental health specialist. Priority order:
   - All 55 High Risk examples (highest priority — these drive the safety metric).
   - All borderline examples flagged in `review_notes` (second priority).
   - Remaining Distressed, Safe, and Unclear (third priority).

2. **Second annotator.** Recruit a second annotator (ideally with clinical training in Arabic) to independently re-label the set. Compute Cohen's kappa.

3. **Resolve disagreements** via specialist arbitration, using the annotation guide as reference.

4. **Update `label_verified` to `true`** only after specialist confirmation, and update `reviewer_type` accordingly.

5. **Augment with real transcripts** when available — anonymized, IRB-approved, with proper consent. The synthesized set should be supplemented (not replaced) by real data.

6. **IRB/ethics review.** Birzeit University's research ethics committee should review the project before any real child data is collected.

---

## Reproducibility

To regenerate the merged file from the batch files:

```bash
cd data/gold/
cat batch_original.jsonl batch_safe.jsonl batch_distressed.jsonl \
    batch_high_risk.jsonl batch_unclear.jsonl > gold_test_set.jsonl
wc -l gold_test_set.jsonl  # should report 200
```

---

## Version history

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-05-08 | Initial 200-example draft. 50 from earlier pass + 150 new examples annotated against `annotation-guide.md` v1.0. Awaiting expert review. |

---

**End of README.**