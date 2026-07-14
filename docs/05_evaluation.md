# 05 — Evaluation & Data Integrity

## Metrics hierarchy (read this first)

SENSE is a safety system. Metrics are ranked accordingly:

1. **High Risk recall** — the headline safety metric. Missing a Red turn is the worst
   possible failure.
2. **Zero dangerous downgrades** — the non-negotiable invariant. Any Red-labeled turn in
   a session must produce a final Red session label. This must hold in **every** run.
3. **Macro F1** — secondary.
4. **Overall accuracy** — reported for reference only. It is *not* an optimization target
   (see "The 41% number" below).

## Datasets

| Set | Size | File | Notes |
|---|---|---|---|
| Gold test set | 544 records | `dataset_sorted_expanded.jsonl` (IDs `ex_001`–`ex_544`) | Never used as training/few-shot data |
| Gold eval subset | n=250 | — | Used in the reported gold evaluation |
| Unseen set | n=75 | — | Held out; measures generalization |

**Expert review flags:** every High Risk example and every ambiguous Unclear example is
flagged `needs_expert_review`. No clinical data artifact is considered *verified* until a
specialist has reviewed it. Unreviewed items are usable for development but must be
labeled as unverified in any report.

## Reported results

| Set | Overall accuracy | High Risk | Dangerous downgrades |
|---|---|---|---|
| Gold (n=250) | 100% | — | 0 |
| Unseen (n=75) | 41% | 100% precision | **0** |

### The 41% number — conservative by design

The unseen-set overall accuracy is low **because the system escalates on ambiguity**:
uncertain inputs are pushed toward Unclear/Distressed instead of defaulting to Safe.
In a triage context, a false escalation costs a specialist a few minutes of review; a
false de-escalation can cost a child the help they needed. The 41% is therefore a
consequence of the safety posture, not a defect — and it coexists with 100% High Risk
precision and zero downgrades on the same set.

## Evaluation protocol (mandatory steps, in order)

1. **Run `check_data_leakage.py`** — required before *every* evaluation cycle. Confirms no
   gold-set example leaked into prompts, RAG resources, or few-shot examples.
2. **Fix stochasticity:** LLM+RAG runs use `temperature=0` and **N=3 iterations per
   example**; disagreements between iterations are logged and inspected.
3. **Hash the gold set:** each run records the **SHA-256** of `dataset_sorted_expanded.jsonl`
   so results are traceable to an exact dataset version.
4. **Save to a timestamped folder** — never overwrite a previous run.
5. Generate metrics and analysis:
   - `classification_metrics.py` — accuracy, per-class precision/recall, macro F1
   - `confusion_matrix.py` — 4×4 tier confusion matrix
   - `error_analysis.py` — per-error inspection, downgrade detection
6. **Verify the invariant:** zero dangerous downgrades. If violated, the run fails
   regardless of every other metric.

## Quick checks

- `smoke_test.py` — fast end-to-end sanity check after any prompt or pipeline change.
- `run_rule_based.py` — runs the rule-based analyzer standalone over a dataset.

## Rules that must never be broken

- The gold set is **never** used as training data, few-shot examples, or RAG content.
- Never report a metric without its dataset SHA-256 and run timestamp.
- Never present unreviewed clinical examples as verified.
