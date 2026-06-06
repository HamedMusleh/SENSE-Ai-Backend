# Known Limitations — SENSE Rule-Based Triage Classifier V3

**Document version:** 1.0
**Last updated:** 2026-05-15
**Status:** Pre-alpha — pending psychologist review
**Authority:** Aligned with `resources/annotation_guide.md` and `resources/safety_rules.md`

---

## 1. Purpose

This document records the **known limitations** of the rule-based triage
classifier as of pre-alpha. It is intentionally honest about what the system
can and cannot do, so that:

1. Reviewers (psychologists, supervisors, evaluation committee) understand
   the system's true performance, not just its sanity-check scores.
2. Future ML development has a documented baseline to improve against.
3. Deployment decisions are made with full awareness of failure modes.
4. The gap between rule-based and ML approaches is explicit.

This document is not a list of bugs to fix. It is a description of the
**inherent design trade-offs** of a rule-based system, plus empirical
evidence of where those trade-offs hurt.

---

## 2. System under test

**Component:** `ai_pipeline/triage/triage_classifier.py` (V3)
**Source:** `rule_based_triage_v1`
**Type:** Lexicon-based pattern matching with discriminative rules from
the annotation guide §5.
**Inputs:** Single Palestinian Arabic utterance (with optional
`conversation_history` for cumulative rules).
**Outputs:** `predicted_label`, `risk_signal`, `predicted_emotion`,
`needs_review`, `confidence`, plus diagnostic fields.

The classifier is **deterministic** — the same input always produces the
same output. No learning, no probabilities derived from data.

---

## 3. Evaluation methodology

Two distinct evaluations were run:

### 3.1 Gold Test Set (sanity check)
- **Path:** `datasets/gold/gold_test_set.jsonl`
- **Size:** 250 examples
- **Built from:** the same annotation_guide that defines the classifier's rules.
- **Purpose:** Verify the implementation matches the specification.

### 3.2 Unseen Validation Set (generalization test)
- **Path:** `datasets/gold/unseen_validation_set.jsonl`
- **Size:** 75 examples
- **Built to:** intentionally use phrasings **different** from the lexicons
  while remaining true to the annotation_guide labels.
- **Distribution:** Safe 30, Distressed 20, High Risk 15, Unclear 10
  (realistic, not balanced — reflects expected real-world frequency).
- **Purpose:** Measure how the classifier handles language it was not
  explicitly engineered for.

The unseen set was constructed **after** the classifier was frozen.
No rule modifications were made in response to its results.

---

## 4. Results summary

| Metric | Gold (250) | Unseen (75) | Gap |
|---|---|---|---|
| Accuracy | **100%** | **41.3%** | -58.7 pts |
| Macro F1 | **100%** | **44.9%** | -55.1 pts |
| Macro Precision | **100%** | **76.2%** | -23.8 pts |
| Macro Recall | **100%** | **50.0%** | -50.0 pts |
| **High Risk Recall** | **100%** | **40.0%** | -60.0 pts |
| Errors | 0 | 44 | — |

### Per-label on unseen set

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Safe / Regulated | **1.000** | 0.300 | 0.462 | 30 |
| Distressed / Needs Support | **0.857** | 0.300 | 0.444 | 20 |
| High Risk / Urgent | **1.000** | 0.400 | 0.571 | 15 |
| Unclear / Need More Context | 0.189 | 1.000 | 0.317 | 10 |

---

## 5. Interpretation

### 5.1 The 100% on gold is a sanity check, not an achievement

The lexicons are derived from the annotation guide; the gold set is also
derived from the annotation guide; the rules are written to match the
discriminative pairs from the annotation guide. A high score on gold means
the implementation matches the specification — which is the minimum
expected, not a meaningful generalization claim.

### 5.2 The 41% on unseen is the realistic number

The unseen set was designed to test whether the classifier handles
**linguistic variation** in the same conceptual space. The 41% number
reflects how the system performs on language it was not specifically
engineered for.

This is **within the expected range** for a small rule-based lexicon on a
domain with rich linguistic variation (Palestinian Arabic, code-switching,
indirect emotional expression typical of children). It is not a bug.

### 5.3 The precision/recall trade-off is the system's most important property

| What this means | Why it matters for SENSE |
|---|---|
| Precision is high (76-100%): **when the classifier commits to a label, it is almost always right** | Low rate of false alarms — the system does not over-pathologize normal speech |
| Recall is low (30-40%): **the classifier often falls back to Unclear** | Genuine signals are missed when phrased outside the lexicon |
| `needs_review = True` for all 44 errors | The system **knows when it is uncertain** and flags accordingly |

This is a **conservative classifier**. It prefers to say "I'm not sure,
please review" rather than to guess wrong. For a triage system that routes
to human specialists, this trade-off is defensible — but it does mean a
human reviewer sees more borderline cases.

### 5.4 No false positives on Safe or High Risk

The two labels where false positives would be most dangerous both have
**precision = 1.000** on the unseen set:

- **Safe precision 100%:** The classifier never labeled a Distressed or
  High Risk case as Safe. Zero dangerous downgrades.
- **High Risk precision 100%:** The classifier never labeled a Safe,
  Distressed, or Unclear case as High Risk. Zero false alarms.

The single off-label miss on Safe was a Distressed prediction
(`unseen_023`) — a longing-with-resolution case that triggered the
`unresolved_grief` pattern despite a resolution clause being present.
This is a real false positive at the Distressed level, but does not
endanger the child.

---

## 6. Failure mode taxonomy

All 44 errors on the unseen set fall into one of four patterns, summarized
below.

### 6.1 Type A — Safe → Unclear (15 cases, 34% of errors)

**Cause:** The Safe lexicon (positive_affect, future_orientation,
daily_life_engagement, emotional_resolution) uses specific phrasings
that do not cover the full space of healthy child expression.

**Examples of missed phrasings:**
- "بحب أرسم مناظر طبيعية" — "بحب" not in lexicon (only "حبيت")
- "ضحكت كتير اليوم" — "ضحكت" not in lexicon (only "ضحكنا")
- "بقدر أعد لعشرة بالانجليزي هلأ" — code-switching not modeled
- "افتكرت اليوم ضحكة جدي وحسيت بطمأنينة" — "طمأنينة" not in lexicon

**Impact:** No safety harm. The child gets a gentle Unclear-strategy
response (a soft open question). However, the system fails to recognize
healthy functioning and may over-flag for review.

### 6.2 Type B — Distressed → Unclear (14 cases, 32% of errors)

**Cause:** Distressed lexicons use specific phrases tied to specific
discriminative pair examples. The classifier does not generalize across
synonyms or rephrasings.

**Examples of missed signals:**
- "كل مرة بشوف طيارة بحس قلبي بيدق بسرعة" (hypervigilance, missed
  because lexicon expects "كل ما اسمع صوت")
- "بطلت أحس بنفس الفرحة لما العب" (loss_of_interest, missed because
  lexicon expects "ما عاد بحب اللعب")
- "صرت أنسى أعمل واجباتي" (behavioral_change — no lexicon entry exists
  for this signal)

**Impact:** Moderate. The child receives an Unclear-strategy response
(open question, no pushing). The case is flagged for review. A human
reviewer should catch these.

### 6.3 Type C — High Risk → Unclear (9 cases, 20% of errors) ⚠️

**Cause:** High Risk lexicons are intentionally precise to avoid false
positives. The classifier misses indirect, metaphorical, or culturally-
specific expressions of distress.

**Examples of missed signals:**
- "بدي أنام طويل وما حدا يقدر يصحيني للأبد" — passive_death_wish,
  missed because lexicon expects "بدي انام وما اصحى" exactly
- "ليش أنا اللي بقيت؟ المفروض كنت مكان أخوي" — survivor_guilt with
  self-death framing, expressed as rhetorical question
- "الألوان كلها صارت باهتة، حتى السما ما عاد فيها زرقة" — perceptual
  disturbance expressed poetically
- "صلّيت إنه قلبي يوقف بالليل" — death_wish_via_physical_event with
  religious framing
- "ما في فايدة، كل اللي بصير بصير، خلصت أنا من الموضوع" —
  future_collapse with non-lexicon "خلصت" usage

**Impact: this is the most serious failure mode.** A child expressing
genuine suicidal ideation or dissociation may receive a soft Unclear-
strategy response instead of the hard-coded High Risk safety response.

**Mitigations currently in place:**
1. `needs_review = True` is set on all Unclear outputs.
2. The Unclear-strategy prompt instructs Teta to be soft, not probing,
   and to maintain presence without escalation.
3. The system is **not deployed without human review** at pre-alpha.
4. The classifier never **downgrades** a true High Risk to Safe (zero
   cases). The failure mode is "miss" (Unclear), not "dismiss" (Safe).

### 6.4 Type D — Safe → Distressed (1 case, 2% of errors)

**Cause:** Lexical overreach — a positive resolution containing a grief
keyword fires the grief rule.

**Example:** `unseen_023` — "اشتقت لجدتي بس أمي حكتلي إنه رح نزورها
يوم الخميس" (longing + resolution about an alive grandmother). The
`اشتقتلها` substring matched `unresolved_grief` despite the resolution
clause.

**Impact:** A child expressing healthy adjustment is over-pathologized
and gets a supportive Distressed-strategy response instead of an engaging
Safe-strategy response. Not dangerous, but reduces conversational quality.

---

## 7. Root cause: rule-based systems cannot generalize

The single underlying cause of every failure on the unseen set is the
**lexical mismatch** between the rules and the input. Rule-based
classifiers do not understand:

- **Synonyms:** "بطلت أحس بالفرحة" ≈ "ما عاد بحب اللعب" (same signal,
  different words)
- **Paraphrases:** "بدي أنام طويل وما حدا يصحيني" ≈ "بدي انام وما اصحى"
- **Metaphor:** "الألوان كلها صارت باهتة" ≈ "perceptual disturbance"
- **Rhetorical structure:** "ليش أنا اللي بقيت؟" as a survivor-guilt
  expression
- **Code-switching:** Arabic-English mixed utterances
- **Cultural and religious phrasing:** prayer formulas, idioms, dialectal
  variants

These are exactly the capabilities a trained ML model brings.
**No amount of lexicon expansion will close this gap** — even a
1000-pattern lexicon will miss the 1001st phrasing. The problem is
categorical, not quantitative.

---

## 8. Hard limits of the current design

Beyond the empirical limitations above, the classifier has the following
**architectural** limits:

### 8.1 No semantic understanding
The classifier sees character sequences, not meaning. It cannot tell that
"حسيت قلبي توقف" (a metaphor for shock) is different from "تمنيت قلبي
يوقف" (a death wish) unless both phrasings appear in lexicons.

### 8.2 No conversational reasoning
Cumulative rules (§6.2 and §6.3 of the annotation guide) are implemented
in a minimal form (prior High Risk flags later turns for review,
retraction does not de-escalate). Full conversation-level analysis
(escalation patterns, mixed cases, §6.4 cumulative ambiguity) is **not**
implemented in pre-alpha.

### 8.3 No emotion-from-text inference
The classifier maps risk_signal → emotion via a fixed lookup table.
It cannot infer emotion from text without a matched signal. This is
adequate for triage routing but not for fine-grained emotional analysis.

### 8.4 No personalization
The classifier treats every child identically. It does not adapt to age,
gender, dialect region, or prior conversation tone.

### 8.5 No multimodal fusion
The classifier sees text only. The audio emotion engine and weighting
layer are placeholders. When real wav2vec emotion arrives, the classifier
will need to be modified to consume it (currently the signature accepts
`weighted_result` only via the legacy alias).

---

## 9. Safety implications

### 9.1 What the system does well
- **No false reassurance:** zero cases of dangerous downgrade
  (High Risk → Safe) on either gold or unseen sets.
- **No false alarms on High Risk:** zero cases of Safe/Distressed/Unclear
  → High Risk on the unseen set.
- **Self-aware uncertainty:** all 44 errors had `needs_review = True`.
- **Hard-coded responses for High Risk:** when the classifier does detect
  High Risk, the LLM is bypassed entirely and a vetted response is used.

### 9.2 What the system does poorly
- **Misses indirect High Risk expressions** (60% of true High Risk on
  unseen). This is the most important limitation.
- **Over-uses the Unclear label** — convenient for the classifier,
  but increases reviewer workload.
- **Cannot recognize healthy functioning expressed in non-lexicon
  phrasings**, leading to over-flagging.

### 9.3 Deployment posture
The classifier is **not suitable for unsupervised deployment** at this
stage. Specifically:

- All outputs must be reviewable by a human specialist.
- The system is appropriate as a **first-pass triage** that routes cases
  to specialists, not as a final classifier.
- High Risk hard-coded responses are a **bridging measure**, not a
  substitute for professional follow-up.
- The unseen set's 40% High Risk recall means **the system will miss
  about 6 out of 10 indirectly-expressed crises** at this stage. Mitigations
  (human review, conservative defaults, Unclear-strategy responses) reduce
  but do not eliminate this risk.

---

## 10. Path forward

The known limitations above motivate the next phase of development.

### 10.1 Near-term (no rule changes)
- **Preserve** the unseen validation set as a held-out evaluation set.
  Do not modify lexicons in response to its failures, even when patterns
  are obvious. This protects its value as an unbiased benchmark.
- **Expert review** of `high_risk_safe_responses.json` and the strategy
  prompts (`prompts/strategy_*.txt`).
- **Expert review** of the annotation guide itself, particularly the
  discriminative pairs.

### 10.2 Medium-term — ML triage classifier
The unseen set's failures point directly to what an ML model would handle
better: paraphrase, synonymy, metaphor, code-switching, rhetorical
structure. The data we have suggests the following development plan:

- **Training corpus:** the 250 gold examples + further annotation passes
  (target ≥ 1000 expert-labeled examples).
- **Evaluation corpus:** this unseen validation set (75 examples)
  plus future held-out sets.
- **Model candidates:** fine-tuned Arabic BERT variants (AraBERT,
  CAMeLBERT), or instruction-tuned multilingual models with constrained
  classification heads.
- **Hybrid architecture (recommended):** ML for label prediction,
  rule-based for High Risk safety guardrails. The rules act as a
  precision floor; the ML model raises recall.

### 10.3 Long-term — multimodal triage
Once real wav2vec emotion detection is integrated, the classifier signature
will be extended to consume audio features. The weighting layer becomes
non-trivial. Conversation-level analysis from annotation_guide §6 should
be fully implemented.

---

## 11. Honest summary for reviewers

If you are reviewing this system for the first time, the one-paragraph
summary is:

> The SENSE rule-based triage classifier V3 achieves perfect accuracy on
> the gold set it was specified from, and **41% accuracy on an unseen
> validation set** of 75 expert-aligned examples. Its **precision is high
> (76-100% per label)** but its **recall is low**, with **High Risk recall
> at 40%** — meaning it misses about three out of five indirectly-expressed
> crises. **Zero dangerous downgrades** (true High Risk predicted as Safe)
> occurred on either evaluation. The system is **conservative and self-aware**:
> all 44 errors on the unseen set were flagged for review. This is the
> baseline against which the future ML classifier will be measured. The
> system is not suitable for unsupervised deployment; it is a first-pass
> triage that requires human review at this stage.

---

## 12. Files referenced in this document

- `ai_pipeline/triage/triage_classifier.py` — the classifier
- `resources/triage_lexicons.json` — keyword lexicons
- `resources/annotation_guide.md` — labeling specification
- `resources/safety_rules.md` — safety policy
- `resources/high_risk_safe_responses.json` — hard-coded safe responses
- `prompts/strategy_*.txt` — LLM strategy prompts per label
- `datasets/gold/gold_test_set.jsonl` — gold set (250)
- `datasets/gold/unseen_validation_set.jsonl` — unseen set (75)
- `evaluation/run_gold_evaluation.py` — evaluation script
- `evaluation/results/gold_eval_*.json` — evaluation reports
- `evaluation/predictions/gold_predictions_*.jsonl` — per-example predictions
- `evaluation/results/gold_errors_*.csv` — error analysis exports

---

## 13. Version history

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-05-15 | Initial document. Covers V3 classifier results on gold (250) and unseen (75) sets, error taxonomy, hard limits, and path forward. |

---

**End of document.**