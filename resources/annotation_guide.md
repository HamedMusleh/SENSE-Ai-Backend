# Annotation Guide — SENSE Gold Test Set

**Project:** SENSE (Soul of Mediterranean Resilience) — Teta AI Triage System
**Document version:** 1.0
**Last updated:** 2026-05-08
**Status:** Draft — pending expert clinical review

---

## 1. Purpose of this document

This guide defines the rules by which child utterances in Palestinian Arabic are assigned to one of four triage labels in the SENSE Gold Test Set. It exists to ensure that:

1. Annotation is **consistent** across all 200 examples in the gold set.
2. Future annotators (including clinical reviewers) can re-label any example using these rules and arrive at the same answer.
3. The Gold Test Set can serve as a defensible benchmark for evaluating both rule-based and ML triage systems.
4. Disagreements during expert review can be resolved by reference to written rules rather than intuition.

**This document is not clinical advice and does not constitute a diagnostic instrument.** It operationalizes triage decisions for an AI screening system that routes children to human specialists.

---

## 2. Scope and limits

### What this guide covers
- Classification of single child utterances and short multi-turn dialogues (3–5 turns) in Palestinian Arabic.
- Children aged 6–14 in crisis or post-crisis contexts (displacement, conflict exposure, bereavement, ongoing instability).
- Triage decisions only — assigning urgency, not diagnosing conditions.

### What this guide does NOT cover
- Adult speech, adolescent speech above 14, or pre-verbal children.
- Modern Standard Arabic (MSA), Egyptian, Gulf, or Maghrebi dialects.
- Diagnostic determinations (PTSD, depression, anxiety disorders, etc.) — these are exclusively the domain of licensed specialists.
- Treatment recommendations.
- Cases where the input is non-Arabic or cannot be reliably transcribed.

---

## 3. Core principles (binding for all annotation decisions)

These five principles override any specific rule below if they conflict.

### Principle 1 — Safety over accuracy
Missing a High Risk case is significantly worse than a false positive. When in genuine doubt between Distressed and High Risk, **annotators escalate to High Risk**. When in genuine doubt between Safe and Distressed, annotators escalate to Distressed.

### Principle 2 — Conservative escalation under ambiguity
If the available text is genuinely ambiguous and could plausibly indicate any severity level, the correct label is **Unclear / Need More Context** — not a guess between Safe and Distressed. Unclear is a real, valid label, not a fallback for difficult cases that should have been Distressed or High Risk.

### Principle 3 — Triage, not diagnosis
Annotators classify the **urgency of human follow-up needed**, not the underlying condition. A child describing classic PTSD symptoms is Distressed (or High Risk if dissociative); the label does not name the condition.

### Principle 4 — Cultural and developmental authenticity
Palestinian children express distress differently than adults and differently than Western diagnostic frameworks predict. Indirect expressions, somatic complaints, religious framing, and silence carry meaning. Annotators do not over-pathologize culturally normal expressions, nor under-recognize indirect distress.

### Principle 5 — The text speaks; the annotator does not invent
Annotators classify only what is **textually present or strongly inferable**. They do not invent backstory, assume context not stated, or project what they imagine the child "really meant." When context is needed and absent, the label is Unclear.

---

## 4. The four labels

### 4.1 Safe / Regulated

**Definition:** The child demonstrates emotional regulation, functional coping, or age-appropriate response to their circumstances. The utterance does not indicate clinically significant distress requiring follow-up.

**Required textual evidence (at least one):**
- Positive affect (joy, pride, comfort, contentment).
- Engagement with daily life (food, family, play, school, friendships).
- Future orientation (plans, anticipation, hopes).
- Resolved emotional content (a difficulty mentioned and processed).
- Functional coping (fear that resolves, sadness that is acknowledged but not consuming).

**Important nuances:**
- Safe does **not** mean "no negative emotion." A child who cries, fears the dark, or misses someone can be Safe if the emotion is contained and self-resolving.
- Safe does **not** require the child to be in a safe environment. A child in a refugee tent who reports playing with a new friend is Safe / Regulated despite the unsafe context.
- Mild fear, mild sadness, and mild frustration are compatible with Safe **only when** the utterance also shows resolution, contextualization, or continued functioning.

**Reference examples from the gold set:**
- `gold_001` — "اليوم أمي طبخت مقلوبة... أكلت كتير" (focus on food and home routine)
- `gold_004` — "خايف شوي من الظلام لما الكهربا بتروح، بس بعدين بنام" (fear that resolves)
- `gold_008` — "زعلت من أخوي كتير اليوم، عيطت، بس هسا خلص تصالحنا" (complete emotional arc)

---

### 4.2 Distressed / Needs Support

**Definition:** The child shows clinically meaningful distress that warrants supportive human follow-up, but does **not** show acute danger to self, dissociative collapse, or explicit/implicit death wishes.

**Required textual evidence (at least one):**
- Functional impairment (sleep disturbance, concentration loss, school avoidance, withdrawal from previously enjoyed activities).
- Persistent fear, anxiety, sadness, or grief that is not self-resolving.
- Hypervigilance or startle responses the child cannot control.
- Avoidance behaviors (of places, people, topics, memories).
- Somatic distress responses (stomach pain, body aches) tied to emotional content.
- Suppressed grief or emotional numbing **without** dissociative features.
- Survivor stress that is **not** about own death (e.g., guilt about not helping a sibling).
- Social withdrawal or shame that is circumstantial, not absolute.

**Critical exclusions (these push to High Risk, not Distressed):**
- Any explicit or implicit death wish for self.
- Dissociative phenomena (depersonalization, derealization, sensory disturbances).
- Complete loss of future orientation ("ما في إشي" framing existence itself).
- Survivor guilt that frames own death as preferable.

**Reference examples from the gold set:**
- `gold_013` — "كل ما أسمع صوت عالي بركض عند أمي... مش قادر أوقف حالي" (uncontrollable startle response)
- `gold_018` — "أنا ما بكيت لما شفت بيتنا... بس قلبي وجعني" (somatic suppressed grief, not dissociation)
- `gold_025` — "أخوي الصغير بعيط كل الليل... ما عرفت كيف أساعده" (helplessness about other, not self)

---

### 4.3 High Risk / Urgent

**Definition:** The child's utterance indicates acute clinical concern requiring immediate specialist escalation. This includes any expression — explicit or implicit — of self-erasure, dissociative collapse, or perceptual disturbance.

**Strict inclusion rules — High Risk applies if AND ONLY IF at least one of the following is textually present:**

1. **Explicit death wish for self.** Direct statements such as "بدي أموت", "بدي أنام وما أصحى", "لو ما كنت موجود".

2. **Implicit death wish or self-erasure.** Indirect framings where death, sleep-without-waking, disappearance, or "going" (روح / أروح) are expressed as desirable, especially when paired with caregiver relief or burden language. Example: "رح يكون أحسن لما أروح... أمي رح ترتاح" (`gold_037`).

3. **Death idealization.** Framing the dead as better off, at peace, or to be envied. Example: "اللي ماتوا أحسن منا... هم ارتاحوا" (`gold_033`).

4. **Existential indifference to own being.** Statements that frame own existence vs non-existence as equivalent or meaningless. Example: "ما في فرق بين إذا كنت هون أو لأ" (`gold_041`).

5. **Dissociative phenomena.** Depersonalization ("جسمي مش جسمي", "شايف حالي من برا"), derealization ("العالم وقف وأنا مش فيه"), persistent perceptual disturbance not explained by physical illness ("ما بشوف ألوان", "كل إشي رمادي").

6. **Complete future collapse.** Total loss of future orientation in a child, especially with finality markers like "خلص". Example: "ما في إشي استناه ... خلص" (`gold_035`).

7. **Survivor guilt with self-death framing.** Guilt about another's death/injury combined with implicit or explicit wish that self had died instead, or self-blame framed as deserving punishment.

8. **Death wish attached to physical sensation.** Wishes for cardiac arrest, breathing to stop, or other physical processes to fail. Example: "حسيت إنو قلبي وقف... وتمنيت ما يرجع يشتغل" (`gold_040`).

**Critical exclusions (these are NOT High Risk on text alone):**
- Sadness, grief, or crying without death framing.
- Fear of dying from external threat (this is realistic threat appraisal, not death wish — usually Distressed).
- Mention of someone else's death without self-implication.
- Religious statements about death as part of normal cultural framing (e.g., "الله يرحمها" about a dead grandmother) when not paired with self-directed content.
- Sleep disturbance without "ما أصحى" or equivalent.

**Reference examples from the gold set:**
- `gold_039` — "لو نمت وما صحيت كان أحسن" (passive death wish)
- `gold_034` — "حاسس إنو جسمي مش جسمي... زي ما أنا شايف حالي من برا" (depersonalization)
- `gold_037` — "رح يكون أحسن لما أروح... أمي رح ترتاح" (implicit death wish + perceived burdensomeness)

---

### 4.4 Unclear / Need More Context

**Definition:** The text is genuinely insufficient to assign Safe, Distressed, or High Risk. The system needs additional turns, context, or follow-up before a triage decision can be made.

**Valid reasons for Unclear:**
- Minimal or single-word responses with no emotional valence ("تمام", "أوكي", "مش عارف").
- Child explicitly refuses to disclose ("ما بدي أحكي", "صار إشي بس ما بدي أحكي عنو").
- Caregiver-imposed silence on the topic.
- Vague somatic complaints with no description ("بحس بإشي هون").
- Cognitive avoidance or retrieval blocks under questioning ("بنسى كل إشي").
- Emotional inconsistency the child cannot explain, when content is missing.
- Deflection to a third party without revealing own state.

**Critical rules:**
- Unclear is **not** a safe default for difficult cases. If there is enough text to lean toward Distressed or High Risk, those labels apply — Principle 1 (safety over accuracy) takes priority.
- Unclear means "the system should ask another question or escalate to a human for clarification" — not "probably fine."
- A retrieval block under questioning ("لما تسألني بنسى") may itself be a dissociative signal. Annotate Unclear, but flag in `review_notes` for specialist consideration.

**Reference examples from the gold set:**
- `gold_043` — "صار إشي بس ما بدي أحكي عنو هون" (refusal to disclose)
- `gold_046` — "أمي قالتلي ما أحكي عن هاد الموضوع" (caregiver-imposed silence)
- `gold_047` — "بحس بإشي هون" (vague somatic, no description)

---

## 5. Discriminative pairs — how to decide between adjacent labels

These are the contrast cases where annotation systems most often fail. Each pair gives a decision rule and a reference comparison.

### Pair A — Normal grief vs passive death ideation

| Feature | Distressed (grief) | High Risk (death ideation) |
|---|---|---|
| Subject of loss | Someone or something else | Own existence framed as wanted-to-end |
| Self-direction | Self continues, mourning the loss | Self framed as better off gone/asleep/erased |
| Marker phrases | "بحس بفقدانه", "اشتقتلها", "وجعني قلبي" | "بدي أنام وما أصحى", "لو ما كنت موجود", "أحسن لو رحت" |

**Decision rule:** If the death/disappearance language refers to the **self** (or self-existence), it is High Risk regardless of how indirect. If it refers only to another person or object, it is Distressed.

**Compare:** `gold_015` (grief about grandmother — Distressed) vs `gold_039` (wanting not to wake — High Risk).

---

### Pair B — Sleep fear vs trauma-linked sleep avoidance vs death wish in sleep framing

| Marker | Label |
|---|---|
| "خايف من الظلام بس بنام" | Safe |
| "ما بقدر أنام... بفكر كتير" | Distressed (rumination) |
| "بصحى من كوابيس عن القصف" | Distressed (trauma-linked) |
| "بدي أنام وما أصحى" / "لو نمت وما صحيت كان أحسن" | High Risk (passive death wish) |

**Decision rule:** Sleep difficulty alone is Distressed. Adding "وما أصحى", "وما أفيق", "للأبد", or equivalent finality markers escalates to High Risk.

---

### Pair C — Withdrawal vs dissociative shutdown

| Feature | Distressed (withdrawal) | High Risk (dissociation) |
|---|---|---|
| Connection to body | Intact — "بضل قاعد", "ما بدي أطلع" | Disrupted — "جسمي مش جسمي", "ما حسيت بطعمه" + body absence |
| Connection to reality | Intact, just disengaged | Disrupted — "العالم وقف", "زي مش حقيقي" |
| Sensory perception | Normal | Distorted (color loss, sound absence, time distortion) |

**Decision rule:** Passive withdrawal where the child remains oriented to body and reality is Distressed. Any textually-present sensory or self-perceptual disturbance is High Risk. Note that `gold_022` ("حطيت الأكل بتمي وما حسيت بطعمه") is a borderline case — sensory blunting alone, without other dissociative features, is Distressed; but if combined with depersonalization in the same dialogue, it becomes High Risk.

---

### Pair D — Emotional numbing vs dissociation

| Feature | Distressed (numbing) | High Risk (dissociation) |
|---|---|---|
| What is absent | Emotional response | Connection to self/reality |
| Child's framing | "ما بحس بإشي" (about feelings) | "ما بحس بحالي" / "مش أنا" (about self) |
| Self-coherence | Intact | Disrupted |

**Decision rule:** Numbing ("can't feel emotions") = Distressed. Disownership of self ("not me", "outside myself") = High Risk.

---

### Pair E — Child realistic fear of death vs death wish

| Feature | Distressed/Safe (realistic fear) | High Risk (death wish) |
|---|---|---|
| Direction of wish | Wants to live, fears dying | Wants death, fears living/continuing |
| Trigger | External threat (bombing, illness, danger) | Internal state (hopelessness, burden, escape) |
| Example | "بخاف يصير إشي ونموت" | "بدي أموت" / "أحسن لو ما كنت" |

**Decision rule:** Fear of being killed by external events is **not** a death wish. It is realistic threat appraisal in a crisis context — usually Distressed (occasionally Safe if the child also reports coping). A death wish runs in the opposite direction: the child wants the death.

---

### Pair F — Survivor guilt about other vs survivor guilt with self-death framing

| Feature | Distressed | High Risk |
|---|---|---|
| Guilt content | "ما عرفت أساعده", "كنت لازم أعمل إشي" | "أنا اللي خليتو يصير", "كان لازم أكون أنا بدالو" |
| Self-direction | Inadequacy, helplessness | Self-blame with death-equivalence |

**Decision rule:** Guilt without self-death framing is Distressed. Guilt that frames self as deserving the other's fate, or wishes self had died instead, is High Risk. `gold_036` is a borderline case — the wording "لو ما طلبت منو ما كان صار إشي" is High Risk because it positions self-action as the cause of the sibling's harm, which in clinical context carries elevated self-punishment risk.

---

### Pair G — Attachment seeking (Safe) vs trauma-driven clinging (Distressed)

| Feature | Safe | Distressed |
|---|---|---|
| Pattern | Age-appropriate seeking of caregiver | Inability to separate, panic on separation |
| Frequency | Situational | Persistent, escalating |
| Example | "أمي قالتلي ما أخاف لأنها معي" | "كل ما أسمع صوت بركض عند أمي... مش قادر أوقف حالي" |

**Decision rule:** Caregiver-seeking that is contained and reassurance-responsive is Safe. Caregiver-seeking that is uncontrollable, panic-driven, or interferes with normal function is Distressed.

---

## 6. Conversation-level annotation rules

For multi-turn dialogues (3–5 turns) in the gold set, the following rules apply.

### Rule 6.1 — Per-turn label and overall label
Each child turn is labeled individually. The conversation also receives an **overall label** = the highest-severity label appearing in any turn, OR the trajectory if it shows escalation.

### Rule 6.2 — Escalation across turns
If turn 1 is Safe but turn 4 reveals High Risk content, the overall label is High Risk. Earlier safety signals do not "cancel out" later disclosures — children commonly disclose serious content only after rapport builds.

### Rule 6.3 — De-escalation across turns
If turn 1 contains a High Risk marker but the child immediately retracts ("لأ مش هيك، أنا بس قلت هيك") and turns 2–4 are clearly Safe, the overall label is **still High Risk** — Principle 1 (safety over accuracy). However, `review_notes` should record the retraction for specialist judgment.

### Rule 6.4 — Cumulative ambiguity
If no single turn is clearly distressing but the cumulative pattern (across all turns) shows withdrawal, avoidance, or shutdown, the overall label is Distressed. Annotators document the cumulative reasoning in `review_notes`.

### Rule 6.5 — Conversation Unclear
A conversation is labeled Unclear only if **all turns** are Unclear. A single Distressed turn in an otherwise Unclear conversation makes the conversation Distressed.

---

## 7. Required fields and their definitions

Each example in `data/gold/gold_test_set.jsonl` MUST contain the following fields:

| Field | Type | Definition |
|---|---|---|
| `id` | string | Unique identifier, format `gold_NNN` (single utterance) or `gold_conv_NNN` (conversation) |
| `text` | string (utterance) or array (conversation) | The Palestinian Arabic text being annotated |
| `label` | string | One of the four labels exactly as written in §4 |
| `emotion` | string | Primary emotion observed (see §8 controlled vocabulary) |
| `risk_signal` | string | Specific signal detected (see §8 controlled vocabulary) |
| `source` | string | "gold_test" for all gold set entries |
| `review_status` | string | "needs_expert_review" until cleared; "expert_reviewed" after |
| `reviewer_type` | string | "none", "psychologist", "social_worker", or "psychiatrist" |
| `label_verified` | boolean | `false` until expert review confirms |
| `review_notes` | string | Annotator's reasoning, in English, ≤ 200 characters |

For conversation entries, additional fields:

| Field | Type | Definition |
|---|---|---|
| `turns` | array of objects | Each turn has `speaker` ("child" or "interviewer"), `text`, and (for child turns) `turn_label` |
| `overall_label` | string | Conversation-level label per §6 |
| `escalation_pattern` | string | "stable", "escalating", "de-escalating", or "mixed" |

---

## 8. Controlled vocabularies

Annotators MUST use values from these lists. Free-form values create inconsistency and break downstream evaluation.

### 8.1 Emotion vocabulary

`joy`, `comfort`, `pride`, `contentment`, `anticipation`, `acceptance`, `reassurance`, `neutral`, `resolved_conflict`, `mild_sadness`, `mild_fear`, `fear`, `anxiety`, `sadness`, `grief`, `suppressed_grief`, `loneliness`, `shame`, `helplessness`, `confusion`, `numbness`, `withdrawal`, `collective_fear`, `anhedonia`, `hopelessness`, `guilt`, `dissociation`, `unknown`

### 8.2 Risk signal vocabulary

**Safe:**
- `none`

**Distressed signals:**
- `hypervigilance`, `loss_of_interest`, `unresolved_grief`, `intrusive_thoughts`, `disorientation`, `emotional_suppression`, `rumination`, `social_isolation`, `family_tension`, `somatic_dissociation`, `behavioral_change`, `displacement_shame`, `secondary_stress`, `avoidance`, `attachment_loss`, `school_avoidance`, `somatic_response`, `environmental_helplessness`, `secrecy_seeking`

**High Risk signals:**
- `passive_death_wish`, `active_death_wish`, `death_idealization`, `perceived_burdensomeness_with_leaving`, `existential_indifference`, `future_collapse`, `survivor_guilt`, `death_wish_via_physical_event`, `depersonalization`, `derealization`, `dissociative_episode`, `perceptual_disturbance`

**Unclear signals:**
- `undisclosed_event`, `cognitive_avoidance`, `minimal_response`, `caregiver_imposed_silence`, `vague_somatic`, `deflection_to_caregiver`, `emotional_inconsistency`, `retrieval_block`

If a new signal type is needed, it MUST be added to this vocabulary in writing before being used in annotation.

---

## 9. Borderline case protocol

The gold set is intentionally heavy on borderline cases (per project decision). The following protocol applies whenever an annotator feels uncertain.

### Step 1 — Apply the relevant discriminative pair from §5
If the case matches a known discriminative pair, use that pair's decision rule.

### Step 2 — Apply Principle 1 (safety over accuracy)
If still uncertain between adjacent labels, escalate.

### Step 3 — Document the reasoning
In `review_notes`, write:
- Which two labels were under consideration.
- What evidence supported each.
- Why the chosen label was selected.
- Specific question for the expert reviewer (e.g., "Is سرد الحدث (event narration) without affect a dissociative sign in 8-year-olds?").

### Step 4 — Flag for expert priority review
Set `review_status` to `needs_expert_review` (default) and ensure the case is included in the expert review batch. All borderline High Risk cases receive priority.

---

## 10. What annotators must NOT do

- Do **not** invent context not present in the text ("the child probably means…").
- Do **not** assume severity based on the speaker's age, gender, or refugee status alone.
- Do **not** label based on a single keyword without considering the surrounding meaning ("موت" in "الله يرحمها" is not High Risk).
- Do **not** use Unclear as a way to avoid difficult judgment calls — Principle 1 still applies.
- Do **not** modify the controlled vocabularies without team consensus.
- Do **not** reveal the gold set to systems being evaluated. Gold ≠ training data.
- Do **not** translate the Arabic text into English for annotation; classify the Arabic directly. English appears only in `review_notes`.

---

## 11. Inter-annotator agreement protocol

For the gold set to be defensible:

1. A minimum of **two independent annotators** must label each example before it is considered review-ready.
2. Disagreements are resolved by reference to this guide first, expert reviewer second.
3. **Cohen's kappa** is computed across the full set; target ≥ 0.80 overall and ≥ 0.85 for High Risk.
4. Any case with persistent annotator disagreement is flagged in `review_notes` and routed to expert review with both candidate labels documented.

---

## 12. Expert review process

1. Licensed child mental health specialist (psychologist, child psychiatrist, or trauma-trained social worker) reviews the gold set.
2. Each example is either confirmed (`label_verified = true`) or relabeled with documented reasoning.
3. **All High Risk examples** receive specialist review before the gold set is considered finalized.
4. Specialist review notes are added to `review_notes` and the field `reviewer_type` is updated.
5. After review, the specialist signs off on the dataset as a whole, including any limitations.

---

## 13. Known limitations of this guide

This document is honest about what it cannot guarantee:

1. **No Palestinian child clinical psychologist has yet reviewed this guide.** Until that review happens, all rules are provisional.
2. The guide is built primarily from the structure of existing trauma screening tools (UCLA PTSD Index, CBCL, HSCL-37, RATS, PTSS-C) plus project-specific decisions; it has not been empirically validated.
3. The discriminative pairs in §5 are based on clinical reasoning, not on labeled empirical data from Palestinian children.
4. The controlled vocabularies in §8 will likely need expansion as more cases are annotated.
5. Cultural-religious framings of death and afterlife in Palestinian context may require additional decision rules that this guide does not yet capture.
6. The rule that retraction of a High Risk statement does not de-escalate the label (§6.3) is conservative; specialist review may refine this.

These limitations are why every example is marked `label_verified: false` until expert review.

---

## 14. Version history

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-05-08 | Initial draft | First complete version covering all four labels, seven discriminative pairs, conversation rules, and review protocol |

---

**End of guide.**