# 04 — Prompt System

All behavior, safety rules, and triage logic live in `prompts/` and `resources/` —
**never hardcoded in Python**. Changing Teta's behavior must never require touching backend code.

## The six prompts

| # | File (role) | Consumed by | Purpose |
|---|---|---|---|
| 1 | **Analyzer** | Triage stage | Defines the four tiers and classification rules for each turn |
| 2 | **Teta persona** | `gpt-5` (Responses API) | The single authoritative definition of who Teta is |
| 3 | **Strategy — Safe/Regulated** | Teta engine | Reply style for green turns: warm, everyday grandmother talk |
| 4 | **Strategy — Distressed** | Teta engine | Reply style for yellow turns: comfort and validation, **no therapeutic techniques** |
| 5 | **Strategy — Unclear** | Teta engine | Gentle, open clarifying questions; never pressures the child |
| 6 | **STT transcription** | `gpt-4o-transcribe` | Dialect guidance for accurate Palestinian-Arabic transcription |

> 🔴 **High Risk has no strategy prompt by design** — Red turns bypass the LLM entirely
> and receive pre-vetted hard-coded responses.

## Teta persona: hard rules

Teta is a warm Palestinian grandmother with **strictly bounded** knowledge and behavior:

1. **Never claims to be human.** No fabricated human life or daily activities
   (the AI-transparency rule, persona §1.5 — e.g., never "كنت بالمطبخ").
2. **Never uses therapeutic techniques** — including breathing exercises, grounding
   exercises, or any clinical intervention. Teta comforts; she does not treat.
3. **Refuses out-of-scope topics without naming them.** Technical, academic, sports, and
   any subject outside an ordinary grandmother's world are declined warmly — but the
   refusal must not mention the subject itself (naming it is itself a scope leak).
4. **Never diagnoses, never labels the child's state clinically.**

## Hardening history (why these rules exist)

The persona prompt went through three documented tightening rounds:

| Round | Failure observed | Fix |
|---|---|---|
| 1 | Teta named out-of-scope subjects while refusing them | Refusals must not identify the topic |
| 2 | Teta offered breathing exercises to distressed children | Blanket prohibition on therapeutic techniques |
| 3 | Teta implied a human life ("I was in the kitchen") | Added AI-transparency rule (§1.5) |

## Cross-prompt consistency — the critical failure mode

Distributed prompt systems can **silently contradict each other**. Two documented incidents:

1. **The breathing-exercise conflict:** the persona prompt prohibited therapeutic
   techniques, while the Distressed strategy prompt *instructed* a breathing exercise.
   The strategy prompt won at runtime — the safety rule was silently undermined.
2. **Out-of-scope handling divergence:** the three strategy files handled off-topic
   requests differently until they were unified.

### Rules derived from these incidents

- **Single source of truth:** persona rules live *only* in the persona prompt. Strategy
  files must never restate (and therefore never contradict) persona rules.
- **Any prompt edit requires a cross-prompt consistency pass:** re-read all six files and
  check for instructions that conflict with the persona's hard rules.
- **Test after every prompt change:** run the smoke test and re-run evaluation on the
  gold set before committing (see docs/05).

## RAG resources (`resources/`)

The `gpt-5` file_search vector store contains: `safety_rules.md`, `annotation_guide.md`,
`triage_labels.md`, `dataset_examples.jsonl`, `borderline_cases.jsonl`. These ground the
model's understanding of the tiers and safety policy; they complement — never override —
the persona's hard rules.
