# 01 — System Architecture

## Overview

SENSE is a three-tier system:

```
┌──────────────┐     HTTP / WebSocket     ┌──────────────┐    ai_adapter.py    ┌──────────────┐
│   Clients    │ ───────────────────────► │   Backend    │ ──────────────────► │  AI Pipeline │
│ Flutter app  │                          │   FastAPI    │   (single bridge)   │ STT→Triage→  │
│ sense_web    │ ◄─────────────────────── │              │ ◄────────────────── │ LLM→TTS      │
└──────────────┘                          └──────────────┘                     └──────────────┘
```

**Architectural rule:** backend code never imports directly from `ai_pipeline/`, `prompts/`,
`datasets/`, or `resources/`. All pipeline access flows through a single adapter file
(`backend/services/ai_adapter.py`). This enabled parallel development across the team.

## Per-turn pipeline

| Stage | Component | Model / Method | Measured time |
|---|---|---|---|
| 1. STT | `ai_pipeline` | `gpt-4o-transcribe` + dialect transcription prompt | ~3.5s |
| 2. Triage | rule-based analyzer | keyword/pattern rules, four tiers | ~0.01s |
| 3. Reply | Teta engine | `gpt-5` (Responses API) + file_search RAG | ~7s |
| 4. TTS | `ai_pipeline` | `gpt-4o-mini-tts`, voice `coral`, speed 1.12× | ~3s |
| **Total** | | | **~13.75s** |

### Triage tiers

| Tier | Label | Behavior |
|---|---|---|
| 🟢 | Safe / Regulated | Normal Teta reply |
| 🟡 | Distressed / Needs Support | Teta reply using the Distressed strategy prompt |
| 🔴 | High Risk / Urgent | **LLM bypassed** — pre-vetted hard-coded response only |
| ⚪ | Unclear / Need More Context | Gentle clarifying reply via the Unclear strategy |

### Safety invariants

1. **High Risk never reaches the LLM.** The reply is selected from pre-vetted hard-coded
   responses. This removes any possibility of the model improvising in a crisis turn.
2. **No downgrades.** Session-level label = max severity seen. Any Red turn ⇒ final session
   label is Red, regardless of later turns.
3. **Conservative escalation.** Ambiguous inputs escalate to Unclear/Distressed rather than
   defaulting to Safe. This is a deliberate design choice (see docs/05_evaluation.md).

## RAG layer

Teta's replies are grounded via OpenAI file_search over a vector store containing:
`safety_rules.md`, `annotation_guide.md`, `triage_labels.md`, `dataset_examples.jsonl`,
`borderline_cases.jsonl`.

## Disabled audio-emotion module

An XLSR/wav2vec2 speech-emotion module was designed but is **disabled**: no Arabic child
emotion speech dataset exists globally, so it could not be trained or validated responsibly.
It is retained as a placeholder returning `{"source": "disabled_for_demo"}` and documented
as a research gap, not a defect.

## Key design decisions (measure, then decide)

| Decision | Evidence | Outcome |
|---|---|---|
| Local → cloud migration | Local CPU pipeline measured 20–40s/turn | Migrated to OpenAI APIs (~13.75s) |
| `gpt-4.1-mini` → `gpt-5` | Persona scope drift observed with mini; cost delta ~$0.001/turn | Adopted `gpt-5` |
| STT mini → `gpt-4o-transcribe` | Dialect accuracy on Palestinian Arabic | Upgraded |
| Emulator mic silence | Reproduced consistently; platform limitation | Built `sense_web.html` fallback |
