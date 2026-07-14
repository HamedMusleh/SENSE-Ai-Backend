# 06 — Developer Guide

## Ownership boundaries

| Area | Owner | Directories |
|---|---|---|
| AI / Pipeline | Hamed Musleh | `ai_pipeline/`, `prompts/`, `resources/`, `datasets/`, `evaluation/` |
| Backend | Bara Mohsen | `backend/` |
| Mobile / Web | Ahmad Zuhd | Flutter app, `sense_web.html` |

**Rule:** you do not modify code outside your area. Cross-area needs go through the
adapter contract (below) or a schema change agreed by both owners.

## The adapter pattern (`backend/services/ai_adapter.py`)

The adapter is the **single bridge** between backend and AI pipeline. Backend code never
imports from `ai_pipeline/`, `prompts/`, `datasets/`, or `resources/` directly.

### Why it exists

- **Parallel development:** backend and frontend progressed against `mock` mode while the
  real pipeline was still being built.
- **Testability:** the six pytest integration tests run in `mock` mode — deterministic,
  free, offline.
- **Swappability:** the entire AI stack was migrated (local models → OpenAI cloud) without
  touching a single backend route.

### Pipeline modes (`PIPELINE_MODE`)

| Mode | STT / LLM / TTS | Use for |
|---|---|---|
| `real` | Live OpenAI calls | Demos, evaluation, production |
| `mock` | Canned deterministic responses | Frontend dev, CI, integration tests |
| `hybrid` | Mix (e.g., real triage + mock TTS) | Debugging a single stage cheaply |

## Separation of concerns

- **Behavior lives in `prompts/` and `resources/`** — persona, strategies, triage rules.
- **Python code is plumbing** — routing, validation, orchestration, API calls.
- If you find yourself encoding Teta's behavior or a safety rule in Python, stop: it
  belongs in a prompt or resource file (with one exception — the High Risk hard-coded
  responses, which are deliberately kept out of the LLM path).

## Change checklist

Before committing **any** change that touches prompts, resources, triage, or the pipeline:

1. `python evaluation/check_data_leakage.py` — must pass.
2. `python smoke_test.py` — end-to-end sanity in `real` (or `hybrid`) mode.
3. `pytest --import-mode=importlib` — all six backend tests green in `mock` mode.
4. If a prompt changed: **cross-prompt consistency pass** over all six files
   (see docs/04 — this is how the breathing-exercise conflict slipped through).
5. If triage logic changed: re-run gold evaluation; confirm **zero dangerous downgrades**
   and High Risk recall unchanged, and save the timestamped results folder.

## Project principles (carry these forward)

- **Measure, then decide.** Every architectural decision (cloud migration, model choice)
  was justified with measured data. Keep it that way.
- **Document limitations honestly.** Disabled XLSR module, emulator mic silence, and the
  41% unseen accuracy are all documented as what they are — research gaps, platform
  limitations, and design choices — not hidden or spun.
- **Safety invariants are not configurable.** No flag, mode, or refactor may ever make a
  Red downgrade possible.
