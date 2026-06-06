# SENSE — Speech Emotion and Neural Support Engine

> An AI-assisted emotional **screening and triage** system for Arabic-speaking
> children (ages 6–14) in crisis environments, with a focus on the
> **Palestinian dialect**.

SENSE listens to a child speaking in their own dialect, estimates the emotional
state and risk level of what they express, replies with warmth through a
grandmother persona (**"Teta"**), and produces a structured report to help a
human specialist prioritise their attention.

> ⚠️ **Important:** SENSE is **not** a diagnostic or clinical tool. It performs
> emotional triage and supportive presence only, and is designed to **assist**
> human specialists — never to replace them.

---

## Project Context

| | |
|---|---|
| **University** | Birzeit University |
| **Faculty** | Faculty of Engineering and Technology |
| **Department** | Electrical and Computer Engineering Department |
| **Course** | ENCS5200 — Introduction to Graduation Project |
| **Stage** | Pre-Alpha — Integrated Working Prototype |

---

## System Architecture

A three-tier system communicating in a strictly layered fashion:

```
Mobile App (Flutter)
   │  records audio, sends via HTTP
   ▼
Backend (FastAPI)  ──►  AI Pipeline
                          │
                          ├─ Whisper STT (Palestinian dialect)
                          ├─ Arabic text preprocessing
                          ├─ Audio emotion (XLSR + prosody)
                          ├─ Triage classifier (4 labels)
                          ├─ Multimodal weighting (safe fusion)
                          ├─ Teta reply (LLM + RAG) OR
                          │  hard-coded response (High Risk)
                          └─ Session analysis (specialist report)
   ◄── returns reply
Mobile App displays Teta's reply
```

**Flow:** `Mobile → Backend → AI Pipeline → Backend → Mobile`

---

## Repository Structure

```
sense-ai-demo/
├── ai_pipeline/          # The complete intelligence layer
│   ├── stt/              #   Whisper speech-to-text
│   ├── audio_emotion/    #   XLSR + prosodic emotion + weighting
│   ├── triage/           #   Rule-based triage classifier
│   ├── tts/              #   Text-to-speech (placeholder, future)
│   └── integration_api.py#   Stable public interface to the pipeline
├── backend/              # FastAPI server (API, orchestration, adapter)
├── datasets/             # Annotation guide + evaluation sets
├── evaluation/           # Triage classifier evaluation
├── prompts/              # Teta persona strategy prompts
├── resources/            # Triage lexicons, safety responses
├── tests/                # Unit and integration tests
├── mobile_app/
│   └── flutter_client/   # Flutter mobile application
└── requirements.txt
```

---

## Key Features

- **Dialect-aware transcription** — locally hosted Whisper model primed for
  Palestinian Arabic (runs locally for privacy).
- **Four-category triage** — Safe / Regulated, Distressed / Needs Support,
  High Risk / Urgent, Unclear / Need More Context.
- **Multimodal fusion** — combines a textual triage decision with an acoustic
  emotional signal under a binding safety policy.
- **Safety-first design** — High Risk cases **never** reach the language model;
  they are served by vetted, pre-written responses.
- **Teta persona** — warm, authentic Palestinian grandmother voice.
- **Specialist report** — risk trajectory, peak risk, per-turn summary, and a
  recommendation at the end of each session.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| AI Pipeline | Python, Whisper, XLSR (wav2vec2), librosa, Transformers, GPT-class LLM + RAG |
| Backend | FastAPI, Uvicorn, Pydantic |
| Mobile | Flutter, Dart, Provider |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Flutter 3.35+
- An Android emulator or a physical Android device

### 1. Backend + AI Pipeline

```bash
cd sense-ai-demo

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your environment (see .env.example) — add your LLM API key

# Run the server (real pipeline mode)
# Windows (PowerShell):
$env:SENSE_PIPELINE_MODE="real"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait until the log shows `Whisper ready` and `Application startup complete`.
The API is then available at `http://127.0.0.1:8000` and interactive docs at
`http://127.0.0.1:8000/docs`.

### 2. Mobile App

```bash
cd mobile_app/flutter_client

flutter pub get

# Launch an emulator, then:
flutter run
```

> **Note on the emulator microphone:** the Android emulator's virtual
> microphone may produce silent recordings (a known emulator limitation). The
> complete flow can be validated using the bundled test audio button, and live
> microphone capture works on physical devices.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/session/start` | Start a new conversation session |
| `POST` | `/api/upload_audio` | Upload an audio turn, get Teta's reply |
| `POST` | `/api/analyze` | End the session, get the specialist report |

---

## Evaluation

The triage classifier was evaluated on two datasets:

| Dataset | Size | Result |
|---|---|---|
| Gold test set | 250 | 100% (sanity check — same source as the rules) |
| Unseen validation | 75 | ~41% overall accuracy |
| — High Risk precision | — | **100%** |
| — Dangerous downgrades | — | **0** (no High Risk ever misclassified as Safe) |

The system is intentionally **conservative**: it escalates under doubt rather
than downgrading, which is the most important property for a safety-critical
triage tool.

---

## Known Limitations

- **Audio emotion signal is weak.** XLSR is a speech-recognition model, not an
  emotion model, and single-speaker acted speech yields overlapping samples.
  This is documented honestly; the fix is a fine-tuned emotion model trained on
  genuine multi-speaker child data (future work).
- **Inference is CPU-bound and slow.** Without a GPU, a single turn can take
  tens of seconds. GPU deployment is a future improvement.
- **High-risk responses await clinical review** by a licensed specialist.

---

## Future Work

- Fine-tuned Arabic child emotion model (real multi-speaker data + ethics approval).
- Machine-learning triage model (using the unseen set as a held-out benchmark).
- Dialect-aware text-to-speech so Teta can reply with an audible voice.
- GPU deployment to reduce per-turn latency.
- Formal psychologist review of high-risk responses and strategy prompts.

---

## Team

| Role | Responsibility |
|---|---|
| **AI / Pipeline Lead** | Whisper STT, audio emotion, triage classifier, weighting, Teta persona, RAG, session analysis, integration interface, datasets & evaluation |
| **Backend Engineer** | FastAPI server, API endpoints, orchestration, adapter, schemas, risk mapping |
| **Mobile Engineer** | Flutter app, UI, microphone recording, audio upload, HTTP layer, Android configuration |

---

## License & Ethics

This project handles sensitive data relating to children and mental health.
The repository is **private** and intended for academic evaluation only. Any
collection of real child data must follow appropriate ethical approval and
parental consent, under specialist supervision.

---

*SENSE — built to listen, designed to keep children safe.*
