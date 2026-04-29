# AuraGuard AI 🧠👓
### Life-Critical Assistive Platform for Alzheimer's Patients

> A wearable AI co-pilot that watches, understands, and speaks — so the patient is never truly alone.

**Hackabull VII — Tech For Good | Health Care & Wellness**

---

## The Problem

Over 6 million Americans live with Alzheimer's disease, and 16+ million family members provide unpaid care. Patients face daily life-threatening risks — leaving the stove on, not recognizing family members, forgetting to hydrate or take medication. Caregivers cannot be present 24/7.

## Our Solution

AuraGuard AI transforms Meta Smart Glasses into a life-critical safety system. The glasses stream a first-person POV to a laptop running three coordinated services:

- A **Vision Engine** that detects hazards and familiar faces locally using ONNX models
- An **AI Brain** that reasons about what it sees and speaks empathetically to the patient
- A **Caregiver Portal** that gives families and clinicians a live and longitudinal view of their loved one's day
- An **Event Audio Webapp** that routes synthesized speech through the glasses speaker via Picture-in-Picture

---

## Architecture

```
Meta Smart Glasses (POV stream via SpecBridge iOS app)
        │  RTMP
        ▼
┌─────────────────────────────┐
│  mediamtx (RTMP server)     │  :1935
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Vision Engine              │  Python + OpenCV  :5000
│  - YuNet face detection     │
│  - SFace face recognition   │
│  - Gemini health detection  │
└────────────┬────────────────┘
             │ POST /event  (JSON Contract)
             ▼
┌─────────────────────────────┐
│  AI Brain (FastAPI)         │  Gemini + ElevenLabs  :8000
│  - Gemini secondary verify  │
│  - Voice script generation  │
│  - ElevenLabs TTS synthesis │
│  - Pygame audio playback    │
│  - MongoDB event logging    │
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌──────────┐   ┌──────────────────┐
│ MongoDB  │   │  Event Audio     │  :8502
│  Atlas   │   │  Webapp          │
│ (events) │   │  (PiP + TTS)     │
└────┬─────┘   └──────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Caregiver Portal (Streamlit)│  :8501
│  - Live event feed           │
│  - Health trend charts       │
└──────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Wearable Hardware | Meta Smart Glasses (POV stream via SpecBridge iOS app) |
| RTMP Server | mediamtx |
| Vision Engine | Python, OpenCV, YuNet + SFace (ONNX), ONNXRuntime, Flask |
| AI Reasoning | Google Gemini 1.5 Flash (multimodal) |
| Voice Synthesis | ElevenLabs + edge-tts (fallback) |
| Audio Playback | Pygame, Web Audio API (PiP) |
| AI Brain API | FastAPI, Uvicorn, Pydantic |
| Live Database | MongoDB Atlas + Motor (async driver) |
| Data Warehouse | Snowflake |
| Caregiver Dashboard | Streamlit, Plotly, Pandas |
| Event Audio Webapp | FastAPI, HTML5 Canvas, Picture-in-Picture API |

---

## Project Structure

```
.
├── run_all.py                        # One-command launcher for all services
├── mediamtx.yml                      # RTMP server configuration
├── requirements.txt                  # All pinned Python dependencies
├── .env.example                      # Environment variable template
├── test_webcam.py                    # Webcam test (no glasses required)
│
├── shared/
│   └── contract.py                   # Canonical Pydantic models (single source of truth)
│
└── services/
    ├── brain/                        # AI Brain — FastAPI service on :8000
    │   ├── main.py                   # App entry point, lifespan, route mounts
    │   ├── config.py                 # Pydantic-settings config loader
    │   ├── models.py                 # Local model aliases
    │   ├── routes/
    │   │   ├── event.py              # POST /event — full processing pipeline
    │   │   └── health.py             # GET /health — MongoDB liveness check
    │   └── services/
    │       ├── audio.py              # Pygame mixer init + playback
    │       ├── elevenlabs.py         # ElevenLabs TTS synthesis
    │       ├── gemini.py             # Gemini verification + voice script generation
    │       └── mongodb.py            # Motor async MongoDB client + writes
    │
    ├── vision/
    │   ├── face_recognition_engine.py  # Vision Engine — YuNet + SFace + Gemini health
    │   └── known_faces/              # Per-person .jpg + .json profile files
    │       ├── ismail.json
    │       ├── mohammed.json
    │       └── taikhoom.json
    │
    └── webapp/
        └── app.py                    # Event Audio Webapp — FastAPI + PiP on :8502
```

---

## Core Features

### 1. Face Recognition
Identifies familiar people using **YuNet** (face detection) and **SFace** (face recognition), both running locally via ONNX/ONNXRuntime — no external API calls. When a known face is detected, the patient hears a personalized voice alert:

> *"Ismail, your son Hussain is here. He is a software engineer living in Tampa. Last time you spoke, he told you about his new job."*

### 2. Health Item Detection
Optionally uses **Google Gemini 1.5 Flash** to detect eating, drinking, and medication intake from the glasses POV. The Brain performs a **two-pass verification** (primary detection + secondary yes/no confirmation) to filter false positives before any alert fires.

### 3. Empathetic Voice Alerts
Voice scripts are synthesized via **ElevenLabs** and played through the glasses speaker. Falls back to **edge-tts** if ElevenLabs is unavailable. Audio is routed through the Event Audio Webapp using the **Web Audio API + Picture-in-Picture** so it keeps playing when SpecBridge is in the foreground on iOS.

### 4. Real-Time Caregiver Dashboard
A **Streamlit** portal polls MongoDB Atlas every 5 seconds and renders a live event feed with color-coded cards:
- 🔵 Blue — identity events (face recognized)
- 🟢 Green — health events (eating, drinking, medication)

### 5. Longitudinal Health Trends
The dashboard queries **Snowflake** for aggregated health event data and renders **Plotly** time-series charts — the kind of data a caregiver would share with a clinician.

### 6. Graceful Degradation
Every service continues operating independently if another fails. The Brain marks events `processing_status: partial_failure` and always returns HTTP 200 to the Vision Engine so the pipeline keeps moving.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [`mediamtx`](https://github.com/bluenviron/mediamtx/releases) binary in PATH
- API keys for Gemini, ElevenLabs, MongoDB Atlas, and Snowflake
- ONNX model files in `tests/vision/models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in every value:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID (e.g. `21m00Tcm4TlvDq8ikWAM`) |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | Database name (e.g. `auraguard`) |
| `MONGODB_COLLECTION` | Collection name (e.g. `events`) |
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier |
| `SNOWFLAKE_USER` | Snowflake username |
| `SNOWFLAKE_PASSWORD` | Snowflake password |
| `SNOWFLAKE_DATABASE` | Snowflake database (e.g. `AURAGUARD`) |
| `SNOWFLAKE_SCHEMA` | Snowflake schema (e.g. `PUBLIC`) |
| `SNOWFLAKE_WAREHOUSE` | Snowflake warehouse (e.g. `COMPUTE_WH`) |
| `PATIENT_NAME` | Patient's first name (e.g. `Ismail`) |
| `PATIENT_ID` | Patient identifier (e.g. `patient_001`) |
| `GLASSES_AUDIO_DEVICE` | System audio device name for the glasses speaker |

### 3. Add Known Faces

For each person the patient should recognize, add two files to `services/vision/known_faces/`:

- `<name>.jpg` — a clear, front-facing photo
- `<name>.json` — a profile with the following structure:

```json
{
  "name": "Hussain",
  "relationship": "son",
  "background": "Software engineer living in Tampa.",
  "last_conversation": "Told you about his new job."
}
```

### 4. Launch All Services

```bash
python run_all.py
```

This starts all services in order with health checks:

| Service | URL | Description |
|---------|-----|-------------|
| mediamtx | `rtmp://localhost:1935/live/stream` | RTMP server for glasses stream |
| AI Brain | `http://localhost:8000` | Event processing API |
| Vision Engine | — | Reads RTMP stream, runs detection |
| Caregiver Portal | `http://localhost:8501` | Live dashboard |
| Event Audio | `http://localhost:8502` | PiP audio webapp |

Press `Ctrl+C` to stop all services cleanly.

---

## Testing Without Meta Smart Glasses

Use your laptop's built-in webcam to test the Vision Engine:

```bash
python test_webcam.py
```

Press `Q` in the video window to quit. The webcam releases automatically on exit.

---

## API Reference

### AI Brain — `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/event` | Submit an event from the Vision Engine for processing |
| `GET` | `/health` | Liveness check — returns `200 ok` or `503 degraded` |
| `GET` | `/status` | Detailed status including MongoDB connection and event count |
| `POST` | `/test/voice` | Synthesize and play a test TTS message |
| `POST` | `/test/event` | Inject a synthetic event to test the full pipeline |

### Event Audio Webapp — `http://localhost:8502`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | PiP audio interface (open in browser on iOS) |
| `POST` | `/ingest` | Receive an event from the Vision Engine |
| `GET` | `/poll` | Poll for new events (called by the frontend every second) |
| `POST` | `/speak` | Synthesize text to audio (ElevenLabs → edge-tts fallback) |

---

## The JSON Contract

All events flow through a shared schema defined in `shared/contract.py`:

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2025-04-25T14:32:00Z",
  "patient_id": "patient_001",
  "type": "health | identity",
  "subtype": "eating | drinking | medicine_taken | face_recognized",
  "confidence": 0.91,
  "image_b64": "<base64-encoded-frame>",
  "metadata": {
    "person_profile": {
      "name": "Hussain",
      "relationship": "son",
      "background": "Software engineer living in Tampa.",
      "last_conversation": "Told you about his new job."
    }
  },
  "source": "vision_engine_v1"
}
```

After processing, the Brain writes an `EventRecord` to MongoDB that adds:

| Field | Description |
|-------|-------------|
| `verified` | Whether Gemini secondary verification passed |
| `voice_script` | The exact text spoken to the patient |
| `processing_status` | `success` or `partial_failure` |
| `processed_at` | ISO timestamp of Brain processing |

> Note: `image_b64` is intentionally excluded from `EventRecord` to keep the database lean.

---

## Vision Engine Details

### Face Recognition Pipeline

1. **YuNet** detects faces in the frame (OpenCV DNN, ONNX model)
2. **SFace** generates a 128-dim embedding (ONNXRuntime, workaround for OpenCV ONNX bug)
3. Cosine similarity is computed against all known face embeddings
4. A match is confirmed if similarity ≥ `0.363` (configurable threshold)
5. A **10-second cooldown** prevents the same person from being re-announced repeatedly
6. Recognition runs every **2 seconds** in a background thread to avoid blocking the display loop

### Health Detection (Optional)

Set `ENABLE_HEALTH_DETECTION=true` in `.env` to enable Gemini-based health scanning:

- Runs every **5 seconds** in a background thread
- Detects: water bottle, cup, glass, mug, food, fork, spoon, sandwich, pill, tablet, medication
- A **120-second cooldown** per subtype prevents alert fatigue
- Detected events are POSTed to the Brain for secondary verification before any alert fires

### Frame Quality Checks

Frames are skipped if:
- Mean brightness < 30 (too dark)
- Laplacian variance < 2 (too blurry)

### Supported Video Sources

- **RTMP stream** (default): `rtmp://localhost/live/stream` — set via `RTMP_STREAM_URL` env var
- **Local webcam**: pass an integer index (e.g. `0`) to `run(video_source=0)`
- **Any OpenCV-compatible source**: file path, HTTP stream, etc.

---

## Brain Service Details

### Processing Pipeline (`POST /event`)

1. **Gemini verification** — health events only; identity events skip this step (confidence already established locally)
2. **Voice script generation** — personalized text based on event type, subtype, and patient name
3. **ElevenLabs TTS synthesis** — converts script to audio bytes
4. **Pygame playback** — routes audio to the configured glasses speaker
5. **MongoDB write** — persists the full `EventRecord` for the dashboard

All steps are wrapped in individual try/except blocks. Any failure sets `processing_status: partial_failure` but the route always returns HTTP 200 so the Vision Engine pipeline keeps moving.

### Health Check (`GET /health`)

Pings MongoDB with a 3-second timeout:
- `200 {"status": "ok"}` — MongoDB reachable
- `503 {"status": "degraded", "reason": "mongodb_unreachable"}` — MongoDB unreachable

### Startup Sequence

1. Load and validate all environment variables (exits with code 1 if any are missing)
2. Initialize Motor MongoDB client
3. Verify MongoDB connectivity (degraded start allowed — logs warning, does not exit)
4. Initialize Pygame mixer (degraded start allowed)
5. Construct ElevenLabs client
6. Mount routes and begin serving requests

---

## Event Audio Webapp Details

The webapp at `:8502` solves a specific iOS constraint: when SpecBridge (the glasses mirror app) is in the foreground, audio from other apps is suppressed. The solution:

1. A **Canvas** element draws a live event card (visible in the PiP window)
2. The canvas stream is combined with a **Web Audio API** destination into a single `MediaStream`
3. A hidden `<video>` element plays this stream
4. The user taps **Enter PiP** — the video floats over SpecBridge
5. Audio synthesized via `/speak` is routed through the `AudioContext` → PiP video stream → glasses speaker

Falls back to the browser's built-in `SpeechSynthesis` API if ElevenLabs is unavailable.

---

## Graceful Degradation Summary

| Failure | Behavior |
|---------|----------|
| MongoDB unreachable at startup | Logs warning, continues in degraded state |
| Pygame init fails | Logs warning, continues without audio |
| Gemini verification fails | `verified=false`, pipeline continues |
| ElevenLabs synthesis fails | `partial_failure`, no audio played |
| Brain unreachable (from Vision Engine) | Vision Engine logs error, continues capture loop |
| Snowflake unreachable (from Dashboard) | Placeholder shown in chart area, live feed continues |
| Unhandled exception in Brain | Global handler returns HTTP 500 with structured JSON |

---

## Team

| Role | Service | Port |
|------|---------|------|
| Vision Lead | `services/vision/` | — |
| AI Architect | `services/brain/` | 8000 |
| Dashboard Lead | `services/webapp/` | 8501 / 8502 |

---

## Impact

Reduced safety incidents, earlier health intervention, and restored dignity for patients who deserve to live independently for as long as possible.

---

*Last updated: 2026-04-29*
