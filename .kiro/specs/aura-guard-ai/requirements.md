# Requirements Document

## Introduction

AuraGuard AI is a life-critical assistive platform for Alzheimer's patients, built for Hackabull VII (Tech For Good / Health Care & Wellness track). Meta Smart Glasses stream a first-person POV to a laptop running three coordinated services on a single machine:

- **Vision Engine** (`services/vision/`, port 5000) — a Python/OpenCV/Flask service that captures frames from the glasses, recognizes familiar faces, and detects food, water, and medicine intake, then POSTs structured event payloads to the Brain.
- **AI Brain** (`services/brain/`, port 8000) — a FastAPI service that receives events from the Vision Engine, uses Google Gemini for multimodal verification of health observations, generates personalized voice scripts, synthesizes speech via ElevenLabs, plays audio through the Meta Smart Glasses speaker, and logs enriched event records to MongoDB Atlas.
- **Caregiver Portal** (`services/dashboard/`, port 8501) — a Streamlit dashboard that displays a real-time event feed from MongoDB Atlas and longitudinal health trends from Snowflake.

All three services share a single JSON Contract that defines the structure of every event flowing through the pipeline. A `run_all.py` script launches all three services simultaneously from a single terminal. A `.env` file provides all secrets and configuration.

Because this system operates in a life-critical context for a vulnerable population, correctness, low latency, graceful degradation under API failure, and clear caregiver visibility are first-class requirements across every service.

---

## Glossary

- **System**: The complete AuraGuard AI platform, comprising the Vision_Engine, Brain, and Caregiver_Portal running together on a single laptop.
- **Vision_Engine**: The Python/OpenCV/Flask service running on port 5000. Captures frames from the Meta Smart Glasses POV stream, performs face recognition and health item detection (food, water, medicine), and POSTs Events to the Brain.
- **Brain**: The FastAPI service running on port 8000. The central coordinator of the AuraGuard AI pipeline. Receives Events from the Vision_Engine, performs multimodal reasoning via Gemini, synthesizes and plays voice alerts, and logs Event_Records to MongoDB_Atlas.
- **Caregiver_Portal**: The Streamlit dashboard running on port 8501. Reads Event_Records from MongoDB_Atlas and health trend data from Snowflake.
- **Meta_Smart_Glasses**: The wearable hardware worn by the Patient. Streams a first-person POV video feed that is mirrored to the laptop via scrcpy or ADB.
- **Event**: A structured JSON payload sent from the Vision_Engine to the Brain describing a detected occurrence in the patient's environment. Defined by the JSON Contract.
- **Event_Type**: The top-level classification of an Event. One of: `health`, `identity`.
- **Event_Subtype**: A more specific classification within an Event_Type. Examples: `face_recognized`, `eating`, `drinking`, `medicine_taken`.
- **Person_Profile**: A structured record stored alongside a known face encoding that contains the person's name, their relationship to the Patient (e.g., "son", "daughter", "doctor"), a short biographical background, and a summary of their last conversation with the Patient.
- **Health_Item**: An object visible in a frame that the Vision_Engine classifies as food, water, or medicine. Used to construct `health` Events.
- **Event_Record**: The enriched document written to MongoDB_Atlas after the Brain has processed an Event. Contains all original Event fields (except `image_b64`) plus `verified`, `voice_script`, `processing_status`, and `processed_at`.
- **JSON_Contract**: The shared data schema agreed upon by all three services. Defines the structure of Event payloads and Event_Records.
- **Gemini**: The Google Gemini multimodal API used by the Brain to verify health observations and by the Vision Engine to detect health items in frames.
- **ElevenLabs**: The voice synthesis API used by the Brain to generate personalized, empathetic spoken audio for the Patient.
- **Pygame**: The Python audio library used by the Brain to play synthesized audio through the Meta_Smart_Glasses speaker.
- **Motor**: The async Python MongoDB driver used by the Brain to write Event_Records to MongoDB_Atlas without blocking the FastAPI event loop.
- **MongoDB_Atlas**: The cloud MongoDB database that stores Event_Records for real-time consumption by the Caregiver_Portal.
- **Snowflake**: The cloud data warehouse that stores aggregated longitudinal health data for trend analysis in the Caregiver_Portal.
- **Patient**: The Alzheimer's patient wearing the Meta_Smart_Glasses. Identified by `patient_id` in the Event payload.
- **Caregiver**: A family member or clinician who monitors the Patient via the Caregiver_Portal.
- **Confidence**: A float in the range [0.0, 1.0] representing the Vision_Engine's certainty about a detected event.
- **Known_Faces_Directory**: A local directory containing reference images of people familiar to the Patient, each paired with a Person_Profile JSON file, used by the Vision_Engine for face recognition.
- **Voice_Script**: The natural-language string passed to ElevenLabs for synthesis. Personalized with the Patient's name and contextually appropriate content.
- **Audio_File**: The MP3 file saved locally by the Brain after ElevenLabs synthesis, played through the Meta_Smart_Glasses speaker via Pygame.
- **Glasses_Audio_Device**: The system audio output device name corresponding to the Meta_Smart_Glasses speaker, configured via the `GLASSES_AUDIO_DEVICE` environment variable.
- **Severity_Color**: The color code used by the Caregiver_Portal to visually distinguish event types: yellow for `health`, green for `identity`.
- **run_all.py**: The Python launcher script that starts all three services simultaneously on a single laptop.

---

## Requirements

---

### Requirement 1: Event Ingestion API

**User Story:** As the Vision Engine, I want to POST a structured event payload to the Brain, so that the Brain can process the event and return a confirmation that it was handled successfully.

#### Acceptance Criteria

1. THE Brain SHALL expose a `POST /event` HTTP endpoint on port 8000.
2. WHEN a POST request is received at `/event`, THE Brain SHALL validate the request body against the Event schema defined in the JSON Contract.
3. THE Brain SHALL accept an Event payload containing: `event_id` (UUID v4 string), `timestamp` (ISO 8601 string), `patient_id` (string), `type` (one of `health`, `identity`), `subtype` (string), `confidence` (float), `image_b64` (base64-encoded string), `metadata` (object), and `source` (string).
4. IF the request body is missing a required field or contains a field with an invalid type, THEN THE Brain SHALL return HTTP 422 with a structured error body identifying the invalid fields.
5. WHEN a valid Event is received, THE Brain SHALL return HTTP 200 with a JSON response body containing: `event_id` (echoed from the request), `status` (`"processed"`), and `message` (a human-readable confirmation string).
6. THE Brain SHALL process each Event end-to-end (Gemini verification, voice synthesis, audio playback, MongoDB logging) before returning the HTTP 200 response.

---

### Requirement 2: Multimodal Verification via Gemini

**User Story:** As the AI Architect, I want the Brain to use Google Gemini to verify health observations from raw image frames, so that the system acts only on confirmed intake events and avoids false alerts.

#### Acceptance Criteria

1. WHEN an Event with `type` equal to `health` is received, THE Brain SHALL send the `image_b64` field and a context-specific prompt to the Gemini API for verification.
2. THE Brain SHALL construct the Gemini prompt using the Event's `subtype` field to ask a targeted question (e.g., for `subtype` = `drinking`: "Is the person in this image drinking water or a beverage? Answer YES or NO and explain briefly." For `subtype` = `medicine_taken`: "Is the person in this image taking medication? Answer YES or NO and explain briefly.").
3. WHEN the Gemini API returns a response, THE Brain SHALL parse the response to extract a boolean verification result (`verified`: true or false).
4. WHEN an Event with `type` equal to `identity` is received, THE Brain SHALL skip the Gemini verification step and set `verified` to `true` by default.
5. IF the Gemini API call fails or returns an error, THEN THE Brain SHALL log the error, set `verified` to `false`, and continue processing the Event without retrying.
6. THE Brain SHALL complete the Gemini API call within 10 seconds; IF the call exceeds 10 seconds, THEN THE Brain SHALL treat it as a failure per criterion 5.

---

### Requirement 3: Contextual Voice Script Generation

**User Story:** As a patient, I want the Brain to speak to me in a calm, personalized voice that addresses me by name and tells me exactly what is happening, so that I can understand and respond to the situation.

#### Acceptance Criteria

1. WHEN an Event with `type` equal to `identity` is processed, THE Brain SHALL generate a Voice_Script that includes: the Patient's name, the recognized person's name, their relationship to the Patient, a one-sentence background about them, and a brief summary of their last conversation (e.g., `"Ismail, your son Hussain is here. He is a software engineer living in Tampa. Last time you spoke, he told you about his new job."`). The Brain SHALL retrieve this information from the `metadata.person_profile` field of the Event.
2. WHEN an Event with `type` equal to `health` and `verified` equal to `true` is processed, THE Brain SHALL generate a Voice_Script that addresses the Patient by name and acknowledges the detected health activity based on `subtype` (e.g., for `subtype` = `drinking`: `"Good job, [Patient_Name]. I can see you are drinking water. Stay hydrated."` For `subtype` = `medicine_taken`: `"[Patient_Name], I see you are taking your medication. Well done."`).
3. WHEN `verified` is `false` for a `health` Event, THE Brain SHALL generate an empty Voice_Script and SHALL NOT trigger voice synthesis or audio playback.
4. THE Brain SHALL retrieve the patient name from the `PATIENT_NAME` environment variable when constructing Voice_Scripts.

---

### Requirement 4: Voice Synthesis via ElevenLabs

**User Story:** As a patient, I want the Brain to synthesize speech using a consistent, calming voice, so that I receive audio alerts that are easy to understand and not alarming.

#### Acceptance Criteria

1. WHEN a non-empty Voice_Script is available after contextual logic, THE Brain SHALL call the ElevenLabs API with the Voice_Script and the configured `ELEVENLABS_VOICE_ID`.
2. THE Brain SHALL save the synthesized audio as an MP3 file to a local temporary directory before playback.
3. THE Brain SHALL name each Audio_File using the `event_id` to ensure uniqueness (e.g., `audio/{event_id}.mp3`).
4. IF the ElevenLabs API call fails or returns an error, THEN THE Brain SHALL log the error and continue processing the Event without audio playback.
5. THE Brain SHALL complete the ElevenLabs API call within 15 seconds; IF the call exceeds 15 seconds, THEN THE Brain SHALL treat it as a failure per criterion 4.

---

### Requirement 5: Audio Playback Through Meta Smart Glasses Speaker

**User Story:** As a patient, I want the synthesized voice alert to play through my glasses speaker so that I hear it privately and immediately without needing to look at or interact with any device.

#### Acceptance Criteria

1. WHEN an Audio_File has been successfully saved, THE Brain SHALL route audio playback to the Meta_Smart_Glasses speaker by selecting it as the target audio output device.
2. THE Brain SHALL identify the Meta_Smart_Glasses speaker by matching the audio output device name against the value configured in the `GLASSES_AUDIO_DEVICE` environment variable.
3. THE Brain SHALL use Pygame mixer initialized with the target device to play the Audio_File through the glasses speaker.
4. IF the Meta_Smart_Glasses speaker device cannot be found or selected at playback time, THEN THE Brain SHALL fall back to the default system audio output, log a warning identifying the fallback, and proceed with playback.
5. THE Brain SHALL block until audio playback is complete before returning the HTTP response to the Vision_Engine.
6. IF Pygame fails to initialize or fails to play the Audio_File on either the glasses speaker or the fallback device, THEN THE Brain SHALL log the error and continue processing the Event without audio playback.
7. THE Brain SHALL delete the Audio_File from the local temporary directory after playback completes successfully.

---

### Requirement 6: Event Logging to MongoDB Atlas

**User Story:** As a Caregiver, I want every processed event to be logged to MongoDB Atlas with full context, so that I can see a real-time feed of my patient's health and identity events in the Caregiver Portal.

#### Acceptance Criteria

1. WHEN an Event has been fully processed (Gemini verification, voice synthesis, audio playback), THE Brain SHALL write an Event_Record to the MongoDB Atlas collection specified by `MONGODB_DB` and `MONGODB_COLLECTION` environment variables.
2. THE Event_Record SHALL contain all fields from the original Event payload plus: `verified` (boolean), `voice_script` (string), `processing_status` (`"success"` or `"partial_failure"`), and `processed_at` (ISO 8601 timestamp).
3. THE Brain SHALL use the Motor async driver for all MongoDB write operations to avoid blocking the FastAPI event loop.
4. IF the MongoDB write fails, THEN THE Brain SHALL log the error and still return HTTP 200 to the Vision_Engine, setting `processing_status` to `"partial_failure"` in the response.
5. THE Brain SHALL complete the MongoDB write within 5 seconds; IF the write exceeds 5 seconds, THEN THE Brain SHALL treat it as a failure per criterion 4.
6. THE Brain SHALL NOT store the raw `image_b64` field in the Event_Record to limit document size in MongoDB Atlas.

---

### Requirement 7: Configuration and Environment Management

**User Story:** As a developer, I want all API keys and service configuration to be loaded from environment variables, so that secrets are never hardcoded and the service can be configured without code changes.

#### Acceptance Criteria

1. THE Brain SHALL load all configuration values from environment variables at startup using `python-dotenv` and `pydantic-settings`.
2. THE Brain SHALL require the following environment variables to be present at startup: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `PATIENT_NAME`, `PATIENT_ID`, and `GLASSES_AUDIO_DEVICE`.
3. IF any required environment variable is missing at startup, THEN THE Brain SHALL log a descriptive error message identifying the missing variable and exit with a non-zero status code.
4. THE Brain SHALL expose a `GET /health` endpoint that returns HTTP 200 with `{"status": "ok"}` when all required configuration is loaded and the MongoDB connection is reachable.
5. IF the MongoDB connection is not reachable at the time of a `GET /health` request, THEN THE Brain SHALL return HTTP 503 with `{"status": "degraded", "reason": "mongodb_unreachable"}`.

---

### Requirement 8: API Contract Compliance

**User Story:** As the Vision Engine, I want the Brain to always return a well-formed JSON response, so that I can reliably parse the result and know whether the event was handled.

#### Acceptance Criteria

1. THE Brain SHALL return `Content-Type: application/json` on all responses from `/event` and `/health`.
2. WHEN an Event is processed successfully with no failures in any downstream step, THE Brain SHALL return `{"event_id": "<id>", "status": "processed", "message": "Event processed successfully."}`.
3. WHEN an Event is processed but one or more downstream steps (Gemini, ElevenLabs, Pygame, MongoDB) fail, THE Brain SHALL return `{"event_id": "<id>", "status": "processed", "message": "Event processed with partial failures."}` and still return HTTP 200.
4. THE Brain SHALL never return HTTP 5xx to the Vision_Engine for failures in downstream integrations (Gemini, ElevenLabs, MongoDB); those failures SHALL be handled internally per their respective requirements.
5. IF an unhandled exception occurs during Event processing, THEN THE Brain SHALL catch it, log the full stack trace, and return HTTP 500 with `{"event_id": "<id>", "status": "error", "message": "Internal server error."}`.

---

### Requirement 9: Startup and Lifecycle

**User Story:** As a developer, I want the Brain service to start cleanly, initialize all connections at startup, and shut down gracefully, so that the service is reliable during the hackathon demo.

#### Acceptance Criteria

1. THE Brain SHALL initialize the Motor MongoDB client and verify connectivity during the FastAPI startup lifespan event.
2. THE Brain SHALL initialize the Pygame mixer and attempt to select the Meta_Smart_Glasses speaker device (identified by `GLASSES_AUDIO_DEVICE`) during the FastAPI startup lifespan event.
3. IF any startup initialization step fails, THEN THE Brain SHALL log the failure and continue starting up in a degraded state rather than refusing to start.
4. WHEN the Brain process receives a shutdown signal, THE Brain SHALL close the Motor MongoDB client connection cleanly.
5. THE Brain SHALL be launchable with the command `uvicorn brain.main:app --host 0.0.0.0 --port 8000`.

---

### Requirement 10: Frame Capture from Meta Smart Glasses

**User Story:** As the Vision Lead, I want the Vision Engine to continuously capture frames from the Meta Smart Glasses POV stream mirrored to the laptop, so that all downstream detection algorithms have a steady supply of current frames to analyze.

#### Acceptance Criteria

1. THE Vision_Engine SHALL capture frames from the video source made available on the laptop by the Meta_Smart_Glasses mirror (via scrcpy or ADB).
2. THE Vision_Engine SHALL open the video source at startup and maintain a continuous capture loop until the process is terminated.
3. IF the video source cannot be opened at startup, THEN THE Vision_Engine SHALL log a descriptive error and exit with a non-zero status code.
4. IF the video source drops or a frame cannot be read during the capture loop, THEN THE Vision_Engine SHALL log the failure and attempt to re-read the next frame without terminating the process.

---

### Requirement 11: Health Item Detection (Food, Water, Medicine)

**User Story:** As a Caregiver, I want the Vision Engine to detect when the patient is eating, drinking, or taking medicine so that the system can log intake events and encourage healthy behaviour.

#### Acceptance Criteria

1. WHEN a new frame is available, THE Vision_Engine SHALL send the frame to Gemini with a prompt asking it to identify whether the patient is eating food, drinking water/a beverage, or taking medicine.
2. WHEN Gemini identifies a health item in the frame, THE Vision_Engine SHALL construct an Event with `type` = `health` and `subtype` set to one of: `eating`, `drinking`, or `medicine_taken` based on the identified item.
3. THE Vision_Engine SHALL set the `confidence` field of the health Event to the confidence value returned or inferred from the Gemini response.
4. THE Vision_Engine SHALL include the base64-encoded source frame in the `image_b64` field of the constructed health Event so the Brain can forward it to Gemini for secondary verification.
5. IF Gemini returns no health item for a given frame, THE Vision_Engine SHALL not construct a health Event for that frame.
6. IF the Gemini call for health item detection fails, THEN THE Vision_Engine SHALL log the failure and skip health detection for that frame without terminating the capture loop.

---

### Requirement 12: Local Face Recognition with Person Profile

**User Story:** As a patient, I want the Vision Engine to recognize familiar faces in my field of view and retrieve their full profile so that the system can tell me who they are, how I know them, and what we last talked about.

#### Acceptance Criteria

1. THE Vision_Engine SHALL load face encodings and their associated Person_Profile records from the Known_Faces_Directory at startup using the `face_recognition` library.
2. Each entry in the Known_Faces_Directory SHALL consist of a reference image file and a paired JSON file containing: `name` (string), `relationship` (string), `background` (string), and `last_conversation` (string).
3. IF the Known_Faces_Directory is empty or does not exist at startup, THEN THE Vision_Engine SHALL log a warning and continue operating with face recognition disabled.
4. WHEN a new frame is available, THE Vision_Engine SHALL attempt to detect and encode any faces present in the frame.
5. WHEN a detected face encoding matches a known encoding within the library's default distance threshold, THE Vision_Engine SHALL construct an Event with `type` = `identity`, `subtype` = `face_recognized`, and `metadata.person_profile` set to the full Person_Profile record for the matched person.
6. WHEN a detected face encoding does not match any known encoding, THE Vision_Engine SHALL not construct an identity Event for that face.
7. THE Vision_Engine SHALL include the base64-encoded source frame in the `image_b64` field of any identity Event it constructs.

---

### Requirement 13: Event Construction and Dispatch

**User Story:** As the Vision Lead, I want the Vision Engine to construct a well-formed JSON Event payload and POST it to the Brain whenever a detection occurs, so that the Brain has all the information it needs to reason about the situation.

#### Acceptance Criteria

1. WHEN a detection (face recognition or health item) triggers event construction, THE Vision_Engine SHALL populate all required JSON_Contract fields: `event_id` (UUID v4), `timestamp` (ISO 8601 UTC), `patient_id` (from `PATIENT_ID` environment variable), `type`, `subtype`, `confidence`, `image_b64`, `metadata`, and `source` (set to `"vision_engine_v1"`). For identity Events, `metadata` SHALL include `person_profile`. For health Events, `metadata` SHALL include `detected_item`.
2. THE Vision_Engine SHALL POST the constructed Event as a JSON body to `http://{BRAIN_HOST}:{BRAIN_PORT}/event`.
3. WHEN the Brain returns HTTP 200, THE Vision_Engine SHALL log the `event_id` and `status` from the response body and continue the capture loop.
4. IF the Brain returns a non-200 response or the POST request fails due to a network error, THEN THE Vision_Engine SHALL log the error including the `event_id` and continue the capture loop without retrying.
5. THE Vision_Engine SHALL complete the POST request within 30 seconds; IF the request exceeds 30 seconds, THEN THE Vision_Engine SHALL treat it as a failure per criterion 4.

---

### Requirement 14: Live Event Feed in Caregiver Portal

**User Story:** As a Caregiver, I want to see a real-time feed of all events detected for my patient so that I can immediately identify health and identity events and take action.

#### Acceptance Criteria

1. THE Caregiver_Portal SHALL query MongoDB_Atlas for the most recent Event_Records on each refresh cycle, ordered by `processed_at` descending.
2. THE Caregiver_Portal SHALL display each Event_Record in a tabular or card layout showing at minimum: `timestamp`, `type`, `subtype`, `confidence`, `verified`, `voice_script`, and `processing_status`.
3. THE Caregiver_Portal SHALL apply Severity_Color coding to each displayed Event_Record: yellow for `type` = `health`, green for `type` = `identity`.
4. THE Caregiver_Portal SHALL auto-refresh the event feed every 5 seconds without requiring a full page reload.
5. IF the MongoDB_Atlas connection is unavailable during a refresh cycle, THEN THE Caregiver_Portal SHALL display the last successfully fetched data and show a visible warning indicating the connection is unavailable.

---

### Requirement 15: Longitudinal Health Trends

**User Story:** As a Caregiver, I want to see charts of my patient's health activity over time so that I can identify patterns and share meaningful data with clinicians.

#### Acceptance Criteria

1. THE Caregiver_Portal SHALL query Snowflake for aggregated health event data using the credentials provided in the `SNOWFLAKE_*` environment variables.
2. THE Caregiver_Portal SHALL render at least one time-series chart using Plotly showing the frequency of `health` events over a configurable time window.
3. THE Caregiver_Portal SHALL render the health trend charts using the pandas DataFrame returned from the Snowflake query as the data source.
4. IF the Snowflake connection is unavailable, THEN THE Caregiver_Portal SHALL display a visible placeholder message in the chart area and continue displaying the live event feed without interruption.
5. THE Caregiver_Portal SHALL refresh the Snowflake-backed charts on each auto-refresh cycle alongside the live event feed.

---

### Requirement 16: JSON Contract Compliance Across Services

**User Story:** As the system architect, I want all three services to produce and consume data that strictly conforms to the shared JSON Contract so that the pipeline is reliable and each service can be developed and tested independently.

#### Acceptance Criteria

1. THE Vision_Engine SHALL produce Event payloads that conform to the JSON_Contract schema, including all required fields with their specified types.
2. THE Brain SHALL validate every incoming Event payload against the JSON_Contract schema and reject non-conforming payloads with HTTP 422.
3. THE Brain SHALL produce Event_Records that conform to the JSON_Contract Event_Record schema, including all required enrichment fields (`verified`, `voice_script`, `processing_status`, `processed_at`).
4. THE Caregiver_Portal SHALL read Event_Records from MongoDB_Atlas using the field names defined in the JSON_Contract Event_Record schema.
5. THE Brain SHALL NOT store the `image_b64` field in the Event_Record written to MongoDB_Atlas, as specified in the JSON_Contract.
6. FOR ALL valid Event payloads produced by the Vision_Engine, THE Brain SHALL be able to parse and process the payload without error (round-trip structural compatibility).

---

### Requirement 17: System Launcher

**User Story:** As a developer, I want a single command to start all three services simultaneously so that the demo can be set up quickly and reliably without opening multiple terminals.

#### Acceptance Criteria

1. THE System SHALL provide a `run_all.py` script at the repository root that launches the Vision_Engine, Brain, and Caregiver_Portal as separate subprocesses.
2. WHEN `run_all.py` is executed, THE System SHALL start all three services within 15 seconds on the demo laptop.
3. THE `run_all.py` script SHALL load environment variables from the `.env` file before launching any service subprocess.
4. WHEN any service subprocess terminates unexpectedly, THE `run_all.py` script SHALL log which service terminated and its exit code, and continue running the remaining services.
5. WHEN `run_all.py` receives a keyboard interrupt (SIGINT), THE System SHALL send a termination signal to all running service subprocesses and wait for them to exit cleanly before the launcher process exits.

---

### Requirement 18: Hardware Mirror Integration

**User Story:** As a developer, I want the Meta Smart Glasses POV to be visible on the laptop screen alongside the dashboard so that judges can see both the patient's perspective and the system's response simultaneously.

#### Acceptance Criteria

1. THE System SHALL support mirroring the Meta_Smart_Glasses video feed to the laptop display using scrcpy or ADB before the Vision_Engine capture loop begins.
2. THE Vision_Engine SHALL be configurable to read from the mirrored video source by setting the video source identifier via an environment variable or configuration parameter.
3. THE System documentation SHALL include step-by-step instructions for establishing the hardware mirror connection before running `run_all.py`.

---

### Requirement 19: Shared Environment Configuration

**User Story:** As a developer, I want all three services to read their configuration from a single `.env` file so that secrets and service addresses are managed in one place and never hardcoded.

#### Acceptance Criteria

1. THE System SHALL use a single `.env` file at the repository root as the authoritative source for all secrets and configuration values for all three services.
2. THE System SHALL provide a `.env.example` file at the repository root listing every required environment variable with placeholder values and inline comments describing each variable's purpose.
3. THE Vision_Engine SHALL read `PATIENT_ID`, `BRAIN_HOST`, and `BRAIN_PORT` from environment variables at startup.
4. THE Brain SHALL read `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `PATIENT_NAME`, `PATIENT_ID`, and `GLASSES_AUDIO_DEVICE` from environment variables at startup.
5. THE Caregiver_Portal SHALL read `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, and `SNOWFLAKE_WAREHOUSE` from environment variables at startup.
6. IF a required environment variable is missing for any service at startup, THEN that service SHALL log a descriptive error identifying the missing variable and exit with a non-zero status code.

---

### Requirement 20: Graceful Degradation

**User Story:** As a Caregiver, I want the system to continue operating even if one service or external API becomes unavailable, so that a partial failure does not cause a complete loss of patient monitoring.

#### Acceptance Criteria

1. IF the Brain is unreachable when the Vision_Engine attempts to POST an Event, THEN THE Vision_Engine SHALL log the failure and continue the capture loop without terminating.
2. IF the Gemini API is unavailable during Brain processing, THEN THE Brain SHALL set `verified` to `false`, skip voice synthesis for health events, log the failure, and continue to write the Event_Record to MongoDB_Atlas.
3. IF the ElevenLabs API is unavailable during Brain processing, THEN THE Brain SHALL log the failure, skip audio playback, and continue to write the Event_Record to MongoDB_Atlas.
4. IF MongoDB_Atlas is unavailable during Brain processing, THEN THE Brain SHALL log the failure and still return HTTP 200 to the Vision_Engine with `processing_status` = `"partial_failure"`.
5. IF MongoDB_Atlas is unavailable when the Caregiver_Portal attempts to refresh the event feed, THEN THE Caregiver_Portal SHALL display the last successfully fetched data and show a visible connection-unavailable warning.
6. IF Snowflake is unavailable when the Caregiver_Portal attempts to refresh health trends, THEN THE Caregiver_Portal SHALL display a placeholder in the chart area and continue displaying the live event feed without interruption.
