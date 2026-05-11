# Requirements Document

## Introduction

The hybrid health detection feature introduces a two-pass detection pipeline into the AuraGuard Vision Engine. Currently, every frame sampled during the 5-second health check interval is sent directly to Gemini for classification, with no local pre-filter. This causes unnecessary cloud API calls, increases latency, and risks alert fatigue when Gemini returns false positives.

The new architecture adds a **Primary Pass** — a fast, lightweight local detector (YOLO-based object detection or a motion/proximity heuristic) that runs on each sampled frame and decides whether the frame is worth escalating. Only frames that pass the local filter are forwarded to the **Secondary Pass**, which calls Gemini 1.5 Flash with a targeted, subtype-specific prompt and a confidence score. The Brain service continues to own final verification and audio alerting.

This is a life-critical patient-monitoring system. Safety and accuracy requirements take precedence over performance optimisations.

---

## Glossary

- **Vision_Engine**: The Python process in `services/vision/face_recognition_engine.py` that reads video frames, runs face recognition, and (optionally) runs health detection.
- **Local_Detector**: The new lightweight YOLO-based (or heuristic) object/motion detector added to the Vision Engine as the Primary Pass.
- **Gemini_Client**: The component inside the Vision Engine that calls the Gemini 1.5 Flash multimodal API as the Secondary Pass.
- **Brain**: The FastAPI service in `services/brain/` that receives events via `POST /event`, runs final Gemini verification, generates voice scripts, and plays audio.
- **Health_Event**: A Pydantic `Event` model instance with `type="health"` and a `subtype` of `drinking`, `eating`, or `medicine_taken`.
- **Primary_Pass**: The local detection step that filters frames before any cloud call is made.
- **Secondary_Pass**: The Gemini 1.5 Flash call that classifies a frame flagged by the Primary Pass.
- **Confidence_Score**: A float in [0.0, 1.0] returned by the Secondary Pass indicating how certain Gemini is that the detected activity is occurring.
- **Health_Cooldown**: The 120-second per-subtype suppression window already enforced in the Vision Engine.
- **Frame_Quality_Check**: The existing brightness and Laplacian-variance gate (`is_frame_usable`) applied before any detection.
- **HEALTH_CHECK_INTERVAL_SECONDS**: The existing 5-second timer that controls how often a frame is sampled for health detection.
- **Detection_Threshold**: A configurable minimum Confidence_Score below which a Secondary Pass result is discarded.
- **Patient**: The single monitored individual wearing the Meta Smart Glasses.

---

## Requirements

### Requirement 1: Local Primary Pass Detection

**User Story:** As a system operator, I want the Vision Engine to run a fast local detector on each sampled health frame, so that only frames containing relevant objects or hand-to-face motion are escalated to the cloud.

#### Acceptance Criteria

1. WHEN a health frame is sampled at the HEALTH_CHECK_INTERVAL_SECONDS interval, THE Vision_Engine SHALL run the Local_Detector on that frame before making any network call.
2. THE Local_Detector SHALL classify each frame as either **flagged** or **not-flagged** within 200 ms on the host CPU.
3. WHEN the Local_Detector classifies a frame as **not-flagged**, THE Vision_Engine SHALL discard the frame and make no Gemini API call for that sample.
4. WHEN the Local_Detector classifies a frame as **flagged**, THE Vision_Engine SHALL pass the frame to the Secondary Pass (Gemini_Client).
5. THE Local_Detector SHALL flag a frame when at least one of the following is detected: a hand in proximity to the face region, or any object from the set {cup, glass, mug, bottle, water bottle, soda can, food item, fork, spoon, sandwich, pill, tablet, medicine packet}.
6. THE Local_Detector SHALL operate without any external API calls or network I/O.
7. IF the Local_Detector raises an unhandled exception, THEN THE Vision_Engine SHALL log the error at WARNING level and fall back to sending the frame directly to the Secondary Pass, preserving the existing behaviour.

---

### Requirement 2: Secondary Pass — Targeted Gemini Verification

**User Story:** As a system operator, I want the Vision Engine to send flagged frames to Gemini 1.5 Flash with a targeted, subtype-specific prompt, so that the cloud call is focused and returns a usable confidence score.

#### Acceptance Criteria

1. WHEN the Primary Pass flags a frame, THE Gemini_Client SHALL send the frame to Gemini 1.5 Flash with a prompt that names the specific suspected activity (e.g., "Is the user taking medication in this frame? Return a confidence score between 0.0 and 1.0.").
2. THE Gemini_Client SHALL derive the targeted prompt from the object(s) detected by the Local_Detector, mapping detected objects to the subtypes defined in `HEALTH_SUBTYPE_MAP`.
3. THE Gemini_Client SHALL parse the Gemini response to extract a numeric Confidence_Score in [0.0, 1.0].
4. IF the Gemini response does not contain a parseable numeric score, THEN THE Gemini_Client SHALL assign a Confidence_Score of 0.0 and log a WARNING.
5. THE Gemini_Client SHALL apply a configurable Detection_Threshold (default 0.6); WHEN the Confidence_Score is below the Detection_Threshold, THE Vision_Engine SHALL discard the result and emit no Health_Event.
6. WHEN the Confidence_Score meets or exceeds the Detection_Threshold, THE Vision_Engine SHALL construct a Health_Event and POST it to the Brain.
7. THE Gemini_Client SHALL complete the Secondary Pass within 10 seconds; IF the call exceeds 10 seconds, THEN THE Vision_Engine SHALL log a TIMEOUT error and discard the frame.
8. THE Gemini_Client SHALL use the model identifier `gemini-1.5-flash` for all Secondary Pass calls.

---

### Requirement 3: Health Event Construction and Dispatch

**User Story:** As a system operator, I want the Vision Engine to construct a well-formed Health_Event and send it to the Brain whenever the two-pass pipeline confirms an activity, so that the Brain can generate an appropriate voice alert.

#### Acceptance Criteria

1. WHEN the Secondary Pass returns a Confidence_Score at or above the Detection_Threshold, THE Vision_Engine SHALL construct a Health_Event with the following fields populated: `event_id` (UUID4), `timestamp` (UTC ISO-8601), `patient_id` (from environment), `type="health"`, `subtype` (mapped from detected object), `confidence` (the Confidence_Score from the Secondary Pass), `image_b64` (JPEG-encoded frame, resized to max 640 px on the longest side), `metadata` containing `detected_item` (the raw object label from the Local_Detector), and `source="vision_engine_v1"`.
2. THE Vision_Engine SHALL POST the Health_Event to the Brain endpoint `POST /event` within 5 seconds of constructing it.
3. IF the Brain POST returns a non-2xx HTTP status or raises a connection error, THEN THE Vision_Engine SHALL log the failure at ERROR level and not retry.
4. THE Vision_Engine SHALL enforce the existing Health_Cooldown: WHEN a Health_Event of a given subtype has been dispatched, THE Vision_Engine SHALL suppress further Health_Events of the same subtype for 120 seconds.
5. THE Vision_Engine SHALL apply the existing Frame_Quality_Check before the Primary Pass; IF the frame fails the quality check, THEN THE Vision_Engine SHALL skip both passes for that sample.

---

### Requirement 4: Configuration and Feature Flag

**User Story:** As a system operator, I want to control the hybrid detection pipeline through environment variables, so that I can tune thresholds and disable the feature without code changes.

#### Acceptance Criteria

1. THE Vision_Engine SHALL read the existing `ENABLE_HEALTH_DETECTION` environment variable; WHILE `ENABLE_HEALTH_DETECTION` is `false`, THE Vision_Engine SHALL not run either pass.
2. THE Vision_Engine SHALL read a `LOCAL_DETECTOR_MODEL` environment variable specifying the path to the YOLO model weights file; IF the variable is unset, THE Vision_Engine SHALL use a default bundled model path.
3. THE Vision_Engine SHALL read a `HEALTH_DETECTION_THRESHOLD` environment variable as a float; IF the variable is unset or unparseable, THE Vision_Engine SHALL use the default value of 0.6.
4. THE Vision_Engine SHALL read a `LOCAL_DETECTOR_CONFIDENCE` environment variable as a float threshold for the Local_Detector's own object-detection confidence; IF the variable is unset, THE Vision_Engine SHALL use a default of 0.4.
5. WHERE `LOCAL_DETECTOR_MODEL` points to a file that does not exist, THE Vision_Engine SHALL log a CRITICAL error at startup and disable health detection for the session.

---

### Requirement 5: Observability and Logging

**User Story:** As a system operator, I want detailed logs for every step of the two-pass pipeline, so that I can diagnose false positives, false negatives, and latency issues in a life-critical monitoring context.

#### Acceptance Criteria

1. WHEN the Local_Detector runs on a frame, THE Vision_Engine SHALL log at DEBUG level: the list of detected objects and their confidence scores, and whether the frame was flagged.
2. WHEN the Secondary Pass is invoked, THE Vision_Engine SHALL log at INFO level: the targeted prompt sent to Gemini, the raw Gemini response text, and the parsed Confidence_Score.
3. WHEN a Health_Event is suppressed by the Health_Cooldown, THE Vision_Engine SHALL log at DEBUG level the subtype and the remaining cooldown duration in seconds.
4. WHEN a Health_Event is suppressed because the Confidence_Score is below the Detection_Threshold, THE Vision_Engine SHALL log at INFO level the subtype, the Confidence_Score, and the Detection_Threshold.
5. WHEN a Health_Event is dispatched to the Brain, THE Vision_Engine SHALL log at INFO level the subtype, Confidence_Score, and the HTTP response status code from the Brain.
6. THE Vision_Engine SHALL record the wall-clock latency of each Secondary Pass call and log it at DEBUG level.

---

### Requirement 6: Safety and Accuracy Constraints

**User Story:** As a patient safety engineer, I want the pipeline to be biased toward false positives over false negatives for the `medicine_taken` subtype, so that missed medication events are minimised in a life-critical context.

#### Acceptance Criteria

1. THE Vision_Engine SHALL use a lower Detection_Threshold for the `medicine_taken` subtype than for `eating` and `drinking`; the default `medicine_taken` threshold SHALL be 0.45 and the default threshold for all other subtypes SHALL be 0.6.
2. WHEN the Local_Detector detects any object from the set {pill, tablet, medicine, medication, medicine packet}, THE Vision_Engine SHALL flag the frame regardless of the object-detection confidence score returned by the Local_Detector.
3. THE Vision_Engine SHALL NOT discard a `medicine_taken` candidate frame solely on the basis of a failed Frame_Quality_Check; IF the frame fails the quality check AND the Local_Detector has already flagged a medicine-related object, THEN THE Vision_Engine SHALL still invoke the Secondary Pass.
4. THE Vision_Engine SHALL preserve the existing 120-second Health_Cooldown for `medicine_taken` events to prevent alert fatigue while maintaining safety sensitivity.

---

### Requirement 7: Backward Compatibility

**User Story:** As a developer, I want the new two-pass pipeline to be a drop-in replacement for the existing `detect_health_activity` function, so that no changes are required in the Brain service or the shared contract.

#### Acceptance Criteria

1. THE Vision_Engine SHALL continue to POST Health_Events to the Brain using the existing `Event` Pydantic model defined in `shared/contract.py` without modification.
2. THE Vision_Engine SHALL continue to honour the `source` field value `"vision_engine_v1"` in all dispatched events.
3. THE Brain SHALL require no code changes to process Health_Events produced by the new two-pass pipeline.
4. THE Vision_Engine SHALL preserve the existing `HEALTH_SUBTYPE_MAP` mapping from raw object labels to subtypes (`drinking`, `eating`, `medicine_taken`).
5. WHEN `ENABLE_HEALTH_DETECTION` is `false`, THE Vision_Engine SHALL behave identically to the current implementation (no health detection of any kind).

---

### Requirement 8: Local Detector Model Loading

**User Story:** As a developer, I want the Local_Detector model to be loaded once at startup and reused across all frames, so that per-frame inference latency is minimised.

#### Acceptance Criteria

1. THE Vision_Engine SHALL load the Local_Detector model into memory once during process initialisation, before the main frame-processing loop starts.
2. IF the Local_Detector model fails to load at startup, THEN THE Vision_Engine SHALL log a CRITICAL error, set `ENABLE_HEALTH_DETECTION` to `false` for the session, and continue running face recognition normally.
3. THE Vision_Engine SHALL reuse the same in-memory Local_Detector model instance for all frames without reloading between frames.
4. THE Local_Detector SHALL be thread-safe when called from the existing background health worker thread.
