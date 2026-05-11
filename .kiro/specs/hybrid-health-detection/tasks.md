# Implementation Plan: Hybrid Health Detection

## Overview

Replace the internals of `detect_health_activity()` in `services/vision/face_recognition_engine.py` with a two-pass pipeline: a fast local YOLO-based Primary Pass that filters frames before any cloud call, followed by a targeted Gemini 1.5 Flash Secondary Pass that returns a structured confidence score. The existing function signature, caller interface, and all downstream contracts remain unchanged.

## Tasks

- [x] 1. Create `LocalDetector` module (`services/vision/local_detector.py`)
  - Define `DetectionResult` dataclass with fields: `flagged`, `detected_objects`, `confidence_scores`, `medicine_flagged`
  - Implement `LocalDetector.__init__` that loads the YOLO model from `model_path` using Ultralytics, reads `LOCAL_DETECTOR_CONFIDENCE` env var (default 0.4)
  - Implement `LocalDetector.run(frame_bgr)` that runs YOLO inference, maps detected labels against `HEALTH_SUBTYPE_MAP`, sets `flagged=True` when any health-relevant object is found, and sets `medicine_flagged=True` for medicine-related objects regardless of YOLO confidence score (safety override)
  - Implement module-level `get_detector()` singleton using a global `_detector` variable — loads once, reuses across all calls
  - Implement startup failure handling: catch model load errors, log CRITICAL, set `_health_detection_disabled = True`
  - _Requirements: 1.1, 1.2, 1.5, 1.6, 4.2, 4.4, 6.2, 8.1, 8.2, 8.3, 8.4_

  - [ ]* 1.1 Write property test: health object detection implies flagging (Property 3)
    - **Property 3: Health object detection implies flagging**
    - For any frame where YOLO detects at least one object whose label maps to a key in `HEALTH_SUBTYPE_MAP`, `DetectionResult.flagged` SHALL be `True`
    - Use `st.sampled_from(list(HEALTH_SUBTYPE_MAP.keys()))` to generate object labels; mock YOLO output
    - **Validates: Requirements 1.5**

  - [ ]* 1.2 Write property test: medicine objects flagged at any YOLO confidence (Property 11)
    - **Property 11: Medicine objects are flagged at any YOLO confidence**
    - For any YOLO detection result containing a medicine-related label at any confidence score (including 0.0), `medicine_flagged` SHALL be `True` and `flagged` SHALL be `True`
    - Use `st.floats(min_value=0.0, max_value=1.0)` for confidence scores; use medicine label set `{pill, tablet, medicine, medication, medicine packet}`
    - **Validates: Requirements 6.2**

  - [ ]* 1.3 Write property test: model loaded exactly once (Property 14)
    - **Property 14: Model is loaded exactly once regardless of frame count**
    - For any N ≥ 1 frames processed, the YOLO model loading function SHALL be called exactly once
    - Use `st.integers(min_value=1, max_value=50)` for frame count; mock `_load_detector` and count calls
    - **Validates: Requirements 8.1, 8.3**

- [x] 2. Create `GeminiHealthClient` module (`services/vision/gemini_health.py`)
  - Implement `ConfidenceResult` dataclass with fields: `score`, `raw_text`, `subtype`
  - Implement `build_health_prompt(subtype)` returning the three subtype-specific prompts defined in the design (drinking, eating, medicine_taken)
  - Implement `parse_confidence_score(response_text)` that extracts the first float matching `\d+\.\d+` or `\d+`, clamps to [0.0, 1.0], returns 0.0 and logs WARNING on parse failure
  - Implement `call_gemini_health(frame_b64, subtype, api_key, timeout=10.0)` using `google.generativeai` with `gemini-1.5-flash`, wrapped in a 10-second timeout; log TIMEOUT error and return `None` on timeout
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 2.8_

  - [ ]* 2.1 Write property test: confidence score parsing always yields [0.0, 1.0] (Property 4)
    - **Property 4: Confidence score parsing always yields a value in [0.0, 1.0]**
    - For any string (including empty, no numbers, multiple numbers, out-of-range numbers), `parse_confidence_score` SHALL return a float in [0.0, 1.0]
    - Use `st.text()` for arbitrary strings and `st.floats().map(str)` for numeric strings
    - **Validates: Requirements 2.3, 2.4**

  - [ ]* 2.2 Write property test: subtype-specific prompts reference correct subtype (Property 15)
    - **Property 15: Subtype-specific prompts reference the correct subtype**
    - For each subtype in `{"drinking", "eating", "medicine_taken"}`, `build_health_prompt(subtype)` SHALL contain a keyword specific to that subtype and SHALL NOT contain keywords specific to other subtypes
    - Use `st.sampled_from(["drinking", "eating", "medicine_taken"])` as input
    - **Validates: Requirements 2.1, 2.2**

- [x] 3. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement per-subtype threshold configuration helpers in `face_recognition_engine.py`
  - Add `_DEFAULT_THRESHOLDS` dict: `{"medicine_taken": 0.45, "eating": 0.6, "drinking": 0.6}`
  - Implement `_get_threshold(subtype)` that reads `HEALTH_DETECTION_THRESHOLD` env var; if set and parseable, returns it for all subtypes uniformly; otherwise returns per-subtype default; logs WARNING on unparseable value
  - Implement `_resolve_subtype(detected_objects)` that maps the first matching object label from `HEALTH_SUBTYPE_MAP` to a subtype string, returning `None` if no match
  - Read `LOCAL_DETECTOR_MODEL` env var at module level; if path does not exist, log CRITICAL and set `_health_detection_disabled = True`
  - Read `HEALTH_DETECTION_THRESHOLD` and `LOCAL_DETECTOR_CONFIDENCE` env vars at module level with documented defaults
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.1_

  - [ ]* 4.1 Write property test: medicine threshold always lower than other subtypes (Property 12)
    - **Property 12: Medicine threshold is always lower than other subtype thresholds**
    - For any valid configuration (env var unset), the effective threshold for `medicine_taken` SHALL be strictly less than for `eating` and `drinking`
    - Use `st.none() | st.floats(min_value=0.0, max_value=1.0).map(str)` to simulate env var presence/absence
    - **Validates: Requirements 6.1**

- [x] 5. Implement `_run_secondary_pass()` and `_dispatch_health_event()` helpers in `face_recognition_engine.py`
  - Implement `_run_secondary_pass(frame_bgr, subtype)` that JPEG-encodes the frame (max 640px longest side), calls `call_gemini_health`, logs INFO with prompt/response/score, logs DEBUG with wall-clock latency, returns `ConfidenceResult` or `None` on timeout
  - Implement `_dispatch_health_event(frame_bgr, subtype, confidence_result, detection_result)` that constructs an `Event` from `shared.contract` with all required fields (UUID4 event_id, UTC ISO-8601 timestamp, patient_id from env, type="health", source="vision_engine_v1", confidence, image_b64, metadata with detected_item), POSTs to Brain within 5 seconds, logs INFO with subtype/score/HTTP status, logs ERROR on non-2xx or connection error without retrying
  - _Requirements: 2.6, 3.1, 3.2, 3.3, 5.2, 5.5, 5.6, 7.1, 7.2_

  - [ ]* 5.1 Write property test: dispatched events always contain all required fields (Property 7)
    - **Property 7: Dispatched events always contain all required fields with correct values**
    - For any valid detection result passing threshold and cooldown gates, the constructed `Event` SHALL be a valid `shared.contract.Event` instance with `type="health"`, `source="vision_engine_v1"`, UUID4 `event_id`, UTC ISO-8601 `timestamp`, and `confidence` equal to the Secondary Pass score
    - Use `st.floats(min_value=0.0, max_value=1.0)` for scores; `st.sampled_from(["drinking", "eating", "medicine_taken"])` for subtypes
    - **Validates: Requirements 3.1, 7.1, 7.2**

- [x] 6. Replace `detect_health_activity()` in `face_recognition_engine.py` with the two-pass pipeline
  - Call `get_detector()` once during module-level initialisation (alongside `_load_face_models()`) so the model is warm before the main loop
  - Replace the function body with the new pipeline: frame quality check → Primary Pass (with fallback on exception) → medicine safety override → subtype resolution → Secondary Pass → threshold gate → cooldown gate → dispatch
  - Implement medicine safety override: if frame fails quality check AND `medicine_flagged=True`, skip quality gate and proceed to Secondary Pass
  - Implement Local Detector exception fallback: catch all exceptions, log WARNING, construct synthetic `DetectionResult(flagged=True, ...)` and continue to Secondary Pass
  - Add all required logging: DEBUG for Primary Pass objects/flagged status, DEBUG for cooldown suppression with remaining seconds, INFO for below-threshold suppression with score/threshold, INFO for dispatched events
  - _Requirements: 1.1, 1.3, 1.4, 1.7, 3.4, 3.5, 5.1, 5.3, 5.4, 5.5, 6.3, 7.4, 7.5_

  - [ ]* 6.1 Write property test: Gemini never called for non-flagged frames (Property 1)
    - **Property 1: Gemini is never called for non-flagged frames**
    - For any frame where `LocalDetector.run()` returns `flagged=False`, the Gemini client SHALL NOT be invoked
    - Mock `LocalDetector.run` to return `DetectionResult(flagged=False, ...)` and assert `call_gemini_health` call count is 0
    - **Validates: Requirements 1.3**

  - [ ]* 6.2 Write property test: Gemini always called for flagged frames when quality passes (Property 2)
    - **Property 2: Gemini is always called for flagged frames (when quality passes)**
    - For any frame where `LocalDetector.run()` returns `flagged=True` and the frame passes quality check (or medicine override applies), Gemini SHALL be invoked exactly once
    - Use `st.booleans()` for quality check result combined with `st.booleans()` for `medicine_flagged`
    - **Validates: Requirements 1.4, 6.3**

  - [ ]* 6.3 Write property test: below-threshold scores never produce events (Property 5)
    - **Property 5: Below-threshold scores never produce events**
    - For any subtype and any confidence score strictly below the configured threshold, the pipeline SHALL construct no `Health_Event` and make no POST to the Brain
    - Use `st.sampled_from(["drinking", "eating", "medicine_taken"])` and `st.floats(min_value=0.0, max_value=0.449)` (below medicine threshold)
    - **Validates: Requirements 2.5**

  - [ ]* 6.4 Write property test: at-or-above-threshold scores produce exactly one event (Property 6)
    - **Property 6: At-or-above-threshold scores produce exactly one event (when cooldown is inactive)**
    - For any subtype and any confidence score at or above the configured threshold, when cooldown is inactive, the pipeline SHALL construct exactly one `Health_Event` and POST it to the Brain
    - Use `st.floats(min_value=0.6, max_value=1.0)` for scores; mock Brain POST to return 200
    - **Validates: Requirements 2.6, 3.1**

  - [ ]* 6.5 Write property test: cooldown suppresses same-subtype events within 120 seconds (Property 8)
    - **Property 8: Cooldown suppresses all same-subtype events within 120 seconds**
    - After one `Health_Event` of a subtype is dispatched, any subsequent detection of the same subtype within 120 seconds SHALL produce no additional dispatch
    - Use `st.floats(min_value=0.0, max_value=119.9)` for elapsed time since last event
    - **Validates: Requirements 3.4**

  - [ ]* 6.6 Write property test: quality-failed frames skipped unless medicine flagged (Property 9)
    - **Property 9: Quality-failed frames are skipped unless medicine is flagged**
    - For any frame that fails `is_frame_usable()`, if `medicine_flagged=False`, neither Local Detector escalation nor Gemini call SHALL occur
    - Mock `is_frame_usable` to return False and `medicine_flagged=False`; assert zero Gemini calls
    - **Validates: Requirements 3.5, 6.3**

  - [ ]* 6.7 Write property test: medicine safety override bypasses quality gate (Property 10)
    - **Property 10: Medicine safety override bypasses quality gate**
    - For any frame that fails `is_frame_usable()`, if `medicine_flagged=True`, the Secondary Pass SHALL still be invoked
    - Mock `is_frame_usable` to return False and `medicine_flagged=True`; assert Gemini is called once
    - **Validates: Requirements 6.3**

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Write unit tests for configuration, startup, and error handling (`tests/vision/test_local_detector.py`, `tests/vision/test_gemini_health.py`, `tests/vision/test_detect_health_activity.py`)
  - [x] 8.1 Write unit tests for `LocalDetector` configuration and startup
    - Test `LOCAL_DETECTOR_MODEL` env var: set to valid path, unset (uses default), set to nonexistent path (CRITICAL log + health detection disabled)
    - Test `LOCAL_DETECTOR_CONFIDENCE` env var: valid float, unset (default 0.4), unparseable value
    - Test model load failure: assert `_health_detection_disabled = True` and face recognition continues
    - _Requirements: 4.2, 4.4, 4.5, 8.2_

  - [x] 8.2 Write unit tests for `GeminiHealthClient` error paths
    - Test Gemini timeout: assert ERROR log and `None` return from `_run_secondary_pass`
    - Test parse failure: assert WARNING log and `score=0.0` returned
    - Test Brain POST non-2xx: assert ERROR log and no retry
    - Test Brain POST connection error: assert ERROR log and no retry
    - _Requirements: 2.4, 2.7, 3.3, 5.2_

  - [ ]* 8.3 Write unit tests for cooldown edge cases
    - Test exactly at 120s boundary (should suppress), just before 120s (suppress), just after 120s (allow)
    - Test `HEALTH_DETECTION_THRESHOLD` env var: valid float overrides all subtypes, unparseable logs WARNING and uses defaults
    - _Requirements: 3.4, 4.3_

  - [ ]* 8.4 Write unit tests for logging assertions
    - Assert DEBUG log when Primary Pass runs (objects list + flagged status)
    - Assert INFO log when Secondary Pass invoked (prompt, raw response, score)
    - Assert DEBUG log when cooldown suppresses (subtype + remaining seconds)
    - Assert INFO log when below-threshold suppresses (subtype, score, threshold)
    - Assert INFO log when event dispatched (subtype, score, HTTP status)
    - Assert DEBUG log for Secondary Pass wall-clock latency
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 9. Write `tests/vision/test_health_properties.py` — all Hypothesis property tests
  - Create the test file with `@settings(max_examples=100)` for standard properties and `@settings(max_examples=200)` for safety-critical properties (10, 11, 12)
  - Annotate each test with `# Feature: hybrid-health-detection, Property N: <property_text>`
  - Collect all property sub-tasks from tasks 1–6 into this single file
  - _Requirements: all_

  - [ ]* 9.1 Write property test: HEALTH_SUBTYPE_MAP round-trip (Property 13)
    - **Property 13: HEALTH_SUBTYPE_MAP round-trip**
    - For any key in `HEALTH_SUBTYPE_MAP`, the mapped value SHALL be one of `{"drinking", "eating", "medicine_taken"}` and the mapping SHALL be deterministic
    - Use `st.sampled_from(list(HEALTH_SUBTYPE_MAP.keys()))` as input; assert value in valid set and repeated calls return same result
    - **Validates: Requirements 7.4**

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The design uses Python with Hypothesis for property-based testing and pytest for unit tests
- Test files live under `tests/vision/` — create the directory if it does not exist
- The YOLO model weights (`yolov8n.pt`) must be downloaded once to `tests/vision/models/` and gitignored
- Property tests for safety-critical properties (10, 11, 12) use `max_examples=200` per the design's testing strategy
- The `detect_health_activity()` function signature is unchanged — this is a drop-in replacement
- `shared/contract.py` is NOT modified — backward compatibility is preserved by construction
