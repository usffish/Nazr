"""
Face Recognition Engine (YuNet + SFace with ONNXRuntime)
---------------------------------------------------------
1. Loads known-face embeddings from local known_faces/*.jpg + *.json files.
2. Uses OpenCV's YuNet for face detection (via DNN).
3. Uses ONNXRuntime for SFace recognition (workaround for OpenCV ONNX bug).
4. No external API calls - all processing is local.
5. Logs events to local JSONL and speaks via edge-tts.
"""

import os
import json
import uuid
import base64
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import queue
import subprocess
import threading
import time
import av
import cv2
import numpy as np
import onnxruntime as ort
import requests as http_requests
from dotenv import load_dotenv
import pygame

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Import canonical HEALTH_SUBTYPE_MAP (lowercase keys) from local_detector for
# use in the new hybrid-detection helpers.  The legacy uppercase HEALTH_SUBTYPE_MAP
# defined below is kept for backward compatibility with detect_health_activity().
from services.vision.local_detector import (  # noqa: E402
    HEALTH_SUBTYPE_MAP as _LOCAL_HEALTH_SUBTYPE_MAP,
    DetectionResult,
    get_detector,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
KNOWN_FACES_DIR = _ROOT / "known_faces"
_MODELS_DIR = _ROOT.parent.parent / "tests" / "vision" / "models"
YUNET_MODEL = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = _MODELS_DIR / "face_recognition_sface_2021dec.onnx"
_EVENT_LOG_DIR = Path(__file__).parent.parent.parent / "tempfiles"
EVENTS_JSONL = _EVENT_LOG_DIR / "events.jsonl"
DETECTION_PAUSED_FLAG = _EVENT_LOG_DIR / "detection_paused"

# ── Config ────────────────────────────────────────────────────────────────────

RECOGNITION_INTERVAL = 2.0  # seconds between recognition attempts
COOLDOWN_SECONDS = 10  # seconds before the same person can be re-announced
CONFIDENCE_THRESHOLD = 0.363  # SFace cosine threshold
ENABLE_HEALTH_DETECTION: bool = os.getenv("ENABLE_HEALTH_DETECTION", "false").lower() == "true"
HEALTH_CHECK_INTERVAL_SECONDS = 5
HEALTH_COOLDOWN_SECONDS = 120

HEALTH_SUBTYPE_MAP = {
    "WATER BOTTLE": "drinking", "SODA CAN": "drinking", "CAN": "drinking",
    "CUP": "drinking", "GLASS": "drinking", "MUG": "drinking",
    "BOTTLE": "drinking", "DRINKING": "drinking",
    "FOOD": "eating", "FORK": "eating", "SPOON": "eating",
    "SANDWICH": "eating", "EATING": "eating",
    "PILL": "medicine_taken", "PILLS": "medicine_taken", "TABLET": "medicine_taken",
    "MEDICINE": "medicine_taken", "MEDICATION": "medicine_taken",
}

_last_health_event: dict[str, float] = {}

# ── Hybrid health detection — module-level configuration ─────────────────────
# Read env vars once at import time so they are available before the main loop.

# Path to the YOLO model weights used by the Local Detector (Primary Pass).
# Default: tests/vision/models/yolov8n.pt (relative to repo root).
_LOCAL_DETECTOR_MODEL_PATH: str = os.getenv(
    "LOCAL_DETECTOR_MODEL",
    str(Path(__file__).parent.parent.parent / "tests" / "vision" / "models" / "yolov8n.pt"),
)

# Per-subtype confidence threshold override.  When set, applies uniformly to
# all subtypes.  When unset, per-subtype defaults in _DEFAULT_THRESHOLDS apply.
# Default: (unset — use per-subtype defaults)
_HEALTH_DETECTION_THRESHOLD_ENV: str | None = os.getenv("HEALTH_DETECTION_THRESHOLD")

# Minimum YOLO object-detection confidence for the Local Detector.
# Default: 0.4  (Requirement 4.4)
_LOCAL_DETECTOR_CONFIDENCE_ENV: str = os.getenv("LOCAL_DETECTOR_CONFIDENCE", "0.4")

# Module-level flag: set to True at startup if the YOLO model file is missing,
# disabling health detection for the session (Requirement 4.5).
_health_detection_disabled: bool = False

if not os.path.exists(_LOCAL_DETECTOR_MODEL_PATH):
    log.critical(
        "CRITICAL: LOCAL_DETECTOR_MODEL path does not exist: %r — "
        "health detection is DISABLED for this session.",
        _LOCAL_DETECTOR_MODEL_PATH,
    )
    _health_detection_disabled = True

# ── Per-subtype threshold defaults (Requirement 6.1) ─────────────────────────

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "medicine_taken": 0.45,
    "eating": 0.6,
    "drinking": 0.6,
}


def _get_threshold(subtype: str) -> float:
    """Return the effective detection threshold for *subtype*.

    If ``HEALTH_DETECTION_THRESHOLD`` env var is set and parseable as a float,
    that value is returned uniformly for all subtypes.  Otherwise the
    per-subtype default from ``_DEFAULT_THRESHOLDS`` is used (falling back to
    0.6 for unknown subtypes).

    Logs a WARNING when the env var is present but cannot be parsed.

    Requirements: 4.3, 6.1
    """
    env_val = os.getenv("HEALTH_DETECTION_THRESHOLD")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            log.warning(
                "Unparseable HEALTH_DETECTION_THRESHOLD=%r, using defaults", env_val
            )
    return _DEFAULT_THRESHOLDS.get(subtype, 0.6)


def _resolve_subtype(detected_objects: list[str]) -> str | None:
    """Map the first matching object label to a health subtype.

    Iterates *detected_objects* in order and returns the subtype for the first
    label that appears in ``HEALTH_SUBTYPE_MAP`` (case-insensitive).  Returns
    ``None`` if no label matches.

    Uses the canonical lowercase ``HEALTH_SUBTYPE_MAP`` imported from
    ``local_detector`` so that labels produced by the Local Detector (which are
    already lowercase) are matched correctly.

    Requirements: 2.2
    """
    for label in detected_objects:
        subtype = _LOCAL_HEALTH_SUBTYPE_MAP.get(label.lower())
        if subtype:
            return subtype
    return None


# ── Thread state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
_speak_lock = threading.Lock()
_recognizing = False
_pending_profile = False
_health_running = False

# ── YuNet + SFace initialization ─────────────────────────────────────────────

def _load_face_models():
    """Load YuNet detector and SFace recognizer using ONNXRuntime."""
    if not YUNET_MODEL.exists():
        log.error("YuNet model not found!")
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")
    if not SFACE_MODEL.exists():
        log.error("SFace model not found!")
        raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}")

    # Load YuNet detector using OpenCV DNN
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3,
    )

    # Load SFace using ONNXRuntime (workaround for OpenCV bug)
    sess = ort.InferenceSession(
        str(SFACE_MODEL),
        providers=['CPUExecutionProvider']
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    log.info("YuNet + SFace models loaded (SFace via ONNXRuntime)")
    return detector, sess, input_name, output_name


_detector, _sface_session, _sface_input_name, _sface_output_name = _load_face_models()
_detector_lock = threading.Lock()

# ── Warm up the Local Detector (Primary Pass) at module init ─────────────────
# Call get_detector() once here so the YOLO model is loaded before the main
# loop starts.  On failure, get_detector() logs CRITICAL and sets
# _health_detection_disabled = True inside local_detector — we mirror that
# flag here so detect_health_activity() can check it.
try:
    get_detector()
except Exception:
    # Failure already logged + _health_detection_disabled set inside local_detector.
    # Mirror the disabled state in this module's flag.
    _health_detection_disabled = True


def _get_face_feature_onnx(img_bgr: np.ndarray) -> np.ndarray | None:
    """Detect face with YuNet and get SFace feature using ONNXRuntime."""
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    # Detect faces with YuNet
    with _detector_lock:
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None

    # Get the highest-confidence face
    best = max(faces, key=lambda f: float(f[14]))
    
    # Align the face using OpenCV (only if recognizer was successfully created)
    aligned = _recognizer.alignCrop(img_bgr, best) if _recognizer is not None else None
    
    # If OpenCV alignment fails, manually crop
    if aligned is None:
        x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        # Add some margin
        margin = int(w * 0.1)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
        aligned = img_bgr[y1:y2, x1:x2]
        aligned = cv2.resize(aligned, (112, 112))

    # Preprocess for SFace: normalize to [-1, 1]
    aligned = aligned.astype(np.float32) / 255.0
    aligned = (aligned - 0.5) / 0.5
    aligned = aligned.transpose(2, 0, 1).flatten()
    aligned = aligned.reshape(1, 3, 112, 112)

    # Run SFace inference
    features = _sface_session.run([_sface_output_name], {_sface_input_name: aligned})[0]
    return features[0]


# ── Also keep OpenCV recognizer for alignment ────────────────────────────────

try:
    _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    _use_onnx_alignment = False
except Exception as e:
    log.warning(f"OpenCV SFace failed: {e}, using manual alignment")
    _recognizer = None
    _use_onnx_alignment = True


def _get_face_feature(img_bgr: np.ndarray) -> np.ndarray | None:
    """Detect the largest face and return SFace feature vector."""
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    with _detector_lock:
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None

    # Pick the highest-confidence face
    best = max(faces, key=lambda f: float(f[14]))

    # Align the face
    if _recognizer is not None:
        try:
            aligned = _recognizer.alignCrop(img_bgr, best)
            feature = _recognizer.feature(aligned)
            return feature
        except Exception as e:
            log.warning(f"OpenCV alignment failed: {e}")

    # Fallback: manual alignment
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    margin = int(w * 0.1)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
    aligned = img_bgr[y1:y2, x1:x2]
    aligned = cv2.resize(aligned, (112, 112))

    # Preprocess for SFace
    aligned = aligned.astype(np.float32) / 255.0
    aligned = (aligned - 0.5) / 0.5
    aligned = aligned.transpose(2, 0, 1).flatten()
    aligned = aligned.reshape(1, 3, 112, 112)

    # Run SFace via ONNXRuntime
    features = _sface_session.run([_sface_output_name], {_sface_input_name: aligned})[0]
    return features[0]


# ── Known-face loader ─────────────────────────────────────────────────────────

def load_known_faces() -> list[dict]:
    """Load known faces from local known_faces/*.jpg + *.json files."""
    known: list[dict] = []

    if not KNOWN_FACES_DIR.exists():
        log.warning("Known faces directory not found: %s", KNOWN_FACES_DIR)
        return known

    for jpg_file in sorted(KNOWN_FACES_DIR.glob("*.jpg")):
        name = jpg_file.stem
        json_file = jpg_file.with_suffix(".json")
        profile: dict = {}
        if json_file.exists():
            try:
                profile = json.loads(json_file.read_text())
            except Exception as e:
                log.warning("Failed to parse %s: %s", json_file.name, e)

        img = cv2.imread(str(jpg_file))
        if img is None:
            log.warning("Could not read %s", jpg_file)
            continue

        feature = _get_face_feature(img)
        if feature is None:
            log.warning("No face detected in %s", jpg_file)
            continue

        known.append({"name": name, "profile": profile, "feature": feature})
        log.info("Loaded face: %s", name)

    log.info("Loaded %d known faces", len(known))
    return known


# ── SFace matching ────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Flatten to 1D if needed
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


def match_with_sface(frame_bgr: np.ndarray, known_faces: list[dict]) -> dict | None:
    """Run YuNet + SFace on frame_bgr and match against known faces."""
    if not known_faces:
        return None

    query_feature = _get_face_feature(frame_bgr)
    if query_feature is None:
        log.debug("No face detected in frame")
        return None

    best_score = 0.0
    best_person = None

    for person in known_faces:
        score = _cosine_similarity(query_feature, person["feature"])
        log.debug("  %s -> cosine=%.3f", person["name"], score)
        if score > best_score:
            best_score = score
            best_person = person

    if best_score >= CONFIDENCE_THRESHOLD and best_person is not None:
        log.info("Matched: %s (cosine=%.3f)", best_person["name"], best_score)
        return best_person["profile"]

    log.debug("No confident match (best=%.3f)", best_score)
    return None


# ── Local event store ─────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def _append_event(event: dict) -> None:
    """Append one event as a JSON line to the local events.jsonl store."""
    _EVENT_LOG_DIR.mkdir(exist_ok=True)
    payload = {k: v for k, v in event.items() if k != "_id"}
    with _write_lock:
        with EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def read_events(n: int = 50) -> list[dict]:
    """Read the most recent n events from the local JSONL store."""
    if not EVENTS_JSONL.exists():
        return []
    lines = EVENTS_JSONL.read_text(encoding="utf-8").splitlines()
    events = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(events) >= n:
            break
    return events


# ── Frame quality check ───────────────────────────────────────────────────────

def is_frame_usable(frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 30:
        log.debug("Frame too dark")
        return False
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 2:
        log.debug("Frame too blurry")
        return False
    return True


# ── Voice script ──────────────────────────────────────────────────────────────

def build_voice_script(profile: dict) -> str:
    patient_name = os.getenv("PATIENT_NAME", "there")
    name = profile.get("name", "someone")
    relationship = profile.get("relationship", "someone you know")
    background = profile.get("background", "")
    last_convo = profile.get("last_conversation", "")

    script = f"{patient_name}, your {relationship} {name} is here."
    if background:
        script += f" {background}."
    if last_convo:
        script += f" Last time you spoke, {last_convo}."
    return script


# ── Event saver ───────────────────────────────────────────────────────────────

_event_log_counter = 0


def save_event_json(event: dict) -> None:
    global _event_log_counter
    _EVENT_LOG_DIR.mkdir(exist_ok=True)
    _event_log_counter += 1
    payload = {k: v for k, v in event.items() if k not in ("image_b64", "_id")}
    path = _EVENT_LOG_DIR / f"event_{_event_log_counter:04d}.json"
    path.write_text(json.dumps(payload, indent=2))
    log.debug("Event JSON saved -> %s", path)
    try:
        http_requests.post("http://localhost:8502/ingest", json=payload, timeout=2)
    except Exception:
        pass


# ── Event logger ──────────────────────────────────────────────────────────────

def log_event(profile: dict, frame_bgr: np.ndarray | None = None) -> str:
    """Append an identity event to the local JSONL store and POST to Brain."""
    voice_script = build_voice_script(profile)

    image_b64 = ""
    if frame_bgr is not None:
        _, buf = cv2.imencode(".jpg", frame_bgr)
        image_b64 = base64.b64encode(buf).decode("utf-8")

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": 1.0,
        "image_b64": image_b64,
        "metadata": {"person_profile": profile},
        "source": "vision_engine_v1",
    }
    _append_event(event)
    save_event_json(event)
    log.info("Event logged for: %s", profile.get("name"))

    # POST to Brain for Gemini verification, TTS, and MongoDB write
    brain_posted = False
    try:
        brain_url = f"http://{os.getenv('BRAIN_HOST', 'localhost')}:{os.getenv('BRAIN_PORT', '8000')}/event"
        resp = http_requests.post(brain_url, json=event, timeout=30)
        if resp.status_code < 300:
            log.info("Identity event sent to Brain: %s -> HTTP %d", profile.get("name"), resp.status_code)
            brain_posted = True
        else:
            log.error("Brain returned non-2xx for identity event: HTTP %d", resp.status_code)
    except Exception as e:
        log.warning("Failed to POST identity event to Brain (will speak locally): %s", e)

    # Return empty string if Brain handled it (Brain does TTS),
    # return voice_script if Brain unreachable (caller will speak locally)
    return "" if brain_posted else voice_script


# ── Audio ─────────────────────────────────────────────────────────────────────

def _play_mp3(tmp_path: str):
    import platform
    played = False
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        played = True
    except Exception:
        pass
    if not played and platform.system() == "Darwin":
        subprocess.run(["afplay", tmp_path], check=False)


def speak(voice_script: str):
    if not _speak_lock.acquire(blocking=False):
        log.info("[Voice skipped - already speaking]")
        return
    tmp_path = None
    try:
        import asyncio, edge_tts

        async def _synth():
            chunks: list[bytes] = []
            async for chunk in edge_tts.Communicate(voice_script, voice="en-US-AriaNeural").stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        audio_bytes = asyncio.run(_synth())
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        _play_mp3(tmp_path)
    except Exception as e:
        log.warning("edge-tts failed: %s", e)
        log.info("[Voice] %s", voice_script)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        _speak_lock.release()


# ── Hybrid health detection helpers ──────────────────────────────────────────

def _run_secondary_pass(frame_bgr: np.ndarray, subtype: str):
    """
    Run the Secondary Pass (Gemini 1.5 Flash) on *frame_bgr* for *subtype*.

    JPEG-encodes the frame (max 640 px on the longest side), calls
    ``call_gemini_health``, logs INFO with prompt/response/score, logs DEBUG
    with wall-clock latency, and returns a :class:`ConfidenceResult` or
    ``None`` on timeout/failure.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 5.2, 5.6
    """
    from services.vision.gemini_health import call_gemini_health, build_health_prompt, ConfidenceResult  # noqa: F401

    # Resize to max 640 px on the longest side
    h, w = frame_bgr.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    prompt = build_health_prompt(subtype)
    log.info(
        "Secondary pass invoked — subtype=%r, prompt=%r",
        subtype,
        prompt,
    )

    api_key = os.getenv("GEMINI_API_KEY", "")
    t_start = time.monotonic()
    result = call_gemini_health(frame_b64=frame_b64, subtype=subtype, api_key=api_key)
    latency_ms = (time.monotonic() - t_start) * 1000.0

    log.debug(
        "Secondary pass latency — subtype=%r, latency=%.1f ms",
        subtype,
        latency_ms,
    )

    if result is not None:
        log.info(
            "Secondary pass result — subtype=%r, raw_text=%r, score=%.4f",
            subtype,
            result.raw_text,
            result.score,
        )

    return result


def _dispatch_health_event(
    frame_bgr: np.ndarray,
    subtype: str,
    confidence_result,
    detection_result,
) -> None:
    """
    Construct a Health_Event and POST it to the Brain.

    Builds a :class:`shared.contract.Event` with all required fields, POSTs
    it to the Brain endpoint within 5 seconds, logs INFO on success, logs
    ERROR on non-2xx or connection error (no retry), and calls
    ``save_event_json`` after a successful POST.

    Requirements: 3.1, 3.2, 3.3, 5.5, 7.1, 7.2
    """
    from shared.contract import Event

    # Resize to max 640 px on the longest side for the event image
    h, w = frame_bgr.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame_bgr)
    image_b64 = base64.b64encode(buf).decode("utf-8")

    detected_item = (
        detection_result.detected_objects[0]
        if detection_result.detected_objects
        else subtype
    )

    event = Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        patient_id=os.getenv("PATIENT_ID", "unknown"),
        type="health",
        subtype=subtype,
        confidence=confidence_result.score,
        image_b64=image_b64,
        metadata={"detected_item": detected_item},
        source="vision_engine_v1",
    )

    brain_url = (
        f"http://{os.getenv('BRAIN_HOST', 'localhost')}:"
        f"{os.getenv('BRAIN_PORT', '8000')}/event"
    )

    try:
        resp = http_requests.post(brain_url, json=event.model_dump(), timeout=5)
        if resp.status_code // 100 == 2:
            log.info(
                "Health event dispatched — subtype=%r, score=%.4f, http_status=%d",
                subtype,
                confidence_result.score,
                resp.status_code,
            )
            save_event_json(event.model_dump())
        else:
            log.error(
                "Brain POST returned non-2xx — subtype=%r, score=%.4f, http_status=%d",
                subtype,
                confidence_result.score,
                resp.status_code,
            )
    except Exception as exc:
        log.error(
            "Brain POST connection error — subtype=%r, score=%.4f, error=%s",
            subtype,
            confidence_result.score,
            exc,
        )


# ── Health activity detector (Gemini - optional) ──────────────────────────────

def detect_health_activity(frame_bgr: np.ndarray) -> None:
    """Run the two-pass hybrid health detection pipeline on *frame_bgr*.

    Pass 1 — Local YOLO detector (Primary Pass): fast, no network I/O.
    Pass 2 — Gemini 1.5 Flash (Secondary Pass): targeted, confidence-scored.

    The function signature is unchanged from the previous implementation so
    that the caller (``_health_worker``) requires no modification.

    Requirements: 1.1, 1.3, 1.4, 1.7, 3.4, 3.5, 5.1, 5.3, 5.4, 5.5, 6.3,
                  7.4, 7.5
    """
    # Guard: health detection disabled at startup (model missing / load error)
    if _health_detection_disabled:
        return

    # ── 1. Frame quality check ────────────────────────────────────────────────
    quality_ok = is_frame_usable(frame_bgr)

    # ── 2. Primary Pass ───────────────────────────────────────────────────────
    try:
        result = get_detector().run(frame_bgr)
        log.debug(
            "Primary pass: flagged=%s, objects=%s, scores=%s",
            result.flagged,
            result.detected_objects,
            result.confidence_scores,
        )
    except Exception as exc:
        log.warning(
            "Local detector failed: %s — falling back to Gemini (synthetic flagged result)",
            exc,
        )
        result = DetectionResult(
            flagged=True,
            detected_objects=[],
            confidence_scores={},
            medicine_flagged=False,
        )

    # ── 3. Medicine safety override / quality gate ────────────────────────────
    # If the frame is unusable AND no medicine object was flagged, skip.
    # (If medicine WAS flagged, we bypass the quality gate — Requirement 6.3.)
    if not quality_ok and not result.medicine_flagged:
        log.debug("Frame quality check failed and no medicine flag — skipping health detection")
        return

    # ── 4. Primary Pass flag check ────────────────────────────────────────────
    if not result.flagged:
        log.debug(
            "Primary pass: not flagged — skipping Secondary Pass. Objects: %s",
            result.detected_objects,
        )
        return

    # ── 5. Resolve subtype from detected objects ──────────────────────────────
    subtype = _resolve_subtype(result.detected_objects)
    if subtype is None:
        return

    # ── 6. Secondary Pass ─────────────────────────────────────────────────────
    confidence_result = _run_secondary_pass(frame_bgr, subtype)
    if confidence_result is None:
        return

    # ── 7. Threshold gate ─────────────────────────────────────────────────────
    threshold = _get_threshold(subtype)
    if confidence_result.score < threshold:
        log.info(
            "Health event suppressed (below threshold) — subtype=%r, score=%.4f, threshold=%.4f",
            subtype,
            confidence_result.score,
            threshold,
        )
        return

    # ── 8. Cooldown gate ──────────────────────────────────────────────────────
    now_ts = datetime.now(timezone.utc).timestamp()
    last_ts = _last_health_event.get(subtype, 0.0)
    elapsed = now_ts - last_ts
    if elapsed < HEALTH_COOLDOWN_SECONDS:
        remaining = HEALTH_COOLDOWN_SECONDS - elapsed
        log.debug(
            "Health event suppressed (cooldown) — subtype=%r, remaining=%.0fs",
            subtype,
            remaining,
        )
        return
    _last_health_event[subtype] = now_ts

    # ── 9. Dispatch ───────────────────────────────────────────────────────────
    _dispatch_health_event(frame_bgr, subtype, confidence_result, result)


# ── Frame generator ───────────────────────────────────────────────────────────

_NET_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://")


# Global video capture for cleanup
_video_capture = None
_video_capture_lock = threading.Lock()


def _cleanup_on_exit():
    """Cleanup function to release webcam on exit."""
    global _video_capture
    if _video_capture is not None:
        with _video_capture_lock:
            if _video_capture is not None:
                _video_capture.release()
                _video_capture = None
        cv2.destroyAllWindows()
        print("Webcam released")


import atexit
atexit.register(_cleanup_on_exit)


def _yield_frames(video_source):
    """Yield BGR numpy frames with automatic reconnect on failure."""
    global _video_capture
    
    use_av = isinstance(video_source, str) and any(
        video_source.startswith(p) for p in _NET_PREFIXES
    )

    if use_av:
        while True:
            try:
                log.info("Connecting to stream: %s", video_source)
                container = av.open(
                    video_source,
                    options={"fflags": "nobuffer", "flags": "low_delay"},
                )
                log.info("Stream connected.")
                for av_frame in container.decode(video=0):
                    yield av_frame.to_ndarray(format="bgr24")
                log.info("Stream ended - reconnecting in 5 s...")
            except Exception as e:
                log.warning("Stream error: %s - reconnecting in 5 s...", e)
            time.sleep(5)
    else:
        global _video_capture
        with _video_capture_lock:
            _video_capture = cv2.VideoCapture(video_source)
        if not _video_capture.isOpened():
            log.error("Could not open video source: %s", video_source)
            return
        while True:
            with _video_capture_lock:
                if _video_capture is None:
                    ret, frame = False, None
                else:
                    ret, frame = _video_capture.read()
            if ret and frame is not None:
                yield frame
            else:
                log.warning("Frame read failed - reconnecting in 3 s...")
                with _video_capture_lock:
                    if _video_capture is not None:
                        _video_capture.release()
                        _video_capture = None
                time.sleep(3)
                with _video_capture_lock:
                    _video_capture = cv2.VideoCapture(video_source)
                if not _video_capture.isOpened():
                    log.error("Failed to reopen: %s", video_source)
                    return


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(video_source=None):
    try:
        known_faces = load_known_faces()

        if not known_faces:
            log.warning("No known faces loaded - add <name>.jpg + <name>.json to known_faces/")

        if video_source is None:
            video_source = os.getenv("RTMP_STREAM_URL", "rtmp://localhost/live/stream")

        log.info("Opening video source: %s", video_source)
        log.info("Recognition engine: YuNet + SFace (ONNXRuntime)")
        log.info("Running. Press Q to quit.")

        frame_count = 0
        last_label = ("Scanning...", (200, 200, 200))
        current_match = None
        last_matched_name = None
        last_match_time = 0.0
        last_recognition_time = 0.0
        last_health_time = 0.0

        global _recognizing, _pending_profile, _health_running

        def _recognition_worker(frame_copy):
            global _recognizing, _pending_profile
            profile = match_with_sface(frame_copy, known_faces)
            with _lock:
                _pending_profile = profile
                _recognizing = False

        def _health_worker(frame_copy):
            global _health_running
            detect_health_activity(frame_copy)
            with _lock:
                _health_running = False

        # Keep only the latest frame
        _frame_q: queue.Queue = queue.Queue(maxsize=1)

        def _reader():
            for f in _yield_frames(video_source):
                if _frame_q.full():
                    try:
                        _frame_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    _frame_q.put_nowait(f)
                except queue.Full:
                    pass

        threading.Thread(target=_reader, daemon=True).start()

        while True:
            try:
                frame = _frame_q.get(timeout=5)
            except queue.Empty:
                log.warning("No frames for 5 s - waiting...")
                continue

            frame_count += 1
            now = time.time()

            # Reset after cooldown
            if current_match is not None and (now - last_match_time) >= COOLDOWN_SECONDS:
                log.info("Cooldown expired - ready to scan again")
                current_match = None
                last_matched_name = None
                last_label = ("Scanning...", (200, 200, 200))

            # Collect result from background recognition worker
            with _lock:
                has_result = _pending_profile is not False
                profile = _pending_profile
                _pending_profile = False

            if has_result and profile is not None and current_match is None:
                name = profile.get("name", "Unknown")
                current_match = name
                last_match_time = now
                if name != last_matched_name:
                    last_matched_name = name
                    frame_snapshot = frame.copy()
                    def _handle_match(p=profile, f=frame_snapshot):
                        voice_script = log_event(p, f)
                        # speak() locally only if Brain is unreachable
                        # (log_event returns "" when Brain handled it)
                        if voice_script:
                            speak(voice_script)
                    threading.Thread(target=_handle_match, daemon=True).start()
                    last_label = (f"Matched: {name}", (0, 255, 0))
                else:
                    last_label = (f"Matched: {name}", (0, 200, 100))
            elif has_result and profile is None and current_match is None:
                last_label = ("Scanning...", (200, 200, 200))

            # Fire recognition on a timer
            detection_paused = DETECTION_PAUSED_FLAG.exists()
            if detection_paused:
                last_label = ("Manual mode - detection paused", (200, 140, 0))
            else:
                with _lock:
                    should_recognise = (
                        current_match is None
                        and not _recognizing
                        and (now - last_match_time) >= COOLDOWN_SECONDS
                        and (now - last_recognition_time) >= RECOGNITION_INTERVAL
                        and known_faces
                        and is_frame_usable(frame)
                    )
                    if should_recognise:
                        _recognizing = True
                        last_recognition_time = now

                if should_recognise:
                    threading.Thread(
                        target=_recognition_worker,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()

            # Optional health scan
            if ENABLE_HEALTH_DETECTION:
                with _lock:
                    should_health = (
                        not _health_running
                        and (now - last_health_time) >= HEALTH_CHECK_INTERVAL_SECONDS
                    )
                    if should_health:
                        _health_running = True
                        last_health_time = now
                if should_health:
                    threading.Thread(target=_health_worker, args=(frame.copy(),), daemon=True).start()

            # Display
            cv2.putText(
                frame, last_label[0], (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_label[1], 2,
            )
            cv2.imshow("AuraGuard - Face Recognition (YuNet+SFace)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        log.info("Stopped.")
    finally:
        _cleanup_on_exit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AuraGuard Vision Engine")
    parser.add_argument(
        "--webcam", action="store_true",
        help="Use laptop webcam (camera index 0) instead of RTMP stream"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam camera index to use (default: 0)"
    )
    args = parser.parse_args()

    if args.webcam:
        log.info("Webcam mode: using camera index %d", args.camera)
        run(video_source=args.camera)
    else:
        run()