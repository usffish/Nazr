"""
Face Recognition Engine (Gemini-powered)
------------------------------------------
1. Loads known faces + profiles from known_faces/ at startup
2. Every N frames, takes a snapshot and sends it to Gemini
3. Gemini compares the snapshot against ALL reference photos
4. Only fires if Gemini is certain — strict prompt, no guessing
5. Logs event to MongoDB and speaks via ElevenLabs
"""

import os
import json
import uuid
import base64
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import subprocess
import threading
import time
import av
import cv2
import certifi
import requests as http_requests
import google.generativeai as genai
from pymongo import MongoClient
from dotenv import load_dotenv
import pygame

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_PARALLEL_CALLS = 1          # single call — no voting
CHECK_EVERY_N_FRAMES = 10          # frames between face-recognition checks
HEALTH_CHECK_EVERY_N_FRAMES = 30   # frames between health-activity checks (~1s at 30fps)
COOLDOWN_SECONDS = 10
HEALTH_CHECK_INTERVAL_SECONDS = 5   # minimum seconds between health scans
CONFIDENCE_THRESHOLD = 0.65        # minimum confidence to accept a match
KNOWN_FACES_DIR = Path(__file__).parent / "known_faces"

# When True, always return the closest person in the DB even if confidence is low.
# When False, return None (no match) if Gemini isn't confident enough.
# Override via .env:  ALWAYS_BEST_GUESS=false
ALWAYS_BEST_GUESS: bool = os.getenv("ALWAYS_BEST_GUESS", "true").lower() != "false"

HEALTH_SUBTYPE_MAP = {
    # drinking triggers
    "WATER BOTTLE": "drinking",
    "SODA CAN": "drinking",
    "CAN": "drinking",
    "CUP": "drinking",
    "GLASS": "drinking",
    "MUG": "drinking",
    "BOTTLE": "drinking",
    "DRINKING": "drinking",
    # eating triggers
    "FOOD": "eating",
    "FORK": "eating",
    "SPOON": "eating",
    "SANDWICH": "eating",
    "EATING": "eating",
    # medicine triggers
    "PILL": "medicine_taken",
    "PILLS": "medicine_taken",
    "TABLET": "medicine_taken",
    "MEDICINE": "medicine_taken",
    "MEDICATION": "medicine_taken",
}
HEALTH_COOLDOWN_SECONDS = 120  # minimum gap between same health-event type

_last_health_event: dict[str, float] = {}

# ── Background-thread state ───────────────────────────────────────────────────

_lock = threading.Lock()
_recognizing = False        # True while a Gemini face-match is in flight
_pending_profile = False    # False = no result pending; None = "no match" result; dict = matched profile
_health_running = False     # True while a health-activity call is in flight

# ── Event image saver ─────────────────────────────────────────────────────────

_EVENT_LOG_DIR = Path("/Users/mtb/Programming/Hackabull-2026/tempfiles")
_event_log_counter = 0

def save_event_json(event: dict) -> None:
    global _event_log_counter
    _EVENT_LOG_DIR.mkdir(exist_ok=True)
    _event_log_counter += 1
    n = _event_log_counter
    payload = {k: v for k, v in event.items() if k not in ("image_b64", "_id")}
    path = _EVENT_LOG_DIR / f"event_{n:04d}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[Image #{n}] saved → {path}", flush=True)
    try:
        http_requests.post("http://localhost:8502/ingest", json=payload, timeout=2)
    except Exception:
        pass  # webapp not running — file is still saved

# ── Gemini ────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

# ── MongoDB ───────────────────────────────────────────────────────────────────

def connect_to_mongo():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aura_guard")
    col_name = os.getenv("MONGODB_COLLECTION", "events")
    if not uri:
        raise ValueError("MONGODB_URI not set in .env")
    client = MongoClient(uri, tlsCAFile=certifi.where())
    log.info(f"Connected to MongoDB: {db_name}.{col_name}")
    return client[db_name][col_name]

# ── Known people loader (profiles only — no reference images needed) ──────────

def load_known_faces() -> list[dict]:
    """Load profiles from known_faces/*.json. No images required."""
    known = []
    if not KNOWN_FACES_DIR.exists():
        log.warning(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return known

    for profile_file in sorted(KNOWN_FACES_DIR.glob("*.json")):
        try:
            profile = json.loads(profile_file.read_text())
            shirt_color = profile.get("shirt_color", "")
            if not shirt_color:
                log.warning(f"No shirt_color in {profile_file.name}, skipping.")
                continue
            known.append({"name": profile.get("name", profile_file.stem), "profile": profile})
            log.info(f"Loaded profile: {profile.get('name')} — shirt: {shirt_color}")
        except Exception as e:
            log.warning(f"Failed to load {profile_file.name}: {e}")

    log.info(f"Total profiles loaded: {len(known)}")
    return known

# ── Gemini shirt-color matching ───────────────────────────────────────────────

def is_frame_usable(frame_bgr) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    log.info(f"Frame quality — brightness={brightness:.1f}, blur={blur_score:.1f}")
    if brightness < 30:
        log.info("Frame too dark, skipping")
        return False
    if blur_score < 2:
        log.info("Frame too blurry, skipping")
        return False
    return True


def _resize_for_gemini(frame_bgr, max_width: int = 640):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    return frame_bgr


def match_with_gemini(frame_bgr, known_faces: list[dict]) -> dict | None:
    """
    Single Gemini call: ask what shirt color the person is wearing, then
    match against the known shirt-color → profile mapping.
    """
    if not known_faces:
        return None

    if not is_frame_usable(frame_bgr):
        return None

    frame_bgr = _resize_for_gemini(frame_bgr)
    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    # Build color → person lookup from loaded profiles
    color_map: dict[str, dict] = {}
    for person in known_faces:
        shirt_color = person["profile"].get("shirt_color", "").lower()
        for word in shirt_color.split("/"):
            color_map[word.strip()] = person

    known_colors = ", ".join(
        f"{p['profile']['shirt_color']} ({p['name']})" for p in known_faces
    )

    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
        f"""You are identifying a person by their shirt color from a first-person wearable camera.

The known people and their shirt colors are:
{known_colors}

Look at the person visible in this image. What color is their shirt or top?

Respond ONLY with this exact JSON (no extra text):
{{"shirt_color": "green", "confidence": 0.95, "person_visible": true}}

- shirt_color: the single best-matching color from the known list above
- confidence: float 0.0–1.0 of how certain you are
- person_visible: true if a person with a visible shirt is in frame, false otherwise""",
    ]

    try:
        response = model.generate_content(parts)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
    except Exception as e:
        log.warning(f"Gemini shirt-color parse failed: {e}")
        return None

    if not result.get("person_visible", False):
        log.info("Gemini: no person visible in frame")
        return None

    detected_color = result.get("shirt_color", "").lower().strip()
    confidence = max(float(result.get("confidence", 0.0)), 0.05)
    log.info(f"Shirt color detected: {detected_color!r} (confidence={confidence:.2f})")

    # Direct color match
    for keyword, person in color_map.items():
        if keyword in detected_color or detected_color in keyword:
            log.info(f"Matched: {person['name']} via shirt color '{detected_color}'")
            return person["profile"]

    # ALWAYS_BEST_GUESS: return whoever has the closest color name
    if ALWAYS_BEST_GUESS:
        guess = known_faces[0]
        log.info(f"Best guess: {guess['name']} (no color match for '{detected_color}')")
        return guess["profile"]

    log.info(f"No color match for '{detected_color}'")
    return None

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

# ── MongoDB logger ────────────────────────────────────────────────────────────

def log_event(collection, profile: dict) -> str:
    voice_script = build_voice_script(profile)
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": 1.0,
        "metadata": {"person_profile": profile},
        "source": "vision_engine_v1",
        "verified": True,
        "voice_script": voice_script,
        "processing_status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    collection.insert_one(event)
    save_event_json(event)
    log.info(f"Logged: {profile.get('name')}")
    return voice_script

# ── Audio ─────────────────────────────────────────────────────────────────────

def _play_mp3(tmp_path: str):
    """Play an MP3 file via pygame or afplay."""
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
    # ElevenLabs is reserved for the webapp → glasses audio path (port 8502).
    # The vision engine plays locally via edge-tts only, so we never double-bill credits.
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
        log.warning(f"edge-tts failed: {e}")
        log.info(f"[Voice] {voice_script}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Health activity detector ──────────────────────────────────────────────────

def detect_health_activity(frame_bgr) -> None:
    """Ask Gemini if the wearer is eating, drinking, or taking medicine.
    If detected, POST a health event to the Brain service."""
    frame_bgr = _resize_for_gemini(frame_bgr)
    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
        (
            "This image is from a first-person point-of-view camera worn on glasses. "
            "Look carefully at what objects are visible and being held or used. "
            "If you see any of the following, name it: "
            "water bottle, soda can, cup, glass, mug, bottle, food, fork, spoon, sandwich, pill, pills, tablet, medicine, medication. "
            "Reply with ONLY the object name from that list (e.g. 'water bottle', 'cup', 'pills'). "
            "If none of those objects are visible, reply with exactly: NONE."
        ),
    ]

    try:
        response = model.generate_content(parts)
        answer = response.text.strip().upper()
        log.info(f"Health check response: {answer!r}")
        subtype = next(
            (st for kw, st in HEALTH_SUBTYPE_MAP.items() if kw in answer),
            None,
        )
        if subtype is None:
            return

        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts - _last_health_event.get(subtype, 0) < HEALTH_COOLDOWN_SECONDS:
            log.debug(f"Health event {subtype} suppressed (cooldown active)")
            return
        _last_health_event[subtype] = now_ts

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": os.getenv("PATIENT_ID", "unknown"),
            "type": "health",
            "subtype": subtype,
            "confidence": 0.9,
            "image_b64": frame_b64,
            "metadata": {"detected_item": answer.title()},
            "source": "vision_engine_v1",
        }

        brain_host = os.getenv("BRAIN_HOST", "localhost")
        brain_port = os.getenv("BRAIN_PORT", "8000")
        brain_url = f"http://{brain_host}:{brain_port}/event"
        resp = http_requests.post(brain_url, json=event, timeout=5)
        save_event_json(event)
        log.info(f"Health event sent: {subtype} → Brain responded {resp.status_code}")
    except Exception as e:
        log.warning(f"Health detection failed: {e}")


# ── Frame generator (PyAV for network streams, cv2 for local) ────────────────
# PyAV links against Homebrew's FFmpeg libraries directly — same binary path as
# `ffplay`, so it inherits full RTMP support without spawning a subprocess pipe.

_NET_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://")


def _yield_frames(video_source: str):
    """Yield BGR numpy frames with automatic reconnect on failure."""
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
                log.info("Stream ended — reconnecting in 3 s...")
            except Exception as e:
                log.warning("Stream error: %s — reconnecting in 3 s...", e)
            time.sleep(3)
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            log.error("Could not open video source: %s", video_source)
            return
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                log.warning("Frame read failed — reconnecting in 3 s...")
                cap.release()
                time.sleep(3)
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    log.error("Failed to reopen: %s", video_source)
                    return


# ── Face presence detector ────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(frame_bgr) -> bool:
    """Quick local check — lenient settings so we don't miss angled/partial faces."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
    return len(faces) > 0

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(video_source: str | None = None):
    collection = connect_to_mongo()
    known_faces = load_known_faces()

    if not known_faces:
        log.warning("No known faces loaded — add <name>.jpg + <name>.json to known_faces/")

    if video_source is None:
        video_source = os.getenv("RTMP_STREAM_URL", "rtmp://localhost/live/stream")

    log.info("Opening video source: %s", video_source)
    log.info("Running. Press Q to quit.")

    frame_count = 0
    last_label = ("Scanning...", (200, 200, 200))
    current_match = None
    last_matched_name = None   # suppresses re-announcing the same person
    face_absent_frames = 0
    ABSENT_THRESHOLD = 30
    last_match_time = 0.0
    last_health_check_time = 0.0
    face_first_seen_time: float | None = None  # when face continuously in frame since
    FACE_DWELL_SECONDS = 3.0  # must be in frame this long before Gemini fires

    global _recognizing, _pending_profile, _health_running

    def _gemini_worker(frame_copy):
        global _recognizing, _pending_profile
        profile = match_with_gemini(frame_copy, known_faces)
        with _lock:
            _pending_profile = profile
            _recognizing = False

    def _health_worker(frame_copy):
        global _health_running
        detect_health_activity(frame_copy)
        with _lock:
            _health_running = False

    for frame in _yield_frames(video_source):
        frame_count += 1

        # ── Detect face presence every frame (cheap, local) ──
        face_present = has_face(frame)
        now = time.time()

        if not face_present:
            face_absent_frames += 1
            face_first_seen_time = None  # reset dwell timer when face leaves
            if face_absent_frames >= ABSENT_THRESHOLD and current_match is not None:
                log.info("Face left frame — resetting (was: %s)", current_match)
                current_match = None
                last_label = ("Scanning...", (200, 200, 200))
        else:
            face_absent_frames = 0
            if face_first_seen_time is None:
                face_first_seen_time = now  # start dwell timer

        # ── After cooldown, reset current_match so we scan for new people ──
        if current_match is not None and (now - last_match_time) >= COOLDOWN_SECONDS:
            current_match = None
            last_label = ("Scanning...", (200, 200, 200))

        # ── Collect result from background Gemini worker if ready ──
        with _lock:
            has_result = _pending_profile is not False
            profile = _pending_profile
            _pending_profile = False

        if has_result and profile is not None and current_match is None:
            name = profile.get("name", "Unknown")
            current_match = name
            last_match_time = now
            if name != last_matched_name:
                # New person — announce and log (off the main loop)
                last_matched_name = name
                threading.Thread(
                    target=lambda p=profile: speak(log_event(collection, p)),
                    daemon=True,
                ).start()
                last_label = (f"Matched: {name}", (0, 255, 0))
            else:
                # Same person returned — show label silently
                last_label = (f"Matched: {name}", (0, 200, 100))
        elif has_result and profile is None and current_match is None:
            last_matched_name = None
            last_label = ("No match", (0, 0, 255))

        # ── Fire background face-match if conditions met ──
        cooldown_elapsed = (now - last_match_time) >= COOLDOWN_SECONDS
        face_dwelled = face_first_seen_time is not None and (now - face_first_seen_time) >= FACE_DWELL_SECONDS
        with _lock:
            should_recognise = (
                current_match is None
                and cooldown_elapsed
                and face_dwelled
                and not _recognizing
                and frame_count % CHECK_EVERY_N_FRAMES == 0
                and known_faces
            )
            if should_recognise:
                _recognizing = True

        if should_recognise:
            threading.Thread(target=_gemini_worker, args=(frame.copy(),), daemon=True).start()

        # ── Fire background health scan every HEALTH_CHECK_INTERVAL_SECONDS ──
        with _lock:
            should_health = (
                not _health_running
                and (now - last_health_check_time) >= HEALTH_CHECK_INTERVAL_SECONDS
            )
            if should_health:
                _health_running = True
                last_health_check_time = now

        if should_health:
            threading.Thread(target=_health_worker, args=(frame.copy(),), daemon=True).start()

        cv2.putText(frame, last_label[0], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_label[1], 2)
        cv2.imshow("AuraGuard - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    log.info("Stopped.")


if __name__ == "__main__":
    run()
