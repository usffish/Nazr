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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import subprocess
import threading
import time
import av
import numpy as np
import cv2
import certifi
import requests as http_requests
import google.generativeai as genai
from pymongo import MongoClient
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
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
model = genai.GenerativeModel("gemini-3.1-pro-preview")

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

# ── Known faces loader ────────────────────────────────────────────────────────

def load_known_faces() -> list[dict]:
    """
    Load reference images and profiles.
    Reads the raw image bytes — no face detection needed at load time.
    Returns list of { name, image_b64, mime_type, profile }
    """
    known = []

    if not KNOWN_FACES_DIR.exists():
        log.warning(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return known

    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    for img_file in sorted(KNOWN_FACES_DIR.iterdir()):
        if img_file.suffix.lower() not in mime_map:
            continue
        profile_file = img_file.with_suffix(".json")
        if not profile_file.exists():
            log.warning(f"No profile JSON for {img_file.name}, skipping.")
            continue

        with open(img_file, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(profile_file) as f:
            profile = json.load(f)

        known.append({
            "name": profile.get("name", img_file.stem),
            "image_b64": image_b64,
            "mime_type": mime_map[img_file.suffix.lower()],
            "profile": profile,
        })
        log.info(f"Loaded reference: {profile.get('name', img_file.stem)}")

    log.info(f"Total known faces loaded: {len(known)}")
    return known

# ── Gemini matching ───────────────────────────────────────────────────────────

def is_frame_usable(frame_bgr) -> bool:
    """Reject frames that are too dark or too blurry."""
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


def _parse_gemini_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _verify_one(frame_b64: str, person: dict) -> float:
    """
    Ask Gemini: is the live frame the same person as this ONE reference photo?
    Returns a confidence float 0.0–1.0.
    """
    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
        {"inline_data": {"mime_type": person["mime_type"], "data": person["image_b64"]}},
        f"""You are a strict face verification system.

Image 1 is a live camera frame.
Image 2 is a reference photo of {person['name']}.

Task: Determine whether the person in Image 1 is the SAME individual as in Image 2.

Focus only on permanent facial features: eye spacing, nose shape, jawline, face geometry, brow ridge.
Ignore lighting, angle, expression, glasses, or hair differences.

Be very conservative — only say yes if you are highly confident.

Respond ONLY with this exact JSON (no extra text):
{{"same_person": true, "confidence": 0.95}}
or
{{"same_person": false, "confidence": 0.10}}

confidence must be a float between 0.0 and 1.0 representing how certain you are.""",
    ]

    response = model.generate_content(parts)
    result = _parse_gemini_json(response.text.strip())
    confidence = float(result.get("confidence", 0.0))
    same = bool(result.get("same_person", False))
    log.info(f"  [{person['name']}] same={same}, confidence={confidence:.2f}")
    return confidence if same else 0.0


def _resize_for_gemini(frame_bgr, max_width: int = 640):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    return frame_bgr


def match_with_gemini(frame_bgr, known_faces: list[dict]) -> dict | None:
    """
    For each known person, fire GEMINI_PARALLEL_CALLS verification calls in
    parallel and tally the votes. The person with the most confident majority
    wins — only accepted if confidence >= CONFIDENCE_THRESHOLD.
    """
    if not known_faces:
        return None

    if not is_frame_usable(frame_bgr):
        log.debug("Frame rejected (quality check failed)")
        return None

    frame_bgr = _resize_for_gemini(frame_bgr)
    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    # Build one task per (person, vote_index)
    tasks = [(person, i) for person in known_faces for i in range(GEMINI_PARALLEL_CALLS)]

    # votes[name] = list of confidence scores where same_person=True
    votes: dict[str, list[float]] = {p["name"]: [] for p in known_faces}

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(_verify_one, frame_b64, person): person for person, _ in tasks}
        for future in as_completed(futures):
            person = futures[future]
            try:
                confidence = future.result()
                if confidence > 0:
                    votes[person["name"]].append(confidence)
            except Exception as e:
                log.warning(f"Vote failed for {person['name']}: {e}")

    # Pick the person with the most positive votes, break ties by avg confidence
    best_person = None
    best_score = (0, 0.0)  # (vote_count, avg_confidence)

    for person in known_faces:
        name = person["name"]
        positive_votes = votes[name]
        count = len(positive_votes)
        avg_conf = sum(positive_votes) / count if count else 0.0
        log.info(f"  [{name}] votes={count}/{GEMINI_PARALLEL_CALLS}, avg_confidence={avg_conf:.2f}")
        if (count, avg_conf) > best_score:
            best_score = (count, avg_conf)
            best_person = person

    majority = GEMINI_PARALLEL_CALLS // 2 + 1  # need more than half
    if best_person and best_score[0] >= majority and best_score[1] >= CONFIDENCE_THRESHOLD:
        log.info(f"Matched: {best_person['name']} ({best_score[0]}/{GEMINI_PARALLEL_CALLS} votes, avg={best_score[1]:.2f})")
        return best_person["profile"]

    log.info(f"No confident majority match (best: {best_person['name'] if best_person else 'none'}, votes={best_score[0]}, avg={best_score[1]:.2f})")
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

def speak(voice_script: str):
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

    if not api_key or api_key == "your_elevenlabs_api_key_here":
        log.info(f"[Voice] {voice_script}")
        return

    tmp_path = None
    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=voice_script,
            model_id="eleven_flash_v2_5",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio:
                f.write(chunk)
            tmp_path = f.name

        # Try pygame first, fall back to afplay on macOS
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

        if not played:
            import subprocess, platform
            if platform.system() == "Darwin":
                subprocess.run(["afplay", tmp_path], check=False)

    except Exception as e:
        log.warning(f"Audio failed: {e}")
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

        if not face_present:
            face_absent_frames += 1
            if face_absent_frames >= ABSENT_THRESHOLD and current_match is not None:
                log.info("Face left frame — resetting (was: %s)", current_match)
                current_match = None
                last_label = ("Scanning...", (200, 200, 200))
        else:
            face_absent_frames = 0

        # ── After cooldown, reset current_match so we scan for new people ──
        now = time.time()
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
        with _lock:
            should_recognise = (
                current_match is None
                and cooldown_elapsed
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
