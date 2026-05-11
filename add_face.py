"""
add_face.py — Capture a face photo from the webcam and add it to known_faces.

Usage:
    python add_face.py

The script will:
  1. Open your webcam
  2. Show a live preview with face detection overlay
  3. Let you press SPACE to capture when the face looks good
  4. Ask for the person's name and profile details
  5. Save <name>.jpg and <name>.json to services/vision/known_faces/

Press Q at any time to quit without saving.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

KNOWN_FACES_DIR = Path(__file__).parent / "services" / "vision" / "known_faces"
MODELS_DIR = Path(__file__).parent / "tests" / "vision" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


def _load_detector(frame_w: int, frame_h: int):
    """Load YuNet face detector if available, else return None."""
    if not YUNET_MODEL.exists():
        return None
    try:
        det = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (frame_w, frame_h),
            score_threshold=0.6, nms_threshold=0.3,
        )
        return det
    except Exception:
        return None


def _detect_faces(detector, frame_bgr):
    """Return list of (x, y, w, h) face boxes, or empty list."""
    if detector is None:
        return []
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    boxes = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        boxes.append((x, y, fw, fh))
    return boxes


def _draw_overlay(frame, boxes, status_text, status_color):
    """Draw face boxes and status text onto frame (in-place)."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Semi-transparent bottom bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, status_text, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, "SPACE: capture   Q: quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def _prompt(label: str, default: str = "") -> str:
    """Prompt the user for input with an optional default."""
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val if val else default


def main():
    print("\n=== AuraGuard — Add Known Face ===\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("ERROR: Could not read from webcam.")
        cap.release()
        sys.exit(1)

    h, w = test_frame.shape[:2]
    detector = _load_detector(w, h)
    if detector is None:
        print("NOTE: YuNet model not found — face detection overlay disabled.")
        print(f"      (Expected at {YUNET_MODEL})")
        print("      You can still capture a photo manually.\n")

    captured_frame = None
    print("Position the person's face in the frame, then press SPACE to capture.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost webcam feed.")
            break

        display = frame.copy()
        boxes = _detect_faces(detector, frame)

        if boxes:
            status = f"Face detected ({len(boxes)}) — press SPACE to capture"
            color = (0, 255, 0)
        else:
            status = "No face detected — adjust position"
            color = (0, 140, 255)

        _draw_overlay(display, boxes, status, color)
        cv2.imshow("AuraGuard — Add Face (SPACE to capture, Q to quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord(' '):
            if detector is not None and not boxes:
                print("  No face detected in this frame — try again.")
                continue
            captured_frame = frame.copy()
            print("  ✓ Photo captured!\n")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        sys.exit(0)

    # Show the captured frame briefly
    cv2.imshow("Captured — press any key to continue", captured_frame)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    # Collect profile info
    print("Enter details for this person:\n")
    name = ""
    while not name:
        name = input("  Name (e.g. Hussain): ").strip()
        if not name:
            print("  Name cannot be empty.")

    relationship  = _prompt("Relationship to patient (e.g. son, daughter, nurse)", "family member")
    background    = _prompt("Background (e.g. Software engineer living in Tampa)", "")
    last_convo    = _prompt("Last conversation topic (e.g. told you about his new job)", "")

    # Derive filename from name (lowercase, spaces → underscores)
    filename_stem = name.lower().replace(" ", "_")
    jpg_path  = KNOWN_FACES_DIR / f"{filename_stem}.jpg"
    json_path = KNOWN_FACES_DIR / f"{filename_stem}.json"

    # Warn if overwriting
    if jpg_path.exists():
        confirm = input(f"\n  '{jpg_path.name}' already exists. Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            sys.exit(0)

    # Save photo
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(jpg_path), captured_frame)

    # Save profile JSON
    profile = {
        "name": name,
        "relationship": relationship,
        "background": background,
        "last_conversation": last_convo,
    }
    json_path.write_text(json.dumps(profile, indent=2))

    print(f"\n✓ Saved photo:   {jpg_path}")
    print(f"✓ Saved profile: {json_path}")
    print(f"\nRestart test_webcam.py — '{name}' will now be recognized.\n")


if __name__ == "__main__":
    main()
