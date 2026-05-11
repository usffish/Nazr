"""
Local Detector — Primary Pass for Hybrid Health Detection
----------------------------------------------------------
Runs a lightweight YOLO-based object detector on each sampled health frame.
Only frames containing health-relevant objects are escalated to the Secondary
Pass (Gemini). No network I/O is performed here.

Requirements: 1.1, 1.2, 1.5, 1.6, 4.2, 4.4, 6.2, 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# ── Try to import Ultralytics YOLO (optional dependency) ─────────────────────

try:
    from ultralytics import YOLO as _YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YOLO = None  # type: ignore
    _ULTRALYTICS_AVAILABLE = False
    log.warning(
        "ultralytics package not installed — LocalDetector will be unavailable. "
        "Install with: pip install ultralytics"
    )

# ── Health-relevant object map ────────────────────────────────────────────────
# Maps lowercase YOLO class labels → health subtype.
# This is the canonical definition used by the Primary Pass.
# face_recognition_engine.py imports HEALTH_SUBTYPE_MAP from here (or keeps its
# own uppercase copy for backward compatibility — both are equivalent).

HEALTH_SUBTYPE_MAP: dict[str, str] = {
    # drinking
    "cup": "drinking",
    "glass": "drinking",
    "mug": "drinking",
    "bottle": "drinking",
    "water bottle": "drinking",
    "soda can": "drinking",
    # eating
    "food item": "eating",
    "food": "eating",
    "fork": "eating",
    "spoon": "eating",
    "sandwich": "eating",
    # medicine_taken  (safety override: flagged at ANY confidence score)
    "pill": "medicine_taken",
    "tablet": "medicine_taken",
    "medicine": "medicine_taken",
    "medication": "medicine_taken",
    "medicine packet": "medicine_taken",
}

# Set of labels that trigger the medicine safety override (Requirement 6.2).
_MEDICINE_LABELS: frozenset[str] = frozenset(
    label for label, subtype in HEALTH_SUBTYPE_MAP.items() if subtype == "medicine_taken"
)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class DetectionResult:
    """Result returned by LocalDetector.run()."""

    flagged: bool
    """True if the frame should be escalated to the Secondary Pass."""

    detected_objects: list[str] = field(default_factory=list)
    """Raw YOLO class labels that were detected above the confidence threshold
    (or medicine labels detected at any confidence)."""

    confidence_scores: dict[str, float] = field(default_factory=dict)
    """Per-label YOLO confidence scores (label → score)."""

    medicine_flagged: bool = False
    """True if any medicine-related object was detected, regardless of YOLO
    confidence score (safety override per Requirement 6.2)."""


# ── LocalDetector ─────────────────────────────────────────────────────────────


class LocalDetector:
    """
    Wraps a YOLO model for health-relevant object detection.

    Thread-safe: Ultralytics YOLO is safe to call from multiple threads when
    the model is loaded once and called with ``model(frame)`` — no shared
    mutable state per call (Requirement 8.4).
    """

    def __init__(self, model_path: str, confidence_threshold: float | None = None) -> None:
        """
        Load the YOLO model from *model_path*.

        Parameters
        ----------
        model_path:
            Path to the YOLO weights file (e.g. ``yolov8n.pt``).
        confidence_threshold:
            Minimum YOLO confidence score for a detection to be considered.
            If ``None``, reads ``LOCAL_DETECTOR_CONFIDENCE`` env var; defaults
            to 0.4 if the env var is unset or unparseable (Requirement 4.4).
        """
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run `pip install ultralytics` to enable local detection."
            )

        # Resolve confidence threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            env_val = os.getenv("LOCAL_DETECTOR_CONFIDENCE", "")
            try:
                self.confidence_threshold = float(env_val) if env_val else 0.4
            except ValueError:
                log.warning(
                    "Unparseable LOCAL_DETECTOR_CONFIDENCE=%r — using default 0.4", env_val
                )
                self.confidence_threshold = 0.4

        log.info(
            "Loading YOLO model from %r (confidence_threshold=%.2f)",
            model_path,
            self.confidence_threshold,
        )
        self._model = _YOLO(model_path)
        log.info("YOLO model loaded successfully from %r", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, frame_bgr: np.ndarray) -> DetectionResult:
        """
        Run YOLO inference on *frame_bgr* and return a :class:`DetectionResult`.

        The method is thread-safe — Ultralytics YOLO does not mutate shared
        state during inference.

        Parameters
        ----------
        frame_bgr:
            A BGR numpy array (as returned by OpenCV).

        Returns
        -------
        DetectionResult
            ``flagged=True`` when at least one health-relevant object is found.
            ``medicine_flagged=True`` when any medicine-related object is
            detected at *any* YOLO confidence score (safety override).
        """
        results = self._model(frame_bgr, verbose=False)

        detected_objects: list[str] = []
        confidence_scores: dict[str, float] = {}
        medicine_flagged = False
        flagged = False

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            # names maps class-index → label string
            names: dict[int, str] = result.names  # type: ignore[assignment]

            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                raw_label: str = names.get(cls_idx, "").lower().strip()

                # Medicine safety override: flag at ANY confidence (Req 6.2)
                if raw_label in _MEDICINE_LABELS:
                    medicine_flagged = True
                    flagged = True
                    # Record the object even if below the normal threshold
                    if raw_label not in confidence_scores or confidence_scores[raw_label] < score:
                        confidence_scores[raw_label] = score
                    if raw_label not in detected_objects:
                        detected_objects.append(raw_label)
                    log.debug(
                        "Medicine object detected: %r (score=%.3f) — safety override applied",
                        raw_label,
                        score,
                    )
                    continue

                # Normal threshold gate for non-medicine objects
                if score < self.confidence_threshold:
                    continue

                if raw_label in HEALTH_SUBTYPE_MAP:
                    flagged = True
                    if raw_label not in confidence_scores or confidence_scores[raw_label] < score:
                        confidence_scores[raw_label] = score
                    if raw_label not in detected_objects:
                        detected_objects.append(raw_label)

        log.debug(
            "Primary pass: flagged=%s, medicine_flagged=%s, objects=%s",
            flagged,
            medicine_flagged,
            detected_objects,
        )

        return DetectionResult(
            flagged=flagged,
            detected_objects=detected_objects,
            confidence_scores=confidence_scores,
            medicine_flagged=medicine_flagged,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_detector: LocalDetector | None = None
_health_detection_disabled: bool = False


def _load_detector() -> LocalDetector:
    """
    Instantiate a :class:`LocalDetector` using environment configuration.

    Reads ``LOCAL_DETECTOR_MODEL`` env var for the model path; falls back to
    ``"yolov8n.pt"`` if unset (Requirement 4.2).

    Raises
    ------
    Exception
        Any exception raised by the YOLO model loader is propagated so that
        :func:`get_detector` can catch it and disable health detection.
    """
    model_path = os.getenv("LOCAL_DETECTOR_MODEL", "yolov8n.pt")
    return LocalDetector(model_path=model_path)


def get_detector() -> LocalDetector:
    """
    Return the module-level :class:`LocalDetector` singleton.

    The model is loaded on the first call and reused for all subsequent calls
    (Requirements 8.1, 8.3). If loading fails, logs CRITICAL, sets
    ``_health_detection_disabled = True``, and re-raises so the caller can
    handle gracefully (Requirement 8.2).

    Returns
    -------
    LocalDetector
        The shared detector instance.

    Raises
    ------
    RuntimeError
        If health detection has been disabled due to a prior load failure.
    Exception
        Any exception from the underlying model loader on first call.
    """
    global _detector, _health_detection_disabled

    if _health_detection_disabled:
        raise RuntimeError(
            "LocalDetector is disabled because the model failed to load at startup."
        )

    if _detector is None:
        try:
            _detector = _load_detector()
        except Exception as exc:
            _health_detection_disabled = True
            log.critical(
                "CRITICAL: Failed to load LocalDetector model — "
                "health detection is DISABLED for this session. Error: %s",
                exc,
            )
            raise

    return _detector
