"""
Property-based tests for the hybrid health detection pipeline.
Each test is annotated with the design property it validates.

Feature: hybrid-health-detection
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the repo root is on sys.path so imports work without installation.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import HEALTH_SUBTYPE_MAP from the canonical source (local_detector).
# This import is lightweight — local_detector only imports numpy and optionally
# ultralytics (which is guarded with a try/except), so no heavy side-effects.
from services.vision.local_detector import HEALTH_SUBTYPE_MAP  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: stub out heavy module-level side-effects in face_recognition_engine
# ---------------------------------------------------------------------------

def _import_engine_with_stubs():
    """
    Import services.vision.face_recognition_engine with all heavy
    module-level side-effects (model loading, pygame, av, etc.) stubbed out.

    Returns the imported module.
    """
    # Stub packages that may not be installed or that trigger hardware access.
    _stub_packages = [
        "av", "pygame", "pygame.mixer", "pygame.mixer.music",
        "onnxruntime", "cv2", "edge_tts",
        "ultralytics",
    ]
    for pkg in _stub_packages:
        if pkg not in sys.modules:
            sys.modules[pkg] = MagicMock()

    # Stub cv2 with the attributes used at module level.
    cv2_mock = MagicMock()
    cv2_mock.FaceDetectorYN.create.return_value = MagicMock()
    cv2_mock.FaceRecognizerSF.create.return_value = MagicMock()
    sys.modules["cv2"] = cv2_mock

    # Stub onnxruntime InferenceSession.
    ort_mock = MagicMock()
    sess_mock = MagicMock()
    sess_mock.get_inputs.return_value = [MagicMock(name="input")]
    sess_mock.get_outputs.return_value = [MagicMock(name="output")]
    ort_mock.InferenceSession.return_value = sess_mock
    sys.modules["onnxruntime"] = ort_mock

    # Stub dotenv so load_dotenv() is a no-op.
    dotenv_mock = MagicMock()
    dotenv_mock.load_dotenv = MagicMock()
    sys.modules["dotenv"] = dotenv_mock

    # Stub pygame.
    pygame_mock = MagicMock()
    sys.modules["pygame"] = pygame_mock

    # Make the YUNET/SFACE model paths appear to exist so _load_face_models
    # does not raise FileNotFoundError.
    with patch("pathlib.Path.exists", return_value=True), \
         patch("os.path.exists", return_value=True):
        # Remove cached module so it re-imports cleanly.
        for mod_name in list(sys.modules.keys()):
            if "face_recognition_engine" in mod_name:
                del sys.modules[mod_name]

        import services.vision.face_recognition_engine as engine  # noqa: PLC0415

    return engine


# Import once for the whole test session.
try:
    _engine = _import_engine_with_stubs()
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    _engine = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Property 12: Medicine threshold is always lower than other subtype thresholds
# ---------------------------------------------------------------------------
# Feature: hybrid-health-detection, Property 12:
#   For any valid configuration (env var unset), the effective threshold for
#   medicine_taken SHALL be strictly less than for eating and drinking.
#
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _engine is None,
    reason=f"Could not import face_recognition_engine: {_IMPORT_ERROR}",
)
@settings(max_examples=10)
@given(
    env_override=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0).map(str),
    )
)
def test_medicine_threshold_always_lower_than_other_subtypes(env_override):
    """
    **Validates: Requirements 6.1**

    Property 12: Medicine threshold is always lower than other subtype thresholds.

    When HEALTH_DETECTION_THRESHOLD is unset (None), the per-subtype defaults
    apply and medicine_taken (0.45) must be strictly less than eating (0.6) and
    drinking (0.6).

    When HEALTH_DETECTION_THRESHOLD IS set to a uniform float, all subtypes
    receive the same value — the property only applies to the default
    (per-subtype) configuration, so we skip the uniform-override case.
    """
    # When the env var is set, all subtypes get the same value uniformly.
    # The "medicine < others" invariant is intentionally only enforced for
    # the per-subtype default configuration (env var absent).
    if env_override is not None:
        return

    # Ensure HEALTH_DETECTION_THRESHOLD is absent from the environment.
    env = {k: v for k, v in os.environ.items() if k != "HEALTH_DETECTION_THRESHOLD"}
    with patch.dict(os.environ, env, clear=True):
        threshold_medicine = _engine._get_threshold("medicine_taken")
        threshold_eating = _engine._get_threshold("eating")
        threshold_drinking = _engine._get_threshold("drinking")

    assert threshold_medicine < threshold_eating, (
        f"medicine_taken threshold ({threshold_medicine}) must be strictly less than "
        f"eating threshold ({threshold_eating})"
    )
    assert threshold_medicine < threshold_drinking, (
        f"medicine_taken threshold ({threshold_medicine}) must be strictly less than "
        f"drinking threshold ({threshold_drinking})"
    )


# ---------------------------------------------------------------------------
# Property 7: Dispatched events always contain all required fields
# ---------------------------------------------------------------------------
# Feature: hybrid-health-detection, Property 7:
#   For any valid detection result passing threshold and cooldown gates, the
#   constructed Event SHALL be a valid shared.contract.Event instance with
#   type="health", source="vision_engine_v1", UUID4 event_id, UTC ISO-8601
#   timestamp, and confidence equal to the Secondary Pass score.
#
# Validates: Requirements 3.1, 7.1, 7.2
# ---------------------------------------------------------------------------

import re
import uuid as _uuid_module
from datetime import datetime, timezone


def _is_uuid4(s: str) -> bool:
    """Return True if *s* is a valid UUID4 string."""
    try:
        val = _uuid_module.UUID(s, version=4)
        return str(val) == s
    except ValueError:
        return False


def _is_utc_iso8601(s: str) -> bool:
    """Return True if *s* is a UTC ISO-8601 timestamp (ends with +00:00 or Z)."""
    # Accept both +00:00 and Z suffixes produced by datetime.isoformat()
    try:
        if s.endswith("Z"):
            datetime.fromisoformat(s[:-1])
            return True
        dt = datetime.fromisoformat(s)
        return dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0
    except (ValueError, AttributeError):
        return False


@pytest.mark.skipif(
    _engine is None,
    reason=f"Could not import face_recognition_engine: {_IMPORT_ERROR}",
)
@settings(max_examples=10)
@given(
    score=st.floats(min_value=0.0, max_value=1.0),
    subtype=st.sampled_from(["drinking", "eating", "medicine_taken"]),
    detected_objects=st.lists(
        st.sampled_from(["cup", "bottle", "fork", "pill", "sandwich"]),
        min_size=0,
        max_size=3,
    ),
)
def test_dispatched_events_contain_all_required_fields(score, detected_objects, subtype):
    """
    **Validates: Requirements 3.1, 7.1, 7.2**

    Property 7: Dispatched events always contain all required fields with
    correct values.

    For any valid detection result passing threshold and cooldown gates, the
    constructed Event SHALL be a valid shared.contract.Event instance with
    type="health", source="vision_engine_v1", UUID4 event_id, UTC ISO-8601
    timestamp, and confidence equal to the Secondary Pass score.
    """
    import numpy as np
    from unittest.mock import MagicMock, patch
    from services.vision.gemini_health import ConfidenceResult
    from services.vision.local_detector import DetectionResult
    from shared.contract import Event

    # Build minimal test inputs
    frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

    confidence_result = ConfidenceResult(
        score=score,
        raw_text=str(score),
        subtype=subtype,
    )
    detection_result = DetectionResult(
        flagged=True,
        detected_objects=detected_objects,
        confidence_scores={obj: 0.9 for obj in detected_objects},
        medicine_flagged=any(
            obj in {"pill", "tablet", "medicine", "medication", "medicine packet"}
            for obj in detected_objects
        ),
    )

    captured_events: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        captured_events.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    # cv2 is mocked globally in this test session; provide a minimal JPEG
    # buffer so base64 encoding works without real OpenCV.
    import io
    try:
        from PIL import Image as _PILImage
        _pil_img = _PILImage.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        _buf_io = io.BytesIO()
        _pil_img.save(_buf_io, format="JPEG")
        _jpeg_bytes = _buf_io.getvalue()
    except ImportError:
        # Minimal valid JPEG (1x1 white pixel) as fallback
        _jpeg_bytes = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
            b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
            b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81"
            b"\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19"
            b"\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86"
            b"\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4"
            b"\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2"
            b"\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9"
            b"\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
            b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2"
            b"\x8a(\x03\xff\xd9"
        )

    import numpy as _np
    _fake_buf = _np.frombuffer(_jpeg_bytes, dtype=_np.uint8)
    _fake_imencode = MagicMock(return_value=(True, _fake_buf))

    with patch("services.vision.face_recognition_engine.cv2.imencode", _fake_imencode), \
         patch("services.vision.face_recognition_engine.cv2.resize", side_effect=lambda img, size: img), \
         patch("services.vision.face_recognition_engine.http_requests.post", side_effect=_fake_post), \
         patch("services.vision.face_recognition_engine.save_event_json"):
        _engine._dispatch_health_event(frame_bgr, subtype, confidence_result, detection_result)

    # Exactly one event must have been POSTed
    assert len(captured_events) == 1, (
        f"Expected exactly 1 event POSTed, got {len(captured_events)}"
    )

    payload = captured_events[0]

    # Validate it round-trips through the shared.contract.Event model
    event = Event(**payload)

    # type must be "health"
    assert event.type == "health", f"Expected type='health', got {event.type!r}"

    # source must be "vision_engine_v1"
    assert event.source == "vision_engine_v1", (
        f"Expected source='vision_engine_v1', got {event.source!r}"
    )

    # event_id must be a valid UUID4
    assert _is_uuid4(event.event_id), (
        f"event_id {event.event_id!r} is not a valid UUID4"
    )

    # timestamp must be UTC ISO-8601
    assert _is_utc_iso8601(event.timestamp), (
        f"timestamp {event.timestamp!r} is not a valid UTC ISO-8601 string"
    )

    # confidence must equal the Secondary Pass score
    assert event.confidence == score, (
        f"Expected confidence={score}, got {event.confidence}"
    )

    # subtype must be passed through unchanged
    assert event.subtype == subtype, (
        f"Expected subtype={subtype!r}, got {event.subtype!r}"
    )

    # metadata must contain detected_item
    assert "detected_item" in event.metadata, (
        f"metadata missing 'detected_item' key: {event.metadata}"
    )

    # detected_item should be first detected object or subtype fallback
    expected_item = detected_objects[0] if detected_objects else subtype
    assert event.metadata["detected_item"] == expected_item, (
        f"Expected detected_item={expected_item!r}, got {event.metadata['detected_item']!r}"
    )


# ---------------------------------------------------------------------------
# Property 13: HEALTH_SUBTYPE_MAP round-trip
# ---------------------------------------------------------------------------
# Feature: hybrid-health-detection, Property 13:
#   For any key in HEALTH_SUBTYPE_MAP, the mapped value SHALL be one of
#   {"drinking", "eating", "medicine_taken"} and the mapping SHALL be
#   deterministic (same key always produces same subtype).
#
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

_VALID_SUBTYPES: frozenset[str] = frozenset({"drinking", "eating", "medicine_taken"})


@settings(max_examples=10)
@given(label=st.sampled_from(list(HEALTH_SUBTYPE_MAP.keys())))
def test_health_subtype_map_round_trip(label: str) -> None:
    """
    **Validates: Requirements 7.4**

    Property 13: HEALTH_SUBTYPE_MAP round-trip.

    For any key in HEALTH_SUBTYPE_MAP:
    1. The mapped value SHALL be one of {"drinking", "eating", "medicine_taken"}.
    2. The mapping SHALL be deterministic — repeated lookups of the same key
       always return the same subtype.
    """
    # First lookup
    value_first = HEALTH_SUBTYPE_MAP[label]

    # Value must be one of the three valid subtypes
    assert value_first in _VALID_SUBTYPES, (
        f"HEALTH_SUBTYPE_MAP[{label!r}] = {value_first!r} is not in "
        f"{_VALID_SUBTYPES}"
    )

    # Second lookup — must be identical (determinism)
    value_second = HEALTH_SUBTYPE_MAP[label]
    assert value_first == value_second, (
        f"HEALTH_SUBTYPE_MAP[{label!r}] returned different values on repeated "
        f"lookups: {value_first!r} vs {value_second!r}"
    )
