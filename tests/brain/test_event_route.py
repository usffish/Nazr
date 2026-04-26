"""
tests/brain/test_event_route.py — Property and parametrized tests for POST /event.

Property 1: Valid Events Are Accepted, Invalid Events Are Rejected
Validates: Requirements 1.2, 1.4, 16.2
"""
from __future__ import annotations

import os
import pytest
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from shared.contract import Event, PersonProfile
from fastapi.testclient import TestClient
from services.brain.main import app


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

person_profile_strategy = st.builds(
    PersonProfile,
    name=st.text(min_size=1, max_size=50),
    relationship=st.text(min_size=1, max_size=30),
    background=st.text(min_size=1, max_size=200),
    last_conversation=st.text(min_size=1, max_size=200),
)

identity_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("identity"),
    subtype=st.just("face_recognized"),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1),
    metadata=st.fixed_dictionaries({"person_profile": person_profile_strategy}),
    source=st.just("vision_engine_v1"),
)

health_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("health"),
    subtype=st.sampled_from(["eating", "drinking", "medicine_taken"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1),
    metadata=st.fixed_dictionaries(
        {"detected_item": st.sampled_from(["food", "water", "medicine"])}
    ),
    source=st.just("vision_engine_v1"),
)

valid_event_strategy = st.one_of(identity_event_strategy, health_event_strategy)


# ---------------------------------------------------------------------------
# Helper: build a TestClient with all infrastructure mocked
# ---------------------------------------------------------------------------

# Placeholder values used only in tests — not real credentials
_TEST_ENV_VARS = {
    "GEMINI_API_KEY": "placeholder-gemini",
    "ELEVENLABS_API_KEY": "placeholder-el",
    "ELEVENLABS_VOICE_ID": "placeholder-voice",
    "MONGODB_URI": "mongodb://localhost:27017",
    "MONGODB_DB": "test_db",
    "MONGODB_COLLECTION": "test_events",
    "PATIENT_NAME": "TestPatient",
    "PATIENT_ID": "test_patient_001",
    "GLASSES_AUDIO_DEVICE": "Test Device",
}


@contextmanager
def _make_client():
    """Context manager that yields a TestClient with all downstream mocked.

    Used by hypothesis tests (which cannot receive pytest fixtures as args).
    Sets the 9 required env vars, redirects pydantic-settings away from the
    real .env file, and patches all downstream service calls.
    """
    from services.brain.config import get_settings
    from pydantic_settings import SettingsConfigDict
    import services.brain.config as config_module

    original_env = {k: os.environ.get(k) for k in _TEST_ENV_VARS}
    original_model_config = config_module.Settings.model_config

    try:
        for k, v in _TEST_ENV_VARS.items():
            os.environ[k] = v

        # Redirect away from real .env to avoid extra-inputs errors from
        # Snowflake/vision/dashboard vars that Settings does not declare.
        config_module.Settings.model_config = SettingsConfigDict(
            env_file="nonexistent.env", env_file_encoding="utf-8"
        )
        get_settings.cache_clear()

        with (
            patch(
                "services.brain.main.init_motor",
                return_value=MagicMock(),
            ),
            patch(
                "services.brain.main.verify_mongodb",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch(
                "services.brain.main.init_pygame",
                return_value=None,
            ),
            patch(
                "elevenlabs.ElevenLabs",
                return_value=MagicMock(),
            ),
        ):
            with TestClient(app) as c:
                yield c
    finally:
        for k, orig in original_env.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig
        config_module.Settings.model_config = original_model_config
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Property 1 — valid events produce HTTP 200
# ---------------------------------------------------------------------------

# Property 1: Valid Events Are Accepted, Invalid Events Are Rejected
# Validates: Requirements 1.2, 1.4, 16.2
@given(valid_event_strategy)
@settings(max_examples=10)
def test_valid_event_returns_200(event: Event):
    """Any well-formed Event payload must be accepted with HTTP 200."""
    payload = event.model_dump()

    with _make_client() as client:
        with (
            patch(
                "services.brain.routes.event.verify_health_event",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch(
                "services.brain.routes.event.generate_voice_script",
                return_value="Test voice script.",
            ),
            patch(
                "services.brain.routes.event.synthesize_audio",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=MagicMock())(),
            ),
            patch(
                "services.brain.routes.event.play_audio",
                return_value=None,
            ),
            patch(
                "services.brain.routes.event.write_event_record",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
        ):
            response = client.post("/event", json=payload)

    assert response.status_code == 200, (
        f"Expected HTTP 200 for valid event, got {response.status_code}. "
        f"Body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 1 — invalid events produce HTTP 422
# ---------------------------------------------------------------------------

# A complete valid payload used as the base for corruption tests
_VALID_PAYLOAD = {
    "event_id": "123e4567-e89b-12d3-a456-426614174000",
    "timestamp": "2025-01-01T00:00:00Z",
    "patient_id": "patient_001",
    "type": "health",
    "subtype": "eating",
    "confidence": 0.9,
    "image_b64": "aGVsbG8=",
    "metadata": {"detected_item": "food"},
    "source": "vision_engine_v1",
}

# Build parametrize cases: (description, payload_dict)
_INVALID_CASES = []

# Missing required fields — remove one field at a time
for _field in _VALID_PAYLOAD:
    _broken = {k: v for k, v in _VALID_PAYLOAD.items() if k != _field}
    _INVALID_CASES.append((f"missing_{_field}", _broken))

# Wrong types / invalid literal values
_INVALID_CASES.extend([
    (
        "confidence_as_string",
        {**_VALID_PAYLOAD, "confidence": "not-a-float"},
    ),
    (
        "type_invalid_literal",
        {**_VALID_PAYLOAD, "type": "unknown_type"},
    ),
    (
        "source_invalid_literal",
        {**_VALID_PAYLOAD, "source": "bad_source"},
    ),
    (
        "event_id_as_int",
        {**_VALID_PAYLOAD, "event_id": 12345},
    ),
    (
        "confidence_as_none",
        {**_VALID_PAYLOAD, "confidence": None},
    ),
    (
        "metadata_as_string",
        {**_VALID_PAYLOAD, "metadata": "not-a-dict"},
    ),
])


# Property 1: Valid Events Are Accepted, Invalid Events Are Rejected
# Validates: Requirements 1.2, 1.4, 16.2
@pytest.mark.parametrize(
    "description,payload",
    _INVALID_CASES,
    ids=[c[0] for c in _INVALID_CASES],
)
def test_invalid_event_returns_422(description, payload, client):
    """Payloads with missing required fields or wrong types must be rejected with HTTP 422."""
    response = client.post("/event", json=payload)
    assert response.status_code == 422, (
        f"Expected HTTP 422 for invalid payload ({description}), "
        f"got {response.status_code}. Body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 2 — HTTP 200 response always contains echoed event_id
# ---------------------------------------------------------------------------

# Property 2: HTTP 200 Response Always Contains Echoed event_id
# Validates: Requirements 1.5, 8.2
@given(valid_event_strategy)
@settings(max_examples=10)
def test_response_echoes_event_id(event: Event):
    """The response body must always echo back the original event_id."""
    payload = event.model_dump()

    with _make_client() as client:
        with (
            patch(
                "services.brain.routes.event.verify_health_event",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch(
                "services.brain.routes.event.generate_voice_script",
                return_value="Test voice script.",
            ),
            patch(
                "services.brain.routes.event.synthesize_audio",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=MagicMock())(),
            ),
            patch(
                "services.brain.routes.event.play_audio",
                return_value=None,
            ),
            patch(
                "services.brain.routes.event.write_event_record",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
        ):
            response = client.post("/event", json=payload)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    assert response.json()["event_id"] == event.event_id, (
        f"Expected echoed event_id={event.event_id!r}, "
        f"got {response.json().get('event_id')!r}"
    )


# ---------------------------------------------------------------------------
# Property 3 — identity events always set verified=True without calling Gemini
# ---------------------------------------------------------------------------

# Property 3: Identity Events Always Set verified=True Without Calling Gemini
# Validates: Requirements 2.4
@given(identity_event_strategy)
@settings(max_examples=10)
def test_identity_event_skips_gemini_and_sets_verified_true(event: Event):
    """Identity events must never call Gemini and must produce verified=True in the record."""
    payload = event.model_dump()

    mock_gemini = MagicMock()  # sync mock — must never be called for identity events
    mock_write = AsyncMock(return_value=True)

    with _make_client() as client:
        with (
            patch("services.brain.routes.event.verify_health_event", mock_gemini),
            patch(
                "services.brain.routes.event.generate_voice_script",
                return_value="Test voice script.",
            ),
            patch(
                "services.brain.routes.event.synthesize_audio",
                AsyncMock(return_value=MagicMock()),
            ),
            patch("services.brain.routes.event.play_audio", return_value=None),
            patch("services.brain.routes.event.write_event_record", mock_write),
        ):
            response = client.post("/event", json=payload)

    # Gemini must never be called for identity events
    mock_gemini.assert_not_called()

    assert response.status_code == 200, (
        f"Expected HTTP 200 for identity event, got {response.status_code}. "
        f"Body: {response.text}"
    )

    # Verify the EventRecord written to MongoDB has verified=True
    assert mock_write.called, "write_event_record should have been called"
    written_record = mock_write.call_args[0][0]  # first positional arg
    assert written_record.verified is True, (
        f"EventRecord.verified must be True for identity events, "
        f"got {written_record.verified!r}"
    )


# ---------------------------------------------------------------------------
# Property 11 — downstream failures never produce HTTP 5xx
# ---------------------------------------------------------------------------

# All combinations of downstream failures to inject
_FAILURE_COMBINATIONS = [
    {"gemini": True,  "elevenlabs": False, "pygame": False, "mongodb": False},
    {"gemini": False, "elevenlabs": True,  "pygame": False, "mongodb": False},
    {"gemini": False, "elevenlabs": False, "pygame": True,  "mongodb": False},
    {"gemini": False, "elevenlabs": False, "pygame": False, "mongodb": True},
    {"gemini": True,  "elevenlabs": True,  "pygame": False, "mongodb": False},
    {"gemini": True,  "elevenlabs": False, "pygame": True,  "mongodb": False},
    {"gemini": True,  "elevenlabs": False, "pygame": False, "mongodb": True},
    {"gemini": False, "elevenlabs": True,  "pygame": True,  "mongodb": False},
    {"gemini": False, "elevenlabs": True,  "pygame": False, "mongodb": True},
    {"gemini": False, "elevenlabs": False, "pygame": True,  "mongodb": True},
    {"gemini": True,  "elevenlabs": True,  "pygame": True,  "mongodb": False},
    {"gemini": True,  "elevenlabs": True,  "pygame": False, "mongodb": True},
    {"gemini": True,  "elevenlabs": False, "pygame": True,  "mongodb": True},
    {"gemini": False, "elevenlabs": True,  "pygame": True,  "mongodb": True},
    {"gemini": True,  "elevenlabs": True,  "pygame": True,  "mongodb": True},
]


# Property 11: Downstream Failures Never Produce HTTP 5xx
# Validates: Requirements 8.3, 8.4, 20.2, 20.3, 20.4
@pytest.mark.parametrize(
    "failures",
    _FAILURE_COMBINATIONS,
    ids=[
        "gemini={gemini},el={elevenlabs},pygame={pygame},mongo={mongodb}".format(**c)
        for c in _FAILURE_COMBINATIONS
    ],
)
def test_downstream_failures_never_produce_5xx(failures, client):
    """Any combination of downstream failures must still return HTTP 200."""

    async def _raise(*args, **kwargs):
        raise Exception("injected failure")

    def _raise_sync(*args, **kwargs):
        raise Exception("injected failure")

    with (
        patch(
            "services.brain.routes.event.verify_health_event",
            side_effect=_raise if failures["gemini"] else AsyncMock(return_value=True),
        ),
        patch(
            "services.brain.routes.event.generate_voice_script",
            return_value="Test script.",
        ),
        patch(
            "services.brain.routes.event.synthesize_audio",
            side_effect=_raise if failures["elevenlabs"] else AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "services.brain.routes.event.play_audio",
            side_effect=_raise_sync if failures["pygame"] else MagicMock(return_value=None),
        ),
        patch(
            "services.brain.routes.event.write_event_record",
            side_effect=_raise if failures["mongodb"] else AsyncMock(return_value=True),
        ),
    ):
        response = client.post("/event", json=_VALID_PAYLOAD)

    assert response.status_code == 200, (
        f"Expected HTTP 200 with failures={failures}, "
        f"got {response.status_code}. Body: {response.text}"
    )
