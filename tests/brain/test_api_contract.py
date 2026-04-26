"""
tests/brain/test_api_contract.py — Property 12: Content-Type header compliance.

Task 13.2:
  Property 12: All Brain Responses Have Content-Type application/json
  Validates: Requirements 8.1

Covers:
  - Valid POST /event requests (hypothesis)
  - Invalid POST /event requests (HTTP 422)
  - GET /health (MongoDB reachable)
  - GET /health (MongoDB unreachable)
  - Requests that trigger the global exception handler (HTTP 500)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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

valid_event_strategy = st.one_of(
    st.builds(
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
    ),
    st.builds(
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
    ),
)

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

# Placeholder env vars for hypothesis tests (cannot use pytest fixtures with @given)
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
    """Yield a TestClient with all downstream mocked (for use in @given tests)."""
    from services.brain.config import get_settings
    from pydantic_settings import SettingsConfigDict
    import services.brain.config as config_module

    original_env = {k: os.environ.get(k) for k in _TEST_ENV_VARS}
    original_model_config = config_module.Settings.model_config

    try:
        for k, v in _TEST_ENV_VARS.items():
            os.environ[k] = v
        config_module.Settings.model_config = SettingsConfigDict(
            env_file="nonexistent.env", env_file_encoding="utf-8"
        )
        get_settings.cache_clear()

        with (
            patch("services.brain.main.init_motor", return_value=MagicMock()),
            patch(
                "services.brain.main.verify_mongodb",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch("services.brain.main.init_pygame", return_value=None),
            patch("elevenlabs.ElevenLabs", return_value=MagicMock()),
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


def _assert_json_content_type(response, label: str = "") -> None:
    ct = response.headers.get("content-type", "")
    assert "application/json" in ct, (
        f"Expected Content-Type: application/json{' (' + label + ')' if label else ''}, "
        f"got {ct!r}. Status: {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Property 12: valid POST /event → Content-Type: application/json
# Uses _make_client() because @given tests cannot receive pytest fixtures
# ---------------------------------------------------------------------------

# Property 12: All Brain Responses Have Content-Type application/json
# Validates: Requirements 8.1
@given(valid_event_strategy)
@settings(max_examples=10)
def test_valid_event_response_has_json_content_type(event: Event):
    with _make_client() as client:
        with (
            patch(
                "services.brain.routes.event.verify_health_event",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch("services.brain.routes.event.generate_voice_script", return_value="script"),
            patch(
                "services.brain.routes.event.synthesize_audio",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=MagicMock())(),
            ),
            patch("services.brain.routes.event.play_audio", return_value=None),
            patch(
                "services.brain.routes.event.write_event_record",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
        ):
            response = client.post("/event", json=event.model_dump())

    _assert_json_content_type(response, "valid POST /event")


# ---------------------------------------------------------------------------
# Property 12: invalid POST /event (HTTP 422) → Content-Type: application/json
# ---------------------------------------------------------------------------

def test_invalid_event_422_has_json_content_type(client):
    # Property 12: All Brain Responses Have Content-Type application/json
    # Validates: Requirements 8.1
    response = client.post("/event", json={"bad": "payload"})
    assert response.status_code == 422
    _assert_json_content_type(response, "invalid POST /event 422")


# ---------------------------------------------------------------------------
# Property 12: GET /health (reachable) → Content-Type: application/json
# ---------------------------------------------------------------------------

def test_health_ok_has_json_content_type(client):
    # Property 12: All Brain Responses Have Content-Type application/json
    # Validates: Requirements 8.1
    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(return_value={"ok": 1})
    client.app.state.motor_client = mock_motor

    response = client.get("/health")
    assert response.status_code == 200
    _assert_json_content_type(response, "GET /health ok")


# ---------------------------------------------------------------------------
# Property 12: GET /health (unreachable) → Content-Type: application/json
# ---------------------------------------------------------------------------

def test_health_degraded_has_json_content_type(client):
    # Property 12: All Brain Responses Have Content-Type application/json
    # Validates: Requirements 8.1
    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(side_effect=Exception("unreachable"))
    client.app.state.motor_client = mock_motor

    response = client.get("/health")
    assert response.status_code == 503
    _assert_json_content_type(response, "GET /health degraded")


# ---------------------------------------------------------------------------
# Property 12: global exception handler (HTTP 500) → Content-Type: application/json
# ---------------------------------------------------------------------------

def test_unhandled_exception_500_has_json_content_type(client):
    # Property 12: All Brain Responses Have Content-Type application/json
    # Validates: Requirements 8.1
    # Patch EventRecord (used outside any try/except in the route) to raise,
    # which triggers the global exception handler and returns HTTP 500.
    # raise_server_exceptions=False is needed so TestClient returns the 500
    # response instead of re-raising the exception.
    from fastapi.testclient import TestClient as _TC
    no_raise_client = _TC(client.app, raise_server_exceptions=False)

    with patch(
        "services.brain.routes.event.EventRecord",
        side_effect=RuntimeError("unexpected boom"),
    ):
        with (
            patch(
                "services.brain.routes.event.verify_health_event",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=True)(),
            ),
            patch(
                "services.brain.routes.event.generate_voice_script",
                return_value="script",
            ),
            patch(
                "services.brain.routes.event.synthesize_audio",
                new_callable=lambda: lambda *a, **kw: AsyncMock(return_value=MagicMock())(),
            ),
            patch("services.brain.routes.event.play_audio", return_value=None),
        ):
            response = no_raise_client.post("/event", json=_VALID_PAYLOAD)

    assert response.status_code == 500
    _assert_json_content_type(response, "global exception handler HTTP 500")
