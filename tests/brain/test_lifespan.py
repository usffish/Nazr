"""
tests/brain/test_lifespan.py — Unit tests for Brain service lifespan behavior.

Tests that:
- MongoDB connectivity failure at startup logs a WARNING but does not raise
- Pygame initialization failure at startup logs a WARNING but does not raise

The TestClient is used as a context manager to trigger the lifespan startup
and shutdown sequences synchronously.

Requirements: 9.3
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.brain.config import get_settings
from services.brain.main import app


def _ensure_elevenlabs_mock() -> None:
    """Inject a fake 'elevenlabs' module into sys.modules if not installed.

    The lifespan does `from elevenlabs import ElevenLabs` inside the function
    body. If the real package is absent we stub it so the import doesn't fail.
    The actual ElevenLabs class is then patched per-test via unittest.mock.patch.
    """
    if "elevenlabs" not in sys.modules:
        fake_module = types.ModuleType("elevenlabs")
        fake_module.ElevenLabs = MagicMock  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = fake_module


# Ensure the stub is in place before any test in this module runs.
_ensure_elevenlabs_mock()

# The 9 required environment variables for the Brain service
REQUIRED_ENV_VARS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "ELEVENLABS_API_KEY": "test-el-key",
    "ELEVENLABS_VOICE_ID": "test-voice-id",
    "MONGODB_URI": "mongodb://localhost:27017",
    "MONGODB_DB": "test_db",
    "MONGODB_COLLECTION": "test_events",
    "PATIENT_NAME": "TestPatient",
    "PATIENT_ID": "test_patient_001",
    "GLASSES_AUDIO_DEVICE": "Test Device",
}


def _set_all_env_vars(monkeypatch) -> None:
    """Inject all 9 required env vars and override the .env file path."""
    from pydantic_settings import SettingsConfigDict
    import services.brain.config as config_module

    for key, value in REQUIRED_ENV_VARS.items():
        monkeypatch.setenv(key, value)

    # Prevent pydantic-settings from reading the real .env file
    monkeypatch.setattr(
        config_module.Settings,
        "model_config",
        SettingsConfigDict(env_file="nonexistent.env", env_file_encoding="utf-8"),
    )


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache before and after every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Test: MongoDB connectivity failure logs WARNING and does not raise
# ---------------------------------------------------------------------------

def test_mongodb_failure_logs_warning_and_continues(monkeypatch, caplog):
    """
    When verify_mongodb raises an Exception at startup, the Brain SHALL log a
    WARNING and continue starting up in a degraded state rather than refusing
    to start.

    Requirements: 9.3
    """
    _set_all_env_vars(monkeypatch)

    with (
        patch(
            "services.brain.main.init_motor",
            return_value=MagicMock(),
        ),
        patch(
            "services.brain.main.verify_mongodb",
            side_effect=Exception("connection refused"),
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
        with caplog.at_level(logging.WARNING):
            # TestClient as context manager triggers lifespan startup/shutdown.
            # If the lifespan raises, this will propagate — the test asserts it
            # does NOT raise.
            with TestClient(app) as client:
                pass  # startup succeeded if we reach here

    # Assert that a WARNING was logged about the MongoDB failure
    warning_records = [
        r for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert warning_records, (
        "Expected at least one WARNING log record when MongoDB connectivity "
        "check raises at startup, but none were found. "
        f"All log records: {[(r.levelname, r.message) for r in caplog.records]}"
    )

    # Confirm the warning message references the MongoDB failure
    warning_messages = " ".join(r.getMessage() for r in warning_records)
    assert any(
        keyword in warning_messages.lower()
        for keyword in ("mongodb", "connection", "exception", "degraded", "startup")
    ), (
        f"WARNING log does not mention MongoDB failure. "
        f"Warning messages: {warning_messages!r}"
    )


# ---------------------------------------------------------------------------
# Test: Pygame initialization failure logs WARNING and does not raise
# ---------------------------------------------------------------------------

def test_pygame_failure_logs_warning_and_continues(monkeypatch, caplog):
    """
    When init_pygame raises an Exception at startup, the Brain SHALL log a
    WARNING and continue starting up in a degraded state rather than refusing
    to start.

    Requirements: 9.3
    """
    _set_all_env_vars(monkeypatch)

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
            side_effect=Exception("no audio device"),
        ),
        patch(
            "elevenlabs.ElevenLabs",
            return_value=MagicMock(),
        ),
    ):
        with caplog.at_level(logging.WARNING):
            # TestClient as context manager triggers lifespan startup/shutdown.
            # If the lifespan raises, this will propagate — the test asserts it
            # does NOT raise.
            with TestClient(app) as client:
                pass  # startup succeeded if we reach here

    # Assert that a WARNING was logged about the Pygame failure
    warning_records = [
        r for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert warning_records, (
        "Expected at least one WARNING log record when Pygame init raises at "
        "startup, but none were found. "
        f"All log records: {[(r.levelname, r.message) for r in caplog.records]}"
    )

    # Confirm the warning message references the Pygame failure
    warning_messages = " ".join(r.getMessage() for r in warning_records)
    assert any(
        keyword in warning_messages.lower()
        for keyword in ("pygame", "audio", "exception", "degraded", "startup")
    ), (
        f"WARNING log does not mention Pygame failure. "
        f"Warning messages: {warning_messages!r}"
    )
