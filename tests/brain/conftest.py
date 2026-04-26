"""
tests/brain/conftest.py — Shared fixtures for Brain service tests.

Provides a TestClient fixture with all downstream services mocked so that
no test makes real API calls to Gemini, ElevenLabs, MongoDB, or Pygame.

Pattern follows the testing-strategy steering file.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from services.brain.main import app


@pytest.fixture
def mock_settings(monkeypatch):
    """Inject all 9 required environment variables via monkeypatch."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice-id")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGODB_DB", "test_db")
    monkeypatch.setenv("MONGODB_COLLECTION", "test_events")
    monkeypatch.setenv("PATIENT_NAME", "TestPatient")
    monkeypatch.setenv("PATIENT_ID", "test_patient_001")
    monkeypatch.setenv("GLASSES_AUDIO_DEVICE", "Test Device")


@pytest.fixture
def client(mock_settings, monkeypatch):
    """
    FastAPI TestClient with all downstream services mocked.

    Patches:
    - services.brain.services.mongodb.init_motor  -> MagicMock (no real connection)
    - services.brain.services.mongodb.verify_mongodb -> AsyncMock returning True
    - services.brain.services.audio.init_pygame  -> returns None (no real Pygame)
    - elevenlabs.ElevenLabs                       -> MagicMock (no real API calls)
    """
    # Clear the lru_cache so monkeypatched env vars take effect
    from services.brain.config import get_settings
    get_settings.cache_clear()

    # Redirect pydantic-settings away from the real .env file so that extra
    # variables in the repo .env (Snowflake, vision_host, etc.) don't cause
    # "Extra inputs are not permitted" validation errors.
    from pydantic_settings import SettingsConfigDict
    import services.brain.config as config_module
    monkeypatch.setattr(
        config_module.Settings,
        "model_config",
        SettingsConfigDict(env_file="nonexistent.env", env_file_encoding="utf-8"),
    )

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
