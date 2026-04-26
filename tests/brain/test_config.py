"""
tests/brain/test_config.py — Unit tests for Brain service config validation.

Tests that:
- All 9 required env vars present → valid Settings instance returned
- Missing any single required env var → sys.exit(1)

Requirements: 7.2, 7.3
"""

from __future__ import annotations

import pytest

from services.brain.config import get_settings, Settings

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


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache before every test so monkeypatched env vars take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_all_vars(monkeypatch) -> None:
    """Helper: inject all 9 required env vars via monkeypatch.

    Also patches Settings.model_config so pydantic-settings reads from a
    non-existent .env file, preventing the real repo .env from being loaded
    (which contains extra vars that Settings rejects as extra_forbidden).
    """
    from pydantic_settings import SettingsConfigDict
    import services.brain.config as config_module

    for key, value in REQUIRED_ENV_VARS.items():
        monkeypatch.setenv(key, value)

    # Override the env_file to a path that doesn't exist so pydantic-settings
    # only reads from the OS environment (which monkeypatch controls).
    monkeypatch.setattr(
        config_module.Settings,
        "model_config",
        SettingsConfigDict(env_file="nonexistent.env", env_file_encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# Happy path: all 9 vars present
# ---------------------------------------------------------------------------

def test_all_vars_present_returns_settings(monkeypatch):
    """When all 9 required env vars are set, get_settings() returns a valid Settings instance."""
    _set_all_vars(monkeypatch)

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.GEMINI_API_KEY == "test-gemini-key"
    assert settings.ELEVENLABS_API_KEY == "test-el-key"
    assert settings.ELEVENLABS_VOICE_ID == "test-voice-id"
    assert settings.MONGODB_URI == "mongodb://localhost:27017"
    assert settings.MONGODB_DB == "test_db"
    assert settings.MONGODB_COLLECTION == "test_events"
    assert settings.PATIENT_NAME == "TestPatient"
    assert settings.PATIENT_ID == "test_patient_001"
    assert settings.GLASSES_AUDIO_DEVICE == "Test Device"


# ---------------------------------------------------------------------------
# Missing single required var -> sys.exit(1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_var", list(REQUIRED_ENV_VARS.keys()))
def test_missing_single_var_causes_exit(monkeypatch, missing_var):
    """
    When any single required env var is absent, get_settings() must call sys.exit(1).

    Requirements: 7.2, 7.3
    """
    _set_all_vars(monkeypatch)
    # Remove the one var under test
    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        get_settings()

    assert excinfo.value.code == 1, (
        f"Expected sys.exit(1) when {missing_var!r} is missing, "
        f"got exit code {excinfo.value.code!r}"
    )
