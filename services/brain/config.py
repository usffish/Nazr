"""
services/brain/config.py — Pydantic-settings configuration for the AI Brain service.

Loads all required environment variables at startup. If any required variable is
missing, logs a descriptive error and exits with code 1 (Requirement 7.3).
"""

from __future__ import annotations

import functools
import logging
import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All required environment variables for the AI Brain service.

    Pydantic-settings will raise ValidationError at construction time if any
    required field is absent from the environment or the .env file.

    Requirements: 7.1, 7.2, 19.4
    """

    GEMINI_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    PATIENT_NAME: str
    PATIENT_ID: str
    GLASSES_AUDIO_DEVICE: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    Constructs Settings() exactly once. If any required environment variable is
    missing, logs each missing field by name and exits with code 1.

    Requirement 7.3: IF any required environment variable is missing at startup,
    THEN THE Brain SHALL log a descriptive error message identifying the missing
    variable and exit with a non-zero status code.
    """
    try:
        return Settings()
    except ValidationError as exc:
        missing_fields = [
            error["loc"][0]
            for error in exc.errors()
            if error.get("type") in ("missing", "value_error")
        ]
        if missing_fields:
            for field in missing_fields:
                logger.error("Missing required environment variable: %s", field)
        else:
            # Log all validation errors if they aren't simple missing-field errors
            logger.error("Configuration validation failed: %s", exc)
        sys.exit(1)
