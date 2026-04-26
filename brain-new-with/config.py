"""
brain/config.py — Pydantic-settings configuration for the AI Brain service.

Loads all required environment variables at startup. If any required variable is
missing, logs a descriptive error and exits with code 1.
"""

from __future__ import annotations

import functools
import logging
import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All required environment variables for the AI Brain service."""

    GEMINI_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    PATIENT_NAME: str
    PATIENT_ID: str
    GLASSES_AUDIO_DEVICE: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    Exits with code 1 if any required environment variable is missing.
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
            logger.error("Configuration validation failed: %s", exc)
        sys.exit(1)
