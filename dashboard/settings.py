import logging
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Resolve .env from project root regardless of working directory
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class DashboardSettings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    PATIENT_NAME: str

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")


def get_settings() -> DashboardSettings:
    try:
        return DashboardSettings()
    except Exception as e:
        logger.error("Missing required environment variables: %s", e)
        sys.exit(1)
