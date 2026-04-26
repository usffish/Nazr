"""
brain/services/mongodb.py — Motor async MongoDB service.

Handles client initialisation, connectivity verification, and EventRecord writes.
image_b64 is never stored — excluded at the EventRecord model level.
"""
from __future__ import annotations

import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient

from brain.models import EventRecord

logger = logging.getLogger(__name__)


def init_motor(uri: str) -> AsyncIOMotorClient:
    """Create and return a Motor async MongoDB client."""
    return AsyncIOMotorClient(uri)


async def verify_mongodb(client: AsyncIOMotorClient) -> bool:
    """Ping MongoDB to verify connectivity. Returns False on any failure."""
    try:
        await client.admin.command("ping")
        logger.info("MongoDB connectivity verified.")
        return True
    except Exception as exc:
        logger.warning("MongoDB connectivity check failed: %s", exc)
        return False


async def write_event_record(
    record: EventRecord,
    client: AsyncIOMotorClient,
    db: str,
    collection: str,
) -> bool:
    """Write an EventRecord to MongoDB with a 5-second timeout.

    Returns True on success, False on timeout or any failure.
    """
    try:
        document = record.model_dump()
        assert "image_b64" not in document, "image_b64 must not be in EventRecord"

        await asyncio.wait_for(
            client[db][collection].insert_one(document),
            timeout=5.0,
        )
        logger.info("EventRecord written to MongoDB (event_id=%s)", record.event_id)
        return True
    except asyncio.TimeoutError:
        logger.error("MongoDB write timed out after 5s (event_id=%s)", record.event_id)
        return False
    except Exception as exc:
        logger.error("MongoDB write failed (event_id=%s): %s", record.event_id, exc)
        return False
