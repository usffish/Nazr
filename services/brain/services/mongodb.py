"""
services/brain/services/mongodb.py — Motor async MongoDB service.

Handles client initialisation, connectivity verification, and Event_Record writes.
All write operations use asyncio.wait_for with a 5-second timeout.
image_b64 is never stored — excluded at the EventRecord model level.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""
from __future__ import annotations

import asyncio
import logging

import certifi
from motor.motor_asyncio import AsyncIOMotorClient

from services.brain.models import EventRecord

logger = logging.getLogger(__name__)


def init_motor(uri: str) -> AsyncIOMotorClient:
    """Create and return a Motor async MongoDB client.

    Requirement 6.3: use Motor async driver for all MongoDB operations.
    """
    return AsyncIOMotorClient(uri, tlsCAFile=certifi.where())


async def verify_mongodb(client: AsyncIOMotorClient) -> bool:
    """Ping the MongoDB admin database to verify connectivity.

    Returns True on success, False on any exception (logs WARNING).
    Used at startup — failure is non-fatal (degraded start allowed).
    """
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
    """Write an EventRecord to MongoDB Atlas.

    Uses asyncio.wait_for with a 5-second timeout. Returns True on success,
    False on timeout or any exception (logs ERROR).

    image_b64 is never present in EventRecord.model_dump() — excluded at
    the model level per Requirement 6.6.

    Requirements: 6.1, 6.2, 6.4, 6.5, 6.6
    """
    try:
        document = record.model_dump()
        # Defensive check — image_b64 must never reach MongoDB
        assert "image_b64" not in document, "image_b64 must not be in EventRecord"

        await asyncio.wait_for(
            client[db][collection].insert_one(document),
            timeout=5.0,
        )
        logger.info("EventRecord written to MongoDB (event_id=%s)", record.event_id)
        return True
    except asyncio.TimeoutError:
        logger.error(
            "MongoDB write timed out after 5s (event_id=%s)", record.event_id
        )
        return False
    except Exception as exc:
        logger.error(
            "MongoDB write failed (event_id=%s): %s", record.event_id, exc
        )
        return False
