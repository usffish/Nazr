"""
brain/models.py — Re-export shim for shared contract models.
"""

from __future__ import annotations

from brain.shared.contract import (
    Event,
    EventRecord,
    EventResponse,
    HealthResponse,
    HealthMetadata,
    IdentityMetadata,
    PersonProfile,
)

__all__ = [
    "Event",
    "EventRecord",
    "EventResponse",
    "HealthResponse",
    "PersonProfile",
    "IdentityMetadata",
    "HealthMetadata",
]
