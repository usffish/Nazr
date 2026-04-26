"""
services/brain/models.py — Re-export shim for shared contract models.

All Brain service modules import from here rather than directly from
shared.contract, keeping intra-service imports clean and consistent.

Requirements: 16.2, 16.3
"""

from __future__ import annotations

from shared.contract import (
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
