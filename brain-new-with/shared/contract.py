"""
brain/shared/contract.py — Canonical Pydantic models for the AuraGuard AI JSON Contract.

This is the single source of truth for all data models shared across services.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class PersonProfile(BaseModel):
    name: str
    relationship: str
    background: str
    last_conversation: str


class IdentityMetadata(BaseModel):
    person_profile: PersonProfile


class HealthMetadata(BaseModel):
    detected_item: str


class Event(BaseModel):
    event_id: str
    timestamp: str
    patient_id: str
    type: Literal["health", "identity"]
    subtype: str
    confidence: float
    image_b64: str
    metadata: dict
    source: Literal["vision_engine_v1"]


class EventRecord(BaseModel):
    event_id: str
    timestamp: str
    patient_id: str
    type: str
    subtype: str
    confidence: float
    # NOTE: image_b64 intentionally excluded — never stored in MongoDB
    metadata: dict
    source: str
    verified: bool
    voice_script: str
    processing_status: Literal["success", "partial_failure"]
    processed_at: str


class EventResponse(BaseModel):
    event_id: str
    status: Literal["processed", "error"]
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    reason: Optional[str] = None
