"""
brain/routes/event.py — POST /event route.

Orchestrates the full Brain processing pipeline:
  1. Gemini verification (health events only)
  2. Voice script generation
  3. ElevenLabs TTS synthesis
  4. Pygame audio playback
  5. MongoDB EventRecord write

All downstream failures are absorbed — the route always returns HTTP 200.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from brain.models import Event, EventRecord, EventResponse
from brain.services.gemini import verify_health_event, generate_voice_script
from brain.services.elevenlabs import synthesize_audio
from brain.services.audio import play_audio
from brain.services.mongodb import write_event_record

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/event", response_model=EventResponse)
async def handle_event(event: Event, request: Request) -> EventResponse:
    """Process an incoming Event from the Vision Engine."""
    request.state.event_id = event.event_id

    settings = request.app.state.settings
    motor_client = request.app.state.motor_client
    elevenlabs_client = request.app.state.elevenlabs_client

    any_failure = False

    # Step 1: Gemini verification (health only; identity always verified=True)
    verified = True
    if event.type == "health":
        try:
            verified = await verify_health_event(
                event.image_b64, event.subtype, settings.GEMINI_API_KEY
            )
        except Exception as exc:
            logger.error("Gemini verification raised unexpectedly: %s", exc)
            verified = False
            any_failure = True

    # Step 2: Voice script generation
    voice_script = ""
    try:
        voice_script = generate_voice_script(event, verified, settings.PATIENT_NAME)
    except Exception as exc:
        logger.error("Voice script generation failed: %s", exc)
        any_failure = True

    # Step 3: ElevenLabs synthesis (skip if no voice script)
    audio_buffer = None
    if voice_script:
        try:
            audio_buffer = await synthesize_audio(
                voice_script, settings.ELEVENLABS_VOICE_ID, elevenlabs_client
            )
            if audio_buffer is None:
                any_failure = True
        except Exception as exc:
            logger.error("ElevenLabs synthesis raised unexpectedly: %s", exc)
            any_failure = True

    # Step 4: Pygame playback (skip if synthesis returned None)
    if audio_buffer is not None:
        try:
            play_audio(audio_buffer)
        except Exception as exc:
            logger.error("Pygame playback raised unexpectedly: %s", exc)
            any_failure = True

    # Step 5: MongoDB write
    processing_status = "partial_failure" if any_failure else "success"
    record = EventRecord(
        event_id=event.event_id,
        timestamp=event.timestamp,
        patient_id=event.patient_id,
        type=event.type,
        subtype=event.subtype,
        confidence=event.confidence,
        metadata=event.metadata,
        source=event.source,
        verified=verified,
        voice_script=voice_script,
        processing_status=processing_status,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        write_success = await write_event_record(
            record, motor_client, settings.MONGODB_DB, settings.MONGODB_COLLECTION
        )
        if not write_success:
            any_failure = True
            processing_status = "partial_failure"
    except Exception as exc:
        logger.error("MongoDB write raised unexpectedly: %s", exc)
        any_failure = True
        processing_status = "partial_failure"

    message = "Event processed with partial failures." if any_failure else "Event processed successfully."

    return EventResponse(
        event_id=event.event_id,
        status="processed",
        message=message,
    )
