"""
services/brain/main.py — FastAPI application entry point for the AI Brain service.

Wires together the lifespan startup/shutdown sequence, global exception handler,
and route mounts. All downstream service initialisation happens in the lifespan
so that app.state is populated before any request is handled.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 8.5
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.brain.config import get_settings
from services.brain.services.audio import init_pygame
from services.brain.services.mongodb import init_motor, verify_mongodb

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    Startup sequence (Requirements 9.1, 9.2, 9.3):
    1. Load validated settings — exits with code 1 if any env var is missing.
    2. Initialise Motor MongoDB client.
    3. Verify MongoDB connectivity — log WARNING on failure, do NOT exit.
    4. Initialise Pygame mixer — log WARNING on failure, do NOT exit.
    5. Construct ElevenLabs client.
    6. Store all clients and settings on app.state.

    Shutdown (Requirement 9.4):
    - Close the Motor MongoDB client cleanly.
    """
    # 1. Load settings
    settings = get_settings()

    # 2. Initialise Motor MongoDB client
    motor_client = init_motor(settings.MONGODB_URI)

    # 3. Verify MongoDB connectivity (degraded start allowed)
    try:
        connected = await verify_mongodb(motor_client)
        if not connected:
            logger.warning(
                "MongoDB connectivity check failed at startup — "
                "continuing in degraded state."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MongoDB connectivity check raised an exception at startup: %s — "
            "continuing in degraded state.",
            exc,
        )

    # 4. Initialise Pygame mixer (degraded start allowed)
    try:
        init_pygame(settings.GLASSES_AUDIO_DEVICE)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Pygame initialisation failed at startup: %s — "
            "continuing in degraded state.",
            exc,
        )

    # 5. Construct ElevenLabs client
    from elevenlabs import ElevenLabs  # type: ignore

    elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    # 6. Store on app.state
    app.state.motor_client = motor_client
    app.state.elevenlabs_client = elevenlabs_client
    app.state.settings = settings

    logger.info("AI Brain startup complete.")

    yield  # application runs

    # Shutdown
    motor_client.close()
    logger.info("AI Brain shutdown complete.")


# FastAPI application

app = FastAPI(
    title="AuraGuard AI Brain",
    description="Central coordinator for the AuraGuard AI assistive platform.",
    version="1.0.0",
    lifespan=lifespan,
)


# Global exception handler (Requirement 8.5)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions, log the full stack trace, and return HTTP 500.

    The response body always includes the event_id from request.state when
    available, falling back to "unknown" for requests that never set it.

    Requirement 8.5: IF an unhandled exception occurs, THE Brain SHALL catch it,
    log the full stack trace, and return HTTP 500 with a structured JSON body.
    """
    event_id = getattr(request.state, "event_id", "unknown")
    logger.exception(
        "Unhandled exception during request processing (event_id=%s)", event_id
    )
    return JSONResponse(
        status_code=500,
        content={
            "event_id": event_id,
            "status": "error",
            "message": "Internal server error.",
        },
    )


# Route mounts

from services.brain.routes.event import router as event_router  # noqa: E402
from services.brain.routes.health import router as health_router  # noqa: E402

app.include_router(event_router)
app.include_router(health_router)


# ── Admin / test endpoints ────────────────────────────────────────────────────

@app.get("/status")
async def service_status(request: Request) -> JSONResponse:
    """Detailed health status for the admin dashboard."""
    from services.brain.services.mongodb import verify_mongodb
    settings = request.app.state.settings
    motor_client = request.app.state.motor_client

    mongo_ok = await verify_mongodb(motor_client)

    try:
        event_count = await motor_client[settings.MONGODB_DB][settings.MONGODB_COLLECTION].count_documents({})
    except Exception:
        event_count = -1

    return JSONResponse({
        "status": "ok",
        "mongodb": "connected" if mongo_ok else "disconnected",
        "event_count": event_count,
        "patient_name": settings.PATIENT_NAME,
        "patient_id": settings.PATIENT_ID,
    })


@app.post("/test/voice")
async def test_voice(request: Request) -> JSONResponse:
    """Synthesize and play a test TTS message to verify audio is working."""
    import asyncio
    from services.brain.services.elevenlabs import synthesize_audio
    from services.brain.services.audio import play_audio

    settings = request.app.state.settings
    elevenlabs_client = request.app.state.elevenlabs_client
    script = (
        f"AuraGuard is online. Hello {settings.PATIENT_NAME}, "
        "all systems are working correctly."
    )

    buffer = await synthesize_audio(script, settings.ELEVENLABS_VOICE_ID, elevenlabs_client)
    if buffer is None:
        return JSONResponse({"status": "error", "message": "TTS synthesis failed"}, status_code=500)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, play_audio, buffer)
    return JSONResponse({"status": "ok", "script": script})


@app.post("/test/event")
async def test_event(request: Request) -> JSONResponse:
    """Inject a synthetic event to test the full processing pipeline."""
    import uuid
    from datetime import datetime, timezone
    from shared.contract import Event
    from services.brain.routes.event import handle_event as process_event

    body = await request.json()
    event_type = body.get("type", "identity")
    subtype = body.get("subtype", "face_recognized")

    if event_type == "identity":
        metadata = {
            "person_profile": {
                "name": body.get("name", "Test Person"),
                "relationship": body.get("relationship", "friend"),
                "background": "Admin panel test injection",
                "last_conversation": "tested the admin panel just now",
            }
        }
    else:
        metadata = {"detected_item": subtype.title()}

    event = Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        patient_id=request.app.state.settings.PATIENT_ID,
        type=event_type,
        subtype=subtype,
        confidence=1.0,
        image_b64="",
        metadata=metadata,
        source="vision_engine_v1",
    )
    return await process_event(event, request)
