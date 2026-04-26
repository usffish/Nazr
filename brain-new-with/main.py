"""
brain/main.py — FastAPI application entry point for the AI Brain service.

Launch with:
    uvicorn brain.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from brain.config import get_settings
from brain.services.audio import init_pygame
from brain.services.mongodb import init_motor, verify_mongodb

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown sequence.

    Startup:
    1. Load validated settings — exits with code 1 if any env var is missing.
    2. Initialise Motor MongoDB client.
    3. Verify MongoDB connectivity — log WARNING on failure, do NOT exit.
    4. Initialise Pygame mixer — log WARNING on failure, do NOT exit.
    5. Construct ElevenLabs client.
    6. Store all clients and settings on app.state.

    Shutdown:
    - Close the Motor MongoDB client cleanly.
    """
    settings = get_settings()

    motor_client = init_motor(settings.MONGODB_URI)

    try:
        connected = await verify_mongodb(motor_client)
        if not connected:
            logger.warning("MongoDB unreachable at startup — continuing in degraded state.")
    except Exception as exc:
        logger.warning("MongoDB check raised at startup: %s — continuing in degraded state.", exc)

    try:
        init_pygame(settings.GLASSES_AUDIO_DEVICE)
    except Exception as exc:
        logger.warning("Pygame init failed at startup: %s — continuing in degraded state.", exc)

    from elevenlabs import ElevenLabs  # type: ignore
    elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    app.state.motor_client = motor_client
    app.state.elevenlabs_client = elevenlabs_client
    app.state.settings = settings

    logger.info("AI Brain startup complete.")

    yield

    motor_client.close()
    logger.info("AI Brain shutdown complete.")


app = FastAPI(
    title="AuraGuard AI Brain",
    description="Central coordinator for the AuraGuard AI assistive platform.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions, log the stack trace, return HTTP 500."""
    event_id = getattr(request.state, "event_id", "unknown")
    logger.exception("Unhandled exception (event_id=%s)", event_id)
    return JSONResponse(
        status_code=500,
        content={"event_id": event_id, "status": "error", "message": "Internal server error."},
    )


from brain.routes.event import router as event_router  # noqa: E402
from brain.routes.health import router as health_router  # noqa: E402

app.include_router(event_router)
app.include_router(health_router)
