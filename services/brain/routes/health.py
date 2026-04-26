"""
services/brain/routes/health.py — GET /health route.

Returns HTTP 200 {"status": "ok"} when MongoDB is reachable,
HTTP 503 {"status": "degraded", "reason": "mongodb_unreachable"} otherwise.

Requirements: 7.4, 7.5
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.brain.models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> JSONResponse:
    """Check service health by pinging MongoDB.

    Requirements:
    - 7.4: Return HTTP 200 {"status": "ok"} when MongoDB is reachable.
    - 7.5: Return HTTP 503 {"status": "degraded", "reason": "mongodb_unreachable"} otherwise.
    """
    motor_client = request.app.state.motor_client
    try:
        await asyncio.wait_for(
            motor_client.admin.command("ping"),
            timeout=3.0,
        )
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception as exc:
        logger.warning("Health check: MongoDB unreachable — %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "reason": "mongodb_unreachable"},
        )
