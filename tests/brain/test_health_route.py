"""
tests/brain/test_health_route.py — Unit tests for GET /health endpoint.

Task 11.2:
  - HTTP 200 {"status": "ok"} when Motor ping succeeds
  - HTTP 503 {"status": "degraded", "reason": "mongodb_unreachable"} when Motor ping raises

Requirements: 7.4, 7.5
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Health endpoint — MongoDB reachable
# ---------------------------------------------------------------------------

def test_health_returns_200_when_mongodb_reachable(client):
    """GET /health returns HTTP 200 {"status": "ok"} when MongoDB ping succeeds.

    Requirement 7.4
    """
    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(return_value={"ok": 1})
    client.app.state.motor_client = mock_motor

    response = client.get("/health")

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    assert response.json() == {"status": "ok"}, (
        f"Expected {{\"status\": \"ok\"}}, got {response.json()}"
    )


# ---------------------------------------------------------------------------
# Health endpoint — MongoDB unreachable
# ---------------------------------------------------------------------------

def test_health_returns_503_when_mongodb_unreachable(client):
    """GET /health returns HTTP 503 with degraded status when MongoDB ping raises.

    Requirement 7.5
    """
    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(side_effect=Exception("connection refused"))
    client.app.state.motor_client = mock_motor

    response = client.get("/health")

    assert response.status_code == 503, (
        f"Expected HTTP 503, got {response.status_code}. Body: {response.text}"
    )
    body = response.json()
    assert body["status"] == "degraded", (
        f"Expected status='degraded', got {body.get('status')!r}"
    )
    assert body["reason"] == "mongodb_unreachable", (
        f"Expected reason='mongodb_unreachable', got {body.get('reason')!r}"
    )


# ---------------------------------------------------------------------------
# Health endpoint — timeout treated as unreachable
# ---------------------------------------------------------------------------

def test_health_returns_503_on_mongodb_timeout(client):
    """GET /health returns HTTP 503 when MongoDB ping times out.

    Requirement 7.5 (timeout is treated as unreachable)
    """
    import asyncio

    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(side_effect=asyncio.TimeoutError())
    client.app.state.motor_client = mock_motor

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
