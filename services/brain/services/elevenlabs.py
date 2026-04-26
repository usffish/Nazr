"""
services/brain/services/elevenlabs.py — ElevenLabs streaming TTS service.

Streams audio from ElevenLabs into an in-memory BytesIO buffer.
No files are written to disk at any point (Requirement 4.2, 5.1).

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elevenlabs import ElevenLabs

logger = logging.getLogger(__name__)


def _collect_audio_chunks(voice_script: str, voice_id: str, client) -> io.BytesIO:
    """Synchronous helper: call ElevenLabs and collect all chunks into a BytesIO buffer.

    The ElevenLabs SDK is synchronous, so this runs in a thread executor.
    No temp files are written — all audio bytes go directly into the buffer.
    """
    audio_stream = client.text_to_speech.convert(
        text=voice_script,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
    )
    buffer = io.BytesIO()
    for chunk in audio_stream:
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    return buffer


async def synthesize_audio(
    voice_script: str,
    voice_id: str,
    client,
) -> io.BytesIO | None:
    """Stream TTS audio from ElevenLabs into an in-memory buffer.

    Uses asyncio.wait_for with a 15-second timeout. Returns None on any
    failure (timeout or exception) — callers must handle None gracefully.

    Requirements: 4.1, 4.3, 4.4, 4.5
    """
    try:
        loop = asyncio.get_event_loop()
        buffer = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _collect_audio_chunks(voice_script, voice_id, client),
            ),
            timeout=15.0,
        )
        return buffer
    except asyncio.TimeoutError:
        logger.error("ElevenLabs synthesis timed out after 15s")
        return None
    except Exception as exc:
        logger.error("ElevenLabs synthesis failed: %s", exc)
        return None
