"""
brain/services/elevenlabs.py — ElevenLabs streaming TTS service.

Streams audio into an in-memory BytesIO buffer. No files written to disk.
"""
from __future__ import annotations

import asyncio
import io
import logging

logger = logging.getLogger(__name__)


def _collect_audio_chunks(voice_script: str, voice_id: str, client) -> io.BytesIO:
    """Synchronous helper: collect all TTS chunks into a BytesIO buffer."""
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

    Returns None on timeout or any failure.
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
