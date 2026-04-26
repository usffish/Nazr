"""
tests/brain/test_audio.py — Property and unit tests for audio services.

Task 7.2: Property 9 — ElevenLabs Streaming Produces Audio Without Temp Files
Task 8.2: Pygame fallback logic unit tests
"""
from __future__ import annotations

import asyncio
import glob
import io
import logging
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from services.brain.services.elevenlabs import synthesize_audio
from services.brain.services.audio import init_pygame, play_audio


# ---------------------------------------------------------------------------
# Property 9: ElevenLabs Streaming Produces Audio Without Temp Files
# Validates: Requirements 4.1, 5.1
# ---------------------------------------------------------------------------

def _make_mock_client(chunks: list[bytes]) -> MagicMock:
    """Build a mock ElevenLabs client whose convert() returns an iterator of chunks."""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.return_value = iter(chunks)
    return mock_client


@given(
    st.lists(st.binary(min_size=1, max_size=256), min_size=1, max_size=20)
)
@settings(max_examples=10)
def test_synthesize_audio_no_temp_files(chunks):
    # Property 9: ElevenLabs Streaming Produces Audio Without Temp Files
    # Validates: Requirements 4.1, 5.1

    # Assert no audio files exist before the call
    assert glob.glob("audio/*.mp3") == []
    assert glob.glob("audio/*.wav") == []

    mock_client = _make_mock_client(chunks)

    # Use a fresh event loop per hypothesis example to avoid loop reuse issues
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            synthesize_audio("Hello world", "test-voice-id", mock_client)
        )
    finally:
        loop.close()

    # Assert no audio files exist after the call
    assert glob.glob("audio/*.mp3") == []
    assert glob.glob("audio/*.wav") == []

    # Assert the result is a BytesIO buffer with the expected content
    assert result is not None
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    expected = b"".join(chunks)
    assert result.read() == expected


# ---------------------------------------------------------------------------
# Task 8.2: Pygame fallback logic unit tests
# Validates: Requirements 5.4, 5.6
# ---------------------------------------------------------------------------

def test_init_pygame_falls_back_to_default_on_device_failure(caplog):
    """When the target device raises on init, init_pygame falls back to default."""
    call_count = 0

    def mock_init_failing_first(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("device not found")
        # second call (fallback, no kwargs) succeeds silently

    with patch("services.brain.services.audio.pygame") as mock_pygame:
        mock_pygame.mixer.pre_init = MagicMock()
        mock_pygame.mixer.init.side_effect = mock_init_failing_first

        with caplog.at_level(logging.WARNING, logger="services.brain.services.audio"):
            init_pygame("NonExistentDevice")

        # Should have called init twice (first failed, second fallback)
        assert mock_pygame.mixer.init.call_count == 2
        # Should have logged a warning about the fallback
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_play_audio_catches_exceptions_and_does_not_reraise():
    """play_audio must catch all exceptions and not re-raise them."""
    buffer = io.BytesIO(b"fake audio data")

    with patch("services.brain.services.audio.pygame") as mock_pygame:
        mock_pygame.mixer.music.load.side_effect = Exception("playback error")

        # Should not raise
        play_audio(buffer)
