"""
brain/services/audio.py — Pygame audio playback service.

Handles device selection at startup and in-memory buffer playback.
Never writes audio to disk.
"""

from __future__ import annotations

import io
import logging

import pygame

logger = logging.getLogger(__name__)


def init_pygame(device: str) -> None:
    """Initialise the Pygame mixer targeting the given audio device.

    Falls back to default system audio if the target device fails.
    Falls back silently (logs error) if both fail.
    """
    if pygame.mixer.get_init():
        pygame.mixer.quit()

    try:
        pygame.mixer.pre_init(devicename=device)
        pygame.mixer.init()
        logger.warning("Pygame mixer initialised with device: %s", device)
    except Exception as exc:
        logger.warning(
            "Pygame mixer failed to init with device '%s': %s — falling back to default",
            device,
            exc,
        )
        try:
            pygame.mixer.pre_init()
            pygame.mixer.init()
            logger.warning("Pygame mixer initialised with default audio device (fallback)")
        except Exception as fallback_exc:
            logger.error("Pygame mixer fallback init also failed: %s", fallback_exc)


def play_audio(buffer: io.BytesIO) -> None:
    """Play an in-memory audio buffer through the Pygame mixer.

    Blocks until playback is complete. Never raises.
    """
    try:
        pygame.mixer.music.load(buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    except Exception as exc:
        logger.error("Pygame playback failed: %s", exc)
