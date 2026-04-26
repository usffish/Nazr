"""
services/brain/services/audio.py — Audio playback service.

Primary: pygame mixer (when available).
Fallback: macOS afplay via subprocess (handles pygame SDL2 conflicts on macOS).
Never writes audio to disk unless afplay fallback is active.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import io
import logging
import os
import platform
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_pygame_ok = False

try:
    import pygame
    _pygame_available = True
except ImportError:
    _pygame_available = False


def init_pygame(device: str) -> None:
    """Initialise pygame mixer, falling back to afplay on failure.

    Requirements: 5.2, 5.3, 5.4
    """
    global _pygame_ok
    if not _pygame_available:
        logger.warning("pygame not installed — audio will use afplay")
        return

    try:
        pygame.mixer.pre_init(devicename=device)
        pygame.mixer.init()
        _pygame_ok = True
        logger.info("Pygame mixer initialised with device: %s", device)
        return
    except Exception as exc:
        logger.warning(
            "Pygame mixer failed to init with device '%s': %s — trying default",
            device, exc,
        )

    try:
        pygame.mixer.init()
        _pygame_ok = True
        logger.warning("Pygame mixer initialised with default audio device (fallback)")
    except Exception as fallback_exc:
        logger.warning(
            "Pygame mixer unavailable: %s — audio will use afplay", fallback_exc
        )


def play_audio(buffer: io.BytesIO) -> None:
    """Play an in-memory audio buffer.

    Tries pygame mixer first; falls back to macOS afplay if unavailable.
    Never raises — all exceptions are caught and logged.

    Requirements: 5.1, 5.3, 5.5, 5.6
    """
    if _pygame_ok:
        try:
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            return
        except Exception as exc:
            logger.warning("Pygame playback failed: %s — trying afplay", exc)

    # macOS fallback via afplay
    if platform.system() == "Darwin":
        try:
            buffer.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(buffer.read())
                tmp_path = f.name
            subprocess.run(["afplay", tmp_path], check=False, capture_output=True)
            os.unlink(tmp_path)
        except Exception as exc:
            logger.error("afplay playback failed: %s", exc)
    else:
        logger.error("No audio backend available on this platform")
