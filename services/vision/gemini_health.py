"""
GeminiHealthClient — Secondary Pass for Hybrid Health Detection
---------------------------------------------------------------
Handles the Gemini 1.5 Flash call for the Secondary Pass of the two-pass
health detection pipeline. Builds subtype-specific prompts, calls the API
with a 10-second timeout, and parses the confidence score from the response.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 2.8
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ConfidenceResult:
    """Result returned by the Secondary Pass (Gemini 1.5 Flash)."""

    score: float
    """Parsed confidence score in [0.0, 1.0]."""

    raw_text: str
    """Raw Gemini response text, preserved for logging and debugging."""

    subtype: str
    """The health subtype that was queried (drinking, eating, medicine_taken)."""


# ── Prompt templates ──────────────────────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "drinking": (
        "Is the person in this image drinking water or a beverage? "
        "Return ONLY a confidence score between 0.0 and 1.0 (e.g. 0.85). No other text."
    ),
    "eating": (
        "Is the person in this image eating food? "
        "Return ONLY a confidence score between 0.0 and 1.0 (e.g. 0.85). No other text."
    ),
    "medicine_taken": (
        "Is the person in this image taking medication, pills, or medicine? "
        "Return ONLY a confidence score between 0.0 and 1.0 (e.g. 0.85). No other text."
    ),
}

# Regex: match a decimal float first, then a bare integer
_SCORE_RE = re.compile(r"\d+\.\d+|\d+")


# ── Public helpers ────────────────────────────────────────────────────────────


def build_health_prompt(subtype: str) -> str:
    """
    Return the subtype-specific prompt for the Secondary Pass.

    Parameters
    ----------
    subtype:
        One of ``"drinking"``, ``"eating"``, or ``"medicine_taken"``.

    Returns
    -------
    str
        The prompt string to send to Gemini.

    Raises
    ------
    KeyError
        If *subtype* is not one of the three recognised values.
    """
    return _PROMPTS[subtype]


def parse_confidence_score(response_text: str) -> float:
    """
    Extract and clamp the first numeric value from *response_text*.

    The function searches for the first occurrence of a decimal float
    (``\\d+\\.\\d+``) or a bare integer (``\\d+``) in the response.  The
    matched value is clamped to the closed interval [0.0, 1.0].

    Parameters
    ----------
    response_text:
        Raw text returned by Gemini.

    Returns
    -------
    float
        Parsed score in [0.0, 1.0], or ``0.0`` if no numeric value is found.
    """
    match = _SCORE_RE.search(response_text)
    if match is None:
        log.warning(
            "parse_confidence_score: no numeric value found in response %r — returning 0.0",
            response_text,
        )
        return 0.0

    raw_value = float(match.group())
    clamped = max(0.0, min(1.0, raw_value))
    return clamped


# ── Secondary Pass call ───────────────────────────────────────────────────────


def call_gemini_health(
    frame_b64: str,
    subtype: str,
    api_key: str,
    timeout: float = 10.0,
) -> ConfidenceResult | None:
    """
    Call Gemini 1.5 Flash with a subtype-specific prompt and return a
    :class:`ConfidenceResult`.

    This function is synchronous and safe to call from a daemon thread (the
    existing ``_health_worker`` thread in ``face_recognition_engine.py``).
    The Gemini call is executed inside a :class:`~concurrent.futures.ThreadPoolExecutor`
    so that a hard timeout can be enforced without blocking the caller thread
    indefinitely.

    Parameters
    ----------
    frame_b64:
        Base-64-encoded JPEG image of the frame to classify.
    subtype:
        One of ``"drinking"``, ``"eating"``, or ``"medicine_taken"``.
    api_key:
        Gemini API key.
    timeout:
        Maximum seconds to wait for the Gemini response (default 10.0).

    Returns
    -------
    ConfidenceResult
        On success: parsed score, raw response text, and subtype.
    None
        On timeout or unrecoverable error.
    """
    import google.generativeai as genai  # imported here to keep module importable without the package

    prompt = build_health_prompt(subtype)
    log.info(
        "Secondary pass — subtype=%r, prompt=%r",
        subtype,
        prompt,
    )

    def _do_call() -> ConfidenceResult:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
            prompt,
        ]
        response = model.generate_content(parts)
        raw_text: str = response.text
        score = parse_confidence_score(raw_text)
        log.info(
            "Secondary pass result — subtype=%r, raw_text=%r, score=%.4f",
            subtype,
            raw_text,
            score,
        )
        return ConfidenceResult(score=score, raw_text=raw_text, subtype=subtype)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_call)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            log.error(
                "Secondary pass timed out after %ss for subtype=%s",
                timeout,
                subtype,
            )
            future.cancel()
            return None
        except Exception as exc:
            log.error(
                "Secondary pass raised an unexpected error for subtype=%s: %s",
                subtype,
                exc,
            )
            return None
