"""
services/brain/services/gemini.py — Gemini verification and voice script generation.

Gemini verification implemented in Task 4.1.
Voice script generation implemented in Task 5.1.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import PIL.Image

from services.brain.models import Event, PersonProfile

logger = logging.getLogger(__name__)

# Voice script templates — defined here so they can be imported by tests
# even before the full implementation is in place.
IDENTITY_TEMPLATE: str = (
    "{patient_name}, {person_name} is here. "
    "{person_name} is your {relationship}. "
    "{background} "
    "Last time you spoke, {last_conversation}"
)

HEALTH_TEMPLATES: dict[str, str] = {
    "drinking": "Good job, {patient_name}. I can see you are drinking water. Stay hydrated.",
    "eating": "{patient_name}, I can see you are eating. Enjoy your meal.",
    "medicine_taken": "{patient_name}, I see you are taking your medication. Well done.",
}

# Subtype-specific verification prompts
_VERIFICATION_PROMPTS: dict[str, str] = {
    "eating": "Is the person in this image eating food? Answer YES or NO and explain briefly.",
    "drinking": "Is the person in this image drinking water or a beverage? Answer YES or NO and explain briefly.",
    "medicine_taken": "Is the person in this image taking medication? Answer YES or NO and explain briefly.",
}


def build_verification_prompt(subtype: str) -> str:
    """Return a subtype-specific YES/NO verification prompt for Gemini.

    Returns a known prompt for eating, drinking, and medicine_taken subtypes.
    Falls back to a generic prompt for any other subtype.
    """
    if subtype in _VERIFICATION_PROMPTS:
        return _VERIFICATION_PROMPTS[subtype]
    return (
        f"Is the person in this image performing the activity: {subtype}? "
        "Answer YES or NO and explain briefly."
    )


def parse_gemini_verified(response_text: str) -> bool:
    """Parse a Gemini response string into a boolean verification result.

    Returns True if the normalized response starts with "YES" or contains
    "YES," or "YES." — matching the design doc specification exactly.
    """
    normalized = response_text.strip().upper()
    return normalized.startswith("YES") or "YES," in normalized or "YES." in normalized


async def _call_gemini(image_b64: str, prompt: str, api_key: str) -> str:
    """Call the Gemini multimodal API with an image and prompt.

    Runs the synchronous google-generativeai SDK in a thread executor to
    avoid blocking the FastAPI event loop.
    """
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    image_data = base64.b64decode(image_b64)
    image = PIL.Image.open(io.BytesIO(image_data))

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content([prompt, image]),
    )
    return response.text


async def verify_health_event(image_b64: str, subtype: str, api_key: str) -> bool:
    """Call Gemini to verify a health event from a base64-encoded image.

    Uses a 10-second timeout via asyncio.wait_for. Returns True if Gemini
    confirms the activity, False on timeout or any exception (logs ERROR).
    """
    prompt = build_verification_prompt(subtype)
    try:
        response_text = await asyncio.wait_for(
            _call_gemini(image_b64, prompt, api_key),
            timeout=10.0,
        )
        return parse_gemini_verified(response_text)
    except asyncio.TimeoutError:
        logger.error("Gemini call timed out after 10s")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini call failed: %s", exc)
        return False


def generate_identity_script(person_profile: PersonProfile, patient_name: str) -> str:
    """Generate a voice script for an identity event using IDENTITY_TEMPLATE.

    Requirement 3.1: script must include patient_name, person name, relationship,
    background, and last_conversation.
    """
    return IDENTITY_TEMPLATE.format(
        patient_name=patient_name,
        person_name=person_profile.name,
        relationship=person_profile.relationship,
        background=person_profile.background,
        last_conversation=person_profile.last_conversation,
    )


def generate_health_script(subtype: str, patient_name: str) -> str:
    """Generate a voice script for a verified health event.

    Returns "" for unknown subtypes (Requirement 3.2, 3.4).
    """
    template = HEALTH_TEMPLATES.get(subtype, "")
    if not template:
        return ""
    return template.format(patient_name=patient_name)


def generate_voice_script(event: Event, verified: bool, patient_name: str) -> str:
    """Generate the appropriate voice script for an event.

    - type="identity" → generate_identity_script (always, regardless of verified)
    - type="health" and verified=True → generate_health_script
    - type="health" and verified=False → return "" (Requirement 3.3)
    """
    if event.type == "identity":
        profile_data = event.metadata.get("person_profile", {})
        person_profile = PersonProfile(**profile_data)
        return generate_identity_script(person_profile, patient_name)
    elif event.type == "health":
        if verified:
            return generate_health_script(event.subtype, patient_name)
        else:
            return ""
    return ""
