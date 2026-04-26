"""
brain/services/gemini.py — Gemini verification and voice script generation.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import PIL.Image

from brain.models import Event, PersonProfile

logger = logging.getLogger(__name__)

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

_VERIFICATION_PROMPTS: dict[str, str] = {
    "eating": "Is the person in this image eating food? Answer YES or NO and explain briefly.",
    "drinking": "Is the person in this image drinking water or a beverage? Answer YES or NO and explain briefly.",
    "medicine_taken": "Is the person in this image taking medication? Answer YES or NO and explain briefly.",
}


def build_verification_prompt(subtype: str) -> str:
    if subtype in _VERIFICATION_PROMPTS:
        return _VERIFICATION_PROMPTS[subtype]
    return (
        f"Is the person in this image performing the activity: {subtype}? "
        "Answer YES or NO and explain briefly."
    )


def parse_gemini_verified(response_text: str) -> bool:
    normalized = response_text.strip().upper()
    return normalized.startswith("YES") or "YES," in normalized or "YES." in normalized


async def _call_gemini(image_b64: str, prompt: str, api_key: str) -> str:
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
    """Call Gemini to verify a health event. Returns False on timeout or error."""
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
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        return False


def generate_identity_script(person_profile: PersonProfile, patient_name: str) -> str:
    return IDENTITY_TEMPLATE.format(
        patient_name=patient_name,
        person_name=person_profile.name,
        relationship=person_profile.relationship,
        background=person_profile.background,
        last_conversation=person_profile.last_conversation,
    )


def generate_health_script(subtype: str, patient_name: str) -> str:
    template = HEALTH_TEMPLATES.get(subtype, "")
    if not template:
        return ""
    return template.format(patient_name=patient_name)


def generate_voice_script(event: Event, verified: bool, patient_name: str) -> str:
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
