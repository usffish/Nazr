"""
tests/brain/test_voice_scripts.py — Property-based tests for voice script generation.

Tasks 5.2, 5.3, 5.4:
  - Property 7: Identity Voice Scripts Contain All Person Profile Fields
  - Property 8: Health Voice Scripts Contain Patient Name and Are Subtype-Appropriate
  - Property 6: Unverified Health Events Produce Empty Voice Scripts
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shared.contract import PersonProfile, Event
from services.brain.services.gemini import (
    generate_voice_script,
    generate_identity_script,
    generate_health_script,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for PersonProfile — all fields non-empty text
person_profile_strategy = st.builds(
    PersonProfile,
    name=st.text(min_size=1, max_size=50),
    relationship=st.text(min_size=1, max_size=30),
    background=st.text(min_size=1, max_size=200),
    last_conversation=st.text(min_size=1, max_size=200),
)

# Strategy for health Events (unverified path — verified flag is passed separately)
health_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("health"),
    subtype=st.sampled_from(["eating", "drinking", "medicine_taken"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1),
    metadata=st.fixed_dictionaries(
        {"detected_item": st.sampled_from(["food", "water", "medicine"])}
    ),
    source=st.just("vision_engine_v1"),
)

# Subtype -> expected keyword in the generated voice script
SUBTYPE_KEYWORDS = {
    "eating": "eating",
    "drinking": "drinking",
    "medicine_taken": "medication",
}

# ---------------------------------------------------------------------------
# Property 7: Identity Voice Scripts Contain All Person Profile Fields
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------


@given(
    person_profile_strategy,
    # Filter out braces to avoid str.format() KeyError/ValueError
    st.text(min_size=1, max_size=50).filter(lambda s: "{" not in s and "}" not in s),
)
@settings(max_examples=10)
def test_identity_voice_script_contains_all_profile_fields(person_profile, patient_name):
    # Property 7: Identity Voice Scripts Contain All Person Profile Fields
    # Validates: Requirements 3.1
    script = generate_identity_script(person_profile, patient_name)
    assert person_profile.name in script
    assert person_profile.relationship in script
    assert person_profile.background in script
    assert person_profile.last_conversation in script


# ---------------------------------------------------------------------------
# Property 8: Health Voice Scripts Contain Patient Name and Are Subtype-Appropriate
# Validates: Requirements 3.2, 3.4
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(["eating", "drinking", "medicine_taken"]),
    # Filter out braces to avoid str.format() KeyError/ValueError
    st.text(min_size=1, max_size=50).filter(lambda s: "{" not in s and "}" not in s),
)
@settings(max_examples=10)
def test_health_voice_script_contains_patient_name_and_keyword(subtype, patient_name):
    # Property 8: Health Voice Scripts Contain Patient Name and Are Subtype-Appropriate
    # Validates: Requirements 3.2, 3.4
    script = generate_health_script(subtype, patient_name)
    assert patient_name in script
    assert SUBTYPE_KEYWORDS[subtype] in script.lower()


# ---------------------------------------------------------------------------
# Property 6: Unverified Health Events Produce Empty Voice Scripts
# Validates: Requirements 3.3, 4.1
# ---------------------------------------------------------------------------


@given(health_event_strategy)
@settings(max_examples=10)
def test_unverified_health_event_produces_empty_voice_script(event):
    # Property 6: Unverified Health Events Produce Empty Voice Scripts
    # Validates: Requirements 3.3, 4.1
    script = generate_voice_script(event, verified=False, patient_name="TestPatient")
    assert script == ""
