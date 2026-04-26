"""
tests/brain/test_event_record.py — Property test for EventRecord transformation.

Property 10: EventRecord Excludes image_b64 and Includes All Enrichment Fields
Validates: Requirements 6.2, 6.6, 16.5
"""
from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from shared.contract import Event, EventRecord


# Strategy for generating valid health Events
health_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("health"),
    subtype=st.sampled_from(["eating", "drinking", "medicine_taken"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1, max_size=100),  # always present in Event
    metadata=st.fixed_dictionaries(
        {"detected_item": st.sampled_from(["food", "water", "medicine"])}
    ),
    source=st.just("vision_engine_v1"),
)

# Strategy for generating valid identity Events
identity_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("identity"),
    subtype=st.just("face_recognized"),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1, max_size=100),  # always present in Event
    metadata=st.fixed_dictionaries(
        {
            "person_profile": st.fixed_dictionaries(
                {
                    "name": st.text(min_size=1, max_size=50),
                    "relationship": st.text(min_size=1, max_size=30),
                    "background": st.text(min_size=1, max_size=100),
                    "last_conversation": st.text(min_size=1, max_size=100),
                }
            )
        }
    ),
    source=st.just("vision_engine_v1"),
)

# Combined strategy covering both event types
event_strategy = st.one_of(health_event_strategy, identity_event_strategy)


def _make_event_record(event: Event) -> EventRecord:
    """Construct an EventRecord from an Event, adding enrichment fields.

    image_b64 is intentionally excluded — enforced at the model level per
    Requirement 6.6.
    """
    return EventRecord(
        event_id=event.event_id,
        timestamp=event.timestamp,
        patient_id=event.patient_id,
        type=event.type,
        subtype=event.subtype,
        confidence=event.confidence,
        # image_b64 intentionally excluded
        metadata=event.metadata,
        source=event.source,
        verified=True,
        voice_script="Test voice script.",
        processing_status="success",
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


@given(event_strategy)
@settings(max_examples=10)
def test_event_record_excludes_image_b64_and_includes_enrichment_fields(event: Event):
    # Property 10: EventRecord Excludes image_b64 and Includes All Enrichment Fields
    # Validates: Requirements 6.2, 6.6, 16.5

    # Confirm the source Event has image_b64
    assert event.image_b64, "Event must have image_b64 for this test to be meaningful"

    record = _make_event_record(event)
    document = record.model_dump()

    # image_b64 must be absent from the EventRecord document
    assert "image_b64" not in document, (
        f"image_b64 must not be present in EventRecord.model_dump(), "
        f"but found it. Keys: {list(document.keys())}"
    )

    # All four enrichment fields must be present
    assert "verified" in document, "EventRecord must contain 'verified'"
    assert "voice_script" in document, "EventRecord must contain 'voice_script'"
    assert "processing_status" in document, "EventRecord must contain 'processing_status'"
    assert "processed_at" in document, "EventRecord must contain 'processed_at'"

    # processing_status must be a valid value
    assert document["processing_status"] in ("success", "partial_failure"), (
        f"processing_status must be 'success' or 'partial_failure', "
        f"got {document['processing_status']!r}"
    )
