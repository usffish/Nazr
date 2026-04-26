"""
tests/brain/test_gemini.py — Property and unit tests for Gemini service functions.

Tasks 4.2 and 4.3:
  - Property 5: Gemini Response Parsing Is Deterministic
  - Property 4: Health Event Gemini Prompts Are Subtype-Specific
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from services.brain.services.gemini import build_verification_prompt, parse_gemini_verified


# ---------------------------------------------------------------------------
# Property 5: Gemini Response Parsing Is Deterministic
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@given(st.text(min_size=0, max_size=100).map(lambda s: "YES" + s))
@settings(max_examples=10)
def test_parse_gemini_verified_yes_prepended(response_text):
    # Property 5: Gemini Response Parsing Is Deterministic
    # Validates: Requirements 2.3
    # Strings constructed by prepending "YES" — parse_gemini_verified must return True
    assert parse_gemini_verified(response_text) is True


@given(
    st.sampled_from(["yes", "Yes", "yEs", "yeS", "YeS", "yES", "YES"]).flatmap(
        lambda prefix: st.text(min_size=0, max_size=100).map(lambda suffix: prefix + suffix)
    )
)
@settings(max_examples=10)
def test_parse_gemini_verified_yes_case_insensitive(response_text):
    # Property 5: Gemini Response Parsing Is Deterministic
    # Validates: Requirements 2.3
    # Strings starting with YES in any case combination — must return True
    assert parse_gemini_verified(response_text) is True


@given(
    st.text(min_size=0, max_size=100).filter(
        lambda s: not s.strip().upper().startswith("YES")
        and "YES," not in s.upper()
        and "YES." not in s.upper()
    )
)
@settings(max_examples=10)
def test_parse_gemini_verified_no(response_text):
    # Property 5: Gemini Response Parsing Is Deterministic
    # Validates: Requirements 2.3
    # Strings that do NOT start with YES and don't contain YES, or YES. — must return False
    assert parse_gemini_verified(response_text) is False


# ---------------------------------------------------------------------------
# Property 4: Health Event Gemini Prompts Are Subtype-Specific
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subtype,expected_keywords,excluded_keywords", [
    (
        "eating",
        ["eating", "food"],
        ["beverage", "medication", "medicine"],
    ),
    (
        "drinking",
        ["drinking", "beverage"],
        ["eating", "food", "medication", "medicine"],
    ),
    (
        "medicine_taken",
        ["medication", "medicine"],
        ["eating", "food", "beverage"],
    ),
])
def test_build_verification_prompt_subtype_specific(subtype, expected_keywords, excluded_keywords):
    # Property 4: Health Event Gemini Prompts Are Subtype-Specific
    # Validates: Requirements 2.2
    prompt = build_verification_prompt(subtype).lower()
    assert any(kw in prompt for kw in expected_keywords), (
        f"Prompt for subtype '{subtype}' missing expected keywords {expected_keywords}. "
        f"Got: {prompt!r}"
    )
    assert not any(kw in prompt for kw in excluded_keywords), (
        f"Prompt for subtype '{subtype}' contains excluded keywords {excluded_keywords}. "
        f"Got: {prompt!r}"
    )
