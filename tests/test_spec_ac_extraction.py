"""Root-cause regression for prompt defect #1: acceptance-criteria extraction.

The reconciler's parser only pulled ``Scenario:`` titles from ```gherkin
fenced blocks, but LLM-authored SPEC_REQUIREMENTS.md writes them as bold
``**Scenario: X**`` headers with Given/When/Then bullets and NO fence — so
every story's ACs were silently dropped, the story-state DB stored none, and
the patching scope block rendered "(none recorded)" (lumina 019fff37). These
pin the robust extractor across the real-world formats.
"""

from __future__ import annotations

from harness.spec_reconciler import (
    _extract_acceptance_criteria,
    parse_spec_requirements,
)


class TestExtractAcceptanceCriteria:
    def test_bold_scenario_headers_the_generator_emits(self):
        # The exact shape from lumina's SPEC_REQUIREMENTS.md.
        body = (
            "**Acceptance Criteria:**\n\n"
            "**Scenario: Add contact with valid details**\n"
            "- Given the Add Contact slide-over is open\n"
            "- When the user submits valid data\n"
            "- Then the contact is created\n\n"
            "**Scenario: Future date of birth is rejected**\n"
            "- Given the form is open\n"
            "- When a future DOB is entered\n"
            "- Then a validation error shows\n"
        )
        acs = _extract_acceptance_criteria(body)
        assert acs == [
            "Add contact with valid details",
            "Future date of birth is rejected",
        ]

    def test_plain_fenced_gherkin_still_works(self):
        body = (
            "```gherkin\n"
            "Scenario: something is true\n"
            "  Given x\n  When y\n  Then z\n"
            "```\n"
        )
        assert _extract_acceptance_criteria(body) == ["something is true"]

    def test_scenario_outline_and_heading_forms(self):
        body = (
            "### Scenario: heading form\n"
            "- Given a\n"
            "**Scenario Outline: outline form**\n"
            "- When b\n"
        )
        assert _extract_acceptance_criteria(body) == [
            "heading form", "outline form",
        ]

    def test_bullet_fallback_when_no_scenarios(self):
        body = (
            "**Acceptance Criteria:**\n"
            "- The dashboard lists contacts by next birthday\n"
            "- Age is shown for each contact\n\n"
            "**Notes:** irrelevant trailing section\n"
        )
        assert _extract_acceptance_criteria(body) == [
            "The dashboard lists contacts by next birthday",
            "Age is shown for each contact",
        ]

    def test_given_when_then_bullets_not_scraped_as_acs(self):
        # Given/When/Then live UNDER scenarios; the scenario titles are the
        # ACs, and the fallback must not fire when scenarios are present.
        body = (
            "**Scenario: only this is an AC**\n"
            "- Given the app is open\n"
            "- When something happens\n"
            "- Then an outcome holds\n"
        )
        assert _extract_acceptance_criteria(body) == ["only this is an AC"]

    def test_genuinely_empty_returns_empty(self):
        assert _extract_acceptance_criteria(
            "As a user, I want a thing, so that benefit.\n"
        ) == []


def test_bold_scenario_spec_populates_story_acs_end_to_end():
    """A full SPEC_REQUIREMENTS.md in the bold-scenario format must yield
    stories WITH acceptance criteria (before the fix: zero)."""
    spec = (
        "# Requirements\n\n"
        "## Epic: EPIC-001 — Root\n\n"
        "### Feature: FEAT-001 — Contacts\n"
        "**Parent epic:** EPIC-001\n\n"
        "#### Story: STORY-001 — Add a new contact\n"
        "**Parent feature:** FEAT-001\n\n"
        "**As a** user\n**I want** to add a contact\n**So that** I track them.\n\n"
        "**Acceptance Criteria:**\n\n"
        "**Scenario: Add contact with valid details**\n"
        "- Given the form is open\n"
        "- When valid data is submitted\n"
        "- Then the contact is created\n\n"
        "**Scenario: Missing name is rejected**\n"
        "- Given the form is open\n"
        "- When first_name is blank\n"
        "- Then a required-field error shows\n"
    )
    parsed = parse_spec_requirements(spec)
    story = next(s for s in parsed["stories"] if s["story_key"] == "STORY-001")
    assert story["acceptance_criteria"] == [
        "Add contact with valid details",
        "Missing name is rejected",
    ]
