"""ADR-0008 action item 2 — measure the premise before acting on it.

The ADR proposes trimming the anchored spec region to a per-story slice,
arguing the loop nodes never read the requirement bodies they are handed. That
argument came from reading code. These tests cover the instrument that checks
it on a real run: it must find genuine use, must not manufacture use out of
shared vocabulary, and must never break a dispatch.
"""

from __future__ import annotations

import pytest

from harness import spec_usage
from harness.spec_usage import (
    build_index,
    measure_call,
    split_spec_region,
)


SPEC = """# Software Requirements Specification

## API conventions

Success responses wrap collections in an envelope.

#### Story: STORY-001 — Display upcoming birthdays with calculated days left

**Parent feature:** FEAT-001

**Acceptance Criteria:**

- Given the application clock is set to 2026-01-15T00:00:00Z and Alice has a
  date of birth of 1990-01-16, the dashboard lists Alice first with one day left.

#### Story: STORY-002 — Add a birthday record

**Parent feature:** FEAT-002

**Acceptance Criteria:**

- Posting a record with an empty last name is rejected with a 422 status and
  no row is written to the employees table at all.
"""

SYSTEM_TAIL = (
    "You are an expert software engineer operating in a sandbox.\n\n"
    "## Rules\n- Never remove existing comments.\n"
)


def _anchored(spec: str = SPEC) -> str:
    return spec + "\n\n---\n\n" + SYSTEM_TAIL


def _msgs(system: str, user: str = "Fix the failing test in db.py."):
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


class TestRegionSplit:
    def test_finds_the_spec_half_of_an_anchored_prompt(self):
        got = split_spec_region(_anchored())
        assert got.startswith("# Software Requirements Specification")
        assert "expert software engineer" not in got

    def test_a_horizontal_rule_inside_the_spec_does_not_truncate_it(self):
        # Spec markdown legitimately contains `---`. Splitting on the FIRST
        # one would hand back a fragment and silently undercount every call.
        spec = SPEC + "\n\n---\n\n## Traceability matrix\n\n| FR-001 | TEST-001 |\n"
        got = split_spec_region(spec + "\n\n---\n\n" + SYSTEM_TAIL)
        assert "Traceability matrix" in got

    def test_a_prompt_with_no_spec_returns_nothing(self):
        assert split_spec_region(SYSTEM_TAIL) == ""

    def test_junk_is_tolerated(self):
        for junk in ("", "---", "no separator here"):
            assert split_spec_region(junk) == ""


class TestIndex:
    def test_blocks_are_requirement_rows(self):
        index = build_index(SPEC)
        assert "STORY-001" in index.req_keys and "STORY-002" in index.req_keys

    def test_an_unparseable_region_still_measures(self):
        # Fail open, not shut: a spec whose headings do not parse degrades to
        # "was any spec-only text echoed", rather than reporting no usage and
        # quietly making the ADR's case for it.
        index = build_index("just prose, no requirement headings at all here")
        assert index.req_keys == ["(unparsed-spec-region)"]


class TestMeasurement:
    def test_a_call_that_ignores_the_spec_reports_unused(self):
        out = measure_call(
            messages=_msgs(_anchored()),
            response_text="I will edit server/app/db.py to bind the session lazily.",
        )
        assert out["used"] is False
        assert out["cited"] == [] and out["echoed"] == []
        assert out["spec_chars"] > 0 and out["req_blocks"] == 2

    def test_naming_a_requirement_counts_as_cited(self):
        out = measure_call(
            messages=_msgs(_anchored()),
            response_text="STORY-002 requires a 422 here, so the test is right.",
        )
        assert out["cited"] == ["STORY-002"] and out["used"] is True

    def test_reproducing_a_requirement_body_counts_as_echoed(self):
        out = measure_call(
            messages=_msgs(_anchored()),
            response_text=(
                "The criterion says the application clock is set to "
                "2026-01-15T00:00:00Z and Alice has a date of birth of "
                "1990-01-16, so the first row must be Alice."
            ),
        )
        assert "STORY-001" in out["echoed"] and out["used"] is True

    def test_content_available_elsewhere_in_the_prompt_is_not_credited(self):
        # The decisive case. If the diagnostics already contain the text, the
        # response may have taken it from there — crediting the spec would
        # invent usage and defeat the whole measurement.
        user = (
            "FAILED test_add.py::test_empty_last_name\n"
            "Posting a record with an empty last name is rejected with a 422 "
            "status and no row is written to the employees table at all. "
            "See STORY-002."
        )
        out = measure_call(
            messages=_msgs(_anchored(), user=user),
            response_text=(
                "STORY-002: posting a record with an empty last name is "
                "rejected with a 422 status and no row is written to the "
                "employees table at all."
            ),
        )
        assert out["cited"] == [], "the id was in the prompt's own diagnostics"
        assert out["echoed"] == [], "the sentence was in the prompt's own diagnostics"
        assert out["used"] is False

    def test_a_lone_boundary_artifact_does_not_score_as_used(self):
        # The instrument's own failure mode, found while building it: a
        # punctuation difference at a window boundary makes ONE window unique
        # to the response even when the sentence came from the diagnostics.
        # A false positive here argues for keeping the spec region, so the
        # threshold exists to stop the measurement flattering that answer.
        user = (
            "FAILED test_add.py::test_empty_last_name...Posting a record "
            "with an empty last name is rejected with a 422 status and no "
            "row is written to the employees table at all."
        )
        out = measure_call(
            messages=_msgs(_anchored(), user=user),
            response_text=(
                "Right: posting a record with an empty last name is "
                "rejected with a 422 status and no row is written to the "
                "employees table at all."
            ),
        )
        assert out["echoed"] == [] and out["used"] is False

    def test_a_prompt_without_a_spec_region_is_not_measured(self):
        assert measure_call(
            messages=_msgs(SYSTEM_TAIL), response_text="anything") is None

    def test_an_empty_response_is_recorded_not_counted(self):
        out = measure_call(messages=_msgs(_anchored()), response_text="")
        assert out["used"] is False and out["empty_response"] is True

    def test_a_non_system_first_message_is_skipped(self):
        assert measure_call(
            messages=[{"role": "user", "content": _anchored()}],
            response_text="x",
        ) is None


class TestNeverBreaksDispatch:
    def test_emit_swallows_a_broken_analyzer(self, monkeypatch):
        def _boom(**_kw):
            raise RuntimeError("measurement exploded")

        monkeypatch.setattr(spec_usage, "measure_call", _boom)
        # Must not raise: a measurement is never worth failing a run for.
        spec_usage.maybe_emit(
            messages=_msgs(_anchored()), response_text="x", role="repair")

    def test_emit_publishes_the_payload(self, monkeypatch):
        seen = {}

        def _capture(name, **fields):
            seen["name"] = name
            seen.update(fields)

        import harness.observability as obs
        monkeypatch.setattr(obs, "emit_event", _capture)
        spec_usage.maybe_emit(
            messages=_msgs(_anchored()),
            response_text="STORY-001 says Alice is first.",
            role="repair",
            cache_family="patching:test_regeneration",
        )
        assert seen["name"] == "spec_region_usage"
        assert seen["role"] == "repair"
        assert seen["cache_family"] == "patching:test_regeneration"
        assert seen["cited"] == ["STORY-001"]


class TestGatewayWiring:
    def test_the_flag_is_off_by_default(self):
        from harness.gateway import create_gateway_from_config
        assert create_gateway_from_config({}).config.measure_spec_usage is False

    def test_the_flag_is_honoured(self):
        from harness.gateway import create_gateway_from_config
        gw = create_gateway_from_config({"debug": {"measure_spec_usage": True}})
        assert gw.config.measure_spec_usage is True
