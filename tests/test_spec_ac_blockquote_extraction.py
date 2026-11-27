"""Acceptance criteria must survive the spec reconciler.

Lumina run ``01a21c72`` finished decomposition with 12 stories and 52
acceptance criteria correctly persisted, and one second later had 12 stories
and ZERO acceptance criteria::

    16:00:15 [decomposition]   created 4 feature(s) and 12 story(ies)
    16:00:15 [spec_reconciler] wrote 4 features, 12 stories, 30 edges

``spec_reconciler`` is authoritative over the AC side table by design — it
exists to contain three real decomposition failure modes (silent renumbering,
dropping stories past ``MAX_STORIES_PER_PASS``, fabricating features). The bug
was in how it re-derived the ACs it then overwrote with:

1.  The generator renders Gherkin inside a markdown blockquote
    (``> Scenario: ...``). Every AC pattern in ``spec_reconciler`` anchors on
    ``^[ \\t]*`` followed by a bullet, a bold/heading marker or ``Scenario`` —
    none tolerate a leading ``>``. So the scenario pass found nothing, the
    ``Acceptance Criteria`` bullet fallback found nothing, and extraction
    returned ``[]`` for all 12 stories.
2.  The reconciler carries ``scope_files`` forward from the matched LLM row
    precisely because the spec is a lossy re-derivation. ``acceptance_criteria``
    was not carried forward, so an unreadable AC block did not merely
    contribute nothing — it DELETED what decomposition got right.

Downstream this reproduced run 20's death shape from a new cause: ADR-0006
generated ``tests/acceptance/test_story_001_acceptance.py`` with no ACs to
anchor it, the tests became protected-test blockers immediately, and repair
burned ``UNSATISFIABLE_TEST`` escapes on STORY-001 and STORY-002.

Run 20 never saw this because it REUSED specs. The first run to regenerate
them surfaced it.
"""

from __future__ import annotations

import os
import sqlite3  # noqa: F401 — imported for parity with sibling reconciler tests
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Fix 1 — blockquote-aware AC extraction
# ---------------------------------------------------------------------------


class TestBlockquotedAcceptanceCriteria:
    def test_blockquoted_scenario_headers_are_found(self):
        from harness.spec_reconciler import _extract_acceptance_criteria

        body = (
            "**Acceptance Criteria:**\n"
            "\n"
            "> Scenario: Birthday with known birth year\n"
            ">   Given today's local date is 2026-03-14\n"
            ">   Then a card shows \"Turning 35 today\"\n"
            "\n"
            "> Scenario: No celebrations today\n"
            ">   Given no contact has a birthday today\n"
            ">   Then the zone shows an empty state\n"
        )
        assert _extract_acceptance_criteria(body) == [
            "Birthday with known birth year",
            "No celebrations today",
        ]

    def test_nested_blockquote_depth_is_stripped(self):
        from harness.spec_reconciler import _extract_acceptance_criteria

        assert _extract_acceptance_criteria("> > Scenario: deep\n") == ["deep"]
        assert _extract_acceptance_criteria(">> Scenario: deeper\n") == ["deeper"]

    def test_blockquoted_bullet_fallback_is_found(self):
        """The ``Acceptance Criteria`` bullet fallback must survive
        blockquoting too — a spec that lists plain criteria rather than
        named scenarios is equally affected."""
        from harness.spec_reconciler import _extract_acceptance_criteria

        body = (
            "**Acceptance Criteria:**\n"
            "\n"
            "> - Rejects a February 30 date with HTTP 422\n"
            "> - Accepts an omitted birth year\n"
        )
        assert _extract_acceptance_criteria(body) == [
            "Rejects a February 30 date with HTTP 422",
            "Accepts an omitted birth year",
        ]

    def test_unquoted_formats_are_unchanged(self):
        """Regression guard: the shapes that already worked must parse
        identically, so this fix cannot be the cause of a future loss."""
        from harness.spec_reconciler import _extract_acceptance_criteria

        assert _extract_acceptance_criteria("**Scenario: bold**\n") == ["bold"]
        assert _extract_acceptance_criteria("- **Scenario: bulleted**\n") == [
            "bulleted"
        ]
        assert _extract_acceptance_criteria("### Scenario: heading\n") == ["heading"]
        assert _extract_acceptance_criteria(
            "```gherkin\nScenario: fenced\n  Given x\n```\n"
        ) == ["fenced"]

    def test_a_story_with_genuinely_no_criteria_still_returns_empty(self):
        from harness.spec_reconciler import _extract_acceptance_criteria

        assert _extract_acceptance_criteria("**As a** user\n**I want** X\n") == []

    def test_blockquote_stripping_is_identity_without_blockquotes(self):
        from harness.spec_reconciler import _strip_blockquotes

        text = "- **Scenario: x**\n  Given y\n"
        assert _strip_blockquotes(text) == text

    def test_full_spec_parses_the_same_quoted_and_unquoted(self):
        """End-to-end: the same spec with and without ``> `` prefixes must
        yield the same ACs. Before the fix the quoted form yielded zero."""
        from harness.spec_reconciler import parse_spec_requirements

        quoted = (
            "# Spec\n\n"
            "## Epic: EPIC-001 — Root\n\n"
            "### Feature: FEAT-001 — A feature\n"
            "**Parent epic:** EPIC-001\n\n"
            "#### Story: STORY-001 — First story\n"
            "**Parent feature:** FEAT-001\n\n"
            "**As a** user\n"
            "**I want** X\n"
            "**So that** Y.\n\n"
            "**Acceptance Criteria:**\n\n"
            "> Scenario: first criterion\n"
            ">   Given a precondition\n"
            ">   Then an outcome\n\n"
            "> Scenario: second criterion\n"
            ">   Given another precondition\n"
            ">   Then another outcome\n"
        )
        unquoted = "\n".join(
            ln[2:] if ln.startswith("> ") else ln for ln in quoted.splitlines()
        )

        q = parse_spec_requirements(quoted)["stories"]
        u = parse_spec_requirements(unquoted)["stories"]
        assert [s["acceptance_criteria"] for s in q] == [
            ["first criterion", "second criterion"]
        ]
        assert [s["acceptance_criteria"] for s in q] == [
            s["acceptance_criteria"] for s in u
        ]


# ---------------------------------------------------------------------------
# Fix 2 — the reconciler never overwrites ACs with nothing
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state_db(monkeypatch):
    """Point ``state_db_path`` at a fresh temp file so ``open_story_db``
    builds the real schema there instead of touching ``~/.harness/state.db``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from harness import story_state

    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


def _write_spec(workspace: str, *, ac_block: str) -> str:
    """One EPIC / FEAT / STORY spec whose AC block is caller-supplied, so a
    test can hand the parser a shape it does not recognise."""
    docs = os.path.join(workspace, "docs")
    os.makedirs(docs, exist_ok=True)
    spec_path = os.path.join(docs, "SPEC_REQUIREMENTS.md")
    with open(spec_path, "w") as f:
        f.write(
            "# Spec\n\n"
            "## Epic: EPIC-001 — Root epic\n\n"
            "### Feature: FEAT-001 — A feature\n"
            "**Parent epic:** EPIC-001\n\n"
            "#### Story: STORY-001 — First story\n"
            "**Parent feature:** FEAT-001\n\n"
            "**As a** user\n"
            "**I want** X\n"
            "**So that** Y.\n\n"
            + ac_block
        )
    return spec_path


# An AC shape no pattern recognises: a markdown table. Stands in for "the
# next markup the generator invents", which is the case fix 2 exists for.
_UNPARSEABLE_AC_BLOCK = (
    "**Acceptance Criteria:**\n\n"
    "| # | Criterion |\n"
    "|---|-----------|\n"
    "| 1 | Something holds |\n"
    "| 2 | Something else holds |\n"
)

_PARSEABLE_AC_BLOCK = (
    "**Acceptance Criteria:**\n\n"
    "> Scenario: spec-authored criterion\n"
    ">   Given a precondition\n"
    ">   Then an outcome\n"
)


def _seed_decomposition_story(workspace_name: str, acs: list[str]) -> None:
    """Mimic what ``decomposition_node`` leaves in the DB just before the
    reconciler runs: one story under one feature, carrying ACs."""
    from harness import story_state

    conn = story_state.open_story_db()
    try:
        story_state.create_features(
            conn, workspace_name,
            [{"feature_key": "FEAT-001", "name": "A feature"}],
        )
        story_state.create_stories(
            conn, workspace_name,
            [{
                "title": "First story",
                "feature": "FEAT-001",
                "description": "seeded by decomposition",
                "acceptance_criteria": list(acs),
                "scope_files": ["server/app/api/thing.py"],
            }],
        )
    finally:
        conn.close()


def _acs_in_db(workspace_name: str, story_key: str) -> list[str]:
    from harness import story_state

    conn = story_state.open_story_db()
    try:
        row = story_state.get_story(conn, workspace_name, story_key)
        assert row is not None, f"{story_key} missing from DB"
        return list(row["acceptance_criteria"])
    finally:
        conn.close()


class TestReconcilerNeverOverwritesAcsWithNothing:
    def test_unparseable_spec_acs_keep_decompositions(
        self, isolated_state_db, tmp_path, caplog,
    ):
        """The run-01a21c72 shape: the spec's AC markup is unreadable, so the
        spec contributes zero ACs. Decomposition's must survive rather than be
        deleted."""
        import logging
        from harness import spec_reconciler
        from harness.decomposition import _ingest_requirements

        ws_name = "ws-keep"
        spec_path = _write_spec(str(tmp_path), ac_block=_UNPARSEABLE_AC_BLOCK)
        with open(spec_path) as f:
            _ingest_requirements(ws_name, ws_name, f.read())

        _seed_decomposition_story(ws_name, ["decomp AC one", "decomp AC two"])

        # Precondition: the parser really cannot read this spec's ACs.
        with open(spec_path) as f:
            parsed = spec_reconciler.parse_spec_requirements(f.read())
        assert parsed["stories"][0]["acceptance_criteria"] == []

        caplog.set_level(logging.WARNING, logger="harness.spec_reconciler")
        from harness import story_state

        conn = story_state.open_story_db()
        try:
            spec_reconciler.reconcile_workspace_from_spec(
                conn, ws_name, spec_path,
            )
        finally:
            conn.close()

        assert _acs_in_db(ws_name, "STORY-001") == [
            "decomp AC one", "decomp AC two",
        ]
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "carrying forward" in joined, joined

    def test_parseable_spec_acs_win_over_decompositions(
        self, isolated_state_db, tmp_path,
    ):
        """The spec stays authoritative whenever it can actually be read —
        fix 2 is a fallback, not a merge."""
        from harness import spec_reconciler, story_state
        from harness.decomposition import _ingest_requirements

        ws_name = "ws-spec-wins"
        spec_path = _write_spec(str(tmp_path), ac_block=_PARSEABLE_AC_BLOCK)
        with open(spec_path) as f:
            _ingest_requirements(ws_name, ws_name, f.read())

        _seed_decomposition_story(ws_name, ["stale decomp AC"])

        conn = story_state.open_story_db()
        try:
            spec_reconciler.reconcile_workspace_from_spec(
                conn, ws_name, spec_path,
            )
        finally:
            conn.close()

        assert _acs_in_db(ws_name, "STORY-001") == ["spec-authored criterion"]

    def test_both_empty_stays_empty_without_error(
        self, isolated_state_db, tmp_path,
    ):
        """No ACs anywhere is a legitimate (if poor) state — the fallback
        must not invent any or raise."""
        from harness import spec_reconciler, story_state
        from harness.decomposition import _ingest_requirements

        ws_name = "ws-both-empty"
        spec_path = _write_spec(str(tmp_path), ac_block=_UNPARSEABLE_AC_BLOCK)
        with open(spec_path) as f:
            _ingest_requirements(ws_name, ws_name, f.read())

        _seed_decomposition_story(ws_name, [])

        conn = story_state.open_story_db()
        try:
            spec_reconciler.reconcile_workspace_from_spec(
                conn, ws_name, spec_path,
            )
        finally:
            conn.close()

        assert _acs_in_db(ws_name, "STORY-001") == []
