"""Regression tests for the unified id canonicalisation boundary.

Prior to 2026-07-04 the DB could hold two different literal string
forms of the same key depending on which node populated the row:

    - ``decomposition_node`` → ``_next_story_key`` → raw form
      ``STORY-1`` / ``STORY-2``
    - ``spec_reconciler_node`` → parser output → canonical form
      ``STORY-001`` / ``STORY-002``

Downstream joins (``get_story``, ``mark_done``, ``link_test_to_ac``,
etc.) all used exact-string match. A query with the "wrong" form
silently missed. Ciod session 523e86a7 hit this via the traceability
audit reporting 0% coverage even after all batches sealed.

The fix picks ONE canonical form — zero-padded (``STORY-001``,
``FR-007``, ``EPIC-042``, ``NFR-SEC-001``, ``STORY-NFR-003``) — and
applies it EVERYWHERE:

  * ``_next_story_key`` allocates canonical form.
  * ``story_state._canon`` / ``_canon_ac`` PAD leading zeros (wrap
    :func:`req_ids.canonicalize_req_key` / ``canonicalize_ac_key``).
  * Every public boundary in ``story_state`` folds its input via
    ``_canon`` / ``_canon_ac`` so raw or canonical input both hit
    the canonical DB row.
  * ``spec_reconciler`` folds spec keys via ``_canon`` at insert.
  * ``@verifies:`` marker parser folds via ``_canon_ac`` so raw
    ``STORY-1.AC-1`` and canonical ``STORY-001.AC-1`` both land on
    the same ``test_verifies_ac`` row.

One canonical form, one storage convention, one code path.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture
def isolated_state_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from harness import story_state
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


class TestCanonHelper:
    """The internal ``_canon`` helper zero-pads so raw and canonical
    inputs both fold to the canonical storage form."""

    def test_pads_zeros_on_story_key(self):
        from harness.story_state import _canon
        assert _canon("STORY-1") == "STORY-001"
        assert _canon("STORY-10") == "STORY-010"
        assert _canon("STORY-100") == "STORY-100"

    def test_idempotent_on_canonical_input(self):
        from harness.story_state import _canon
        assert _canon("STORY-001") == "STORY-001"
        assert _canon("STORY-042") == "STORY-042"

    def test_handles_nfr_and_epic_and_feat(self):
        from harness.story_state import _canon
        assert _canon("EPIC-1") == "EPIC-001"
        assert _canon("FEAT-1") == "FEAT-001"
        assert _canon("FR-7") == "FR-007"
        assert _canon("STORY-NFR-3") == "STORY-NFR-003"
        assert _canon("NFR-SEC-1") == "NFR-SEC-001"

    def test_empty_and_none_safe(self):
        from harness.story_state import _canon
        assert _canon("") == ""

    def test_ac_key_pads_story_part(self):
        from harness.story_state import _canon_ac
        assert _canon_ac("STORY-1.AC-1") == "STORY-001.AC-1"
        assert _canon_ac("STORY-1.AC-01") == "STORY-001.AC-1"
        assert _canon_ac("STORY-001.AC-1") == "STORY-001.AC-1"


class TestNextStoryKeyReturnsCanonical:
    """``_next_story_key`` allocates the canonical form so every row
    the counter inserts is byte-identical to what the spec parser
    would emit for the same numeric id."""

    def test_first_key_is_canonical(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        keys = story_state.create_stories(
            conn, "ws",
            [{"title": "T", "feature": "test"}],
        )
        assert keys == ["STORY-001"]
        conn.close()

    def test_counter_advances_in_canonical_form(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        keys = story_state.create_stories(
            conn, "ws",
            [{"title": f"T{i}", "feature": "test"} for i in range(3)],
        )
        assert keys == ["STORY-001", "STORY-002", "STORY-003"]
        conn.close()


class TestGetStoryAcceptsBothForms:
    """``get_story`` folds the input at the boundary — a caller with
    raw ``STORY-1`` must hit the same row as a caller with canonical
    ``STORY-001``. Ciod 523e86a7 regression: mixed inputs across the
    codebase silently missed the row."""

    def test_both_forms_reach_same_row(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        story_state.create_stories(
            conn, "ws",
            [{"title": "T1", "feature": "test"}],
        )
        by_raw = story_state.get_story(conn, "ws", "STORY-1")
        by_canonical = story_state.get_story(conn, "ws", "STORY-001")
        assert by_raw is not None
        assert by_canonical is not None
        assert by_raw["id"] == by_canonical["id"]
        conn.close()


class TestMarkFunctionsAcceptBothForms:
    def test_mark_done_folds_raw(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        story_state.create_stories(
            conn, "ws",
            [{"title": "T1", "feature": "test"}],
        )
        # Pass the RAW form; the fold makes it hit the canonical row.
        story_state.mark_done(conn, "ws", "STORY-1")
        row = story_state.get_story(conn, "ws", "STORY-001")
        assert row is not None
        assert row["status"] == "done"
        conn.close()

    def test_mark_in_progress_folds_raw(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        story_state.create_stories(
            conn, "ws",
            [{"title": "T1", "feature": "test"}],
        )
        n = story_state.mark_in_progress(conn, "ws", "STORY-1")
        assert n == 1, "raw input should hit the canonical row (rowcount=1)"
        conn.close()


class TestLinkStoryToRequirementsFoldsReqKeys:
    """Both tables now store canonical form — the boundary fold
    handles either input direction uniformly."""

    def test_raw_req_key_hits_canonical_row(self, isolated_state_db):
        from harness import story_state
        conn = story_state.open_story_db()
        story_state.ensure_feature(conn, "ws", "test", name="F")
        story_state.create_stories(
            conn, "ws",
            [{"title": "T", "feature": "test"}],
        )
        story_state.create_requirements(
            conn, "ws",
            [{"req_key": "FR-001", "kind": "fr", "title": "X"}],
        )
        story = story_state.get_story(conn, "ws", "STORY-001")
        # Caller passes RAW ``FR-1``; the fn zero-pads to hit ``FR-001``.
        inserted = story_state.link_story_to_requirements(
            conn, "ws", story["id"], ["FR-1"],
        )
        assert inserted == 1
        conn.close()


class TestVerifiesMarkerFoldsAcKeys:
    """The ``@verifies:`` marker parser folds every AC key to the
    canonical storage form so both LLM-written shapes hit the same
    ``test_verifies_ac`` row."""

    def test_raw_marker_folded_to_canonical(self):
        from harness.test_generation import _parse_verifies_marker
        assert _parse_verifies_marker(
            "# @verifies: STORY-1.AC-1, STORY-1.AC-2\n"
        ) == ["STORY-001.AC-1", "STORY-001.AC-2"]

    def test_canonical_marker_preserved(self):
        from harness.test_generation import _parse_verifies_marker
        assert _parse_verifies_marker(
            "# @verifies: STORY-001.AC-1\n"
        ) == ["STORY-001.AC-1"]

    def test_mixed_forms_fold_consistently(self):
        from harness.test_generation import _parse_verifies_marker
        assert _parse_verifies_marker(
            "# @verifies: STORY-1.AC-1, STORY-001.AC-2\n"
        ) == ["STORY-001.AC-1", "STORY-001.AC-2"]
