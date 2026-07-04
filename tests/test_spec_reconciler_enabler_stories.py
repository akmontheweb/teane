"""Regression tests for the enabler-story heading fix.

Ciod v12 traceability audit reported 15/26 requirement coverage.
Investigation showed the 6 ``STORY-NFR-*`` requirements were missing
from the ``stories`` table entirely — the parser's story regex
required exactly 4 hashes (``####``) but the ciod spec uses 3
(``### Enabler Story: STORY-NFR-001 — Performance``) per SAFe
convention.

The fix loosens the story-heading regex to accept 3-6 hashes so
enabler stories written at either level land in the ``stories``
table. All 6 STORY-NFR requirements can then identity-link via the
earlier ``reconcile_workspace_from_spec`` fix.
"""

from __future__ import annotations

from harness.spec_reconciler import _STORY_RE, parse_spec_requirements


class TestStoryHeadingLevels:
    def test_level_4_regular_story(self):
        m = _STORY_RE.match("#### Story: STORY-001 — Log in")
        assert m and m.group(1) == "STORY-001"

    def test_level_4_enabler_story(self):
        # Existing shape — must still match.
        m = _STORY_RE.match("#### Enabler Story: STORY-NFR-006 — Caching")
        assert m and m.group(1) == "STORY-NFR-006"

    def test_level_3_enabler_story(self):
        # The ciod v12 pathology — level-3 enabler heading. Prior
        # to the fix this silently skipped and STORY-NFR-001..006
        # never landed in the stories table.
        m = _STORY_RE.match("### Enabler Story: STORY-NFR-001 — Perf")
        assert m and m.group(1) == "STORY-NFR-001"

    def test_level_2_rejected(self):
        # Page-title level shouldn't be a story. Bound the
        # permissiveness.
        m = _STORY_RE.match("## Story: STORY-002 — Wrong level")
        assert m is None

    def test_no_hash_rejected(self):
        m = _STORY_RE.match("Story: STORY-001 — Missing hash")
        assert m is None

    def test_permissive_whitespace(self):
        # Between hashes, "Story:", and the id — tolerate extras.
        m = _STORY_RE.match("####  Story:  STORY-003  —  Whitespace")
        assert m and m.group(1) == "STORY-003"


class TestEnablerStoryEndToEnd:
    """A minimal SAFe spec with a level-3 Enabler Story: after the
    fix, ``parse_spec_requirements`` returns the story so
    ``reconcile_workspace_from_spec`` can insert it."""

    _MIN_SPEC = (
        "# Spec\n\n"
        "## Epic: EPIC-001 — Root\n\n"
        "### Feature: FEAT-001 — A feature\n"
        "**Parent epic:** EPIC-001\n\n"
        "#### Story: STORY-001 — Regular story\n"
        "**Parent feature:** FEAT-001\n"
        "**As a** user\n"
        "**I want** X\n"
        "**So that** Y.\n\n"
        "## Enabler Stories\n"
        "### Enabler Story: STORY-NFR-001 — Performance\n"
        "**Parent feature:** FEAT-001\n"
        "**As a** operator\n"
        "**I want** SLA\n"
        "**So that** users are happy.\n"
    )

    def test_parser_surfaces_both_stories(self):
        parsed = parse_spec_requirements(self._MIN_SPEC)
        story_keys = {s["story_key"] for s in parsed["stories"]}
        # Pre-fix: only STORY-001 (level-4) would surface. Post-fix
        # the enabler at level-3 also lands.
        assert "STORY-001" in story_keys
        assert "STORY-NFR-001" in story_keys, (
            "Enabler-story at level-3 must be picked up by the parser "
            "— this is the ciod v12 15/26 traceability regression."
        )

    def test_enabler_story_carries_title(self):
        parsed = parse_spec_requirements(self._MIN_SPEC)
        by_key = {s["story_key"]: s for s in parsed["stories"]}
        assert by_key["STORY-NFR-001"]["title"] == "Performance"

    def test_enabler_story_attaches_to_parent_feature(self):
        # The ``**Parent feature:** FEAT-001`` line inside the
        # story body still parses; ``feature`` is populated so the
        # story lands under FEAT-001, not orphaned to PLATFORM.
        parsed = parse_spec_requirements(self._MIN_SPEC)
        by_key = {s["story_key"]: s for s in parsed["stories"]}
        assert by_key["STORY-NFR-001"]["feature"] == "FEAT-001"


class TestStructuralParentInference:
    """The reconciler now writes ``story_satisfies_req`` edges for
    each story's parent feature AND the feature's parent epic, so
    structural EPIC / FEAT requirement rows are traced by their
    descendants. Closes the ciod v12 15/26 gap on epic + feature
    coverage (5 rows recovered)."""

    _SPEC = (
        "# Spec\n\n"
        "## Epic: EPIC-001 — Root epic\n\n"
        "### Feature: FEAT-001 — Alpha\n"
        "**Parent epic:** EPIC-001\n\n"
        "#### Story: STORY-001 — Alpha 1\n"
        "**Parent feature:** FEAT-001\n"
        "**As a** u\n**I want** x\n**So that** y.\n\n"
        "### Feature: FEAT-002 — Beta\n"
        "**Parent epic:** EPIC-001\n\n"
        "#### Story: STORY-002 — Beta 1\n"
        "**Parent feature:** FEAT-002\n"
        "**As a** u\n**I want** x\n**So that** y.\n"
    )

    def test_feature_parent_epic_parsed(self):
        parsed = parse_spec_requirements(self._SPEC)
        by_key = {f["feature_key"]: f for f in parsed["features"]}
        assert by_key["FEAT-001"]["parent_epic"] == "EPIC-001"
        assert by_key["FEAT-002"]["parent_epic"] == "EPIC-001"

    def test_feature_without_parent_epic_marker_returns_none(self):
        spec = (
            "### Feature: FEAT-009 — Orphan\n"
            "no parent-epic line here.\n"
        )
        parsed = parse_spec_requirements(spec)
        assert parsed["features"][0]["parent_epic"] is None

    def test_reconciler_writes_feature_and_epic_edges(self, tmp_path, monkeypatch):
        # End-to-end: after reconcile, story_satisfies_req contains
        # rows linking each story to (a) its own key, (b) its parent
        # feature, (c) its grandparent epic. Locks in the ciod
        # 15/26 → 26/26 recovery path.
        import os
        import tempfile
        from harness import story_state, spec_reconciler
        from harness.decomposition import _ingest_requirements

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
        try:
            # Point the audit at a workspace-tagged app_name; write
            # the spec to disk so ``reconcile_workspace_from_spec``
            # can read it.
            docs = tmp_path / "docs"
            docs.mkdir()
            spec_path = docs / "SPEC_REQUIREMENTS.md"
            spec_path.write_text(self._SPEC)
            _ingest_requirements("app-parent", "app-parent", self._SPEC)
            conn = story_state.open_story_db()
            try:
                summary = spec_reconciler.reconcile_workspace_from_spec(
                    conn, "app-parent", str(spec_path),
                )
                # 2 stories × (self + feature + epic) = 6 links written
                # (may be fewer if any candidate wasn't in the requirements
                # table; here all three levels exist).
                assert summary["story_satisfies_req_written"] >= 6
                # Verify by joining: EPIC-001 has at least one link.
                epic_covered = conn.execute(
                    "SELECT COUNT(*) FROM story_satisfies_req l "
                    "JOIN requirements r ON r.id = l.requirement_id "
                    "WHERE r.workspace='app-parent' AND r.req_key='EPIC-001'"
                ).fetchone()[0]
                assert epic_covered >= 1, (
                    "EPIC-001 must be linked to at least one story via "
                    "the structural parent inference — this is the ciod "
                    "v12 15/26 recovery."
                )
                feat_covered = conn.execute(
                    "SELECT COUNT(*) FROM story_satisfies_req l "
                    "JOIN requirements r ON r.id = l.requirement_id "
                    "WHERE r.workspace='app-parent' AND r.req_key='FEAT-001'"
                ).fetchone()[0]
                assert feat_covered >= 1
            finally:
                conn.close()
        finally:
            os.unlink(tmp.name)
