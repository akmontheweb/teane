"""Regression: spec_reconciler must not crash on duplicate keys.

lumina 019ff55f died at decomposition when ``_insert_features`` hit a
``UNIQUE(workspace, feature_key)`` violation — a spec parse emitted two
features sharing a key, the raw INSERT raised ``sqlite3.IntegrityError``,
``reconcile_workspace_from_spec`` aborted, and the build proceeded with ZERO
stories → no code generated → ``persistent_build_failure`` at HITL.

Both the feature insert and the story insert now dedupe by key (first wins),
turning a fatal crash into a self-heal that mirrors the known duplicate
story_key behaviour.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from harness import story_state, spec_reconciler


@pytest.fixture
def isolated_state_db(monkeypatch):
    """Point ``state_db_path`` at a fresh temp file so ``open_story_db``
    creates the real schema there, never touching ``~/.harness/state.db``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


class TestInsertFeaturesDedup:
    def test_duplicate_feature_key_first_wins_no_crash(self, isolated_state_db, caplog):
        conn = story_state.open_story_db()
        try:
            caplog.set_level(logging.WARNING, logger="harness.spec_reconciler")
            now = story_state._utcnow_iso()
            features = [
                {"feature_key": "FEAT-001", "name": "First", "description": "a"},
                {"feature_key": "FEAT-001", "name": "Dup", "description": "b"},
                {"feature_key": "FEAT-002", "name": "Other", "description": "c"},
            ]
            # Before the fix this raised sqlite3.IntegrityError on the 2nd row.
            ids = spec_reconciler._insert_features(conn, "ws", features, now)

            assert set(ids) == {"FEAT-001", "FEAT-002"}
            rows = conn.execute(
                "SELECT feature_key, name FROM features WHERE workspace='ws'"
                " ORDER BY feature_key"
            ).fetchall()
            assert [r[0] for r in rows] == ["FEAT-001", "FEAT-002"]
            # First wins: the surviving FEAT-001 row is the first-seen name.
            assert dict(rows)["FEAT-001"] == "First"
        finally:
            conn.close()

        assert any(
            "duplicate feature_key" in r.getMessage() and "FEAT-001" in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_distinct_keys_all_inserted(self, isolated_state_db):
        conn = story_state.open_story_db()
        try:
            now = story_state._utcnow_iso()
            features = [
                {"feature_key": f"FEAT-00{i}", "name": f"F{i}", "description": ""}
                for i in range(1, 4)
            ]
            ids = spec_reconciler._insert_features(conn, "ws", features, now)
            assert len(ids) == 3
        finally:
            conn.close()


def _write_spec_with_duplicate_story(workspace: str) -> str:
    """A spec whose feature contains the SAME story key twice — the story
    analogue of the feature_key crash."""
    docs = os.path.join(workspace, "docs")
    os.makedirs(docs, exist_ok=True)
    spec_path = os.path.join(docs, "SPEC_REQUIREMENTS.md")
    story_block = (
        "#### Story: STORY-001 — {title}\n"
        "**Parent feature:** FEAT-001\n"
        "\n"
        "**As a** user\n"
        "**I want** X\n"
        "**So that** Y.\n"
        "\n"
        "```gherkin\n"
        "Scenario: something is true\n"
        "  Given a precondition\n"
        "  When action happens\n"
        "  Then outcome holds\n"
        "```\n"
        "\n"
    )
    with open(spec_path, "w") as f:
        f.write(
            "# Spec\n\n"
            "## Epic: EPIC-001 — Root epic\n\n"
            "### Feature: FEAT-001 — A feature\n"
            "**Parent epic:** EPIC-001\n\n"
            + story_block.format(title="First copy")
            + story_block.format(title="Duplicate copy")
        )
    return spec_path


def test_reconcile_survives_duplicate_story_key(isolated_state_db, tmp_path):
    """Reconcile must complete (not raise IntegrityError) when the parsed
    spec carries a duplicate story_key, and write the story exactly once."""
    from harness.decomposition import _ingest_requirements

    workspace = str(tmp_path)
    spec_path = _write_spec_with_duplicate_story(workspace)
    with open(spec_path) as f:
        _ingest_requirements(workspace, workspace, f.read())

    conn = story_state.open_story_db()
    try:
        summary = spec_reconciler.reconcile_workspace_from_spec(
            conn, workspace, spec_path,
        )
        # Survived the UNIQUE constraint; STORY-001 written exactly once.
        rows = conn.execute(
            "SELECT story_key, COUNT(*) FROM stories WHERE workspace=?"
            " GROUP BY story_key",
            (workspace,),
        ).fetchall()
    finally:
        conn.close()

    assert all(c == 1 for _, c in rows), rows
    assert summary["stories_written"] >= 1
