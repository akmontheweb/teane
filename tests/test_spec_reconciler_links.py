"""Regression test for the ciod session 523e86a7 traceability loop.

Root cause of that 376-iteration ``traceability_block`` HITL loop:
``spec_reconciler_node`` called ``_wipe_workspace`` which cascades-
deletes ``story_satisfies_req`` rows via ``story_id`` FK, then
re-inserted stories via ``_insert_story`` but never repopulated
the link table. Every downstream traceability audit reported 0%
requirement coverage. Auto-resume could not fix a DB-level gap
so the run pinged HITL forever until manual kill.

Fix (2026-07-04): ``reconcile_workspace_from_spec`` now writes
``story_satisfies_req`` edges as part of the same transaction —
identity link when a requirement of the same key exists (SAFe
convention), plus any additional ``requirement_keys`` carried over
from the LLM's original story match.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest


@pytest.fixture
def isolated_state_db(monkeypatch):
    """Point ``state_db_path`` at a fresh temp file so
    ``open_story_db`` creates the real schema there. Avoids touching
    the shared ``~/.harness/state.db``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from harness import story_state
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


def _write_spec(workspace: str) -> str:
    """Minimal SAFe-style spec with 1 EPIC + 1 FEAT + 2 STORIES,
    each STORY carrying one acceptance criterion. Matches the shape
    ``parse_spec_requirements`` recognises."""
    docs = os.path.join(workspace, "docs")
    os.makedirs(docs, exist_ok=True)
    spec_path = os.path.join(docs, "SPEC_REQUIREMENTS.md")
    with open(spec_path, "w") as f:
        f.write(
            "# Spec\n"
            "\n"
            "## Epic: EPIC-001 — Root epic\n"
            "\n"
            "### Feature: FEAT-001 — A feature\n"
            "**Parent epic:** EPIC-001\n"
            "\n"
            "#### Story: STORY-001 — First story\n"
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
            "#### Story: STORY-002 — Second story\n"
            "**Parent feature:** FEAT-001\n"
            "\n"
            "**As a** user\n"
            "**I want** P\n"
            "**So that** Q.\n"
            "\n"
            "```gherkin\n"
            "Scenario: another thing is true\n"
            "  Given a precondition\n"
            "  When action happens\n"
            "  Then outcome holds\n"
            "```\n"
        )
    return spec_path


def test_spec_drift_logged_as_warning(caplog):
    """Finsearch session 44c5e194 root cause B1: LLM stories with no
    spec match used to be an INFO log the run silently ignored.
    Bumped to WARNING with actionable text — the operator can act on
    it before burning hours on a build with orphaned scope."""
    import logging
    from harness.spec_reconciler import _match_llm_to_spec

    caplog.set_level(logging.WARNING, logger="harness.spec_reconciler")
    spec = [
        {"story_key": "STORY-001", "title": "Real spec story"},
    ]
    llm = [
        {"story_key": "STORY-001", "title": "Real spec story"},
        {"story_key": "STORY-999", "title": "Hallucinated feature"},
    ]
    _match_llm_to_spec(spec, llm)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("SPEC DRIFT" in r.message for r in warnings), warnings
    joined = " ".join(r.getMessage() for r in warnings)
    assert "STORY-999" in joined
    assert "Hallucinated" in joined


def test_reconciler_writes_story_satisfies_req_edges(isolated_state_db, tmp_path):
    """The identity link — story ``STORY-001`` satisfies requirement
    ``STORY-001`` — must be written for every spec-authored story.
    Without it, ``traceability.audit_workspace`` reports 0% coverage
    and the end-of-session gate loops forever under headless
    auto-resume."""
    from harness import story_state, spec_reconciler
    from harness.decomposition import _ingest_requirements

    workspace = str(tmp_path)
    spec_path = _write_spec(workspace)
    with open(spec_path) as f:
        spec_text = f.read()

    # Ingest requirements first — mirrors what decomposition_node does
    # before it dispatches the LLM. Populates the ``requirements`` table
    # so the reconciler has something to link against.
    _ingest_requirements("ws-A", "ws-A", spec_text)

    conn = story_state.open_story_db()
    try:
        # Pre-seed a "before reconcile" state that mimics the LLM's
        # decomposition output: a stub story with a completely
        # different key (so we can prove the reconciler wiped it).
        # We don't need to be fancy — the reconciler's ``_wipe_workspace``
        # is the important behaviour under test.
        story_state.create_features(
            conn, "ws-A",
            [{"feature_key": "FEAT-999", "name": "stub", "description": ""}],
        )
        story_state.create_stories(
            conn, "ws-A",
            [{
                "title": "stub story", "feature": "FEAT-999",
                "acceptance_criteria": ["AC1"],
                "requirement_keys": [],
                "depends_on": [], "scope_files": [],
            }],
        )

        summary = spec_reconciler.reconcile_workspace_from_spec(
            conn, "ws-A", spec_path,
        )
    finally:
        conn.close()

    # Sanity: stories are written from the spec.
    assert summary["stories_written"] == 2

    # Regression: the link table is populated. Without the fix this
    # count is 0 and the traceability audit reports 0% coverage.
    assert summary["story_satisfies_req_written"] >= 2

    # Verify the link content — each STORY-nnn story identity-links to
    # the STORY-nnn requirement.
    conn = story_state.open_story_db()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT s.story_key AS sk, r.req_key AS rk "
            "FROM story_satisfies_req l "
            "JOIN stories s ON s.id = l.story_id "
            "JOIN requirements r ON r.id = l.requirement_id "
            "WHERE s.workspace = 'ws-A' "
            "ORDER BY s.story_key, r.req_key"
        ).fetchall()
    finally:
        conn.close()

    pairs = {(row["sk"], row["rk"]) for row in rows}
    # stories table stores keys in raw form (``STORY-N``) while the
    # requirements table stores them zero-padded (``STORY-NNN``). Both
    # conventions are stable at the boundary — see
    # ``story_state._canon`` / ``req_ids.canonicalize_req_key``.
    assert ("STORY-001", "STORY-001") in pairs
    assert ("STORY-002", "STORY-002") in pairs


def test_traceability_audit_reports_full_coverage_after_reconcile(
    isolated_state_db, tmp_path,
):
    """End-to-end proof: after ``reconcile_workspace_from_spec`` runs
    on a minimal SAFe spec, ``traceability.audit_workspace`` reports
    100% requirement coverage for the story-kind requirements. Locks
    in the loop-recovery fix — a run through this path must NOT
    produce a ``has_failures()==True`` report for its own
    self-referential story→requirement mapping."""
    from harness import story_state, spec_reconciler, traceability
    from harness.decomposition import _ingest_requirements

    workspace = str(tmp_path)
    spec_path = _write_spec(workspace)
    with open(spec_path) as f:
        spec_text = f.read()
    _ingest_requirements("ciod-loop", "ciod-loop", spec_text)

    conn = story_state.open_story_db()
    try:
        # Reconcile with an empty LLM-side (no pre-existing stories).
        spec_reconciler.reconcile_workspace_from_spec(
            conn, "ciod-loop", spec_path,
        )
    finally:
        conn.close()

    # Point the audit at the workspace whose basename matches "ciod-loop".
    ws_root = tmp_path / "ciod-loop"
    ws_root.mkdir(exist_ok=True)

    # ``audit_workspace`` derives the app_name from the workspace path
    # basename via ``app_name_for_workspace``. Route it directly to
    # match our test workspace tag.
    report = traceability.audit_workspace(str(ws_root))
    assert report is not None
    # STORY requirements should have 100% coverage — every STORY-nnn
    # in the spec gets an identity link written by the reconciler.
    story_reqs_untraced = [
        u for u in report.untraced
        if u.kind == "safe_story"
    ]
    assert story_reqs_untraced == [], (
        f"Expected zero untraced safe_story requirements after reconcile; "
        f"got {[u.req_id for u in story_reqs_untraced]}. "
        f"This is the ciod 523e86a7 loop regression."
    )
