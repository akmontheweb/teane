"""P1 fail-fast requirement-coverage gate in ``spec_reconciler_node``.

The end-of-run traceability gate (``installation_doc_node``) blocks a build
whose requirements aren't all covered — but only after the entire build and
token budget are spent (lumina 019f82af spent ~90 minutes reaching it).
``spec_reconciler_node`` now runs the SAME coverage audit right after
decomposition+reconcile and, when ``enforce_reqs`` is on and a requirement
has no satisfying story, sets ``early_req_coverage_gap`` + ``exit_code=1`` so
the router ENDs immediately.

These tests drive the node end-to-end (real reconcile against a minimal spec)
and monkeypatch the coverage query to control the gap, isolating the P1 gate
logic from spec-parsing details.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from harness import spec_reconciler, story_state


@pytest.fixture
def isolated_state_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


_MIN_SPEC = (
    "# Spec\n\n"
    "## Epic: EPIC-001 — Root epic\n\n"
    "### Feature: FEAT-001 — A feature\n"
    "**Parent epic:** EPIC-001\n\n"
    "#### Story: STORY-001 — First story\n"
    "**Parent feature:** FEAT-001\n\n"
    "**As a** user\n**I want** X\n**So that** Y.\n\n"
    "```gherkin\n"
    "Scenario: something is true\n"
    "  Given a precondition\n"
    "  When action happens\n"
    "  Then outcome holds\n"
    "```\n"
)


def _prepare(workspace: str) -> None:
    """Write a minimal valid spec and ingest its requirements so the
    reconciler in the node succeeds (a failed reconcile never reaches the
    coverage gate)."""
    from harness.decomposition import _ingest_requirements

    docs = os.path.join(workspace, "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "SPEC_REQUIREMENTS.md"), "w") as f:
        f.write(_MIN_SPEC)
    app = story_state.app_name_for_workspace(workspace)
    _ingest_requirements(app, app, _MIN_SPEC)


def test_clean_coverage_does_not_block(isolated_state_db, tmp_path, monkeypatch):
    ws = str(tmp_path)
    _prepare(ws)
    monkeypatch.setattr(
        story_state, "requirements_without_satisfying_story", lambda c, w: [],
    )
    out = spec_reconciler.spec_reconciler_node({"workspace_path": ws})
    assert out["node_state"]["reconciled"] is True
    assert "early_req_coverage_gap" not in out["node_state"]
    assert "exit_code" not in out


def test_gap_fails_fast_when_enforced(isolated_state_db, tmp_path, monkeypatch):
    ws = str(tmp_path)
    _prepare(ws)
    monkeypatch.setattr(
        story_state, "requirements_without_satisfying_story",
        lambda c, w: [{"req_key": "FR-009", "kind": "fr", "title": "Uncovered"}],
    )
    # enforce_reqs defaults to True (no config).
    out = spec_reconciler.spec_reconciler_node({"workspace_path": ws})
    assert out["node_state"]["early_req_coverage_gap"] is True
    assert out["exit_code"] == 1


def test_gap_is_advisory_when_enforce_reqs_false(
    isolated_state_db, tmp_path, monkeypatch,
):
    ws = str(tmp_path)
    _prepare(ws)
    monkeypatch.setattr(
        story_state, "requirements_without_satisfying_story",
        lambda c, w: [{"req_key": "FR-009", "kind": "fr", "title": "Uncovered"}],
    )
    out = spec_reconciler.spec_reconciler_node({
        "workspace_path": ws,
        "harness_config": {"traceability": {"enforce_reqs": False}},
    })
    # Reported but not blocked — the build proceeds.
    assert "early_req_coverage_gap" not in out["node_state"]
    assert "exit_code" not in out


def test_coverage_check_error_is_fail_open(
    isolated_state_db, tmp_path, monkeypatch,
):
    ws = str(tmp_path)
    _prepare(ws)

    def _boom(conn, workspace):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(
        story_state, "requirements_without_satisfying_story", _boom,
    )
    # A coverage-check failure must never block on its own — the end-of-run
    # gate remains the backstop.
    out = spec_reconciler.spec_reconciler_node({"workspace_path": ws})
    assert out["node_state"]["reconciled"] is True
    assert "early_req_coverage_gap" not in out["node_state"]
    assert "exit_code" not in out
