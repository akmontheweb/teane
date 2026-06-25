"""Unit tests for harness.graph._build_story_preamble (step 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import story_state
from harness.graph import _build_story_preamble


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws_dir = tmp_path / "preamble-ws"
    ws_dir.mkdir()
    ws = str(ws_dir)
    app = story_state.app_name_for_workspace(ws)
    conn = story_state.open_story_db()
    try:
        story_state.create_features(conn, app, [
            {"feature_key": "auth", "name": "Auth"},
        ])
        story_state.create_stories(conn, app, [{
            "title": "Add /register endpoint",
            "feature": "auth",
            "description": "POST /register creates a user.",
            "acceptance_criteria": [
                "POST /register with valid payload returns 201",
                "Duplicate email returns 409",
            ],
            "scope_files": ["src/auth/register.py"],
            "external_ref": None,
        }, {
            "title": "CR-bridged story",
            "feature": "auth",
            "acceptance_criteria": ["Some criterion"],
            "external_ref": "CR-7",
        }])
    finally:
        conn.close()
    return ws


# ---------------------------------------------------------------------------
# No story → empty preamble (the default — preserves today's flow)
# ---------------------------------------------------------------------------

def test_no_active_story_returns_empty():
    state = {"workspace_path": "/tmp/nowhere", "current_story_id": ""}
    assert _build_story_preamble(state, "patching") == ""


def test_missing_workspace_returns_empty():
    state = {"current_story_id": "STORY-1"}
    assert _build_story_preamble(state, "patching") == ""


def test_unknown_story_id_returns_empty(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-999",
    }
    assert _build_story_preamble(state, "patching") == ""


# ---------------------------------------------------------------------------
# Active story → preamble includes scope + acceptance criteria + markers
# ---------------------------------------------------------------------------

def test_preamble_includes_key_title_and_criteria(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-1",
        "story_scope_files": ["src/auth/register.py"],
    }
    preamble = _build_story_preamble(state, "patching")
    assert "STORY-1" in preamble
    assert "Add /register endpoint" in preamble
    assert "POST /register with valid payload returns 201" in preamble
    assert "Duplicate email returns 409" in preamble
    assert "src/auth/register.py" in preamble


def test_preamble_marker_contract_for_patching(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-1",
    }
    preamble = _build_story_preamble(state, "patching")
    assert "# STORY-1: " in preamble
    assert "// STORY-1: " in preamble
    # Not the test naming rule on the patching phase.
    assert "test_story_1_" not in preamble


def test_preamble_includes_test_naming_rule_in_tests_phase(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-1",
    }
    preamble = _build_story_preamble(state, "tests")
    assert "test_story_1_" in preamble


def test_preamble_emits_dual_markers_for_cr_bridged_story(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-2",
    }
    preamble = _build_story_preamble(state, "patching")
    assert "STORY-2" in preamble
    assert "CR-7" in preamble
    assert "BOTH" in preamble


def test_preamble_uses_state_scope_files_over_db_when_set(workspace: str):
    """The state's story_scope_files takes precedence; this lets
    story_loop_node override per-resume edits."""
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-1",
        "story_scope_files": ["override/path.py"],
    }
    preamble = _build_story_preamble(state, "patching")
    assert "override/path.py" in preamble
    assert "src/auth/register.py" not in preamble


def test_preamble_warns_about_cross_story_coupling(workspace: str):
    state = {
        "workspace_path": workspace,
        "current_story_id": "STORY-1",
        "story_scope_files": ["src/auth/register.py"],
    }
    preamble = _build_story_preamble(state, "patching")
    assert "cross-story coupling" in preamble.lower() or "TRACEABILITY.md" in preamble
