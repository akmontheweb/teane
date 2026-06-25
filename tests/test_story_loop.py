"""Unit tests for harness/story_loop.py (batch_planner + story_loop nodes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness import story_loop, story_state


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws-loop"
    ws.mkdir()
    return str(ws)


def _app(workspace: str) -> str:
    return story_state.app_name_for_workspace(workspace)


def _seed_stories(workspace: str, items: list[dict[str, Any]]) -> None:
    """Seed stories under a default ``test`` feature unless each item
    already specifies a ``feature`` key. Auto-creates any referenced
    feature_keys on the fly so test bodies don't have to."""
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        # Collect every feature_key referenced (defaulting to "test").
        feature_keys: set[str] = set()
        for item in items:
            item.setdefault("feature", "test")
            feature_keys.add(item["feature"])
        for fkey in sorted(feature_keys):
            story_state.ensure_feature(conn, app, fkey, name=fkey)
        story_state.create_stories(conn, app, items)
    finally:
        conn.close()


def _state(workspace: str, **extra: Any) -> dict[str, Any]:
    base = {
        "workspace_path": workspace,
        "session_id": "sess-1",
        "story_batch_size": 5,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# batch_planner_node
# ---------------------------------------------------------------------------

def test_batch_planner_with_no_stories_reports_all_complete(workspace: str):
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["current_batch_id"] == 0
    assert out["node_state"]["batch_planned"] is False
    assert out["node_state"]["all_complete"] is True
    assert out["node_state"]["reason"] == "no_stories"


def test_batch_planner_creates_batch_for_independent_stories(workspace: str):
    _seed_stories(workspace, [
        {"title": "A"}, {"title": "B"}, {"title": "C"},
    ])
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["batch_planned"] is True
    assert out["current_batch_id"] > 0
    assert set(out["node_state"]["story_keys"]) == {"STORY-1", "STORY-2", "STORY-3"}


def test_batch_planner_honors_batch_size(workspace: str):
    _seed_stories(workspace, [{"title": f"t{i}"} for i in range(7)])
    out = story_loop.batch_planner_node(_state(workspace, story_batch_size=3))
    assert out["node_state"]["batch_size"] == 3


def test_batch_planner_honors_dependencies(workspace: str):
    _seed_stories(workspace, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-1"]},
    ])
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["story_keys"] == ["STORY-1"]


def test_batch_planner_reports_all_complete_when_every_story_done(workspace: str):
    _seed_stories(workspace, [{"title": "T"}])
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-1")
    finally:
        conn.close()
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["all_complete"] is True
    assert out["node_state"]["done_count"] == 1


def test_batch_planner_reports_stall_when_only_blocked_remains(workspace: str):
    _seed_stories(workspace, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-1"]},
    ])
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_blocked(conn, app, "STORY-1")
    finally:
        conn.close()
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["batch_planned"] is False
    assert out["node_state"]["stalled"] is True
    assert "STORY-1" in out["node_state"]["outstanding_deps"]


def test_batch_planner_writes_batch_row(workspace: str):
    _seed_stories(workspace, [{"title": "T"}])
    out = story_loop.batch_planner_node(_state(workspace))
    batch_id = out["current_batch_id"]
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT session_id, status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row == ("sess-1", "running")


# ---------------------------------------------------------------------------
# Feature-first slicing — the v4 contract
# ---------------------------------------------------------------------------

def test_batch_planner_never_crosses_feature_boundary(workspace: str):
    """A batch contains stories from exactly ONE feature, even when
    other features have ready stories that would fit under the batch
    size cap."""
    _seed_stories(workspace, [
        # Feature 'auth' first (lower feature.id).
        {"title": "auth-1", "feature": "auth"},
        {"title": "auth-2", "feature": "auth"},
        # Feature 'billing' second — should NOT be mixed into the same batch.
        {"title": "billing-1", "feature": "billing"},
        {"title": "billing-2", "feature": "billing"},
    ])
    out = story_loop.batch_planner_node(
        _state(workspace, story_batch_size=10),
    )
    assert out["node_state"]["batch_planned"] is True
    keys = out["node_state"]["story_keys"]
    # Only the auth feature's two stories — billing waits its turn.
    assert set(keys) == {"STORY-1", "STORY-2"}
    assert out["node_state"]["feature_key"] == "auth"
    # The batches row carries the feature_id.
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT feature_id FROM batches WHERE id = ?",
            (out["current_batch_id"],),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] is not None


def test_batch_planner_splits_large_feature_across_batches(workspace: str):
    """A feature with more stories than ``story_batch_size`` lands in
    multiple batches — all tagged with the same feature_id."""
    _seed_stories(workspace, [
        {"title": f"big-{i}", "feature": "big"} for i in range(7)
    ])
    out1 = story_loop.batch_planner_node(
        _state(workspace, story_batch_size=3),
    )
    assert len(out1["node_state"]["story_keys"]) == 3
    fid1 = out1["node_state"]["feature_id"]
    # Mark the first batch's stories done so the next planner pass
    # advances to the remaining feature stories.
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        for k in out1["node_state"]["story_keys"]:
            story_state.mark_done(conn, app, k)
    finally:
        conn.close()
    out2 = story_loop.batch_planner_node(
        _state(workspace, story_batch_size=3),
    )
    # Still the same feature — never mixed with anything else.
    assert out2["node_state"]["feature_id"] == fid1
    assert len(out2["node_state"]["story_keys"]) == 3


def test_batch_planner_advances_to_next_feature_when_first_done(workspace: str):
    """After feature A's stories are all done, the next batch picks
    feature B's stories."""
    _seed_stories(workspace, [
        {"title": "auth-1", "feature": "auth"},
        {"title": "billing-1", "feature": "billing"},
    ])
    # First batch: feature auth.
    out1 = story_loop.batch_planner_node(_state(workspace))
    assert out1["node_state"]["feature_key"] == "auth"
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-1")
    finally:
        conn.close()
    # Second batch: feature billing now picks up.
    out2 = story_loop.batch_planner_node(_state(workspace))
    assert out2["node_state"]["feature_key"] == "billing"
    assert out2["node_state"]["story_keys"] == ["STORY-2"]


# ---------------------------------------------------------------------------
# story_loop_node
# ---------------------------------------------------------------------------

def test_story_loop_no_batch_id_is_safe(workspace: str):
    out = story_loop.story_loop_node(_state(workspace, current_batch_id=0))
    assert out["node_state"]["batch_complete"] is True
    assert out["node_state"]["reason"] == "no_batch_id"


def test_story_loop_picks_first_planned_story(workspace: str):
    _seed_stories(workspace, [
        {"title": "A", "scope_files": ["a.py"], "acceptance_criteria": ["AC-A"]},
        {"title": "B", "scope_files": ["b.py"], "acceptance_criteria": ["AC-B"]},
    ])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["current_story_id"] == "STORY-1"
    assert out["story_scope_files"] == ["a.py"]
    assert out["node_state"]["batch_complete"] is False
    assert out["node_state"]["acceptance_criteria"] == ["AC-A"]

    # STORY-1 now in_progress
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-1")
    finally:
        conn.close()
    assert s["status"] == "in_progress"


def test_story_loop_resumes_in_progress_before_planned(workspace: str):
    """If the batch has both in_progress and planned, the loop returns
    the in_progress one first so a resumed session continues mid-story."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # Pretend STORY-2 was started, then process died before STORY-1.
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-2")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["current_story_id"] == "STORY-2"


def test_story_loop_returns_complete_when_all_done(workspace: str):
    _seed_stories(workspace, [{"title": "A"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-1")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["node_state"]["batch_complete"] is True
    assert out["current_story_id"] == ""
    assert out["node_state"]["blocked_count"] == 0

    # Batch was marked complete
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "complete"


def test_story_loop_reports_blocked_count(workspace: str):
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-1")
        story_state.mark_blocked(conn, app, "STORY-2")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["node_state"]["batch_complete"] is True
    assert out["node_state"]["blocked_count"] == 1

    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "complete_with_blocks"


# ---------------------------------------------------------------------------
# Layer 2 — per-story zero-patch auto-advance
# ---------------------------------------------------------------------------

def test_story_loop_auto_completes_story_at_zero_patch_cap(workspace: str):
    """After STORY_ZERO_PATCH_CAP consecutive zero-patch rounds against
    the same story, story_loop_node marks it done and advances to the
    next story so the batch makes progress."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # Simulate STORY-1 in-progress with 3 zero-patch rounds accumulated.
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-1")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-1",
        loop_counter={
            "story_zero_patch_rounds": {"STORY-1": story_loop.STORY_ZERO_PATCH_CAP}
        },
    ))

    # Advanced to STORY-2.
    assert out["current_story_id"] == "STORY-2"
    assert out["node_state"]["batch_complete"] is False
    assert out["node_state"]["auto_completed_story"] == "STORY-1"
    assert out["node_state"]["auto_completed_zero_rounds"] == (
        story_loop.STORY_ZERO_PATCH_CAP
    )

    # STORY-1 should now be done in the DB.
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-1")
    finally:
        conn.close()
    assert s["status"] == "done"

    # Counter cleared.
    assert out["loop_counter"]["story_zero_patch_rounds"] == {}


def test_story_loop_does_not_auto_complete_below_cap(workspace: str):
    """Below the cap, the story stays in_progress and is re-selected
    (the in_progress story always wins the ORDER BY)."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-1")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-1",
        loop_counter={
            "story_zero_patch_rounds": {
                "STORY-1": story_loop.STORY_ZERO_PATCH_CAP - 1
            }
        },
    ))

    assert out["current_story_id"] == "STORY-1"
    assert out["node_state"].get("auto_completed_story") is None
    # Counter survives unchanged.
    assert out["loop_counter"]["story_zero_patch_rounds"]["STORY-1"] == (
        story_loop.STORY_ZERO_PATCH_CAP - 1
    )

    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-1")
    finally:
        conn.close()
    assert s["status"] == "in_progress"


def test_story_loop_auto_complete_respects_cap_override(workspace: str):
    """state['story_zero_patch_cap'] overrides the default cap."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-1")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-1",
        story_zero_patch_cap=1,
        loop_counter={"story_zero_patch_rounds": {"STORY-1": 1}},
    ))

    assert out["node_state"]["auto_completed_story"] == "STORY-1"


def test_story_loop_auto_complete_finishes_batch_when_last_story(workspace: str):
    """Auto-completing the only remaining story flips the batch to
    complete (not complete_with_blocks — auto-advance is a normal
    completion, not a defect)."""
    _seed_stories(workspace, [{"title": "A"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-1")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-1",
        loop_counter={
            "story_zero_patch_rounds": {"STORY-1": story_loop.STORY_ZERO_PATCH_CAP}
        },
    ))

    assert out["node_state"]["batch_complete"] is True
    assert out["node_state"]["blocked_count"] == 0
    assert out["current_story_id"] == ""

    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "complete"


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def test_route_after_batch_planner_proceeds_when_planned():
    state = {"node_state": {"batch_planned": True}}
    assert story_loop.route_after_batch_planner(state) == "story_loop_node"


def test_route_after_batch_planner_falls_through_when_all_complete():
    state = {"node_state": {"batch_planned": False, "all_complete": True}}
    assert story_loop.route_after_batch_planner(state) == "traceability_node"


def test_route_after_batch_planner_falls_through_on_stall():
    state = {"node_state": {"batch_planned": False, "stalled": True}}
    assert story_loop.route_after_batch_planner(state) == "traceability_node"


def test_route_after_story_loop_proceeds_to_patching_when_story_picked():
    # Phase F removed story_test_first_node; the loop hands the story
    # straight to patching_node and lets _build_story_preamble carry
    # the acceptance criteria into the LLM prompt.
    state = {"node_state": {"batch_complete": False}}
    assert story_loop.route_after_story_loop(state) == "patching_node"


def test_route_after_story_loop_enters_verification_on_completion():
    # Phase E.3: when the batch is fully patched, story_loop_node hands
    # off to speculative_node which is the entry to the per-batch
    # verification chain (speculative → test_gen → lintgate → compile →
    # code_review). Batch sealing happens via route_after_code_review
    # → batch_commit_node, NOT directly from story_loop_node.
    state = {"node_state": {"batch_complete": True}}
    assert story_loop.route_after_story_loop(state) == "speculative_node"
