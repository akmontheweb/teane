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

def test_batch_planner_with_no_stories_routes_to_hitl(workspace: str):
    """An empty story DB is failure, not completion — the pipeline must
    not silently progress when decomposition produced no stories."""
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["current_batch_id"] == 0
    assert out["node_state"]["batch_planned"] is False
    assert out["node_state"]["all_complete"] is False
    assert out["node_state"]["decomposition_missing"] is True
    assert out["exit_code"] == 1


def test_batch_planner_creates_batch_for_independent_stories(workspace: str):
    _seed_stories(workspace, [
        {"title": "A"}, {"title": "B"}, {"title": "C"},
    ])
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["batch_planned"] is True
    assert out["current_batch_id"] > 0
    assert set(out["node_state"]["story_keys"]) == {"STORY-001", "STORY-002", "STORY-003"}


def test_batch_planner_honors_batch_size(workspace: str):
    _seed_stories(workspace, [{"title": f"t{i}"} for i in range(7)])
    out = story_loop.batch_planner_node(_state(workspace, story_batch_size=3))
    assert out["node_state"]["batch_size"] == 3


def test_batch_planner_honors_dependencies(workspace: str):
    _seed_stories(workspace, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-001"]},
    ])
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["story_keys"] == ["STORY-001"]


def test_batch_planner_reports_all_complete_when_every_story_done(workspace: str):
    _seed_stories(workspace, [{"title": "T"}])
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-001")
    finally:
        conn.close()
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["all_complete"] is True
    assert out["node_state"]["done_count"] == 1


def test_batch_planner_reports_stall_when_only_blocked_remains(workspace: str):
    _seed_stories(workspace, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-001"]},
    ])
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_blocked(conn, app, "STORY-001")
    finally:
        conn.close()
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["node_state"]["batch_planned"] is False
    assert out["node_state"]["stalled"] is True
    assert "STORY-001" in out["node_state"]["outstanding_deps"]


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
    assert set(keys) == {"STORY-001", "STORY-002"}
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
        story_state.mark_done(conn, app, "STORY-001")
    finally:
        conn.close()
    # Second batch: feature billing now picks up.
    out2 = story_loop.batch_planner_node(_state(workspace))
    assert out2["node_state"]["feature_key"] == "billing"
    assert out2["node_state"]["story_keys"] == ["STORY-002"]


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
    assert out["current_story_id"] == "STORY-001"
    assert out["story_scope_files"] == ["a.py"]
    assert out["node_state"]["batch_complete"] is False
    assert out["node_state"]["acceptance_criteria"] == ["AC-A"]

    # STORY-001 now in_progress
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-001")
    finally:
        conn.close()
    assert s["status"] == "in_progress"


def test_story_loop_resumes_in_progress_before_planned(workspace: str):
    """If the batch has both in_progress and planned, the loop returns
    the in_progress one first so a resumed session continues mid-story."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # Pretend STORY-002 was started, then process died before STORY-001.
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-002")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["current_story_id"] == "STORY-002"


def test_story_loop_returns_complete_when_all_done(workspace: str):
    _seed_stories(workspace, [{"title": "A"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["node_state"]["batch_complete"] is True
    assert out["current_story_id"] == ""
    assert out["node_state"]["blocked_count"] == 0

    # batch_commit_node is the sole sealer now — story_loop_node only
    # signals batch_complete=True so the per-batch verification chain
    # can run. The batch row stays at its in-flight status (``running``,
    # whatever batch_planner_node sets) until verification clears it.
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] != "complete"


def test_story_loop_reports_blocked_count(workspace: str):
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, "STORY-001")
        story_state.mark_blocked(conn, app, "STORY-002")
    finally:
        conn.close()

    out = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out["node_state"]["batch_complete"] is True
    assert out["node_state"]["blocked_count"] == 1

    # The blocked_count is surfaced via node_state so the verification
    # chain can seal as "complete_with_blocks" later; story_loop_node
    # itself no longer writes the batch row (batch_commit_node is the
    # single sealer).
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] != "complete"
    assert row[0] != "complete_with_blocks"


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

    # Simulate STORY-001 in-progress with 3 zero-patch rounds accumulated.
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        loop_counter={
            "story_zero_patch_rounds": {"STORY-001": story_loop.STORY_ZERO_PATCH_CAP}
        },
    ))

    # Advanced to STORY-002.
    assert out["current_story_id"] == "STORY-002"
    assert out["node_state"]["batch_complete"] is False
    assert out["node_state"]["auto_completed_story"] == "STORY-001"
    assert out["node_state"]["auto_completed_zero_rounds"] == (
        story_loop.STORY_ZERO_PATCH_CAP
    )

    # STORY-001 should now be done in the DB.
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-001")
    finally:
        conn.close()
    assert s["status"] == "done"

    # Counter cleared.
    assert out["loop_counter"]["story_zero_patch_rounds"] == {}


def test_auto_complete_resets_global_zero_patch_counter(workspace: str):
    """lumina 019f82af: auto-completing a vacuous story cleared the per-story
    counter but left the GLOBAL consecutive_zero_patch_rounds at the cap, so a
    downstream node fired a residual zero_patch_loop HITL. Auto-complete must
    reset the global counter too, since the stall is resolved."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, _app(workspace), "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        loop_counter={
            "story_zero_patch_rounds": {"STORY-001": story_loop.STORY_ZERO_PATCH_CAP},
            "consecutive_zero_patch_rounds": story_loop.STORY_ZERO_PATCH_CAP,
        },
    ))

    assert out["node_state"]["auto_completed_story"] == "STORY-001"
    # The global counter that fed the residual HITL is now cleared.
    assert out["loop_counter"]["consecutive_zero_patch_rounds"] == 0


def test_story_loop_does_not_auto_complete_below_cap(workspace: str):
    """Below the per-story zero-patch cap, the story is NOT auto-completed
    (it stays in_progress in the DB), but the cursor still advances to the
    next story in the batch. The cursor-advance is driven by
    ``batch_patched_story_keys`` (Phase E.3 fix) — the previous story's
    patching turn already ran, so picking it again would just re-patch the
    same code without making progress. The per-story zero-patch cap is now
    defensive: it would only fire if state were carried across runs (e.g.,
    via checkpoint resume) such that the same story re-entered story_loop
    with a non-empty rounds counter."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        loop_counter={
            "story_zero_patch_rounds": {
                "STORY-001": story_loop.STORY_ZERO_PATCH_CAP - 1
            }
        },
    ))

    # Phase E.3 cursor-advance: STORY-001 is recorded as patched and STORY-002
    # is picked next; STORY-001 is NOT re-selected even though it's still
    # in_progress in the DB.
    assert out["current_story_id"] == "STORY-002"
    assert out["node_state"].get("auto_completed_story") is None
    assert "STORY-001" in out["batch_patched_story_keys"]
    # Counter survives unchanged — auto-complete didn't fire.
    assert out["loop_counter"]["story_zero_patch_rounds"]["STORY-001"] == (
        story_loop.STORY_ZERO_PATCH_CAP - 1
    )

    # STORY-001 is still in_progress in the DB; batch_commit_node will seal
    # it as ``done`` after the per-batch verification chain runs.
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, "STORY-001")
    finally:
        conn.close()
    assert s["status"] == "in_progress"


def test_story_loop_retries_story_on_zero_patch_when_below_cap(workspace: str):
    """A1 fix (2026-07-11): a patching turn that produced 0 real patches
    must give the same story another shot with the patcher's rejection
    feedback in the message trail, up to the ``story_zero_patch_cap``.
    Before A1, ``patched_keys.append(cur_story_id)`` fired
    unconditionally, so a 0-patch story was burned from the batch queue
    on first attempt and the retry budget was dead code. Finsearch
    STORY-002/003/004 all advanced with 0 patches and never got a
    retry."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # First entry — picks STORY-001.
    out1 = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out1["current_story_id"] == "STORY-001"

    # patching_node ran on STORY-001 with 0 real patches. The just-
    # finished turn stamps ``node_state.patch_success=0`` on state and
    # increments ``loop_counter.story_zero_patch_rounds[STORY-001]``.
    # story_loop must NOT append STORY-001 to patched_keys and must
    # re-pick it so the LLM gets another shot.
    out2 = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        batch_patched_story_keys=[],
        node_state={"patch_success": 0, "current_node": "patching"},
        loop_counter={"story_zero_patch_rounds": {"STORY-001": 1}},
    ))
    assert out2["current_story_id"] == "STORY-001"
    assert out2["batch_patched_story_keys"] == []


def test_story_loop_advances_after_zero_patch_cap_hit(workspace: str):
    """A1: once ``story_zero_patch_rounds[cur_story_id] >= cap``, the
    story auto-completes and story_loop MUST advance to the next
    eligible story — the retry loop bounded by the cap prevents an
    infinite loop for a truly vacuous story."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # STORY-001 has burned its cap (default 3). Advance.
    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        batch_patched_story_keys=[],
        node_state={"patch_success": 0, "current_node": "patching"},
        loop_counter={"story_zero_patch_rounds": {"STORY-001": 3}},
    ))
    assert out["current_story_id"] == "STORY-002"
    assert "STORY-001" in out["batch_patched_story_keys"]


def test_story_loop_advances_when_patch_success_positive(workspace: str):
    """A1: when patching produced ≥1 real patch, cur_story_id must be
    marked patched-this-pass and the cursor advances — the retry loop
    only fires for the 0-success case."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        batch_patched_story_keys=[],
        node_state={"patch_success": 5, "current_node": "patching"},
    ))
    assert out["current_story_id"] == "STORY-002"
    assert out["batch_patched_story_keys"] == ["STORY-001"]


def test_story_loop_advances_cursor_after_patching_turn(workspace: str):
    """Regression for the 2026-06-26 session-burning bug. Before the fix,
    ``story_loop_node ⇄ patching_node`` could loop on the same story
    forever because ``_next_story_in_batch`` ordered in_progress rows
    first and nothing marked the just-patched story ``done`` between
    iterations. The only guard was a global zero-patch tripwire that
    fired only after the patcher gave up — a session-burning failure
    mode (~$18 + 80 min lost in the incident).

    Contract under test: when ``story_loop_node`` re-enters with a
    ``current_story_id`` set (i.e., patching just ran on it), the cursor
    advances to the next ready story in the batch, never re-selecting the
    same one — regardless of whether the previous patching turn applied
    patches, returned zero, or failed mid-flight."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}, {"title": "C"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    # First story_loop entry — picks STORY-001.
    out1 = story_loop.story_loop_node(
        _state(workspace, current_batch_id=batch_id)
    )
    assert out1["current_story_id"] == "STORY-001"
    assert out1["batch_patched_story_keys"] == []

    # Simulate the LangGraph channel layer: the next call sees
    # current_story_id="STORY-001" (patching_node didn't change it) and
    # batch_patched_story_keys is still empty until story_loop_node runs.
    # The patching_node turn is a no-op for this test — we only care that
    # story_loop_node advances the cursor on re-entry.
    out2 = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        batch_patched_story_keys=[],
    ))
    assert out2["current_story_id"] == "STORY-002"
    assert out2["batch_patched_story_keys"] == ["STORY-001"]

    # Third entry — STORY-002 just patched; STORY-003 is next.
    out3 = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-002",
        batch_patched_story_keys=["STORY-001"],
    ))
    assert out3["current_story_id"] == "STORY-003"
    assert sorted(out3["batch_patched_story_keys"]) == [
        "STORY-001", "STORY-002",
    ]

    # Fourth entry — STORY-003 patched, no stories left → batch_complete.
    out4 = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-003",
        batch_patched_story_keys=["STORY-001", "STORY-002"],
    ))
    assert out4["current_story_id"] == ""
    assert out4["node_state"]["batch_complete"] is True
    assert sorted(out4["batch_patched_story_keys"]) == [
        "STORY-001", "STORY-002", "STORY-003",
    ]


def test_batch_planner_resets_patched_keys_for_new_batch(workspace: str):
    """A new batch must start with an empty ``batch_patched_story_keys``
    so the cursor is free to pick any ready story by sequence. Without
    this reset, leftover keys from the previous batch would silently
    skip ready stories."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    out = story_loop.batch_planner_node(_state(workspace))
    assert out["batch_patched_story_keys"] == []


def test_story_loop_auto_complete_respects_cap_override(workspace: str):
    """state['story_zero_patch_cap'] overrides the default cap."""
    _seed_stories(workspace, [{"title": "A"}, {"title": "B"}])
    planned = story_loop.batch_planner_node(_state(workspace))
    batch_id = planned["current_batch_id"]

    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        story_zero_patch_cap=1,
        loop_counter={"story_zero_patch_rounds": {"STORY-001": 1}},
    ))

    assert out["node_state"]["auto_completed_story"] == "STORY-001"


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
        story_state.mark_in_progress(conn, app, "STORY-001")
    finally:
        conn.close()

    out = story_loop.story_loop_node(_state(
        workspace,
        current_batch_id=batch_id,
        current_story_id="STORY-001",
        loop_counter={
            "story_zero_patch_rounds": {"STORY-001": story_loop.STORY_ZERO_PATCH_CAP}
        },
    ))

    assert out["node_state"]["batch_complete"] is True
    assert out["node_state"]["blocked_count"] == 0
    assert out["current_story_id"] == ""

    # batch_commit_node is the sole sealer now; story_loop_node only
    # marks batch_complete on node_state so verification can run.
    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] != "complete"


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


def test_route_after_batch_planner_diverts_to_hitl_on_decomposition_missing():
    state = {"node_state": {"batch_planned": False, "decomposition_missing": True}}
    assert (
        story_loop.route_after_batch_planner(state) == "human_intervention_node"
    )


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
