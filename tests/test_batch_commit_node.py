"""Tests for batch_commit_node — per-batch sealing logic.

The node fires after the per-batch verification pipeline (compile /
review / test / security / regression) has passed. Tests cover:

- Marks all pending stories in the batch as ``done``.
- Leaves already-blocked stories alone; reports them in the count.
- Reports ``complete`` vs ``complete_with_blocks`` batch status.
- Optional git commit when ``commit_on_story`` flag is set.
- Records ``commits`` rows per constituent story.
- Persists committed_sha onto the batches row.
- Resets batch-scoped state cursor + per-batch loop counters.
- Skips cleanly when no batch is active.
- Router always returns ``batch_planner_node``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


from harness import story_state
from harness.story_loop import batch_commit_node, route_after_batch_commit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: str) -> None:
    """Bootstrap a git repo with author identity so commits succeed in CI."""
    subprocess.run(
        ["git", "init"], cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "init.defaultBranch", "main"],
        cwd=path, check=True, capture_output=True,
    )


def _seed_batch(
    workspace: str,
    *,
    session_id: str = "sess-1",
    stories: list[dict[str, Any]] | None = None,
) -> tuple[int, list[str]]:
    """Create a batch with the given stories. Returns (batch_id, keys)."""
    if stories is None:
        stories = [
            {"title": "Add login"},
            {"title": "Add logout"},
        ]
    app = story_state.app_name_for_workspace(workspace)
    conn = story_state.open_story_db()
    try:
        # v4: every story needs a feature. Auto-seed a single ``test``
        # feature and default unset stories to it.
        story_state.ensure_feature(conn, app, "test", name="Test feature")
        for s in stories:
            s.setdefault("feature", "test")
        keys = story_state.create_stories(conn, app, stories)
        bid = story_state.start_batch(conn, app, session_id, keys)
    finally:
        conn.close()
    return bid, keys


def _app(workspace: str) -> str:
    return story_state.app_name_for_workspace(workspace)


def _state(
    workspace: str, batch_id: int, **overrides: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "workspace_path": workspace,
        "session_id": "sess-1",
        "current_batch_id": batch_id,
        "current_story_id": "",
        "story_scope_files": [],
        "story_modified_baseline": [],
        "batch_modified_files": [],
        "commit_on_story": False,
        "loop_counter": {
            "patching": 5, "repair": 3, "compiler": 2, "total_repairs": 8,
            "review_code": 1, "review_spec": 0,
            "consecutive_zero_patch_rounds": 0,
            "missing_dep_consecutive_same": 0,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkipPaths:
    def test_skips_when_no_batch_id(self, tmp_path: Path):
        out = batch_commit_node(
            {"workspace_path": str(tmp_path), "current_batch_id": 0}
        )
        assert out["node_state"]["skipped"] is True
        assert out["node_state"]["reason"] == "no_batch_or_workspace"

    def test_skips_when_no_workspace(self, tmp_path: Path):
        out = batch_commit_node({"current_batch_id": 5})
        assert out["node_state"]["skipped"] is True


class TestStoryMarking:
    def test_marks_all_pending_stories_done(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, keys = _seed_batch(ws)
        out = batch_commit_node(_state(ws, bid))

        assert out["node_state"]["batch_id"] == bid
        assert out["node_state"]["marked_done"] == len(keys)
        assert out["node_state"]["blocked_count"] == 0

        app = _app(ws)
        conn = story_state.open_story_db()
        try:
            for k in keys:
                assert story_state.get_story(conn, app, k)["status"] == "done"
        finally:
            conn.close()

    def test_leaves_already_blocked_stories_blocked(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, keys = _seed_batch(ws, stories=[
            {"title": "A"}, {"title": "B"}, {"title": "C"},
        ])
        # Pre-block STORY-002 (e.g. a per-story repair-cap failure
        # carried into this batch).
        app = _app(ws)
        conn = story_state.open_story_db()
        try:
            story_state.mark_blocked(conn, app, "STORY-002")
        finally:
            conn.close()

        out = batch_commit_node(_state(ws, bid))
        assert out["node_state"]["marked_done"] == 2  # STORY-001 + STORY-003
        assert out["node_state"]["blocked_count"] == 1

        conn = story_state.open_story_db()
        try:
            assert story_state.get_story(conn, app, "STORY-001")["status"] == "done"
            assert story_state.get_story(conn, app, "STORY-002")["status"] == "blocked"
            assert story_state.get_story(conn, app, "STORY-003")["status"] == "done"
        finally:
            conn.close()

    def test_batch_status_complete_when_no_blocks(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws)
        batch_commit_node(_state(ws, bid))
        conn = story_state.open_story_db()
        try:
            row = conn.execute(
                "SELECT status FROM batches WHERE id = ?", (bid,)
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "complete"

    def test_batch_status_with_blocks_when_some_blocked(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws)
        app = _app(ws)
        conn = story_state.open_story_db()
        try:
            story_state.mark_blocked(conn, app, "STORY-001")
        finally:
            conn.close()
        batch_commit_node(_state(ws, bid))
        conn = story_state.open_story_db()
        try:
            row = conn.execute(
                "SELECT status FROM batches WHERE id = ?", (bid,)
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "complete_with_blocks"

    def test_resolves_open_defects_per_story(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws, stories=[{"title": "Solo"}])
        app = _app(ws)
        conn = story_state.open_story_db()
        try:
            story_state.record_defect(
                conn, workspace=app, story_key="STORY-001", session_id="sess-1",
                severity="compile", summary="leftover open defect",
            )
        finally:
            conn.close()

        batch_commit_node(_state(ws, bid))

        conn = story_state.open_story_db()
        try:
            row = conn.execute(
                "SELECT status FROM defects ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "resolved"


class TestStateReset:
    def test_resets_batch_scoped_state(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws)
        out = batch_commit_node(_state(
            ws, bid,
            current_story_id="STORY-001",
            story_scope_files=["a.py"],
            story_modified_baseline=["b.py"],
            batch_modified_files=["x.py", "y.py"],
        ))
        assert out["current_batch_id"] == 0
        assert out["current_story_id"] == ""
        assert out["story_scope_files"] == []
        assert out["story_modified_baseline"] == []
        assert out["batch_modified_files"] == []
        # The pre-reset batch file list is still reported in node_state
        # for telemetry / log inspection.
        assert out["node_state"]["batch_files"] == ["x.py", "y.py"]

    def test_resets_per_batch_loop_counters(self, tmp_path: Path):
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws)
        out = batch_commit_node(_state(ws, bid))
        for key in (
            "patching", "repair", "compiler", "total_repairs",
            "review_spec", "review_code",
            "consecutive_zero_patch_rounds", "missing_dep_consecutive_same",
        ):
            assert out["loop_counter"][key] == 0, (
                f"counter {key} should be reset to 0 between batches"
            )


class TestGitCommit:
    def test_no_commit_when_flag_off(self, tmp_path: Path):
        ws = str(tmp_path)
        _init_git_repo(ws)
        (Path(ws) / "auth.py").write_text("x\n")
        bid, _ = _seed_batch(ws)
        out = batch_commit_node(_state(ws, bid, commit_on_story=False))
        assert out["node_state"]["committed_sha"] is None

    def test_no_commit_when_not_a_git_repo(self, tmp_path: Path):
        # commit_on_story is set but the workspace isn't a repo —
        # non-fatal, batch still seals, SHA is None.
        ws = str(tmp_path)
        bid, _ = _seed_batch(ws)
        out = batch_commit_node(_state(ws, bid, commit_on_story=True))
        assert out["node_state"]["committed_sha"] is None
        # Stories still marked done — commit failure shouldn't block sealing.
        app = _app(ws)
        conn = story_state.open_story_db()
        try:
            assert story_state.get_story(conn, app, "STORY-001")["status"] == "done"
        finally:
            conn.close()

    def test_commits_when_flag_on_and_repo_has_changes(self, tmp_path: Path):
        ws = str(tmp_path)
        _init_git_repo(ws)
        (Path(ws) / "auth.py").write_text("x\n")
        bid, keys = _seed_batch(ws, stories=[
            {"title": "Add login"}, {"title": "Add logout"},
        ])

        out = batch_commit_node(_state(
            ws, bid,
            commit_on_story=True,
            batch_modified_files=["auth.py"],
        ))
        sha = out["node_state"]["committed_sha"]
        assert sha is not None
        assert len(sha) == 40

        # Verify the persisted SHA on the batch row + one shared
        # commits row with story_id NULL (batch-level commit) whose
        # message lists every constituent story.
        conn = story_state.open_story_db()
        try:
            row = conn.execute(
                "SELECT committed_sha FROM batches WHERE id = ?", (bid,)
            ).fetchone()
            assert row[0] == sha
            rows = conn.execute(
                "SELECT story_id, message FROM commits WHERE sha = ?", (sha,)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] is None
            assert rows[0][1].startswith(f"BATCH-{bid}:")
            assert "STORY-001" in rows[0][1] and "STORY-002" in rows[0][1]
        finally:
            conn.close()

        # Commit message lists the batch + constituent stories.
        log = subprocess.run(
            ["git", "log", "--pretty=%s"],
            cwd=ws, capture_output=True, text=True, check=True,
        )
        first_line = log.stdout.splitlines()[0]
        assert first_line.startswith(f"BATCH-{bid}:")
        assert "STORY-001" in first_line
        assert "STORY-002" in first_line


class TestRouter:
    def test_route_after_batch_commit_returns_batch_planner(self):
        assert route_after_batch_commit({}) == "batch_planner_node"
