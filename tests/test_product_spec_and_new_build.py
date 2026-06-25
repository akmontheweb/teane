"""Tests for the product_spec_dir mandate and --new-build cleanup.

Covers:
- _resolve_product_spec_dir (relative vs absolute paths, ~ expansion).
- _spec_dir_workspace_top_level (inside, nested, outside the workspace).
- _load_consolidated_product_spec (missing dir, empty dir, single .txt,
  multiple .txt → alphabetical consolidation).
- _perform_new_build_reset (delete non-preserved entries, preserve the
  workspace-rooted spec folder, commit on master, delete orphan
  agent/patch-* branches).
"""

from __future__ import annotations

import os
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Path resolution + workspace-top-level helpers
# ---------------------------------------------------------------------------

class TestValidateProductSpecDirName:
    """The value must be a bare folder name suitable for use directly
    under the workspace root. Validation rejects path separators,
    absolute paths, `..` traversal, `~` shorthand, and `.`/`..` themselves."""

    def test_simple_name_passes(self):
        from harness.cli import _validate_product_spec_dir_name
        assert _validate_product_spec_dir_name("product_spec") is None

    def test_alphanumeric_with_dashes_passes(self):
        from harness.cli import _validate_product_spec_dir_name
        assert _validate_product_spec_dir_name("my-spec_1") is None

    def test_whitespace_trimmed_then_validated(self):
        from harness.cli import _validate_product_spec_dir_name
        assert _validate_product_spec_dir_name("  product_spec  ") is None

    def test_empty_string_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name("")
        assert msg is not None and "non-empty" in msg

    def test_absolute_path_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name("/home/me/specs")
        assert msg is not None and "no leading" in msg

    def test_relative_path_with_slash_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name("docs/spec")
        assert msg is not None and "path separators" in msg

    def test_backslash_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name("docs\\spec")
        assert msg is not None and "path separators" in msg

    def test_tilde_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name("~/specs")
        assert msg is not None
        # Either rejected as ~ or as path separator (~/ contains /).
        assert "~" in msg or "path separators" in msg

    def test_dot_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        assert _validate_product_spec_dir_name(".") is not None

    def test_double_dot_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        assert _validate_product_spec_dir_name("..") is not None

    def test_non_string_rejected(self):
        from harness.cli import _validate_product_spec_dir_name
        msg = _validate_product_spec_dir_name(123)  # type: ignore[arg-type]
        assert msg is not None and "must be a string" in msg


class TestResolveProductSpecDir:
    """Resolution joins the bare name with the workspace root. The name
    has already been validated separately; this is pure path arithmetic."""

    def test_name_joined_with_workspace(self):
        from harness.cli import _resolve_product_spec_dir
        assert _resolve_product_spec_dir("/work", "product_spec") == "/work/product_spec"

    def test_strips_whitespace(self):
        from harness.cli import _resolve_product_spec_dir
        assert _resolve_product_spec_dir("/work", "  product_spec  ") == "/work/product_spec"


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

class TestLoadConsolidatedProductSpec:
    def test_missing_directory_returns_none(self, tmp_path, capsys):
        from harness.cli import _load_consolidated_product_spec
        spec_dir = tmp_path / "does-not-exist"
        result = _load_consolidated_product_spec(str(tmp_path), str(spec_dir))
        assert result is None
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_empty_directory_returns_none(self, tmp_path, capsys):
        from harness.cli import _load_consolidated_product_spec
        spec_dir = tmp_path / "product_spec"
        spec_dir.mkdir()
        result = _load_consolidated_product_spec(str(tmp_path), str(spec_dir))
        assert result is None
        err = capsys.readouterr().err
        assert "no .txt files" in err

    def test_single_txt_file_returns_consolidated_content(self, tmp_path):
        from harness.cli import _load_consolidated_product_spec
        spec_dir = tmp_path / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("Build a TODO app.")
        result = _load_consolidated_product_spec(str(tmp_path), str(spec_dir))
        assert result is not None
        assert "main.txt" in result
        assert "Build a TODO app." in result
        assert "consolidated from 1 file(s)" in result

    def test_multiple_txt_files_alphabetical_order(self, tmp_path):
        from harness.cli import _load_consolidated_product_spec
        spec_dir = tmp_path / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "z_last.txt").write_text("Z content")
        (spec_dir / "a_first.txt").write_text("A content")
        (spec_dir / "m_middle.txt").write_text("M content")
        result = _load_consolidated_product_spec(str(tmp_path), str(spec_dir))
        assert result is not None
        # alphabetical: a, m, z
        a_pos = result.index("A content")
        m_pos = result.index("M content")
        z_pos = result.index("Z content")
        assert a_pos < m_pos < z_pos
        assert "consolidated from 3 file(s)" in result

    def test_non_txt_files_ignored(self, tmp_path):
        from harness.cli import _load_consolidated_product_spec
        spec_dir = tmp_path / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("Real content.")
        (spec_dir / "ignored.md").write_text("Markdown skipped")
        (spec_dir / "also_ignored.json").write_text('{"foo": "bar"}')
        result = _load_consolidated_product_spec(str(tmp_path), str(spec_dir))
        assert result is not None
        assert "Real content." in result
        assert "Markdown skipped" not in result
        assert '"foo"' not in result

    # External-spec test removed: product_spec_dir is now mandated to live
    # at the workspace root. See TestValidateProductSpecDirName.


# ---------------------------------------------------------------------------
# --new-build reset
# ---------------------------------------------------------------------------

def _init_repo(path: str) -> None:
    """Initialise a git repo with master HEAD and a couple of files."""
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "-b", "master"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", path, "config", "user.email", "test@example.com"],
                   check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", path, "commit", "--allow-empty", "-q", "-m", "init"],
                   check=True)


class TestPerformNewBuildReset:
    def test_deletes_non_preserved_entries_and_keeps_spec(self, tmp_path):
        from harness.cli import _perform_new_build_reset
        ws = tmp_path / "ws"
        ws.mkdir()
        _init_repo(str(ws))
        # Create stale files + a product_spec/ folder + a .gitignore.
        (ws / "stale_main.py").write_text("print('stale')")
        (ws / "stale_dir").mkdir()
        (ws / "stale_dir" / "thing.txt").write_text("stale")
        (ws / ".gitignore").write_text("__pycache__")
        spec_dir = ws / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("Build a TODO app.")
        # Commit them so they're "stale-on-master".
        subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(ws), "commit", "-q", "-m", "stale"], check=True)

        _perform_new_build_reset(str(ws), "product_spec")

        # product_spec/ + .git/ preserved; everything else deleted.
        assert (ws / "product_spec" / "main.txt").exists()
        assert (ws / ".git").is_dir()
        assert not (ws / "stale_main.py").exists()
        assert not (ws / "stale_dir").exists()
        assert not (ws / ".gitignore").exists()

        # A commit landed on master with the deletions.
        log = subprocess.run(
            ["git", "-C", str(ws), "log", "--oneline"],
            capture_output=True, text=True, check=True,
        )
        assert "--new-build reset" in log.stdout

    def test_orphan_patch_branches_deleted(self, tmp_path):
        from harness.cli import _perform_new_build_reset
        ws = tmp_path / "ws"
        ws.mkdir()
        _init_repo(str(ws))
        spec_dir = ws / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("spec")
        subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(ws), "commit", "-q", "-m", "spec"], check=True)

        # Create three orphan branches.
        for sid in ("abc123", "def456", "789xyz"):
            subprocess.run(
                ["git", "-C", str(ws), "branch", f"agent/patch-{sid}"],
                check=True, capture_output=True,
            )

        _perform_new_build_reset(str(ws), "product_spec")

        # All orphans gone.
        branches = subprocess.run(
            ["git", "-C", str(ws), "branch"],
            capture_output=True, text=True, check=True,
        ).stdout
        assert "agent/patch-" not in branches

    def test_non_git_workspace_is_a_no_op(self, tmp_path):
        # No git repo → log a warning and return without touching anything.
        from harness.cli import _perform_new_build_reset
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "stale.py").write_text("survives because no git")
        spec_dir = ws / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("spec")

        _perform_new_build_reset(str(ws), "product_spec")

        assert (ws / "stale.py").exists()


# ---------------------------------------------------------------------------
# Checkpoint + JSONL purge (--new-build true)
# ---------------------------------------------------------------------------

def _make_summary(thread_id: str, workspace_path: str):
    """Build a minimal CheckpointSummary matching the storage dataclass."""
    from harness.storage import CheckpointSummary
    return CheckpointSummary(
        thread_id=thread_id,
        session_id=thread_id,
        workspace_path=workspace_path,
    )


class _FakeCheckpointer:
    """Stub for HarnessAsyncSqliteSaver. Records every adelete_thread call."""
    def __init__(self):
        self.deleted: list[str] = []
        self.closed = False
        # Mimic the .conn.close() coroutine the production code calls.
        outer = self
        class _Conn:
            async def close(self):
                outer.closed = True
        self.conn = _Conn()

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class TestPurgeWorkspaceCheckpoints:
    @pytest.mark.asyncio
    async def test_no_db_file_is_no_op(self, tmp_path, caplog):
        from harness.cli import _purge_workspace_checkpoints
        config = {"persistence": {"db_path": str(tmp_path / "missing.db")}}
        await _purge_workspace_checkpoints(str(tmp_path / "ws"), config)
        # Nothing to assert beyond "no crash" — function logs and returns.

    @pytest.mark.asyncio
    async def test_matching_sessions_deleted_others_skipped(
        self, tmp_path, monkeypatch,
    ):
        # Set up a fake DB file so the no-op early-return doesn't fire.
        db_file = tmp_path / "checkpoints.db"
        db_file.write_text("")  # presence is all that's checked
        ws = str(tmp_path / "target_ws")
        os.makedirs(ws)
        other_ws = str(tmp_path / "other_ws")
        os.makedirs(other_ws)

        sessions = [
            _make_summary("sess-match-1", ws),
            _make_summary("sess-other", other_ws),
            _make_summary("sess-match-2", ws),
            _make_summary("sess-no-path", ""),
        ]

        async def fake_list_all_sessions(db_path: str, limit: int = 50):
            return list(sessions)

        fake_checkpointer = _FakeCheckpointer()
        async def fake_from_db_path(*, db_path: str, ttl_days: int):
            return fake_checkpointer

        # Patch the names that _purge_workspace_checkpoints imports at runtime.
        from harness import storage
        monkeypatch.setattr(storage, "list_all_sessions", fake_list_all_sessions)
        monkeypatch.setattr(
            storage.HarnessAsyncSqliteSaver, "from_db_path", fake_from_db_path,
        )

        from harness.cli import _purge_workspace_checkpoints
        await _purge_workspace_checkpoints(ws, {"persistence": {"db_path": str(db_file)}})

        # Only the matching sessions got deleted; other workspace + empty
        # workspace_path entries were skipped.
        assert sorted(fake_checkpointer.deleted) == ["sess-match-1", "sess-match-2"]
        assert fake_checkpointer.closed is True

    @pytest.mark.asyncio
    async def test_jsonl_logs_removed_for_matching_sessions(
        self, tmp_path, monkeypatch,
    ):
        db_file = tmp_path / "checkpoints.db"
        db_file.write_text("")
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        other_ws = str(tmp_path / "other_ws")
        os.makedirs(other_ws)
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # Logs for the matching session + a rotated backup + an unrelated session.
        (log_dir / "sess-match.jsonl").write_text("entry1")
        (log_dir / "sess-match.jsonl.1").write_text("entry0")
        (log_dir / "sess-other.jsonl").write_text("entryOther")

        async def fake_list_all_sessions(db_path: str, limit: int = 50):
            return [
                _make_summary("sess-match", ws),
                _make_summary("sess-other", other_ws),
            ]

        fake_checkpointer = _FakeCheckpointer()
        async def fake_from_db_path(*, db_path: str, ttl_days: int):
            return fake_checkpointer

        from harness import storage
        monkeypatch.setattr(storage, "list_all_sessions", fake_list_all_sessions)
        monkeypatch.setattr(
            storage.HarnessAsyncSqliteSaver, "from_db_path", fake_from_db_path,
        )

        from harness.cli import _purge_workspace_checkpoints
        await _purge_workspace_checkpoints(
            ws,
            {
                "persistence": {"db_path": str(db_file)},
                "logging": {"log_dir": str(log_dir)},
            },
        )

        # Matching session's .jsonl + rotation gone; other session's log intact.
        assert not (log_dir / "sess-match.jsonl").exists()
        assert not (log_dir / "sess-match.jsonl.1").exists()
        assert (log_dir / "sess-other.jsonl").exists()

    @pytest.mark.asyncio
    async def test_realpath_match_handles_symlinks(self, tmp_path, monkeypatch):
        # If the workspace is reached via a symlink alias, realpath should
        # still match the stored canonical path.
        db_file = tmp_path / "checkpoints.db"
        db_file.write_text("")
        real_ws = tmp_path / "real_ws"
        real_ws.mkdir()
        symlink_ws = tmp_path / "alias_ws"
        try:
            os.symlink(str(real_ws), str(symlink_ws))
        except OSError:
            pytest.skip("symlinks unavailable on this platform")

        async def fake_list_all_sessions(db_path: str, limit: int = 50):
            return [_make_summary("sess-by-real", str(real_ws))]

        fake_checkpointer = _FakeCheckpointer()
        async def fake_from_db_path(*, db_path: str, ttl_days: int):
            return fake_checkpointer

        from harness import storage
        monkeypatch.setattr(storage, "list_all_sessions", fake_list_all_sessions)
        monkeypatch.setattr(
            storage.HarnessAsyncSqliteSaver, "from_db_path", fake_from_db_path,
        )

        from harness.cli import _purge_workspace_checkpoints
        await _purge_workspace_checkpoints(
            str(symlink_ws),
            {"persistence": {"db_path": str(db_file)}},
        )
        # The session stored against the canonical path was matched and deleted.
        assert fake_checkpointer.deleted == ["sess-by-real"]

    @pytest.mark.asyncio
    async def test_no_matching_sessions_no_deletions(self, tmp_path, monkeypatch):
        db_file = tmp_path / "checkpoints.db"
        db_file.write_text("")
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        other_ws = str(tmp_path / "other_ws")
        os.makedirs(other_ws)

        async def fake_list_all_sessions(db_path: str, limit: int = 50):
            return [_make_summary("sess-other", other_ws)]

        fake_checkpointer = _FakeCheckpointer()
        async def fake_from_db_path(*, db_path: str, ttl_days: int):
            return fake_checkpointer

        from harness import storage
        monkeypatch.setattr(storage, "list_all_sessions", fake_list_all_sessions)
        monkeypatch.setattr(
            storage.HarnessAsyncSqliteSaver, "from_db_path", fake_from_db_path,
        )

        from harness.cli import _purge_workspace_checkpoints
        await _purge_workspace_checkpoints(ws, {"persistence": {"db_path": str(db_file)}})
        # Helper returns early when no matches — checkpointer never opened.
        assert fake_checkpointer.deleted == []
        assert fake_checkpointer.closed is False


# ---------------------------------------------------------------------------
# Preview helpers + HITL confirmation gate
# ---------------------------------------------------------------------------

class TestListWorkspaceEntriesToDelete:
    def test_lists_non_preserved_entries(self, tmp_path):
        from harness.cli import _list_workspace_entries_to_delete
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / ".git").mkdir()
        (ws / "product_spec").mkdir()
        (ws / "stale.py").write_text("x")
        (ws / "old_dir").mkdir()

        result = _list_workspace_entries_to_delete(str(ws), "product_spec")
        assert sorted(result) == ["old_dir", "stale.py"]

    def test_empty_workspace_returns_empty(self, tmp_path):
        from harness.cli import _list_workspace_entries_to_delete
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _list_workspace_entries_to_delete(str(ws), "product_spec") == []

    def test_missing_workspace_returns_empty(self, tmp_path):
        from harness.cli import _list_workspace_entries_to_delete
        # Path doesn't exist — return [] rather than crash.
        result = _list_workspace_entries_to_delete(
            str(tmp_path / "does-not-exist"), "product_spec",
        )
        assert result == []


class TestListOrphanPatchBranches:
    def test_lists_agent_patch_branches(self, tmp_path):
        from harness.cli import _list_orphan_patch_branches
        ws = tmp_path / "ws"
        ws.mkdir()
        _init_repo(str(ws))
        for sid in ("abc", "def", "ghi"):
            subprocess.run(
                ["git", "-C", str(ws), "branch", f"agent/patch-{sid}"],
                check=True, capture_output=True,
            )
        result = _list_orphan_patch_branches(str(ws))
        assert sorted(result) == [
            "agent/patch-abc", "agent/patch-def", "agent/patch-ghi",
        ]

    def test_no_orphans_returns_empty(self, tmp_path):
        from harness.cli import _list_orphan_patch_branches
        ws = tmp_path / "ws"
        ws.mkdir()
        _init_repo(str(ws))
        assert _list_orphan_patch_branches(str(ws)) == []

    def test_non_git_returns_empty(self, tmp_path):
        from harness.cli import _list_orphan_patch_branches
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _list_orphan_patch_branches(str(ws)) == []


class TestListWorkspaceCheckpointSessions:
    @pytest.mark.asyncio
    async def test_filters_by_realpath(self, tmp_path, monkeypatch):
        # Stub list_all_sessions to return a mix of matching + non-matching.
        from harness.cli import _list_workspace_checkpoint_sessions
        from harness import storage

        db_file = tmp_path / "checkpoints.db"
        db_file.write_text("")
        ws = str(tmp_path / "target_ws")
        os.makedirs(ws)
        other = str(tmp_path / "other_ws")
        os.makedirs(other)

        sessions = [
            _make_summary("sess-match-1", ws),
            _make_summary("sess-other", other),
            _make_summary("sess-match-2", ws),
            _make_summary("sess-no-path", ""),
        ]

        async def fake_list_all_sessions(db_path, limit=50):
            return list(sessions)
        monkeypatch.setattr(storage, "list_all_sessions", fake_list_all_sessions)

        result = await _list_workspace_checkpoint_sessions(
            ws, {"persistence": {"db_path": str(db_file)}},
        )
        assert sorted(s.thread_id for s in result) == ["sess-match-1", "sess-match-2"]

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self, tmp_path):
        from harness.cli import _list_workspace_checkpoint_sessions
        result = await _list_workspace_checkpoint_sessions(
            str(tmp_path / "ws"),
            {"persistence": {"db_path": str(tmp_path / "missing.db")}},
        )
        assert result == []


class TestNewBuildPreviewPrinting:
    def test_preview_lists_every_section(self, tmp_path, capsys):
        from harness.cli import _print_new_build_preview
        _print_new_build_preview(
            workspace_path="/tmp/ws",
            spec_dirname="product_spec",
            files_to_delete=["stale.py", "old_dir"],
            orphan_branches=["agent/patch-abc"],
            checkpoint_sessions=[_make_summary("sess-1", "/tmp/ws")],
        )
        err = capsys.readouterr().err
        assert "--new-build true — REVIEW BEFORE PROCEEDING" in err
        assert "/tmp/ws" in err
        assert "product_spec" in err
        assert "stale.py" in err
        assert "old_dir" in err
        assert "agent/patch-abc" in err
        assert "sess-1" in err

    def test_preview_shows_none_when_empty(self, capsys):
        from harness.cli import _print_new_build_preview
        _print_new_build_preview(
            workspace_path="/tmp/ws",
            spec_dirname="product_spec",
            files_to_delete=[],
            orphan_branches=[],
            checkpoint_sessions=[],
        )
        err = capsys.readouterr().err
        assert "Workspace files to delete: none." in err
        assert "Orphan agent/patch-* branches: none." in err
        assert "Checkpoint sessions for this workspace: none." in err
        assert "Story state DB for this workspace: none." in err
        assert "Repo index rows for this workspace: none." in err


# ---------------------------------------------------------------------------
# state.db purge (--new-build true)
# ---------------------------------------------------------------------------

class TestPurgeStateDb:
    def test_no_state_db_is_a_no_op(self, tmp_path):
        from harness.story_state import purge_state_db
        # Global state.db doesn't exist yet (no open ever happened).
        out = purge_state_db(str(tmp_path / "ws"))
        assert all(v == 0 for v in out.values())

    def test_deletes_only_target_app_rows(self, tmp_path):
        """The purge wipes rows for this workspace's app name but
        leaves every other app's rows intact."""
        from harness.story_state import (
            app_name_for_workspace, create_stories, list_stories,
            open_story_db, purge_state_db,
        )
        ws_target = tmp_path / "target-app"
        ws_other = tmp_path / "other-app"
        ws_target.mkdir()
        ws_other.mkdir()
        app_target = app_name_for_workspace(str(ws_target))
        app_other = app_name_for_workspace(str(ws_other))

        from harness.story_state import ensure_feature
        conn = open_story_db()
        try:
            ensure_feature(conn, app_target, "test", name="test")
            ensure_feature(conn, app_other, "test", name="test")
            create_stories(conn, app_target, [
                {"title": "T1", "feature": "test"},
                {"title": "T2", "feature": "test"},
            ])
            create_stories(conn, app_other, [
                {"title": "O1", "feature": "test"},
            ])
        finally:
            conn.close()

        out = purge_state_db(str(ws_target))
        assert out["stories"] == 2

        conn = open_story_db()
        try:
            assert list_stories(conn, app_target) == []
            assert [s["title"] for s in list_stories(conn, app_other)] == ["O1"]
        finally:
            conn.close()

    def test_purge_followed_by_open_starts_fresh(self, tmp_path):
        from harness.story_state import (
            app_name_for_workspace, create_stories, list_stories,
            open_story_db, purge_state_db,
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        app = app_name_for_workspace(str(ws))
        from harness.story_state import ensure_feature
        conn = open_story_db()
        try:
            ensure_feature(conn, app, "test", name="test")
            create_stories(conn, app, [{"title": "first", "feature": "test"}])
        finally:
            conn.close()

        purge_state_db(str(ws))

        conn2 = open_story_db()
        try:
            assert list_stories(conn2, app) == []
        finally:
            conn2.close()

    def test_purge_returns_per_table_counts(self, tmp_path):
        """The returned dict carries row counts for every workspace
        table so callers can log a meaningful summary."""
        from harness.story_state import (
            app_name_for_workspace, create_stories, link_file,
            open_story_db, purge_state_db, record_commit, record_defect,
            start_batch,
        )
        from harness.story_state import ensure_feature
        ws = tmp_path / "counts-ws"
        ws.mkdir()
        app = app_name_for_workspace(str(ws))
        conn = open_story_db()
        try:
            ensure_feature(conn, app, "test", name="test")
            create_stories(conn, app, [{"title": "T", "feature": "test"}])
            bid = start_batch(conn, app, "sess-1", ["STORY-1"])
            link_file(conn, app, "STORY-1", "a.py", "code", batch_id=bid)
            record_defect(
                conn, workspace=app, story_key="STORY-1",
                session_id="sess-1", severity="x", summary="y",
            )
            record_commit(
                conn, workspace=app, sha="abc", story_key="STORY-1",
                session_id="sess-1", message="x",
            )
        finally:
            conn.close()

        out = purge_state_db(str(ws))
        assert out["stories"] == 1
        assert out["batches"] == 1
        assert out["file_links"] == 1
        assert out["defects"] == 1
        assert out["commits"] == 1


# ---------------------------------------------------------------------------
# repo_index workspace purge (--new-build true)
# ---------------------------------------------------------------------------

class TestPurgeRepoIndex:
    def test_no_db_is_a_no_op(self, tmp_path, monkeypatch):
        from harness.repo_index import RepoIndexConfig, purge_workspace
        monkeypatch.setattr(
            "harness.repo_index.RepoIndexConfig",
            lambda: RepoIndexConfig(index_dir=str(tmp_path / "missing")),
        )
        assert purge_workspace(str(tmp_path / "ws")) == (0, 0)

    def test_deletes_only_target_workspace_rows(self, tmp_path):
        import sqlite3
        from harness.repo_index import (
            RepoIndexConfig, _db_path, _workspace_id, purge_workspace,
        )
        cfg = RepoIndexConfig(index_dir=str(tmp_path / "idx"))
        os.makedirs(cfg.index_dir, exist_ok=True)
        db = _db_path(cfg)
        ws_target = "/tmp/ws-target"
        ws_other = "/tmp/ws-other"
        wid_target = _workspace_id(ws_target)
        wid_other = _workspace_id(ws_other)

        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS repo_meta (
                workspace_id TEXT PRIMARY KEY,
                backend TEXT NOT NULL,
                idf_json TEXT,
                built_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS repo_chunks (
                workspace_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                file_sha TEXT NOT NULL,
                content TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, file_path, chunk_index)
            );
        """)
        for wid in (wid_target, wid_other):
            conn.execute(
                "INSERT INTO repo_meta(workspace_id, backend, built_at, "
                "chunk_count) VALUES(?, 'tfidf', '2026-06-24', 2)",
                (wid,),
            )
            for i in range(2):
                conn.execute(
                    "INSERT INTO repo_chunks(workspace_id, file_path, "
                    "chunk_index, file_sha, content, vector_json) "
                    "VALUES(?, 'f.py', ?, 'sha', 'x', '{}')",
                    (wid, i),
                )
        conn.commit()
        conn.close()

        meta_n, chunk_n = purge_workspace(ws_target, cfg)
        assert meta_n == 1
        assert chunk_n == 2

        conn = sqlite3.connect(db)
        try:
            remaining_meta = {
                row[0] for row in
                conn.execute("SELECT workspace_id FROM repo_meta")
            }
            remaining_chunks = {
                row[0] for row in
                conn.execute("SELECT workspace_id FROM repo_chunks")
            }
        finally:
            conn.close()
        assert remaining_meta == {wid_other}
        assert remaining_chunks == {wid_other}


# ---------------------------------------------------------------------------
# _perform_new_build_reset calls state.db purge
# ---------------------------------------------------------------------------

class TestNewBuildResetWipesStateDb:
    def test_app_rows_gone_after_reset(self, tmp_path):
        from harness.cli import _perform_new_build_reset
        from harness.story_state import (
            app_name_for_workspace, create_stories, list_stories,
            open_story_db,
        )
        ws = tmp_path / "reset-ws"
        ws.mkdir()
        _init_repo(str(ws))
        spec_dir = ws / "product_spec"
        spec_dir.mkdir()
        (spec_dir / "main.txt").write_text("spec")
        subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(ws), "commit", "-q", "-m", "spec"], check=True,
        )

        # Seed prior-session rows for THIS workspace AND a sibling.
        app_target = app_name_for_workspace(str(ws))
        ws_other = tmp_path / "untouched"
        ws_other.mkdir()
        app_other = app_name_for_workspace(str(ws_other))
        from harness.story_state import ensure_feature
        conn = open_story_db()
        try:
            ensure_feature(conn, app_target, "test", name="test")
            ensure_feature(conn, app_other, "test", name="test")
            create_stories(
                conn, app_target,
                [{"title": "prior session", "feature": "test"}],
            )
            create_stories(
                conn, app_other,
                [{"title": "leave me alone", "feature": "test"}],
            )
        finally:
            conn.close()

        _perform_new_build_reset(str(ws), "product_spec")

        # Target rows are gone. Sibling app's rows survive.
        conn = open_story_db()
        try:
            assert list_stories(conn, app_target) == []
            assert [s["title"] for s in list_stories(conn, app_other)] == [
                "leave me alone"
            ]
        finally:
            conn.close()

    def test_state_db_purge_runs_even_without_git(self, tmp_path):
        from harness.cli import _perform_new_build_reset
        from harness.story_state import (
            app_name_for_workspace, create_stories, list_stories,
            open_story_db,
        )
        ws = tmp_path / "no-git-ws"
        ws.mkdir()
        # No `_init_repo` — _perform_new_build_reset's git-mode branch
        # returns early after the state.db purge step.
        app = app_name_for_workspace(str(ws))
        from harness.story_state import ensure_feature
        conn = open_story_db()
        try:
            ensure_feature(conn, app, "test", name="test")
            create_stories(conn, app, [{"title": "prior", "feature": "test"}])
        finally:
            conn.close()

        _perform_new_build_reset(str(ws), "product_spec")

        conn = open_story_db()
        try:
            assert list_stories(conn, app) == []
        finally:
            conn.close()
