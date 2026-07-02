"""End-to-end test for ``teane purge --all``.

Confirms the ``--all`` branch wipes every store the harness owns:
checkpoint DB, state.db (stories/features/batches/defects/etc.), repo
index, and JSONL session logs. Regression guard for the class of bug
where ``--all`` deleted only the checkpoint DB and left stale story /
index / log state behind.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest


def _min_config(tmp_path: Path) -> dict:
    """Smallest config that ``validate_config_strict`` accepts, with
    persistence + logging + repo_index pointed into ``tmp_path``."""
    return {
        "allow_network": True,
        "product_spec_dir": "product_spec",
        "sandbox": {"backend": "auto"},
        "token_budget": {"hard_cap_usd": 2.0},
        "persistence": {"db_path": str(tmp_path / "checkpoints.db")},
        "logging": {"log_dir": str(tmp_path / "logs")},
        "repo_index": {"enabled": True, "index_dir": str(tmp_path / "repo_index")},
        "models": {
            "openai:gpt-4o-mini": {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "api_key": "",
            },
        },
        "model_routing": {
            "planning_primary": "openai:gpt-4o-mini",
            "patching_primary": "openai:gpt-4o-mini",
            "repair_primary": "openai:gpt-4o-mini",
        },
    }


def _seed_checkpoints_db(db_path: Path) -> None:
    """Create a minimal checkpoints DB with rows in both tables so
    ``purge_checkpoints`` has something to delete."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE checkpoints ("
            "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, "
            "metadata BLOB)"
        )
        conn.execute(
            "CREATE TABLE writes ("
            "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES ('t1','','c1',NULL,'','','')"
        )
        conn.execute(
            "INSERT INTO writes VALUES ('t1','','c1','task',0,'ch','','')"
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_cmd_purge_all_wipes_every_store(tmp_path, monkeypatch):
    from harness import cli as cli_mod
    from harness import story_state
    from harness.repo_index import RepoIndexConfig, build_index, get_stats

    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")

    # 1. Canonical config JSON — discover_config will validate this.
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_min_config(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(cfg_path))

    # 2. Workspace directory used as CWD.
    workspace = tmp_path / "ws-app"
    workspace.mkdir()

    # 3. Seed checkpoints.db.
    checkpoints_db = tmp_path / "checkpoints.db"
    _seed_checkpoints_db(checkpoints_db)

    # 4. Seed state.db via story_state.
    conn = story_state.open_story_db()
    try:
        app = story_state.app_name_for_workspace(str(workspace))
        story_state.ensure_feature(conn, app, "test", name="Test feature")
        story_state.create_stories(conn, app, [{
            "title": "S1", "feature": "test",
            "acceptance_criteria": ["AC1"],
        }])
    finally:
        conn.close()
    assert sqlite3.connect(str(story_state.state_db_path())).execute(
        "SELECT COUNT(*) FROM stories"
    ).fetchone()[0] == 1

    # 5. Seed repo_index.db pointing at the tmp index_dir declared in
    #    _min_config so cmd_purge's RepoIndexConfig.from_config picks it up.
    index_dir = tmp_path / "repo_index"
    (workspace / "hello.py").write_text("def hi(): return 1\n")
    rcfg = RepoIndexConfig(
        enabled=True, top_k=3, chunk_lines=50, index_dir=str(index_dir),
    )
    build_index(str(workspace), rcfg)
    assert get_stats(str(workspace), rcfg) is not None

    # 6. Seed JSONL logs.
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "sess-a.jsonl").write_text('{"e":1}\n')
    (log_dir / "sess-a.jsonl.1").write_text('{"e":2}\n')
    (log_dir / "sess-b.jsonl").write_text('{"e":3}\n')

    # 7. Auto-confirm the destructive prompt.
    class _YesChannel:
        def confirm(self, msg, default=False):
            return True

    monkeypatch.setattr("harness.hitl.get_channel", lambda: _YesChannel())

    # 8. Run cmd_purge --all.
    args = argparse.Namespace(
        all=True, session_id=None, workspace=str(workspace),
    )
    rc = await cli_mod.cmd_purge(args)
    assert rc == 0

    # 9. Every store must be empty now.
    ck_conn = sqlite3.connect(str(checkpoints_db))
    try:
        assert ck_conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert ck_conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0
    finally:
        ck_conn.close()

    state_conn = sqlite3.connect(str(story_state.state_db_path()))
    try:
        assert state_conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 0
        assert state_conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
    finally:
        state_conn.close()

    assert get_stats(str(workspace), rcfg) is None
    assert list(log_dir.glob("*.jsonl*")) == []


@pytest.mark.asyncio
async def test_cmd_purge_all_cancelled_leaves_stores_untouched(tmp_path, monkeypatch):
    """When the confirm prompt is declined, no store may be touched."""
    from harness import cli as cli_mod
    from harness import story_state

    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_min_config(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(cfg_path))

    workspace = tmp_path / "ws-app"
    workspace.mkdir()
    _seed_checkpoints_db(tmp_path / "checkpoints.db")
    conn = story_state.open_story_db()
    try:
        app = story_state.app_name_for_workspace(str(workspace))
        story_state.ensure_feature(conn, app, "test", name="Test feature")
    finally:
        conn.close()

    class _NoChannel:
        def confirm(self, msg, default=False):
            return False

    monkeypatch.setattr("harness.hitl.get_channel", lambda: _NoChannel())

    args = argparse.Namespace(
        all=True, session_id=None, workspace=str(workspace),
    )
    rc = await cli_mod.cmd_purge(args)
    assert rc == 0

    # Checkpoint rows still present — confirms we bailed out before purge.
    ck_conn = sqlite3.connect(str(tmp_path / "checkpoints.db"))
    try:
        assert ck_conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 1
    finally:
        ck_conn.close()
    state_conn = sqlite3.connect(str(story_state.state_db_path()))
    try:
        assert state_conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
    finally:
        state_conn.close()
