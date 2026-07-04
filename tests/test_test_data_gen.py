"""Phase 3 tests for ``harness.test_data_gen``.

Covers:
- detect_flow_kind: agile (rows in state.db) vs waterfall (none)
- gather_schema_context: reads spec files, populates story list when agile
- compute_cache_key: stable across permutations
- write_seed_fixture / cached_fixture_path: cache hit / miss round-trip
- apply_seed_to_sqlite + reset_sqlite_db: round-trip + truncate
- generate_seed_data: validates shape; pluggable generator returns custom data
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from harness import story_state
from harness.test_data_gen import (
    FLOW_AGILE,
    FLOW_WATERFALL,
    SchemaContext,
    apply_seed_to_sqlite,
    cached_fixture_path,
    compute_cache_key,
    detect_flow_kind,
    fallback_seed,
    gather_schema_context,
    generate_seed_data,
    reset_sqlite_db,
    write_seed_fixture,
)


# ---------------------------------------------------------------------------
# detect_flow_kind
# ---------------------------------------------------------------------------


def test_detect_flow_kind_waterfall_without_state_db(tmp_path: Path) -> None:
    workspace = tmp_path / "myapp"
    workspace.mkdir()
    assert detect_flow_kind(str(workspace)) == FLOW_WATERFALL


def test_detect_flow_kind_agile_when_stories_present(tmp_path: Path) -> None:
    workspace = tmp_path / "myapp"
    workspace.mkdir()
    _seed_one_story(workspace)
    assert detect_flow_kind(str(workspace)) == FLOW_AGILE


# ---------------------------------------------------------------------------
# gather_schema_context
# ---------------------------------------------------------------------------


def test_gather_context_reads_spec_files(tmp_path: Path) -> None:
    workspace = tmp_path / "myapp"
    docs = workspace / "docs"
    docs.mkdir(parents=True)
    (docs / "SPEC_REQUIREMENTS.md").write_text("# FR-001\nUsers can log in.\n")
    (docs / "SPEC_DATA_MODEL.md").write_text("# users\nid, email\n")

    ctx = gather_schema_context(str(workspace))

    assert ctx.flow_kind == FLOW_WATERFALL
    assert "docs/SPEC_REQUIREMENTS.md" in ctx.spec_excerpts
    assert "docs/SPEC_DATA_MODEL.md" in ctx.spec_excerpts
    assert "log in" in ctx.spec_excerpts["docs/SPEC_REQUIREMENTS.md"]
    assert ctx.stories == []


def test_gather_context_populates_agile_stories(tmp_path: Path) -> None:
    workspace = tmp_path / "myapp"
    workspace.mkdir()
    _seed_one_story(workspace)

    ctx = gather_schema_context(str(workspace))

    assert ctx.flow_kind == FLOW_AGILE
    assert len(ctx.stories) == 1
    story = ctx.stories[0]
    assert story["story_key"] == "STORY-001"
    assert "STORY-001.AC-1" in story["acceptance_criteria_keys"]


# ---------------------------------------------------------------------------
# compute_cache_key — stable, workspace-path-independent
# ---------------------------------------------------------------------------


def test_cache_key_independent_of_workspace_path(tmp_path: Path) -> None:
    ctx_a = SchemaContext(
        workspace_path="/path/A",
        flow_kind=FLOW_WATERFALL,
        spec_excerpts={"x.md": "hello"},
    )
    ctx_b = SchemaContext(
        workspace_path="/different/B",
        flow_kind=FLOW_WATERFALL,
        spec_excerpts={"x.md": "hello"},
    )
    assert compute_cache_key(ctx_a) == compute_cache_key(ctx_b)


def test_cache_key_changes_when_spec_changes(tmp_path: Path) -> None:
    ctx_a = SchemaContext(
        workspace_path="/x", flow_kind=FLOW_WATERFALL,
        spec_excerpts={"x.md": "v1"},
    )
    ctx_b = SchemaContext(
        workspace_path="/x", flow_kind=FLOW_WATERFALL,
        spec_excerpts={"x.md": "v2"},
    )
    assert compute_cache_key(ctx_a) != compute_cache_key(ctx_b)


def test_cache_key_changes_when_flow_kind_flips(tmp_path: Path) -> None:
    ctx_w = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL)
    ctx_a = SchemaContext(workspace_path="/x", flow_kind=FLOW_AGILE)
    assert compute_cache_key(ctx_w) != compute_cache_key(ctx_a)


# ---------------------------------------------------------------------------
# generate_seed_data + fallback_seed + custom generators
# ---------------------------------------------------------------------------


def test_fallback_seed_marks_meta_row() -> None:
    ctx = SchemaContext(
        workspace_path="/x", flow_kind=FLOW_AGILE,
        spec_excerpts={"docs/SPEC_REQUIREMENTS.md": "hi"},
        stories=[{"story_key": "STORY-001"}],
    )
    seed = fallback_seed(ctx)
    meta = seed["tables"]["_teane_test_meta"][0]
    assert meta["flow_kind"] == FLOW_AGILE
    assert meta["story_count"] == 1
    assert "docs/SPEC_REQUIREMENTS.md" in meta["spec_files"]


def test_generate_seed_data_uses_custom_generator() -> None:
    ctx = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL)
    called: list[SchemaContext] = []

    def gen(c: SchemaContext) -> dict:
        called.append(c)
        return {"tables": {"users": [{"id": "1", "email": "alice@test"}]}}

    seed = generate_seed_data(ctx, generator=gen)
    assert called == [ctx]
    assert seed["tables"]["users"][0]["email"] == "alice@test"


def test_generate_seed_data_rejects_bad_shape() -> None:
    ctx = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL)
    with pytest.raises(ValueError):
        generate_seed_data(ctx, generator=lambda c: "not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_seed_data(ctx, generator=lambda c: {"tables": "wrong"})
    with pytest.raises(ValueError):
        generate_seed_data(ctx, generator=lambda c: {"tables": {"x": [1, 2]}})


# ---------------------------------------------------------------------------
# Fixture cache: write + check + invalidate
# ---------------------------------------------------------------------------


def test_write_then_cache_hit(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    seed = {"tables": {"users": [{"id": "1"}]}}
    key = "abc123"
    seed_path = write_seed_fixture(str(workspace), seed, key)
    assert os.path.isfile(seed_path)
    assert cached_fixture_path(str(workspace), key) == seed_path


def test_cache_miss_when_key_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    write_seed_fixture(str(workspace), {"tables": {}}, "key-v1")
    assert cached_fixture_path(str(workspace), "key-v2") is None


def test_cache_miss_when_files_absent(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert cached_fixture_path(str(workspace), "any") is None


# ---------------------------------------------------------------------------
# SQLite lifecycle: apply seed + reset
# ---------------------------------------------------------------------------


def test_apply_seed_to_sqlite_inserts_rows(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test.db")
    seed = {"tables": {
        "users": [
            {"id": "1", "email": "alice@test"},
            {"id": "2", "email": "bob@test", "_verifies": "STORY-001.AC-1"},
        ],
    }}
    count = apply_seed_to_sqlite(db_path, seed)
    assert count == 2

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, email FROM users ORDER BY id").fetchall()
        assert rows == [("1", "alice@test"), ("2", "bob@test")]
        # _verifies stripped before insert
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        assert "_verifies" not in cols
    finally:
        conn.close()


def test_apply_seed_handles_nested_json_values(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test.db")
    seed = {"tables": {
        "events": [{"id": "1", "payload": {"k": "v", "n": 7}}],
    }}
    apply_seed_to_sqlite(db_path, seed)
    conn = sqlite3.connect(db_path)
    try:
        (payload,) = conn.execute("SELECT payload FROM events").fetchone()
        assert json.loads(payload) == {"k": "v", "n": 7}
    finally:
        conn.close()


def test_reset_sqlite_db_drops_user_tables(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test.db")
    apply_seed_to_sqlite(db_path, {"tables": {
        "users": [{"id": "1"}], "items": [{"sku": "x"}],
    }})
    dropped = reset_sqlite_db(db_path)
    assert dropped == 2
    conn = sqlite3.connect(db_path)
    try:
        remaining = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert remaining == []
    finally:
        conn.close()


def test_reset_sqlite_db_noop_for_missing_file(tmp_path: Path) -> None:
    assert reset_sqlite_db(str(tmp_path / "absent.db")) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_one_story(workspace: Path) -> None:
    """Insert a single feature → story → AC chain for the workspace into the
    isolated state.db (the autouse conftest fixture has already redirected
    TEANE_STATE_DB into tmp_path)."""
    conn = story_state.open_story_db(workspace_path=str(workspace))
    try:
        app = story_state.app_name_for_workspace(str(workspace))
        story_state.create_features(
            conn, app, [{"feature_key": "FEAT-001", "name": "auth", "description": "auth flow"}],
        )
        feature_id = story_state.get_feature_by_key(conn, app, "FEAT-001")["id"]
        now = "2026-06-30T00:00:00+00:00"
        conn.execute(
            "INSERT INTO stories(workspace, story_key, feature_id, title, "
            "description, depends_on, scope_files, status, external_ref, "
            "build_kind, cr_ids, created_at) "
            "VALUES(?, 'STORY-001', ?, 'login', 'log in', '[]', '[]', "
            "'planned', NULL, 'greenfield', '[]', ?)",
            (app, feature_id, now),
        )
        story_id = conn.execute(
            "SELECT id FROM stories WHERE workspace=? AND story_key='STORY-001'",
            (app,),
        ).fetchone()[0]
        story_state.create_acceptance_criteria(
            conn, app, int(story_id),
            [{"ac_key": "STORY-001.AC-1", "text": "user submits valid creds", "ordinal": 1}],
        )
        conn.commit()
    finally:
        conn.close()
