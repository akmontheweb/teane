"""Unit tests for harness/story_state.py — the global, multi-workspace story DB."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from harness.story_state import (
    BUILD_KIND_CR,
    BUILD_KIND_GREENFIELD,
    SCHEMA_VERSION,
    _read_schema_version,
    app_name_for_workspace,
    complete_batch,
    create_features,
    create_stories,
    ensure_feature,
    files_for_batch,
    get_planned_stories,
    get_story,
    link_file,
    list_batches_for_cr,
    list_stories,
    list_stories_for_cr,
    mark_blocked,
    mark_done,
    mark_in_progress,
    open_story_db,
    record_commit,
    record_defect,
    record_test_run,
    regenerate_markdown_views,
    resolve_defects_for_story,
    set_batch_committed_sha,
    start_batch,
    state_db_path,
)


def _seed_feature(conn, app: str, key: str = "test", name: str = "Test feature") -> int:
    """Create a feature if missing; return its feature_id. Test helper
    used by every create_stories(...) call below — v4 requires every
    story to belong to a feature."""
    return ensure_feature(conn, app, key, name=name)


def _create_stories(conn, app, items, **kwargs):
    """Test wrapper around ``create_stories`` that auto-seeds the
    ``test`` feature and assigns it to any item that didn't specify
    one. Keeps the original tests focused on story semantics without
    having to repeat the feature-seed boilerplate everywhere."""
    _seed_feature(conn, app)
    for item in items:
        item.setdefault("feature", "test")
    return create_stories(conn, app, items, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> str:
    """Workspace folder path. The basename is the app name used to
    scope rows in the global state.db (e.g. ``ws-abc12``)."""
    ws = tmp_path / "ws-app"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def app(workspace: str) -> str:
    """The app-name identifier derived from ``workspace``'s basename."""
    return app_name_for_workspace(workspace)


@pytest.fixture
def conn(workspace: str):
    c = open_story_db(workspace)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema / open
# ---------------------------------------------------------------------------

def test_open_creates_global_db():
    conn = open_story_db()
    try:
        assert os.path.exists(state_db_path())
    finally:
        conn.close()


def test_open_is_idempotent():
    open_story_db().close()
    open_story_db().close()
    conn = open_story_db()
    try:
        # All expected tables exist.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()
    expected = {
        "schema_meta", "features", "stories", "batches", "batch_stories",
        "defects", "test_runs", "file_links", "commits",
    }
    assert expected.issubset(tables)


def test_schema_version_persisted(conn: sqlite3.Connection):
    assert _read_schema_version(conn) == SCHEMA_VERSION


def test_pragmas_applied(conn: sqlite3.Connection):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_app_name_from_basename(tmp_path):
    assert app_name_for_workspace(str(tmp_path / "my-app")) == "my-app"
    # Trailing slash stripped.
    assert app_name_for_workspace(str(tmp_path / "other") + "/") == "other"


def test_app_name_rejects_empty():
    with pytest.raises(ValueError):
        app_name_for_workspace("")
    with pytest.raises(ValueError):
        app_name_for_workspace("/")
    with pytest.raises(ValueError):
        app_name_for_workspace(".")


# ---------------------------------------------------------------------------
# Stories CRUD
# ---------------------------------------------------------------------------

def test_create_stories_assigns_sequential_keys(conn, app):
    keys = _create_stories(conn, app, [
        {"title": "Add login"},
        {"title": "Add logout"},
        {"title": "Add password reset"},
    ])
    assert keys == ["STORY-1", "STORY-2", "STORY-3"]


def test_create_stories_defaults_to_greenfield(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    s = get_story(conn, app, "STORY-1")
    assert s["build_kind"] == BUILD_KIND_GREENFIELD
    assert s["cr_ids"] == []


def test_create_stories_persists_json_columns(conn, app):
    create_features(conn, app, [{"feature_key": "users", "name": "Users"}])
    _create_stories(conn, app, [{
        "title": "Auth",
        "feature": "users",
        "acceptance_criteria": ["AC1", "AC2"],
        "depends_on": [],
        "scope_files": ["auth.py", "tests/test_auth.py"],
        "external_ref": "CR-7",
    }])
    s = get_story(conn, app, "STORY-1")
    assert s is not None
    assert s["acceptance_criteria"] == ["AC1", "AC2"]
    assert s["scope_files"] == ["auth.py", "tests/test_auth.py"]
    assert s["external_ref"] == "CR-7"
    assert s["feature_key"] == "users"
    assert s["feature_name"] == "Users"
    assert s["feature_id"] is not None
    assert s["status"] == "planned"


def test_create_stories_rejects_missing_title(conn, app):
    _seed_feature(conn, app)
    with pytest.raises(ValueError):
        _create_stories(conn, app, [{"feature": "test"}])


def test_create_stories_rejects_missing_feature(conn, app):
    """v4 schema makes feature mandatory."""
    with pytest.raises(ValueError, match="feature"):
        create_stories(conn, app, [{"title": "Orphan"}])


def test_create_stories_rejects_unknown_feature(conn, app):
    """Stories must reference a feature_key that exists in the DB."""
    with pytest.raises(ValueError, match="ghost"):
        _create_stories(conn, app, [{
            "title": "T", "feature": "ghost",
        }])


def test_status_transitions(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    mark_in_progress(conn, app, "STORY-1")
    assert get_story(conn, app, "STORY-1")["status"] == "in_progress"
    assert get_story(conn, app, "STORY-1")["started_at"] is not None
    mark_done(conn, app, "STORY-1")
    assert get_story(conn, app, "STORY-1")["status"] == "done"
    assert get_story(conn, app, "STORY-1")["completed_at"] is not None


def test_mark_in_progress_idempotent_on_resume(conn, app):
    """Calling mark_in_progress twice (resume scenario) refreshes
    started_at instead of silently no-oping."""
    _create_stories(conn, app, [{"title": "T"}])
    assert mark_in_progress(conn, app, "STORY-1") == 1
    first_started = get_story(conn, app, "STORY-1")["started_at"]
    # Second call against an already-in_progress story still matches.
    assert mark_in_progress(conn, app, "STORY-1") == 1
    assert get_story(conn, app, "STORY-1")["started_at"] >= first_started


def test_mark_blocked(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    mark_blocked(conn, app, "STORY-1")
    assert get_story(conn, app, "STORY-1")["status"] == "blocked"


def test_list_stories_filter_by_status(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}, {"title": "C"}])
    mark_done(conn, app, "STORY-2")
    assert [s["story_key"] for s in list_stories(conn, app, status="planned")] == [
        "STORY-1", "STORY-3"
    ]
    assert [s["story_key"] for s in list_stories(conn, app, status="done")] == [
        "STORY-2"
    ]


# ---------------------------------------------------------------------------
# Cross-workspace isolation — the new global-DB contract
# ---------------------------------------------------------------------------

def test_two_workspaces_each_own_story_1(conn, tmp_path):
    """Two workspaces can each have a ``STORY-1`` simultaneously."""
    app_a = app_name_for_workspace(str(tmp_path / "alpha"))
    app_b = app_name_for_workspace(str(tmp_path / "beta"))
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    _create_stories(conn, app_a, [{"title": "alpha story 1"}])
    _create_stories(conn, app_b, [{"title": "beta story 1"}])
    assert get_story(conn, app_a, "STORY-1")["title"] == "alpha story 1"
    assert get_story(conn, app_b, "STORY-1")["title"] == "beta story 1"


def test_list_stories_scopes_by_workspace(conn, tmp_path):
    app_a = app_name_for_workspace(str(tmp_path / "alpha"))
    app_b = app_name_for_workspace(str(tmp_path / "beta"))
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    _create_stories(conn, app_a, [{"title": "A1"}, {"title": "A2"}])
    _create_stories(conn, app_b, [{"title": "B1"}])
    assert [s["title"] for s in list_stories(conn, app_a)] == ["A1", "A2"]
    assert [s["title"] for s in list_stories(conn, app_b)] == ["B1"]


def test_mark_done_in_one_workspace_does_not_touch_other(conn, tmp_path):
    app_a = app_name_for_workspace(str(tmp_path / "alpha"))
    app_b = app_name_for_workspace(str(tmp_path / "beta"))
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    _create_stories(conn, app_a, [{"title": "A"}])
    _create_stories(conn, app_b, [{"title": "B"}])
    mark_done(conn, app_a, "STORY-1")
    assert get_story(conn, app_a, "STORY-1")["status"] == "done"
    assert get_story(conn, app_b, "STORY-1")["status"] == "planned"


# ---------------------------------------------------------------------------
# Dependency-aware planning
# ---------------------------------------------------------------------------

def test_get_planned_stories_honors_depends_on(conn, app):
    _create_stories(conn, app, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-1"]},
        {"title": "Polish", "depends_on": ["STORY-2"]},
    ])
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-1"]  # only Base is unblocked

    mark_done(conn, app, "STORY-1")
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-2"]

    mark_done(conn, app, "STORY-2")
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-3"]


def test_get_planned_stories_returns_parallel_independent(conn, app):
    _create_stories(conn, app, [
        {"title": "A"},
        {"title": "B"},
        {"title": "C", "depends_on": ["STORY-1", "STORY-2"]},
    ])
    ready = sorted(s["story_key"] for s in get_planned_stories(conn, app))
    assert ready == ["STORY-1", "STORY-2"]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def test_batch_lifecycle(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    batch_id = start_batch(conn, app, "sess-1", ["STORY-1", "STORY-2"])
    row = conn.execute(
        "SELECT session_id, status, workspace FROM batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    assert row == ("sess-1", "running", app)
    seqs = conn.execute(
        "SELECT sequence FROM batch_stories WHERE batch_id = ? ORDER BY sequence",
        (batch_id,),
    ).fetchall()
    assert [s[0] for s in seqs] == [1, 2]
    complete_batch(conn, batch_id)
    row = conn.execute(
        "SELECT status, completed_at FROM batches WHERE id = ?", (batch_id,)
    ).fetchone()
    assert row[0] == "complete"
    assert row[1] is not None


def test_batch_skips_unknown_story_keys(conn, app):
    _create_stories(conn, app, [{"title": "A"}])
    batch_id = start_batch(conn, app, "sess-1", ["STORY-1", "STORY-99"])
    count = conn.execute(
        "SELECT COUNT(*) FROM batch_stories WHERE batch_id = ?", (batch_id,)
    ).fetchone()[0]
    assert count == 1


def test_start_batch_defaults_to_greenfield(conn, app):
    _create_stories(conn, app, [{"title": "A"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-1"])
    row = conn.execute(
        "SELECT build_kind, cr_ids FROM batches WHERE id = ?", (bid,),
    ).fetchone()
    assert row[0] == BUILD_KIND_GREENFIELD
    assert row[1] is None


# ---------------------------------------------------------------------------
# Change-request tagging (v3)
# ---------------------------------------------------------------------------

def test_create_stories_tags_as_cr(conn, app):
    _create_stories(
        conn, app, [{"title": "Add 2FA"}],
        build_kind=BUILD_KIND_CR, cr_ids=[2],
    )
    s = get_story(conn, app, "STORY-1")
    assert s["build_kind"] == BUILD_KIND_CR
    assert s["cr_ids"] == [2]


def test_start_batch_tags_as_cr(conn, app):
    _create_stories(
        conn, app, [{"title": "T"}],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    bid = start_batch(
        conn, app, "sess-1", ["STORY-1"],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    row = conn.execute(
        "SELECT build_kind, cr_ids FROM batches WHERE id = ?", (bid,),
    ).fetchone()
    assert row[0] == BUILD_KIND_CR
    import json as _json
    assert _json.loads(row[1]) == [3]


def test_list_stories_for_cr_filters_by_cr_id(conn, app):
    _create_stories(conn, app, [{"title": "greenfield"}])
    _create_stories(
        conn, app, [{"title": "added by CR-2"}],
        build_kind=BUILD_KIND_CR, cr_ids=[2],
    )
    _create_stories(
        conn, app, [{"title": "added by CR-5"}],
        build_kind=BUILD_KIND_CR, cr_ids=[5],
    )
    titles = [s["title"] for s in list_stories_for_cr(conn, app, 2)]
    assert titles == ["added by CR-2"]
    titles = [s["title"] for s in list_stories_for_cr(conn, app, 5)]
    assert titles == ["added by CR-5"]
    assert list_stories_for_cr(conn, app, 99) == []


def test_list_batches_for_cr_filters_by_cr_id(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    # Greenfield batch.
    start_batch(conn, app, "sess-1", ["STORY-1"])
    # CR-3 batch.
    _create_stories(
        conn, app, [{"title": "CR-3 thing"}],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    start_batch(
        conn, app, "sess-2", ["STORY-2"],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    batches = list_batches_for_cr(conn, app, 3)
    assert len(batches) == 1
    assert batches[0]["build_kind"] == BUILD_KIND_CR
    assert batches[0]["cr_ids"] == [3]


def test_invalid_build_kind_rejected(conn, app):
    with pytest.raises(ValueError):
        _create_stories(conn, app, [{"title": "T"}], build_kind="bogus")


# ---------------------------------------------------------------------------
# Defects, test runs, file links, commits
# ---------------------------------------------------------------------------

def test_record_and_resolve_defect(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    did = record_defect(
        conn,
        workspace=app,
        story_key="STORY-1",
        session_id="sess-1",
        severity="compile",
        summary="syntax error in foo.py",
        diagnostic={"line": 42},
    )
    row = conn.execute(
        "SELECT status, summary, diagnostic_json FROM defects WHERE id = ?",
        (did,),
    ).fetchone()
    assert row[0] == "open"
    assert "foo.py" in row[1]
    assert "42" in row[2]

    n = resolve_defects_for_story(conn, app, "STORY-1")
    assert n == 1
    assert conn.execute(
        "SELECT status FROM defects WHERE id = ?", (did,)
    ).fetchone()[0] == "resolved"


def test_record_defect_without_story(conn, app):
    did = record_defect(
        conn,
        workspace=app,
        story_key=None,
        session_id="sess-1",
        severity="security",
        summary="orphan finding",
    )
    row = conn.execute(
        "SELECT story_id, workspace FROM defects WHERE id = ?", (did,)
    ).fetchone()
    assert row[0] is None
    assert row[1] == app


def test_record_test_run(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    record_test_run(
        conn,
        workspace=app,
        story_key="STORY-1",
        session_id="sess-1",
        phase="tests_first",
        exit_code=1,
        passed=2, failed=1, errors=0,
        stdout_tail="FAILED: test_foo",
    )
    row = conn.execute(
        "SELECT phase, exit_code, passed, failed FROM test_runs"
    ).fetchone()
    assert row == ("tests_first", 1, 2, 1)


def test_link_file_dedups(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    link_file(conn, app, "STORY-1", "auth.py", "code")
    link_file(conn, app, "STORY-1", "auth.py", "code")  # duplicate kind
    link_file(conn, app, "STORY-1", "auth.py", "test")  # different kind, allowed
    rows = conn.execute(
        "SELECT path, kind FROM file_links ORDER BY kind"
    ).fetchall()
    assert sorted(rows) == [("auth.py", "code"), ("auth.py", "test")]


def test_link_file_unknown_story_is_noop(conn, app):
    link_file(conn, app, "STORY-99", "auth.py", "code")
    assert conn.execute("SELECT COUNT(*) FROM file_links").fetchone()[0] == 0


def test_link_file_stamps_batch_id(conn, app):
    _create_stories(conn, app, [{"title": "Auth"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-1"])
    link_file(conn, app, "STORY-1", "auth.py", "code", batch_id=bid)
    row = conn.execute(
        "SELECT batch_id, workspace FROM file_links WHERE path = ?",
        ("auth.py",),
    ).fetchone()
    assert row[0] == bid
    assert row[1] == app


def test_link_file_updates_batch_id_on_reapply(conn, app):
    """If a later batch touches the same (story, path, kind), the
    stamp updates so 'last batch that touched this file' is queryable."""
    _create_stories(conn, app, [{"title": "Auth"}])
    b1 = start_batch(conn, app, "sess-1", ["STORY-1"])
    link_file(conn, app, "STORY-1", "auth.py", "code", batch_id=b1)
    b2 = start_batch(conn, app, "sess-1", ["STORY-1"])
    link_file(conn, app, "STORY-1", "auth.py", "code", batch_id=b2)
    row = conn.execute(
        "SELECT batch_id FROM file_links WHERE path = ?", ("auth.py",)
    ).fetchone()
    assert row[0] == b2


def test_link_file_without_batch_id_preserves_existing_stamp(conn, app):
    """Calling link_file without batch_id after a batch-stamped row
    exists must not clear the stamp — the unstamped call is a no-op
    when the (story, path, kind) tuple already exists."""
    _create_stories(conn, app, [{"title": "Auth"}])
    b1 = start_batch(conn, app, "sess-1", ["STORY-1"])
    link_file(conn, app, "STORY-1", "auth.py", "code", batch_id=b1)
    link_file(conn, app, "STORY-1", "auth.py", "code")  # no batch_id
    row = conn.execute(
        "SELECT batch_id FROM file_links WHERE path = ?", ("auth.py",)
    ).fetchone()
    assert row[0] == b1


def test_files_for_batch_returns_only_stamped_files(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    b1 = start_batch(conn, app, "sess-1", ["STORY-1", "STORY-2"])
    link_file(conn, app, "STORY-1", "a.py", "code", batch_id=b1)
    link_file(conn, app, "STORY-2", "b.py", "code", batch_id=b1)
    # Unstamped row (legacy / out-of-band touch) must not appear
    # in files_for_batch results.
    link_file(conn, app, "STORY-1", "stale.py", "doc")
    rows = files_for_batch(conn, b1)
    assert sorted(rows) == [
        ("STORY-1", "a.py", "code"),
        ("STORY-2", "b.py", "code"),
    ]


def test_set_batch_committed_sha_persists(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-1"])
    set_batch_committed_sha(conn, bid, "deadbeefcafe")
    row = conn.execute(
        "SELECT committed_sha FROM batches WHERE id = ?", (bid,)
    ).fetchone()
    assert row[0] == "deadbeefcafe"


def test_set_batch_committed_sha_unknown_batch_is_noop(conn):
    set_batch_committed_sha(conn, 9999, "abc")
    assert conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0


def test_record_commit(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    record_commit(
        conn,
        workspace=app,
        sha="abc1234567", story_key="STORY-1",
        session_id="sess-1", message="STORY-1: add auth",
    )
    row = conn.execute(
        "SELECT sha, message, story_id, workspace FROM commits"
    ).fetchone()
    assert row[0] == "abc1234567"
    assert row[1] == "STORY-1: add auth"
    assert row[2] is not None
    assert row[3] == app


# ---------------------------------------------------------------------------
# Markdown view regeneration
# ---------------------------------------------------------------------------

def test_regenerate_markdown_views_empty(conn, workspace):
    stories_path, trace_path = regenerate_markdown_views(conn, workspace)
    assert os.path.exists(stories_path)
    assert os.path.exists(trace_path)
    assert "No stories yet" in Path(stories_path).read_text()
    assert "Traceability matrix" in Path(trace_path).read_text()


def test_regenerate_markdown_views_renders_stories_and_traceability(
    conn, app, workspace
):
    create_features(conn, app, [{"feature_key": "auth", "name": "Auth"}])
    _create_stories(conn, app, [
        {"title": "Add login", "feature": "auth",
         "acceptance_criteria": ["redirect after login"]},
        {"title": "Add logout", "feature": "auth", "depends_on": ["STORY-1"]},
    ])
    mark_done(conn, app, "STORY-1")
    link_file(conn, app, "STORY-1", "src/auth.py", "code")
    link_file(conn, app, "STORY-1", "tests/test_auth.py", "test")
    record_defect(
        conn, workspace=app, story_key="STORY-1", session_id="s1",
        severity="lint", summary="line too long",
    )
    record_commit(
        conn, workspace=app, sha="abcdef0123456789", story_key="STORY-1",
        session_id="s1", message="STORY-1: add login",
    )

    stories_path, trace_path = regenerate_markdown_views(conn, workspace)
    stories_md = Path(stories_path).read_text()
    trace_md = Path(trace_path).read_text()

    assert "STORY-1" in stories_md
    assert "STORY-2" in stories_md
    assert "auth" in stories_md
    assert "redirect after login" in stories_md
    assert "STORY-1" in trace_md
    assert "src/auth.py" in trace_md
    assert "tests/test_auth.py" in trace_md
    assert "abcdef0" in trace_md
    assert "line too long" in trace_md


def test_regenerate_markdown_views_scoped_to_workspace(conn, tmp_path):
    """Two apps share the global DB; each app's docs only reflect
    its own stories."""
    ws_a = tmp_path / "alpha"
    ws_b = tmp_path / "beta"
    ws_a.mkdir()
    ws_b.mkdir()
    _create_stories(conn, "alpha", [{"title": "alpha-only"}])
    _create_stories(conn, "beta", [{"title": "beta-only"}])
    regenerate_markdown_views(conn, str(ws_a))
    regenerate_markdown_views(conn, str(ws_b))
    alpha_md = (ws_a / "docs" / "STORIES.md").read_text()
    beta_md = (ws_b / "docs" / "STORIES.md").read_text()
    assert "alpha-only" in alpha_md and "beta-only" not in alpha_md
    assert "beta-only" in beta_md and "alpha-only" not in beta_md


def test_regenerate_markdown_views_is_deterministic(conn, app, workspace):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    p1, _ = regenerate_markdown_views(conn, workspace)
    first = Path(p1).read_text()
    p2, _ = regenerate_markdown_views(conn, workspace)
    second = Path(p2).read_text()
    assert first == second


def test_regenerate_markdown_views_creates_docs_dir(workspace, app):
    conn = open_story_db()
    try:
        _create_stories(conn, app, [{"title": "T"}])
        regenerate_markdown_views(conn, workspace)
    finally:
        conn.close()
    assert os.path.isdir(os.path.join(workspace, "docs"))
