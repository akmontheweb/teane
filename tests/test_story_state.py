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
    acs_without_verifying_test,
    app_name_for_workspace,
    complete_batch,
    create_acceptance_criteria,
    create_features,
    create_requirements,
    create_stories,
    ensure_feature,
    ensure_requirement,
    files_for_batch,
    get_ac_by_key,
    get_planned_stories,
    get_requirement_by_key,
    get_story,
    link_file,
    link_story_to_requirements,
    link_test_to_ac,
    list_acceptance_criteria,
    list_batches_for_cr,
    list_requirements,
    list_stories,
    list_stories_for_cr,
    mark_blocked,
    mark_done,
    mark_in_progress,
    open_story_db,
    purge_state_db,
    record_commit,
    record_defect,
    record_test_run,
    regenerate_markdown_views,
    requirements_without_satisfying_story,
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
        # v5 traceability tables
        "requirements", "acceptance_criteria",
        "story_satisfies_req", "test_verifies_ac",
    }
    assert expected.issubset(tables)


def test_schema_version_persisted(conn: sqlite3.Connection):
    assert _read_schema_version(conn) == SCHEMA_VERSION


def test_open_migrates_legacy_v3_db(isolated_state_db):
    """A pre-existing v3 DB (``stories.epic``, no ``feature_id``,
    schema_version=3) MUST migrate cleanly when reopened. Regression
    guard for the ordering bug where ``_SCHEMA_SQL`` ran BEFORE
    ``_apply_migrations`` and crashed on ``no such column:
    feature_id`` while building v4 indexes against the still-v3
    ``stories`` table."""
    # Materialise a v3-shaped DB on disk at the isolated path.
    db_path = str(isolated_state_db)
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta(key, value) VALUES ('schema_version', '3');
        CREATE TABLE stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace TEXT NOT NULL,
            story_key TEXT NOT NULL,
            epic TEXT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'planned',
            created_at TEXT NOT NULL
        );
        INSERT INTO stories(workspace, story_key, epic, title, status, created_at)
        VALUES ('legacy-ws', 'STORY-001', 'legacy-epic', 'old story', 'done', '2026-01-01');
        """
    )
    raw.commit()
    raw.close()

    # Reopen via the public API — must NOT raise.
    conn = open_story_db()
    try:
        assert _read_schema_version(conn) == SCHEMA_VERSION
        cols = {c[1] for c in conn.execute("PRAGMA table_info(stories)").fetchall()}
        assert "feature_id" in cols
        assert "epic" not in cols
        # v3 rows are dropped — clean-slate migration is documented.
        assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 0
        # features table exists in v4.
        feature_cols = {c[1] for c in conn.execute("PRAGMA table_info(features)").fetchall()}
        assert "feature_key" in feature_cols
    finally:
        conn.close()


def test_open_is_idempotent_after_migration(isolated_state_db):
    """Reopening a migrated DB MUST be a no-op (no exceptions, no row
    loss). Pins the post-migration steady state."""
    # First open lands at v4.
    conn = open_story_db()
    conn.close()
    # Second open re-runs the same path — must not blow up or reset.
    conn = open_story_db()
    try:
        assert _read_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


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
    assert keys == ["STORY-001", "STORY-002", "STORY-003"]


def test_create_stories_defaults_to_greenfield(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    s = get_story(conn, app, "STORY-001")
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
    s = get_story(conn, app, "STORY-001")
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
    mark_in_progress(conn, app, "STORY-001")
    assert get_story(conn, app, "STORY-001")["status"] == "in_progress"
    assert get_story(conn, app, "STORY-001")["started_at"] is not None
    mark_done(conn, app, "STORY-001")
    assert get_story(conn, app, "STORY-001")["status"] == "done"
    assert get_story(conn, app, "STORY-001")["completed_at"] is not None


def test_mark_in_progress_idempotent_on_resume(conn, app):
    """Calling mark_in_progress twice (resume scenario) refreshes
    started_at instead of silently no-oping."""
    _create_stories(conn, app, [{"title": "T"}])
    assert mark_in_progress(conn, app, "STORY-001") == 1
    first_started = get_story(conn, app, "STORY-001")["started_at"]
    # Second call against an already-in_progress story still matches.
    assert mark_in_progress(conn, app, "STORY-001") == 1
    assert get_story(conn, app, "STORY-001")["started_at"] >= first_started


def test_mark_blocked(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    mark_blocked(conn, app, "STORY-001")
    assert get_story(conn, app, "STORY-001")["status"] == "blocked"


def test_list_stories_filter_by_status(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}, {"title": "C"}])
    mark_done(conn, app, "STORY-002")
    assert [s["story_key"] for s in list_stories(conn, app, status="planned")] == [
        "STORY-001", "STORY-003"
    ]
    assert [s["story_key"] for s in list_stories(conn, app, status="done")] == [
        "STORY-002"
    ]


# ---------------------------------------------------------------------------
# Cross-workspace isolation — the new global-DB contract
# ---------------------------------------------------------------------------

def test_two_workspaces_each_own_story_1(conn, tmp_path):
    """Two workspaces can each have a ``STORY-001`` simultaneously."""
    app_a = app_name_for_workspace(str(tmp_path / "alpha"))
    app_b = app_name_for_workspace(str(tmp_path / "beta"))
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    _create_stories(conn, app_a, [{"title": "alpha story 1"}])
    _create_stories(conn, app_b, [{"title": "beta story 1"}])
    assert get_story(conn, app_a, "STORY-001")["title"] == "alpha story 1"
    assert get_story(conn, app_b, "STORY-001")["title"] == "beta story 1"


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
    mark_done(conn, app_a, "STORY-001")
    assert get_story(conn, app_a, "STORY-001")["status"] == "done"
    assert get_story(conn, app_b, "STORY-001")["status"] == "planned"


# ---------------------------------------------------------------------------
# Dependency-aware planning
# ---------------------------------------------------------------------------

def test_get_planned_stories_honors_depends_on(conn, app):
    _create_stories(conn, app, [
        {"title": "Base"},
        {"title": "Feature", "depends_on": ["STORY-001"]},
        {"title": "Polish", "depends_on": ["STORY-002"]},
    ])
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-001"]  # only Base is unblocked

    mark_done(conn, app, "STORY-001")
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-002"]

    mark_done(conn, app, "STORY-002")
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert ready == ["STORY-003"]


def test_get_planned_stories_returns_parallel_independent(conn, app):
    _create_stories(conn, app, [
        {"title": "A"},
        {"title": "B"},
        {"title": "C", "depends_on": ["STORY-001", "STORY-002"]},
    ])
    ready = sorted(s["story_key"] for s in get_planned_stories(conn, app))
    assert ready == ["STORY-001", "STORY-002"]


def test_get_planned_stories_includes_reopened(conn, app):
    """Phase 7 BUG #4 regression: story_reopen_node flips drifted
    DONE stories to 'reopened'. The planner MUST pick those up or
    the reopen mechanism is silently no-op."""
    from harness.story_state import mark_reopened
    _create_stories(conn, app, [
        {"title": "Login"}, {"title": "Logout"},
    ])
    mark_done(conn, app, "STORY-001")
    mark_done(conn, app, "STORY-002")
    # Spec drifted — story_reopen_node flips STORY-001.
    assert mark_reopened(conn, app, "STORY-001") == 1
    ready = [s["story_key"] for s in get_planned_stories(conn, app)]
    assert "STORY-001" in ready
    assert "STORY-002" not in ready  # still done


def test_mark_in_progress_accepts_reopened(conn, app):
    """Phase 7 BUG #4 regression: mark_in_progress must transition
    'reopened' rows too, not just ('planned', 'in_progress')."""
    from harness.story_state import mark_reopened
    _create_stories(conn, app, [{"title": "A"}])
    mark_done(conn, app, "STORY-001")
    mark_reopened(conn, app, "STORY-001")
    # Planner picks it up and tries to mark it in_progress.
    moved = mark_in_progress(conn, app, "STORY-001")
    assert moved == 1
    s = get_story(conn, app, "STORY-001")
    assert s["status"] == "in_progress"


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def test_batch_lifecycle(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    batch_id = start_batch(conn, app, "sess-1", ["STORY-001", "STORY-002"])
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
    batch_id = start_batch(conn, app, "sess-1", ["STORY-001", "STORY-099"])
    count = conn.execute(
        "SELECT COUNT(*) FROM batch_stories WHERE batch_id = ?", (batch_id,)
    ).fetchone()[0]
    assert count == 1


def test_start_batch_defaults_to_greenfield(conn, app):
    _create_stories(conn, app, [{"title": "A"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-001"])
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
    s = get_story(conn, app, "STORY-001")
    assert s["build_kind"] == BUILD_KIND_CR
    assert s["cr_ids"] == [2]


def test_start_batch_tags_as_cr(conn, app):
    _create_stories(
        conn, app, [{"title": "T"}],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    bid = start_batch(
        conn, app, "sess-1", ["STORY-001"],
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
    start_batch(conn, app, "sess-1", ["STORY-001"])
    # CR-3 batch.
    _create_stories(
        conn, app, [{"title": "CR-3 thing"}],
        build_kind=BUILD_KIND_CR, cr_ids=[3],
    )
    start_batch(
        conn, app, "sess-2", ["STORY-002"],
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
        story_key="STORY-001",
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

    n = resolve_defects_for_story(conn, app, "STORY-001")
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
        story_key="STORY-001",
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
    link_file(conn, app, "STORY-001", "auth.py", "code")
    link_file(conn, app, "STORY-001", "auth.py", "code")  # duplicate kind
    link_file(conn, app, "STORY-001", "auth.py", "test")  # different kind, allowed
    rows = conn.execute(
        "SELECT path, kind FROM file_links ORDER BY kind"
    ).fetchall()
    assert sorted(rows) == [("auth.py", "code"), ("auth.py", "test")]


def test_link_file_unknown_story_is_noop(conn, app):
    link_file(conn, app, "STORY-099", "auth.py", "code")
    assert conn.execute("SELECT COUNT(*) FROM file_links").fetchone()[0] == 0


def test_link_file_stamps_batch_id(conn, app):
    _create_stories(conn, app, [{"title": "Auth"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-001"])
    link_file(conn, app, "STORY-001", "auth.py", "code", batch_id=bid)
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
    b1 = start_batch(conn, app, "sess-1", ["STORY-001"])
    link_file(conn, app, "STORY-001", "auth.py", "code", batch_id=b1)
    b2 = start_batch(conn, app, "sess-1", ["STORY-001"])
    link_file(conn, app, "STORY-001", "auth.py", "code", batch_id=b2)
    row = conn.execute(
        "SELECT batch_id FROM file_links WHERE path = ?", ("auth.py",)
    ).fetchone()
    assert row[0] == b2


def test_link_file_without_batch_id_preserves_existing_stamp(conn, app):
    """Calling link_file without batch_id after a batch-stamped row
    exists must not clear the stamp — the unstamped call is a no-op
    when the (story, path, kind) tuple already exists."""
    _create_stories(conn, app, [{"title": "Auth"}])
    b1 = start_batch(conn, app, "sess-1", ["STORY-001"])
    link_file(conn, app, "STORY-001", "auth.py", "code", batch_id=b1)
    link_file(conn, app, "STORY-001", "auth.py", "code")  # no batch_id
    row = conn.execute(
        "SELECT batch_id FROM file_links WHERE path = ?", ("auth.py",)
    ).fetchone()
    assert row[0] == b1


def test_files_for_batch_returns_only_stamped_files(conn, app):
    _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    b1 = start_batch(conn, app, "sess-1", ["STORY-001", "STORY-002"])
    link_file(conn, app, "STORY-001", "a.py", "code", batch_id=b1)
    link_file(conn, app, "STORY-002", "b.py", "code", batch_id=b1)
    # Unstamped row (legacy / out-of-band touch) must not appear
    # in files_for_batch results.
    link_file(conn, app, "STORY-001", "stale.py", "doc")
    rows = files_for_batch(conn, b1)
    assert sorted(rows) == [
        ("STORY-001", "a.py", "code"),
        ("STORY-002", "b.py", "code"),
    ]


def test_set_batch_committed_sha_persists(conn, app):
    _create_stories(conn, app, [{"title": "T"}])
    bid = start_batch(conn, app, "sess-1", ["STORY-001"])
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
        sha="abc1234567", story_key="STORY-001",
        session_id="sess-1", message="STORY-001: add auth",
    )
    row = conn.execute(
        "SELECT sha, message, story_id, workspace FROM commits"
    ).fetchone()
    assert row[0] == "abc1234567"
    assert row[1] == "STORY-001: add auth"
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
        {"title": "Add logout", "feature": "auth", "depends_on": ["STORY-001"]},
    ])
    mark_done(conn, app, "STORY-001")
    link_file(conn, app, "STORY-001", "src/auth.py", "code")
    link_file(conn, app, "STORY-001", "tests/test_auth.py", "test")
    record_defect(
        conn, workspace=app, story_key="STORY-001", session_id="s1",
        severity="lint", summary="line too long",
    )
    record_commit(
        conn, workspace=app, sha="abcdef0123456789", story_key="STORY-001",
        session_id="s1", message="STORY-001: add login",
    )

    stories_path, trace_path = regenerate_markdown_views(conn, workspace)
    stories_md = Path(stories_path).read_text()
    trace_md = Path(trace_path).read_text()

    assert "STORY-001" in stories_md
    assert "STORY-002" in stories_md
    assert "auth" in stories_md
    assert "redirect after login" in stories_md
    assert "STORY-001" in trace_md
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


# ---------------------------------------------------------------------------
# v5 traceability — requirements / AC / link helpers / audit queries
# ---------------------------------------------------------------------------


def _seed_story(conn, app: str, *, title: str = "S", ac=None) -> dict:
    """Helper: ensure feature, create one story (optionally with AC), return its row."""
    keys = _create_stories(conn, app, [{
        "title": title,
        "acceptance_criteria": list(ac or []),
    }])
    return get_story(conn, app, keys[0])


def test_create_requirements_idempotent_upserts_title(conn, app):
    create_requirements(conn, app, [
        {"req_key": "FR-001", "kind": "fr", "title": "Original"},
    ])
    # Re-insert with a new title — UPSERT path, not UNIQUE error.
    create_requirements(conn, app, [
        {"req_key": "FR-001", "kind": "fr", "title": "Revised"},
    ])
    row = get_requirement_by_key(conn, app, "FR-001")
    assert row["title"] == "Revised"
    assert len(list_requirements(conn, app)) == 1


def test_create_requirements_rejects_invalid_kind(conn, app):
    with pytest.raises(ValueError, match="kind="):
        create_requirements(conn, app, [
            {"req_key": "FR-001", "kind": "bogus", "title": "x"},
        ])


def test_create_requirements_rejects_missing_req_key(conn, app):
    with pytest.raises(ValueError, match="req_key"):
        create_requirements(conn, app, [{"kind": "fr", "title": "x"}])


def test_list_requirements_filters_by_kind(conn, app):
    create_requirements(conn, app, [
        {"req_key": "FR-001", "kind": "fr", "title": "x"},
        {"req_key": "NFR-SEC-001", "kind": "nfr", "title": "y"},
        {"req_key": "US-01-02", "kind": "us", "title": "z"},
    ])
    assert [r["req_key"] for r in list_requirements(conn, app, kind="fr")] == ["FR-001"]
    assert {r["req_key"] for r in list_requirements(conn, app)} == {
        "FR-001", "NFR-SEC-001", "US-01-02",
    }


def test_ensure_requirement_creates_on_miss_returns_id_on_hit(conn, app):
    rid = ensure_requirement(conn, app, "CR-7", kind="cr_synthetic", title="CR 7")
    assert rid > 0
    # Second call returns the same id, no duplicate row.
    rid2 = ensure_requirement(conn, app, "CR-7", kind="cr_synthetic", title="ignored")
    assert rid2 == rid
    assert len(list_requirements(conn, app)) == 1


def test_link_story_to_requirements_rejects_unknown_key(conn, app):
    s = _seed_story(conn, app, title="S1")
    create_requirements(conn, app, [{"req_key": "FR-001", "kind": "fr", "title": "x"}])
    with pytest.raises(ValueError, match="FR-099"):
        link_story_to_requirements(conn, app, s["id"], ["FR-001", "FR-099"])
    # Atomicity check: FR-001 must NOT have been linked when FR-099 was missing.
    rows = conn.execute(
        "SELECT requirement_id FROM story_satisfies_req WHERE story_id = ?",
        (s["id"],),
    ).fetchall()
    assert rows == []


def test_link_story_to_requirements_idempotent(conn, app):
    s = _seed_story(conn, app, title="S1")
    create_requirements(conn, app, [{"req_key": "FR-001", "kind": "fr", "title": "x"}])
    assert link_story_to_requirements(conn, app, s["id"], ["FR-001"]) == 1
    # Second call: idempotent — no new row.
    assert link_story_to_requirements(conn, app, s["id"], ["FR-001"]) == 0


def test_create_stories_persists_ac_to_side_table(conn, app):
    s = _seed_story(conn, app, ac=["AC text 1", "AC text 2"])
    acs = list_acceptance_criteria(conn, app, s["id"])
    assert [a["ac_key"] for a in acs] == [f"{s['story_key']}.AC-1", f"{s['story_key']}.AC-2"]
    assert [a["text"] for a in acs] == ["AC text 1", "AC text 2"]
    assert [a["ordinal"] for a in acs] == [1, 2]


def test_list_stories_backfills_acceptance_criteria_from_side_table(conn, app):
    _seed_story(conn, app, title="S1", ac=["a", "b"])
    _seed_story(conn, app, title="S2", ac=["c"])
    stories = list_stories(conn, app)
    by_title = {s["title"]: s for s in stories}
    assert by_title["S1"]["acceptance_criteria"] == ["a", "b"]
    assert by_title["S2"]["acceptance_criteria"] == ["c"]


def test_create_acceptance_criteria_upserts_text_preserving_id(conn, app):
    s = _seed_story(conn, app, ac=["first"])
    first = list_acceptance_criteria(conn, app, s["id"])[0]
    # UPSERT — re-write the text under the same ac_key.
    create_acceptance_criteria(conn, app, s["id"], [
        {"ac_key": first["ac_key"], "text": "rewritten", "ordinal": 1},
    ])
    after = list_acceptance_criteria(conn, app, s["id"])
    assert len(after) == 1
    assert after[0]["id"] == first["id"]  # id preserved across UPSERT
    assert after[0]["text"] == "rewritten"


def test_get_ac_by_key_round_trip(conn, app):
    s = _seed_story(conn, app, ac=["only AC"])
    ac_key = f"{s['story_key']}.AC-1"
    row = get_ac_by_key(conn, app, ac_key)
    assert row is not None
    assert row["text"] == "only AC"
    assert get_ac_by_key(conn, app, "NOT-AN-AC") is None


def test_link_test_to_ac_idempotent(conn, app):
    s = _seed_story(conn, app, ac=["AC1"])
    ac = list_acceptance_criteria(conn, app, s["id"])[0]
    assert link_test_to_ac(conn, app, "tests/test_x.py", ac["id"]) is True
    assert link_test_to_ac(conn, app, "tests/test_x.py", ac["id"]) is False
    # Different function name → distinct row, both kept.
    assert link_test_to_ac(conn, app, "tests/test_x.py", ac["id"], "test_a") is True


def test_requirements_without_satisfying_story_surfaces_gaps(conn, app):
    create_requirements(conn, app, [
        {"req_key": "FR-001", "kind": "fr", "title": "covered"},
        {"req_key": "FR-002", "kind": "fr", "title": "untraced"},
    ])
    s = _seed_story(conn, app, title="S1")
    link_story_to_requirements(conn, app, s["id"], ["FR-001"])
    untraced = requirements_without_satisfying_story(conn, app)
    assert [r["req_key"] for r in untraced] == ["FR-002"]


def test_requirements_without_satisfying_story_empty_when_clean(conn, app):
    assert requirements_without_satisfying_story(conn, app) == []


def test_acs_without_verifying_test_groups_by_story(conn, app):
    s1 = _seed_story(conn, app, title="S1", ac=["a", "b"])
    s2 = _seed_story(conn, app, title="S2", ac=["c"])
    # Verify only S1.AC-1.
    s1_acs = list_acceptance_criteria(conn, app, s1["id"])
    link_test_to_ac(conn, app, "tests/test_s1.py", s1_acs[0]["id"])
    untested = acs_without_verifying_test(conn, app)
    assert [(u["story_key"], u["ac_key"]) for u in untested] == [
        (s1["story_key"], f"{s1['story_key']}.AC-2"),
        (s2["story_key"], f"{s2['story_key']}.AC-1"),
    ]


def test_acs_without_verifying_test_empty_when_clean(conn, app):
    assert acs_without_verifying_test(conn, app) == []


def test_migrate_v4_to_v5_drops_legacy_acceptance_criteria_column(isolated_state_db):
    """Seed a v4-shape DB on disk; reopen; assert v5 schema landed and
    the legacy JSON column is gone."""
    db_path = state_db_path()
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE schema_meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            INSERT INTO schema_meta(key, value) VALUES ('schema_version', '4');
            CREATE TABLE stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace TEXT NOT NULL,
                story_key TEXT NOT NULL,
                feature_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                acceptance_criteria TEXT NOT NULL DEFAULT '[]',
                depends_on TEXT NOT NULL DEFAULT '[]',
                scope_files TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'planned',
                external_ref TEXT,
                build_kind TEXT NOT NULL DEFAULT 'greenfield',
                cr_ids TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                UNIQUE(workspace, story_key)
            );
            INSERT INTO stories(
                workspace, story_key, title, acceptance_criteria, created_at
            ) VALUES ('demo', 'STORY-001', 'legacy v4', '["old"]', '2026-01-01T00:00:00');
        """)
        conn.commit()
    finally:
        conn.close()
    # Reopen — migration must run.
    conn = open_story_db()
    try:
        assert _read_schema_version(conn) == SCHEMA_VERSION
        cols = {c[1] for c in conn.execute("PRAGMA table_info(stories)").fetchall()}
        assert "acceptance_criteria" not in cols
        # v4 rows are dropped per the clean-slate precedent.
        assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 0
        # v5 side tables present.
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='requirements'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_purge_state_db_clears_v5_tables(workspace, app):
    """purge_state_db must DELETE rows from all four new v5 tables."""
    conn = open_story_db()
    try:
        _seed_feature(conn, app)
        keys = create_stories(conn, app, [{
            "title": "S", "feature": "test",
            "acceptance_criteria": ["AC1"],
        }])
        sid = get_story(conn, app, keys[0])["id"]
        create_requirements(conn, app, [
            {"req_key": "FR-001", "kind": "fr", "title": "x"},
        ])
        link_story_to_requirements(conn, app, sid, ["FR-001"])
        ac = list_acceptance_criteria(conn, app, sid)[0]
        link_test_to_ac(conn, app, "tests/test_x.py", ac["id"])
    finally:
        conn.close()

    counts = purge_state_db(workspace)
    assert counts["requirements"] >= 1
    assert counts["acceptance_criteria"] >= 1
    assert counts["story_satisfies_req"] >= 1
    assert counts["test_verifies_ac"] >= 1

    conn = open_story_db()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM requirements WHERE workspace = ?", (app,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM acceptance_criteria WHERE workspace = ?", (app,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM test_verifies_ac WHERE workspace = ?", (app,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_purge_state_db_all_wipes_every_workspace(tmp_path):
    """purge_state_db_all must clear rows across ALL workspaces, not just one."""
    from harness.story_state import purge_state_db_all

    ws_a = tmp_path / "ws-alpha"
    ws_a.mkdir()
    ws_b = tmp_path / "ws-beta"
    ws_b.mkdir()
    app_a = app_name_for_workspace(str(ws_a))
    app_b = app_name_for_workspace(str(ws_b))

    conn = open_story_db()
    try:
        for a in (app_a, app_b):
            _seed_feature(conn, a)
            keys = create_stories(conn, a, [{
                "title": "S", "feature": "test",
                "acceptance_criteria": ["AC1"],
            }])
            sid = get_story(conn, a, keys[0])["id"]
            create_requirements(conn, a, [
                {"req_key": "FR-001", "kind": "fr", "title": "x"},
            ])
            link_story_to_requirements(conn, a, sid, ["FR-001"])
    finally:
        conn.close()

    counts = purge_state_db_all()
    assert counts["stories"] >= 2
    assert counts["features"] >= 2
    assert counts["requirements"] >= 2

    conn = open_story_db()
    try:
        for table in ("features", "stories", "requirements",
                      "acceptance_criteria", "story_satisfies_req"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n == 0, f"{table} still has {n} row(s) after global purge"
    finally:
        conn.close()


def test_purge_state_db_all_no_db_returns_zeros(tmp_path, monkeypatch):
    """Missing DB file must return the empty counts dict without raising."""
    from harness.story_state import purge_state_db_all

    monkeypatch.setenv("TEANE_STATE_DB", str(tmp_path / "does-not-exist.db"))
    counts = purge_state_db_all()
    assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# rollback_unlinked_done_stories — traceability rollback safety net (A6)
# ---------------------------------------------------------------------------

def test_rollback_unlinked_done_stories_downgrades_only_unlinked(conn, app):
    """Finsearch session 44c5e194 root cause A6: when the traceability
    gate forces END with non-zero exit, done-but-unlinked stories must
    be downgraded to blocked so TRACEABILITY.md reflects the failure
    instead of a false-positive green row."""
    from harness.story_state import rollback_unlinked_done_stories

    keys = _create_stories(conn, app, [
        {"title": "A"}, {"title": "B"}, {"title": "C"},
    ])
    # Mark all done, but only STORY-002 gets a file_link.
    for k in keys:
        mark_done(conn, app, k)
    link_file(conn, app, "STORY-002", "src/b.py", "code")

    rolled = rollback_unlinked_done_stories(conn, app, session_id="sess-1")
    assert sorted(rolled) == ["STORY-001", "STORY-003"]

    # STORY-002 stays done; the other two flip to blocked.
    assert get_story(conn, app, "STORY-001")["status"] == "blocked"
    assert get_story(conn, app, "STORY-002")["status"] == "done"
    assert get_story(conn, app, "STORY-003")["status"] == "blocked"

    # Defects recorded with the expected severity so post-run inspection
    # can group them.
    defs = conn.execute(
        "SELECT severity, story_id FROM defects "
        "WHERE session_id = 'sess-1' AND status = 'open'"
    ).fetchall()
    assert len(defs) == 2
    assert all(d[0] == "traceability_rollback" for d in defs)


def test_rollback_unlinked_done_stories_noop_when_all_linked(conn, app):
    from harness.story_state import rollback_unlinked_done_stories

    keys = _create_stories(conn, app, [{"title": "A"}, {"title": "B"}])
    for k in keys:
        mark_done(conn, app, k)
        link_file(conn, app, k, f"src/{k.lower()}.py", "code")

    assert rollback_unlinked_done_stories(conn, app, "sess-1") == []
    for k in keys:
        assert get_story(conn, app, k)["status"] == "done"


def test_rollback_leaves_non_done_stories_alone(conn, app):
    """Planned / in_progress / blocked stories are outside scope — the
    rollback only touches ``done`` rows that lack file_links."""
    from harness.story_state import rollback_unlinked_done_stories

    keys = _create_stories(conn, app, [
        {"title": "planned"}, {"title": "wip"}, {"title": "blocked"},
        {"title": "done-linked"}, {"title": "done-unlinked"},
    ])
    mark_in_progress(conn, app, keys[1])
    mark_blocked(conn, app, keys[2])
    mark_done(conn, app, keys[3])
    link_file(conn, app, keys[3], "src/x.py", "code")
    mark_done(conn, app, keys[4])

    rolled = rollback_unlinked_done_stories(conn, app, "sess-1")
    assert rolled == [keys[4]]
    # Every other story is untouched.
    assert get_story(conn, app, keys[0])["status"] == "planned"
    assert get_story(conn, app, keys[1])["status"] == "in_progress"
    assert get_story(conn, app, keys[2])["status"] == "blocked"
    assert get_story(conn, app, keys[3])["status"] == "done"
