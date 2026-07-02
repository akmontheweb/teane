"""SQLite-backed state store for Agile-style story decomposition.

This module owns the data layer for teane's per-story TDD flow:

- The **stories** table is the canonical record of work units the
  decomposition LLM derived from ``SPEC_REQUIREMENTS.md``. Each row
  carries acceptance criteria, dependencies, and scope hints.
- **batches** group stories the planner picked for one execution
  round; **batch_stories** records sequence within a batch.
- **defects** capture per-story repair-cap failures (compile / lint /
  test / review / security) so failed acceptances are queryable
  rather than buried in log files.
- **test_runs** record acceptance-test outcomes per story phase
  (tests_first / repair / final).
- **file_links** map stories to the files they touched (code / test
  / doc / infra) — the spine of the traceability matrix.
- **commits** optionally record git SHAs per story when
  ``agile_defaults.commit_on_story`` is enabled in config.json. The DB
  is the source of truth; git is a snapshot layer on top.

**Global, multi-workspace DB.** The store lives at a single shared
location (``~/.harness/state.db`` by default; override with the
``TEANE_STATE_DB`` env var) and carries rows for every workspace teane
has touched. Each row is scoped by a ``workspace`` column whose value
is the workspace folder's basename — that name is treated as the
**app name** and is assumed unique across the operator's machine.
Two workspaces sharing the same basename WILL collide.

All graph nodes go through the typed helpers here — the LLM never
composes SQL. Every helper takes the workspace identifier explicitly
so the SQL stays scoped to the calling app's rows.
``regenerate_markdown_views`` rebuilds ``docs/STORIES.md`` and
``docs/TRACEABILITY.md`` from the DB; teane never edits those
markdown files directly.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 5
"""Bump when adding columns. Add a ``_migrate_to_vN`` function and
register it in ``_MIGRATIONS`` below. Forward-only.

v5 (industry-grade traceability — schema-v5 plan):
- New ``requirements`` table — FR/NFR/US identifiers parsed from
  ``docs/SPEC_REQUIREMENTS.md`` become first-class rows. ``req_key``
  is the literal token (``FR-007``, ``NFR-SEC-001``, ``US-03-02``);
  ``kind`` records which token family it came from so audit views
  can group naturally.
- New ``acceptance_criteria`` table — promotes the old
  ``stories.acceptance_criteria`` JSON column into per-row records
  with stable ``ac_key`` identifiers (``STORY-3.AC-2``). Lets the
  test-gen marker contract (``# @verifies: STORY-3.AC-2``) point at
  a real PK. The legacy JSON column is dropped from v5.
- New link tables ``story_satisfies_req`` (story → requirement, M:N)
  and ``test_verifies_ac`` (test file → AC, M:N). These are the
  edges the SQL coverage audit (``harness/traceability.py``) joins
  against to surface untraced FRs and untested ACs.
- The v4→v5 migration drops every v4 row outright, mirroring the
  v3→v4 precedent. v5 introduces two new LLM prompt contracts
  (decomposition cites ``requirement_keys``; test-gen emits
  ``@verifies`` markers) so old rows can't carry the new edges
  retroactively. Re-decomposition is required.

v4 (feature-first decomposition):
- New ``features`` table — every story now belongs to exactly one
  feature. The decomposition LLM emits ``features`` (mandatory) +
  ``stories`` (each carrying a ``feature`` key ref) instead of the
  previous optional ``epics`` grouping.
- ``stories.feature_id`` FK replaces the old free-text ``stories.epic``
  column. The ``idx_stories_epic`` index is gone; ``idx_stories_feature``
  takes its place.
- ``batches.feature_id`` records which feature a batch belongs to.
  Invariant: a batch never spans features. Small features land in one
  batch; large features may need multiple batches (all tagged with the
  same feature_id), but no batch ever mixes stories from two features.
- The v3→v4 migration drops the legacy tables outright (clean slate).
  Operators upgrading from v3 lose prior story/batch history — which
  is acceptable per the v4 product decision; the decomposition shape
  changed enough that the old rows can't be meaningfully reinterpreted.

v3 (multi-workspace global DB + CR tagging):
- ``stories``, ``batches``, ``defects``, ``test_runs``, ``file_links``,
  ``commits`` each carry a ``workspace TEXT NOT NULL`` column scoping
  the row to its owning app.
- ``stories.story_key`` is no longer globally unique — only unique
  WITHIN a workspace. The composite ``UNIQUE(workspace, story_key)``
  replaces the old single-column UNIQUE.
- ``stories`` and ``batches`` carry ``build_kind`` (``greenfield`` or
  ``cr``) plus ``cr_ids`` (JSON list of integer CR ids ingested in the
  run that created the row). Together these mark a row as an
  incremental change-request layer on top of the original build, so
  traceability views can show "STORY-7 was added by CR-2" without
  losing the greenfield history.

v2 (per-batch verification pipeline, historical — applied to a v1 DB):
- ``file_links.batch_id`` — which batch last touched this file
  (for batch-level repair attribution).
- ``batches.committed_sha`` — git SHA of the BATCH-N commit, when
  git-commit-on-batch is enabled.
"""

CR_FEATURE_KEY = "change-request"
"""Synthetic feature_key used by the CR-bridge in graph.py for
change-request-derived stories that don't fit into a planner-emitted
feature. One row per workspace, created lazily on first CR bridge."""

BUILD_KIND_GREENFIELD = "greenfield"
BUILD_KIND_CR = "cr"
_VALID_BUILD_KINDS = frozenset({BUILD_KIND_GREENFIELD, BUILD_KIND_CR})

# Default path for the harness-global state.db. Override at runtime
# with the ``TEANE_STATE_DB`` env var — pytest's autouse fixture
# (tests/conftest.py) uses that to redirect each test's DB into its
# own ``tmp_path`` so test runs never touch the operator's real file.
_DEFAULT_STATE_DB_PATH = "~/.harness/state.db"

_INVALID_BASENAME = frozenset({"", ".", ".."})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    feature_key TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(workspace, feature_key)
);
CREATE INDEX IF NOT EXISTS idx_features_workspace ON features(workspace);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    story_key TEXT NOT NULL,
    feature_id INTEGER REFERENCES features(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    description TEXT,
    -- v4 had a JSON acceptance_criteria column here; v5 promotes it
    -- to the acceptance_criteria side table below.
    depends_on TEXT NOT NULL DEFAULT '[]',           -- JSON list[story_key]
    scope_files TEXT NOT NULL DEFAULT '[]',          -- JSON list[path]
    status TEXT NOT NULL DEFAULT 'planned',
    external_ref TEXT,
    build_kind TEXT NOT NULL DEFAULT 'greenfield',
    cr_ids TEXT,                                     -- JSON list[int] | NULL
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(workspace, story_key)
);
CREATE INDEX IF NOT EXISTS idx_stories_workspace ON stories(workspace);
CREATE INDEX IF NOT EXISTS idx_stories_status ON stories(workspace, status);
CREATE INDEX IF NOT EXISTS idx_stories_feature ON stories(workspace, feature_id);
CREATE INDEX IF NOT EXISTS idx_stories_build_kind
    ON stories(workspace, build_kind);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    session_id TEXT NOT NULL,
    feature_id INTEGER REFERENCES features(id) ON DELETE SET NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    committed_sha TEXT,
    build_kind TEXT NOT NULL DEFAULT 'greenfield',
    cr_ids TEXT                                       -- JSON list[int] | NULL
);
CREATE INDEX IF NOT EXISTS idx_batches_workspace ON batches(workspace);
CREATE INDEX IF NOT EXISTS idx_batches_feature ON batches(workspace, feature_id);
CREATE INDEX IF NOT EXISTS idx_batches_build_kind
    ON batches(workspace, build_kind);

CREATE TABLE IF NOT EXISTS batch_stories (
    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    PRIMARY KEY (batch_id, story_id)
);

CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    story_id INTEGER REFERENCES stories(id) ON DELETE SET NULL,
    session_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    diagnostic_json TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_defects_workspace ON defects(workspace);
CREATE INDEX IF NOT EXISTS idx_defects_story ON defects(story_id);
CREATE INDEX IF NOT EXISTS idx_defects_status ON defects(workspace, status);

CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    story_id INTEGER REFERENCES stories(id) ON DELETE SET NULL,
    session_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    stdout_tail TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_test_runs_workspace ON test_runs(workspace);

CREATE TABLE IF NOT EXISTS file_links (
    workspace TEXT NOT NULL,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    batch_id INTEGER REFERENCES batches(id) ON DELETE SET NULL,
    PRIMARY KEY (story_id, path, kind)
);
CREATE INDEX IF NOT EXISTS idx_file_links_workspace ON file_links(workspace);
CREATE INDEX IF NOT EXISTS idx_file_links_path ON file_links(path);
CREATE INDEX IF NOT EXISTS idx_file_links_batch ON file_links(batch_id);

CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    story_id INTEGER REFERENCES stories(id) ON DELETE SET NULL,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_commits_workspace ON commits(workspace);
CREATE INDEX IF NOT EXISTS idx_commits_story ON commits(story_id);

-- v5 traceability tables ----------------------------------------------------
CREATE TABLE IF NOT EXISTS requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    req_key TEXT NOT NULL,           -- FR-007 / NFR-SEC-001 / US-03-02 / CR-7
    kind TEXT NOT NULL,              -- 'fr' | 'nfr' | 'us' | 'cr_synthetic'
    title TEXT,
    body TEXT,
    source_path TEXT,                -- 'docs/SPEC_REQUIREMENTS.md'
    source_line INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(workspace, req_key)
);
CREATE INDEX IF NOT EXISTS idx_requirements_workspace ON requirements(workspace);
CREATE INDEX IF NOT EXISTS idx_requirements_kind ON requirements(workspace, kind);

CREATE TABLE IF NOT EXISTS acceptance_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace TEXT NOT NULL,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    ac_key TEXT NOT NULL,            -- 'STORY-3.AC-2'
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    UNIQUE(story_id, ac_key)
);
CREATE INDEX IF NOT EXISTS idx_ac_workspace ON acceptance_criteria(workspace);
CREATE INDEX IF NOT EXISTS idx_ac_story ON acceptance_criteria(story_id);

CREATE TABLE IF NOT EXISTS story_satisfies_req (
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    requirement_id INTEGER NOT NULL REFERENCES requirements(id) ON DELETE CASCADE,
    PRIMARY KEY (story_id, requirement_id)
);
CREATE INDEX IF NOT EXISTS idx_ssr_req ON story_satisfies_req(requirement_id);

-- test_function_name defaults to '' (not NULL) so the composite PK
-- treats file-level markers as a stable triple; INSERT OR IGNORE would
-- otherwise create a duplicate row each time (SQLite's UNIQUE treats
-- NULLs as distinct).
CREATE TABLE IF NOT EXISTS test_verifies_ac (
    workspace TEXT NOT NULL,
    test_path TEXT NOT NULL,
    test_function_name TEXT NOT NULL DEFAULT '',
    ac_id INTEGER NOT NULL REFERENCES acceptance_criteria(id) ON DELETE CASCADE,
    PRIMARY KEY (test_path, test_function_name, ac_id)
);
CREATE INDEX IF NOT EXISTS idx_tva_workspace ON test_verifies_ac(workspace);
CREATE INDEX IF NOT EXISTS idx_tva_ac ON test_verifies_ac(ac_id);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    # Mirrors harness/web_state.py:99 — WAL + busy-timeout avoids
    # ``database is locked`` if the dashboard ever reads while the
    # graph writes from another thread.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except sqlite3.DatabaseError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Workspace / path helpers
# ---------------------------------------------------------------------------

def state_db_path() -> str:
    """Return the active state.db path.

    Resolution order:
      1. ``TEANE_STATE_DB`` env var (used by tests to isolate per
         tmp_path; could also be used by an operator to point at a
         shared store on a network mount).
      2. ``_DEFAULT_STATE_DB_PATH`` — ``~/.harness/state.db``.
    """
    override = os.environ.get("TEANE_STATE_DB", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.expanduser(_DEFAULT_STATE_DB_PATH)


def app_name_for_workspace(workspace_path: str) -> str:
    """Derive the workspace's app-name identifier.

    The app name is the workspace folder's basename. teane assumes
    these basenames are unique across the operator's machine — two
    workspaces with the same folder name WILL share rows in state.db.
    """
    s = (workspace_path or "").rstrip("/\\")
    base = os.path.basename(s) if s else ""
    if base in _INVALID_BASENAME:
        raise ValueError(
            f"workspace_path={workspace_path!r} has no usable basename; "
            "cannot derive an app name for state.db scoping."
        )
    return base


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def _read_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _write_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Clean-slate migration: drop legacy story/batch tables so the v4
    schema can recreate them with the feature-first shape.

    The product decision (2026-06-25) was that v3 history doesn't carry
    forward — the decomposition shape changed enough that old rows can't
    be meaningfully reinterpreted, so existing DBs reset rather than
    migrate row-by-row.

    No-op on a fresh DB (no ``stories`` table) and on an already-v4 DB
    (no ``epic`` column on ``stories``). The surrounding
    ``open_story_db`` runs ``_SCHEMA_SQL`` AFTER this migration, so we
    just drop the legacy tables here and let the caller re-create the
    v4 shape.
    """
    try:
        cols = conn.execute("PRAGMA table_info(stories)").fetchall()
    except sqlite3.DatabaseError:
        return
    if not cols:
        # Fresh DB — no stories table yet. _SCHEMA_SQL will create the
        # v4 shape from scratch after we return.
        return
    if not any(c[1] == "epic" for c in cols):
        # Already migrated — stories table is v4-shaped.
        return
    for table in (
        "commits", "test_runs", "file_links", "defects",
        "batch_stories", "batches", "stories", "features",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Clean-slate migration: drop legacy v4 tables so the v5 schema
    recreates them with the traceability shape.

    v5 introduces two new LLM contracts (decomposition cites
    ``requirement_keys`` per story; test-gen emits ``@verifies``
    markers per test file) so v4 rows can't be re-interpreted to carry
    the new ``story_satisfies_req`` / ``test_verifies_ac`` edges —
    re-decomposition + re-test-gen is the only way to populate them.
    Following the v3→v4 precedent, this migration drops v4 state;
    ``_SCHEMA_SQL`` recreates everything from scratch.

    No-op on a fresh DB (no ``stories`` table) and on an already-v5
    DB (stories no longer carries the legacy ``acceptance_criteria``
    column, indicating AC has been promoted out of JSON).
    """
    try:
        cols = conn.execute("PRAGMA table_info(stories)").fetchall()
    except sqlite3.DatabaseError:
        return
    if not cols:
        return
    if not any(c[1] == "acceptance_criteria" for c in cols):
        return
    for table in (
        "test_verifies_ac", "story_satisfies_req",
        "acceptance_criteria", "requirements",
        "commits", "test_runs", "file_links", "defects",
        "batch_stories", "batches", "stories", "features",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


_MIGRATIONS: list[tuple[int, Any]] = [
    (4, _migrate_v3_to_v4),
    (5, _migrate_v4_to_v5),
]
"""(target_version, callable(conn) -> None). Append-only; never rewrite
history. v1 → v2 and v2 → v3 only existed for per-workspace DBs that
v3 explicitly ignores — fresh global DBs land at SCHEMA_VERSION
straight from ``_SCHEMA_SQL`` with no migration steps. v3 → v4 is the
feature-first reset described in ``_migrate_v3_to_v4``; v4 → v5 is the
traceability reset described in ``_migrate_v4_to_v5``."""


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = _read_schema_version(conn)
    for target, fn in _MIGRATIONS:
        if current < target:
            fn(conn)
            current = target
            _write_schema_version(conn, current)
    if current < SCHEMA_VERSION:
        _write_schema_version(conn, SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Open / purge
# ---------------------------------------------------------------------------

def purge_state_db(workspace_path: str) -> dict[str, int]:
    """Delete every row in state.db tied to ``workspace_path``'s app
    name, leaving rows for other workspaces untouched.

    Used by ``--new-build`` so the next run starts as a fresh thread
    for THIS workspace while preserving every other app's history.
    Best-effort: a missing DB or open failure logs and returns zeros.

    Returns a dict ``{table: rows_deleted}`` for stories / batches /
    defects / test_runs / file_links / commits so the caller can log
    a clear summary.
    """
    counts: dict[str, int] = {
        "features": 0, "stories": 0, "batches": 0, "defects": 0,
        "test_runs": 0, "file_links": 0, "commits": 0,
        "requirements": 0, "acceptance_criteria": 0,
        "story_satisfies_req": 0, "test_verifies_ac": 0,
    }
    try:
        app = app_name_for_workspace(workspace_path)
    except ValueError as exc:
        logger.warning("[story_state] purge skipped: %s", exc)
        return counts

    db = state_db_path()
    if not os.path.isfile(db):
        return counts

    try:
        conn = sqlite3.connect(db)
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "[story_state] Could not open %s to purge app %r: %s",
            db, app, exc,
        )
        return counts

    try:
        _apply_sqlite_pragmas(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Order matters: child tables first so parent deletes don't
            # trip a FK warning. v5 chain (deepest first):
            #   test_verifies_ac → acceptance_criteria → stories
            #   story_satisfies_req → requirements / stories
            #   file_links / test_runs / defects / commits → stories
            #   batch_stories → batches / stories
            #   batches / stories / requirements / features
            for table in (
                "test_verifies_ac",
                "story_satisfies_req",
                "acceptance_criteria",
                "file_links", "test_runs", "defects", "commits",
                "batch_stories",  # transitively via batches+stories
            ):
                if table == "batch_stories":
                    cur = conn.execute(
                        "DELETE FROM batch_stories WHERE batch_id IN "
                        "(SELECT id FROM batches WHERE workspace = ?) "
                        "OR story_id IN "
                        "(SELECT id FROM stories WHERE workspace = ?)",
                        (app, app),
                    )
                elif table == "story_satisfies_req":
                    # No workspace column on the link table — scope by
                    # FK to stories.
                    cur = conn.execute(
                        "DELETE FROM story_satisfies_req WHERE story_id IN "
                        "(SELECT id FROM stories WHERE workspace = ?)",
                        (app,),
                    )
                else:
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE workspace = ?", (app,),
                    )
                # batch_stories isn't in the counts dict (it's a join
                # table without a workspace column of its own).
                if table in counts:
                    counts[table] = cur.rowcount or 0
            for table in ("batches", "stories", "requirements", "features"):
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE workspace = ?", (app,),
                )
                counts[table] = cur.rowcount or 0
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    total = sum(counts.values())
    if total:
        logger.info(
            "[story_state] Purged %d row(s) for app %r from %s "
            "(stories=%d, batches=%d, defects=%d, test_runs=%d, "
            "file_links=%d, commits=%d, requirements=%d, "
            "acceptance_criteria=%d, story_satisfies_req=%d, "
            "test_verifies_ac=%d).",
            total, app, db,
            counts["stories"], counts["batches"], counts["defects"],
            counts["test_runs"], counts["file_links"], counts["commits"],
            counts["requirements"], counts["acceptance_criteria"],
            counts["story_satisfies_req"], counts["test_verifies_ac"],
        )
    return counts


def purge_state_db_all() -> dict[str, int]:
    """Delete every row from every table in state.db across ALL workspaces.

    Used by ``teane purge --all``. Preserves the DB file and schema so
    subsequent runs skip re-migration. Best-effort: a missing DB or open
    failure logs and returns zeros.

    Returns a dict ``{table: rows_deleted}`` mirroring the per-workspace
    variant so callers can log a symmetric summary.
    """
    counts: dict[str, int] = {
        "features": 0, "stories": 0, "batches": 0, "defects": 0,
        "test_runs": 0, "file_links": 0, "commits": 0,
        "requirements": 0, "acceptance_criteria": 0,
        "story_satisfies_req": 0, "test_verifies_ac": 0,
        "batch_stories": 0,
    }
    db = state_db_path()
    if not os.path.isfile(db):
        return counts

    try:
        conn = sqlite3.connect(db)
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "[story_state] Could not open %s for global purge: %s", db, exc,
        )
        return counts

    try:
        _apply_sqlite_pragmas(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Deepest FK children first, then parents.
            for table in (
                "test_verifies_ac",
                "story_satisfies_req",
                "acceptance_criteria",
                "file_links", "test_runs", "defects", "commits",
                "batch_stories",
                "batches", "stories", "requirements", "features",
            ):
                try:
                    cur = conn.execute(f"DELETE FROM {table}")
                except sqlite3.OperationalError:
                    # Table absent (older DB / partial migration) — skip.
                    continue
                counts[table] = cur.rowcount or 0
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    total = sum(counts.values())
    if total:
        logger.info(
            "[story_state] Global purge removed %d row(s) from %s.",
            total, db,
        )
    return counts


def workspace_is_agile_managed(workspace_path: str) -> bool:
    """Return True when ``workspace_path`` has at least one row in the
    ``stories`` table of the global state DB — i.e. a prior ``teane
    build`` (or ``patch``) engaged Agile mode on this workspace.

    Used by ``cmd_patch`` as the default for the ``--agile`` tri-state
    flag. Soft-fails to False on any error (missing DB, locked file,
    invalid workspace path) — the operator can always force agile mode
    with ``--agile true`` regardless.
    """
    try:
        app = app_name_for_workspace(workspace_path)
    except ValueError:
        return False
    db = state_db_path()
    if not os.path.isfile(db):
        return False
    try:
        conn = sqlite3.connect(db)
    except sqlite3.DatabaseError:
        return False
    try:
        cur = conn.execute(
            "SELECT 1 FROM stories WHERE workspace = ? LIMIT 1", (app,),
        )
        return cur.fetchone() is not None
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def open_story_db(workspace_path: Optional[str] = None) -> sqlite3.Connection:
    """Open (creating + migrating) the harness-global story DB.

    The ``workspace_path`` argument is accepted for back-compat with
    call sites that already pass it (and so the same import path keeps
    working) but is NOT used to derive the DB location any more — the
    DB is a single file shared across every workspace, scoped per row
    via the ``workspace`` column.

    Caller closes. Pattern mirrors ``harness/web_state.py:115`` —
    close the connection if anything in schema setup raises so we
    don't leak fds on a half-initialised DB.
    """
    del workspace_path  # accepted for back-compat; not used
    path = state_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _apply_sqlite_pragmas(conn)
        # Ensure schema_meta exists so _apply_migrations can read /
        # write schema_version even on a legacy v3 DB whose tables
        # predate the migration framework. Without this bootstrap
        # ``_write_schema_version`` would crash with "no such table:
        # schema_meta" on the very first migration of a v3 DB.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        # Apply migrations BEFORE _SCHEMA_SQL: _SCHEMA_SQL creates v4
        # indexes that reference v4-only columns (e.g.
        # ``stories.feature_id``). If a v3 DB still has the legacy
        # ``stories`` table on disk, ``executescript`` blows up with
        # ``no such column: feature_id`` before the v3→v4 migration
        # ever gets a chance to drop the legacy tables. Running
        # migrations first leaves a clean slate for the v4 schema.
        _apply_migrations(conn)
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    except Exception:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return conn


# ---------------------------------------------------------------------------
# Feature CRUD
# ---------------------------------------------------------------------------

def _row_to_feature(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "feature_key": row[1],
        "name": row[2],
        "description": row[3],
        "created_at": row[4],
    }


_FEATURE_COLS = "id, feature_key, name, description, created_at"


def create_features(
    conn: sqlite3.Connection,
    workspace: str,
    items: Iterable[dict[str, Any]],
) -> list[str]:
    """Insert features and return the assigned feature_keys in order.

    Each item supports: ``feature_key`` (required, must be unique within
    ``workspace`` — typically a short slug like ``"auth"`` or
    ``"billing"``), ``name`` (required, human-readable label),
    ``description`` (optional). Idempotent on duplicate ``feature_key``
    — the existing row is left untouched and its key is returned, so
    callers can safely re-run decomposition without tripping a UNIQUE
    violation.
    """
    created: list[str] = []
    now = _utcnow_iso()
    for item in items:
        feature_key = (item.get("feature_key") or "").strip()
        if not feature_key:
            raise ValueError("feature requires 'feature_key'")
        name = (item.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"feature {feature_key!r} requires a non-empty 'name'"
            )
        conn.execute(
            "INSERT INTO features(workspace, feature_key, name, "
            "description, created_at) VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(workspace, feature_key) DO NOTHING",
            (workspace, feature_key, name, item.get("description"), now),
        )
        created.append(feature_key)
    conn.commit()
    return created


def list_features(
    conn: sqlite3.Connection, workspace: str,
) -> list[dict[str, Any]]:
    """All features for ``workspace`` ordered by insertion."""
    rows = conn.execute(
        f"SELECT {_FEATURE_COLS} FROM features WHERE workspace = ? "
        "ORDER BY id",
        (workspace,),
    ).fetchall()
    return [_row_to_feature(r) for r in rows]


def get_feature_by_key(
    conn: sqlite3.Connection, workspace: str, feature_key: str,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        f"SELECT {_FEATURE_COLS} FROM features "
        "WHERE workspace = ? AND feature_key = ?",
        (workspace, feature_key),
    ).fetchone()
    return _row_to_feature(row) if row else None


def get_feature(
    conn: sqlite3.Connection, workspace: str, feature_id: int,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        f"SELECT {_FEATURE_COLS} FROM features "
        "WHERE workspace = ? AND id = ?",
        (workspace, int(feature_id)),
    ).fetchone()
    return _row_to_feature(row) if row else None


def ensure_feature(
    conn: sqlite3.Connection,
    workspace: str,
    feature_key: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> int:
    """Return the feature_id for ``feature_key`` in ``workspace``,
    creating the row lazily if it doesn't exist. Used by the CR-bridge
    in graph.py so the synthetic ``change-request`` feature gets seeded
    the first time a workspace ingests a CR.
    """
    existing = get_feature_by_key(conn, workspace, feature_key)
    if existing is not None:
        return int(existing["id"])
    create_features(
        conn, workspace,
        [{
            "feature_key": feature_key,
            "name": name or feature_key,
            "description": description,
        }],
    )
    again = get_feature_by_key(conn, workspace, feature_key)
    if again is None:
        raise RuntimeError(
            f"ensure_feature: failed to materialise feature {feature_key!r}"
        )
    return int(again["id"])


# ---------------------------------------------------------------------------
# Requirements CRUD (v5)
# ---------------------------------------------------------------------------

_VALID_REQ_KINDS = frozenset({
    # Waterfall / ISO 29148 families.
    "fr", "nfr", "us",
    # Agile / SAFe families (Phase 8 — emitted by the SAFe Path A
    # of harness/skills/docgen/requirements_doc.md).
    "epic", "feat", "safe_story", "safe_nfr_story",
    # Synthetic kind for change-request-bridged requirements.
    "cr_synthetic",
})


def _row_to_requirement(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "req_key": row[1],
        "kind": row[2],
        "title": row[3],
        "body": row[4],
        "source_path": row[5],
        "source_line": row[6],
        "created_at": row[7],
    }


_REQUIREMENT_COLS = (
    "id, req_key, kind, title, body, source_path, source_line, created_at"
)


def create_requirements(
    conn: sqlite3.Connection,
    workspace: str,
    items: Iterable[dict[str, Any]],
) -> list[str]:
    """Insert requirements parsed from ``docs/SPEC_REQUIREMENTS.md``.

    Each item supports: ``req_key`` (required — literal FR-007 /
    NFR-SEC-001 / US-03-02 / CR-7 token), ``kind`` (required — one of
    ``fr``, ``nfr``, ``us``, ``cr_synthetic``), ``title``, ``body``,
    ``source_path``, ``source_line``. Idempotent: on duplicate
    ``req_key`` within a workspace, ``title``/``body`` are upserted so
    spec edits propagate without an operator purge. Returns the
    ``req_key``s of every requested row (whether freshly inserted or
    already present).
    """
    created: list[str] = []
    now = _utcnow_iso()
    for item in items:
        req_key = (item.get("req_key") or "").strip()
        if not req_key:
            raise ValueError("requirement requires 'req_key'")
        kind = (item.get("kind") or "").strip()
        if kind not in _VALID_REQ_KINDS:
            raise ValueError(
                f"requirement {req_key!r} kind={kind!r} not in "
                f"{sorted(_VALID_REQ_KINDS)}"
            )
        conn.execute(
            "INSERT INTO requirements(workspace, req_key, kind, title, "
            "body, source_path, source_line, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(workspace, req_key) DO UPDATE SET "
            "title = excluded.title, body = excluded.body, "
            "source_path = excluded.source_path, "
            "source_line = excluded.source_line",
            (
                workspace, req_key, kind,
                item.get("title"), item.get("body"),
                item.get("source_path"), item.get("source_line"),
                now,
            ),
        )
        created.append(req_key)
    conn.commit()
    return created


def list_requirements(
    conn: sqlite3.Connection,
    workspace: str,
    *,
    kind: Optional[str] = None,
) -> list[dict[str, Any]]:
    """All requirements for ``workspace``, optionally filtered by ``kind``."""
    if kind is not None:
        rows = conn.execute(
            f"SELECT {_REQUIREMENT_COLS} FROM requirements "
            "WHERE workspace = ? AND kind = ? ORDER BY id",
            (workspace, kind),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_REQUIREMENT_COLS} FROM requirements "
            "WHERE workspace = ? ORDER BY id",
            (workspace,),
        ).fetchall()
    return [_row_to_requirement(r) for r in rows]


def get_requirement_by_key(
    conn: sqlite3.Connection, workspace: str, req_key: str,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        f"SELECT {_REQUIREMENT_COLS} FROM requirements "
        "WHERE workspace = ? AND req_key = ?",
        (workspace, req_key),
    ).fetchone()
    return _row_to_requirement(row) if row else None


def ensure_requirement(
    conn: sqlite3.Connection,
    workspace: str,
    req_key: str,
    *,
    kind: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    source_path: Optional[str] = None,
    source_line: Optional[int] = None,
) -> int:
    """Return the requirement_id for ``req_key`` in ``workspace``,
    creating it lazily if absent. Used by the CR-bridge to seed a
    synthetic ``CR-N`` requirement the first time a CR is ingested,
    so bridged stories can satisfy the same uniform link contract as
    spec-derived stories.
    """
    existing = get_requirement_by_key(conn, workspace, req_key)
    if existing is not None:
        return int(existing["id"])
    create_requirements(
        conn, workspace,
        [{
            "req_key": req_key, "kind": kind,
            "title": title, "body": body,
            "source_path": source_path, "source_line": source_line,
        }],
    )
    again = get_requirement_by_key(conn, workspace, req_key)
    if again is None:
        raise RuntimeError(
            f"ensure_requirement: failed to materialise {req_key!r}"
        )
    return int(again["id"])


def link_story_to_requirements(
    conn: sqlite3.Connection,
    workspace: str,
    story_id: int,
    req_keys: Iterable[str],
) -> int:
    """Insert ``story_satisfies_req`` edges for each (story, req_key) pair.

    All keys must already exist in ``requirements`` for ``workspace``;
    unknown keys raise ``ValueError`` BEFORE any row is inserted so
    the contract is all-or-nothing (no partial state on failure).
    Idempotent on duplicate edges (composite PK). Returns the count
    of edges actually inserted (excludes pre-existing).
    """
    keys = [k for k in req_keys if k]
    if not keys:
        return 0
    placeholders = ",".join(["?"] * len(keys))
    rows = conn.execute(
        f"SELECT req_key, id FROM requirements WHERE workspace = ? "
        f"AND req_key IN ({placeholders})",
        (workspace, *keys),
    ).fetchall()
    found = {r[0]: r[1] for r in rows}
    missing = [k for k in keys if k not in found]
    if missing:
        raise ValueError(
            f"link_story_to_requirements: unknown req_key(s) for "
            f"workspace={workspace!r}: {sorted(set(missing))}"
        )
    inserted = 0
    for key in keys:
        cur = conn.execute(
            "INSERT OR IGNORE INTO story_satisfies_req"
            "(story_id, requirement_id) VALUES(?, ?)",
            (int(story_id), int(found[key])),
        )
        inserted += cur.rowcount or 0
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Acceptance-criteria CRUD (v5)
# ---------------------------------------------------------------------------

def _row_to_ac(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "story_id": row[1],
        "ac_key": row[2],
        "text": row[3],
        "ordinal": row[4],
    }


_AC_COLS = "id, story_id, ac_key, text, ordinal"


def create_acceptance_criteria(
    conn: sqlite3.Connection,
    workspace: str,
    story_id: int,
    items: Iterable[dict[str, Any]],
) -> list[str]:
    """Insert AC rows for a story. Each item: ``ac_key`` (e.g.
    ``"STORY-3.AC-2"``), ``text``, ``ordinal``. UPSERTs on
    (story_id, ac_key) so re-decomposition rewrites the text without
    a UNIQUE violation; the row id is preserved across upserts so
    existing ``test_verifies_ac`` edges survive.
    """
    created: list[str] = []
    for item in items:
        ac_key = (item.get("ac_key") or "").strip()
        if not ac_key:
            raise ValueError("acceptance_criterion requires 'ac_key'")
        text = item.get("text")
        if not text:
            raise ValueError(
                f"acceptance_criterion {ac_key!r} requires non-empty 'text'"
            )
        ordinal = item.get("ordinal")
        if ordinal is None:
            raise ValueError(
                f"acceptance_criterion {ac_key!r} requires 'ordinal'"
            )
        conn.execute(
            "INSERT INTO acceptance_criteria"
            "(workspace, story_id, ac_key, text, ordinal) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(story_id, ac_key) DO UPDATE SET "
            "text = excluded.text, ordinal = excluded.ordinal",
            (workspace, int(story_id), ac_key, text, int(ordinal)),
        )
        created.append(ac_key)
    conn.commit()
    return created


def list_acceptance_criteria(
    conn: sqlite3.Connection, workspace: str, story_id: int,
) -> list[dict[str, Any]]:
    """All AC rows for one story, ordered by ordinal."""
    rows = conn.execute(
        f"SELECT {_AC_COLS} FROM acceptance_criteria "
        "WHERE workspace = ? AND story_id = ? ORDER BY ordinal",
        (workspace, int(story_id)),
    ).fetchall()
    return [_row_to_ac(r) for r in rows]


def get_ac_by_key(
    conn: sqlite3.Connection, workspace: str, ac_key: str,
) -> Optional[dict[str, Any]]:
    """Resolve an ``ac_key`` (e.g. ``STORY-3.AC-2``) to its full row.
    Returns None if no AC matches. Used by the test-gen marker parser
    to convert ``@verifies: STORY-3.AC-2`` strings into FK ids before
    inserting link rows.
    """
    row = conn.execute(
        f"SELECT {_AC_COLS} FROM acceptance_criteria "
        "WHERE workspace = ? AND ac_key = ?",
        (workspace, ac_key),
    ).fetchone()
    return _row_to_ac(row) if row else None


def link_test_to_ac(
    conn: sqlite3.Connection,
    workspace: str,
    test_path: str,
    ac_id: int,
    test_function_name: str = "",
) -> bool:
    """Record that ``test_path`` (optionally a specific test function)
    verifies ``ac_id``. Idempotent on the composite PK. Returns True
    when a new row was inserted, False on duplicate.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO test_verifies_ac"
        "(workspace, test_path, test_function_name, ac_id) "
        "VALUES(?, ?, ?, ?)",
        (workspace, test_path, test_function_name, int(ac_id)),
    )
    conn.commit()
    return bool(cur.rowcount)


# ---------------------------------------------------------------------------
# Traceability audit queries (v5) — called by harness/traceability.py
# ---------------------------------------------------------------------------

def requirements_without_satisfying_story(
    conn: sqlite3.Connection, workspace: str,
) -> list[dict[str, Any]]:
    """Requirements that no story satisfies — the untraced-FR gap set.
    Used by the SQL audit replacing the old text-grep at
    ``harness/traceability.py``.
    """
    rows = conn.execute(
        f"SELECT {_REQUIREMENT_COLS} FROM requirements r "
        "WHERE r.workspace = ? AND NOT EXISTS ("
        "  SELECT 1 FROM story_satisfies_req ssr "
        "  WHERE ssr.requirement_id = r.id"
        ") ORDER BY r.id",
        (workspace,),
    ).fetchall()
    return [_row_to_requirement(r) for r in rows]


def acs_without_verifying_test(
    conn: sqlite3.Connection, workspace: str,
) -> list[dict[str, Any]]:
    """Acceptance criteria that no test marker covers — the untested-AC
    gap set. Each row carries the parent ``story_key`` so the audit
    report can group by story without an extra join.
    """
    rows = conn.execute(
        "SELECT ac.id, ac.story_id, ac.ac_key, ac.text, ac.ordinal, "
        "s.story_key, s.title "
        "FROM acceptance_criteria ac "
        "JOIN stories s ON s.id = ac.story_id "
        "WHERE ac.workspace = ? AND NOT EXISTS ("
        "  SELECT 1 FROM test_verifies_ac tva WHERE tva.ac_id = ac.id"
        ") ORDER BY s.id, ac.ordinal",
        (workspace,),
    ).fetchall()
    return [
        {
            "id": r[0], "story_id": r[1], "ac_key": r[2],
            "text": r[3], "ordinal": r[4],
            "story_key": r[5], "story_title": r[6],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Story CRUD
# ---------------------------------------------------------------------------

def _next_story_key(conn: sqlite3.Connection, workspace: str) -> str:
    """Allocate the next ``STORY-N`` key within ``workspace``.

    Keys are unique WITHIN a workspace; two workspaces can each have
    a ``STORY-1`` simultaneously — that's why every join carries the
    workspace filter.
    """
    row = conn.execute(
        "SELECT story_key FROM stories WHERE workspace = ? "
        "ORDER BY id DESC LIMIT 1",
        (workspace,),
    ).fetchone()
    if row is None:
        return "STORY-1"
    last = row[0]
    try:
        n = int(last.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        n = conn.execute(
            "SELECT COUNT(*) FROM stories WHERE workspace = ?", (workspace,),
        ).fetchone()[0]
    return f"STORY-{n + 1}"


def create_stories(
    conn: sqlite3.Connection,
    workspace: str,
    items: Iterable[dict[str, Any]],
    *,
    build_kind: str = BUILD_KIND_GREENFIELD,
    cr_ids: Optional[Iterable[int]] = None,
) -> list[str]:
    """Insert decomposition-LLM output. Returns assigned story_keys in order.

    Each item supports: ``title`` (required), ``feature`` (required —
    feature_key ref to a row in the ``features`` table; the feature must
    already exist via ``create_features`` or ``ensure_feature``),
    ``description``, ``acceptance_criteria`` (list[str]), ``depends_on``
    (list[story_key]), ``scope_files`` (list[str]), ``external_ref``.
    Story keys are assigned sequentially within ``workspace`` — caller
    does not specify them.

    ``build_kind`` is ``greenfield`` (default — initial build) or ``cr``
    (a change-request increment on top of an existing build).
    ``cr_ids`` lists the integer CR ids ingested in the run that
    created these stories; ignored when ``build_kind='greenfield'``.
    """
    build_kind = _validate_build_kind(build_kind)
    if build_kind == BUILD_KIND_GREENFIELD:
        cr_ids_json = None
    else:
        cr_ids_json = _serialise_cr_ids(cr_ids)
    created: list[str] = []
    now = _utcnow_iso()
    feature_id_cache: dict[str, int] = {}
    for item in items:
        if not item.get("title"):
            raise ValueError("story requires 'title'")
        feature_key = (item.get("feature") or "").strip()
        if not feature_key:
            raise ValueError(
                f"story {item.get('title')!r} requires a 'feature' key — "
                "v4 schema makes feature mandatory; create the feature "
                "first via create_features() or ensure_feature()."
            )
        if feature_key not in feature_id_cache:
            row = get_feature_by_key(conn, workspace, feature_key)
            if row is None:
                raise ValueError(
                    f"story {item.get('title')!r} references feature "
                    f"{feature_key!r} which has not been created. Call "
                    "create_features() with this feature_key first."
                )
            feature_id_cache[feature_key] = int(row["id"])
        feature_id = feature_id_cache[feature_key]
        key = _next_story_key(conn, workspace)
        cur = conn.execute(
            """
            INSERT INTO stories(
                workspace, story_key, feature_id, title, description,
                depends_on, scope_files,
                status, external_ref, build_kind, cr_ids, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)
            """,
            (
                workspace,
                key,
                feature_id,
                item["title"],
                item.get("description"),
                json.dumps(list(item.get("depends_on") or [])),
                json.dumps(list(item.get("scope_files") or [])),
                item.get("external_ref"),
                build_kind,
                cr_ids_json,
                now,
            ),
        )
        story_id = cur.lastrowid
        # AC moved to its own side table in v5. We accept the same
        # ``acceptance_criteria: list[str]`` input shape callers used
        # under v4 — each string is materialised as one row, with
        # ``ac_key`` synthesised from the story key + ordinal so the
        # test-gen ``@verifies`` marker contract has a stable PK to
        # point at.
        ac_strings = list(item.get("acceptance_criteria") or [])
        if ac_strings and story_id is not None:
            create_acceptance_criteria(
                conn, workspace, int(story_id),
                [
                    {
                        "ac_key": f"{key}.AC-{i + 1}",
                        "text": text,
                        "ordinal": i + 1,
                    }
                    for i, text in enumerate(ac_strings)
                ],
            )
        created.append(key)
    conn.commit()
    return created


def _row_to_story(
    row: sqlite3.Row | tuple,
    *,
    acceptance_criteria: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Hydrate a stories row joined with features on feature_id.

    The SELECT must use ``_STORY_COLS`` (which joins ``features`` via
    ``stories.feature_id``) so columns line up. ``feature_key`` and
    ``feature_name`` may be None for CR-bridge stories that pre-date
    the synthetic feature seed (defensive — current code never inserts
    a story with NULL feature_id, but the FK uses ON DELETE SET NULL).

    v5: ``acceptance_criteria`` is no longer a JSON column on the row
    — it lives in the ``acceptance_criteria`` side table. The dict's
    ``"acceptance_criteria"`` key is preserved (callers of
    ``list_stories`` / ``get_story`` depend on the shape) but is
    populated from the optional ``acceptance_criteria`` kwarg. The
    public hydrators backfill that kwarg with a follow-up batch query
    so the per-row default of ``[]`` only surfaces in unit tests that
    call ``_row_to_story`` directly.
    """
    return {
        "id": row[0],
        "story_key": row[1],
        "feature_id": row[2],
        "feature_key": row[3],
        "feature_name": row[4],
        "title": row[5],
        "description": row[6],
        "acceptance_criteria": list(acceptance_criteria or []),
        "depends_on": json.loads(row[7] or "[]"),
        "scope_files": json.loads(row[8] or "[]"),
        "status": row[9],
        "external_ref": row[10],
        "build_kind": row[11] or BUILD_KIND_GREENFIELD,
        "cr_ids": json.loads(row[12]) if row[12] else [],
        "created_at": row[13],
        "started_at": row[14],
        "completed_at": row[15],
    }


_STORY_COLS = (
    "s.id, s.story_key, s.feature_id, f.feature_key, f.name, "
    "s.title, s.description, "
    "s.depends_on, s.scope_files, s.status, "
    "s.external_ref, s.build_kind, s.cr_ids, "
    "s.created_at, s.started_at, s.completed_at"
)


def _ac_strings_by_story_id(
    conn: sqlite3.Connection,
    workspace: str,
    story_ids: Iterable[int],
) -> dict[int, list[str]]:
    """One-shot lookup: ``{story_id: [ac_text, ...]}`` ordered by ordinal.

    Avoids N+1 reads when ``list_stories`` hydrates a batch of rows.
    Empty input → empty dict, no SQL issued.
    """
    ids = [int(s) for s in story_ids]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT story_id, text FROM acceptance_criteria "
        f"WHERE workspace = ? AND story_id IN ({placeholders}) "
        f"ORDER BY story_id, ordinal",
        (workspace, *ids),
    ).fetchall()
    out: dict[int, list[str]] = {sid: [] for sid in ids}
    for sid, text in rows:
        out[sid].append(text)
    return out
_STORY_FROM = "stories s LEFT JOIN features f ON s.feature_id = f.id"


def _validate_build_kind(build_kind: str) -> str:
    if build_kind not in _VALID_BUILD_KINDS:
        raise ValueError(
            f"build_kind={build_kind!r} not in {sorted(_VALID_BUILD_KINDS)}"
        )
    return build_kind


def _serialise_cr_ids(cr_ids: Optional[Iterable[int]]) -> Optional[str]:
    if cr_ids is None:
        return None
    coerced = [int(x) for x in cr_ids]
    return json.dumps(coerced) if coerced else None


def list_stories(
    conn: sqlite3.Connection,
    workspace: str,
    *,
    status: Optional[str] = None,
    feature_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return stories for ``workspace`` joined with their feature rows.

    Optionally filter by ``status`` (e.g. ``"planned"``, ``"done"``)
    and/or ``feature_id`` (constrain to a single feature — used by the
    batch planner to pick stories from one feature at a time).
    """
    where = ["s.workspace = ?"]
    params: list[Any] = [workspace]
    if status is not None:
        where.append("s.status = ?")
        params.append(status)
    if feature_id is not None:
        where.append("s.feature_id = ?")
        params.append(int(feature_id))
    sql = (
        f"SELECT {_STORY_COLS} FROM {_STORY_FROM} "
        f"WHERE {' AND '.join(where)} ORDER BY s.id"
    )
    rows = conn.execute(sql, tuple(params)).fetchall()
    ac_map = _ac_strings_by_story_id(conn, workspace, [r[0] for r in rows])
    return [
        _row_to_story(r, acceptance_criteria=ac_map.get(r[0], []))
        for r in rows
    ]


def get_story(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        f"SELECT {_STORY_COLS} FROM {_STORY_FROM} "
        "WHERE s.workspace = ? AND s.story_key = ?",
        (workspace, story_key),
    ).fetchone()
    if row is None:
        return None
    ac_map = _ac_strings_by_story_id(conn, workspace, [row[0]])
    return _row_to_story(row, acceptance_criteria=ac_map.get(row[0], []))


def get_planned_stories(
    conn: sqlite3.Connection, workspace: str,
) -> list[dict[str, Any]]:
    """Stories ready to be picked. Honors depends_on — a story whose
    deps are not all 'done' is not returned even if status='planned'.

    Also returns ``reopened`` rows so ``story_reopen_node``'s output
    actually gets re-planned by the batch planner. ``reopened`` is
    a distinct status from ``planned`` so the operator can see (in
    STORIES.md and via ``teane status``) which stories were flipped
    back by a spec-drift verdict vs. which never shipped at all.
    """
    rows = list_stories(conn, workspace, status="planned")
    rows.extend(list_stories(conn, workspace, status="reopened"))
    done_keys = {
        s["story_key"] for s in list_stories(conn, workspace, status="done")
    }
    ready: list[dict[str, Any]] = []
    for s in rows:
        deps = s["depends_on"]
        if all(d in done_keys for d in deps):
            ready.append(s)
    return ready


def mark_in_progress(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> int:
    """Transition a planned/in-progress story into the in_progress state.

    Resumed sessions hit this on a story already in_progress — narrowing
    the WHERE to status='planned' would silently no-op and leak a stale
    started_at. Allow both states so the started_at stamp is refreshed
    deterministically. Returns affected rowcount so callers can detect
    a vanished story_key (rowcount=0).
    """
    cur = conn.execute(
        "UPDATE stories SET status = 'in_progress', started_at = ? "
        "WHERE workspace = ? AND story_key = ? "
        "AND status IN ('planned', 'in_progress', 'reopened')",
        (_utcnow_iso(), workspace, story_key),
    )
    conn.commit()
    return cur.rowcount or 0


def mark_done(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> None:
    conn.execute(
        "UPDATE stories SET status = 'done', completed_at = ? "
        "WHERE workspace = ? AND story_key = ?",
        (_utcnow_iso(), workspace, story_key),
    )
    conn.commit()


def mark_reopened(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> int:
    """Flip a DONE story back to ``reopened`` so the story loop picks it up
    again. Used by ``story_reopen_node`` when a patch-flow spec revision
    invalidates a previously-shipped story's acceptance criteria.

    Returns affected rowcount (0 = vanished story or already non-DONE).
    """
    cur = conn.execute(
        "UPDATE stories SET status = 'reopened', completed_at = NULL "
        "WHERE workspace = ? AND story_key = ? AND status = 'done'",
        (workspace, story_key),
    )
    conn.commit()
    return cur.rowcount or 0


def mark_blocked(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> None:
    conn.execute(
        "UPDATE stories SET status = 'blocked' "
        "WHERE workspace = ? AND story_key = ?",
        (workspace, story_key),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def start_batch(
    conn: sqlite3.Connection,
    workspace: str,
    session_id: str,
    story_keys: list[str],
    *,
    build_kind: str = BUILD_KIND_GREENFIELD,
    cr_ids: Optional[Iterable[int]] = None,
    feature_id: Optional[int] = None,
) -> int:
    """Open a new batch row and seed its ``batch_stories`` membership.

    ``build_kind`` (``greenfield`` or ``cr``) and ``cr_ids`` (JSON list
    of int CR ids ingested in this run) tag the batch as an
    incremental change-request layer or as part of the initial build.

    ``feature_id`` ties the batch to its owning feature row. A batch
    never spans features (enforced by ``batch_planner_node``), so the
    column is non-NULL in normal operation; left optional only for
    test fixtures that bypass the planner.
    """
    build_kind = _validate_build_kind(build_kind)
    cr_ids_json = (
        None if build_kind == BUILD_KIND_GREENFIELD
        else _serialise_cr_ids(cr_ids)
    )
    cur = conn.execute(
        "INSERT INTO batches"
        "(workspace, session_id, feature_id, started_at, status, "
        "build_kind, cr_ids) "
        "VALUES(?, ?, ?, ?, 'running', ?, ?)",
        (
            workspace, session_id,
            None if feature_id is None else int(feature_id),
            _utcnow_iso(), build_kind, cr_ids_json,
        ),
    )
    batch_id = cur.lastrowid
    if batch_id is None:
        raise RuntimeError("failed to allocate batch id")
    for seq, key in enumerate(story_keys, start=1):
        story_row = conn.execute(
            "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
            (workspace, key),
        ).fetchone()
        if story_row is None:
            continue
        conn.execute(
            "INSERT INTO batch_stories(batch_id, story_id, sequence) "
            "VALUES(?, ?, ?)",
            (batch_id, story_row[0], seq),
        )
    conn.commit()
    return batch_id


def list_stories_for_cr(
    conn: sqlite3.Connection, workspace: str, cr_id: int,
) -> list[dict[str, Any]]:
    """All stories in ``workspace`` whose ``cr_ids`` JSON list contains
    ``cr_id``. Used by traceability views ("show me what CR-2 added").

    Implemented in Python rather than via JSON SQL because SQLite's
    json1 extension isn't guaranteed available everywhere and the row
    counts here are small (tens to hundreds per workspace).
    """
    out: list[dict[str, Any]] = []
    target = int(cr_id)
    for s in list_stories(conn, workspace):
        if target in (s.get("cr_ids") or []):
            out.append(s)
    return out


def list_batches_for_cr(
    conn: sqlite3.Connection, workspace: str, cr_id: int,
) -> list[dict[str, Any]]:
    """All batches in ``workspace`` whose ``cr_ids`` JSON list contains
    ``cr_id``. Same in-Python filter as ``list_stories_for_cr`` for
    the same json1 availability reason."""
    target = int(cr_id)
    rows = conn.execute(
        "SELECT id, session_id, started_at, completed_at, status, "
        "committed_sha, build_kind, cr_ids "
        "FROM batches WHERE workspace = ? ORDER BY id",
        (workspace,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        cr_list = json.loads(r[7]) if r[7] else []
        if target not in cr_list:
            continue
        out.append({
            "id": r[0],
            "session_id": r[1],
            "started_at": r[2],
            "completed_at": r[3],
            "status": r[4],
            "committed_sha": r[5],
            "build_kind": r[6],
            "cr_ids": cr_list,
        })
    return out


def complete_batch(
    conn: sqlite3.Connection, batch_id: int, status: str = "complete",
) -> None:
    conn.execute(
        "UPDATE batches SET status = ?, completed_at = ? WHERE id = ?",
        (status, _utcnow_iso(), batch_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Defects, test runs, file links, commits
# ---------------------------------------------------------------------------

def record_defect(
    conn: sqlite3.Connection,
    *,
    workspace: str,
    story_key: Optional[str],
    session_id: str,
    severity: str,
    summary: str,
    diagnostic: Optional[Any] = None,
) -> int:
    story_id = None
    if story_key:
        row = conn.execute(
            "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
            (workspace, story_key),
        ).fetchone()
        story_id = row[0] if row else None
    cur = conn.execute(
        """
        INSERT INTO defects(workspace, story_id, session_id, severity,
                            summary, diagnostic_json, status, created_at)
        VALUES(?, ?, ?, ?, ?, ?, 'open', ?)
        """,
        (
            workspace,
            story_id,
            session_id,
            severity,
            summary,
            json.dumps(diagnostic) if diagnostic is not None else None,
            _utcnow_iso(),
        ),
    )
    conn.commit()
    if cur.lastrowid is None:
        raise RuntimeError("failed to allocate defect id")
    return cur.lastrowid


def resolve_defects_for_story(
    conn: sqlite3.Connection, workspace: str, story_key: str,
) -> int:
    row = conn.execute(
        "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
        (workspace, story_key),
    ).fetchone()
    if row is None:
        return 0
    cur = conn.execute(
        "UPDATE defects SET status = 'resolved', resolved_at = ? "
        "WHERE story_id = ? AND status = 'open'",
        (_utcnow_iso(), row[0]),
    )
    conn.commit()
    return cur.rowcount or 0


def record_test_run(
    conn: sqlite3.Connection,
    *,
    workspace: str,
    story_key: Optional[str],
    session_id: str,
    phase: str,
    exit_code: int,
    passed: int = 0,
    failed: int = 0,
    errors: int = 0,
    stdout_tail: Optional[str] = None,
) -> None:
    story_id = None
    if story_key:
        row = conn.execute(
            "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
            (workspace, story_key),
        ).fetchone()
        story_id = row[0] if row else None
    conn.execute(
        """
        INSERT INTO test_runs(workspace, story_id, session_id, phase,
                              exit_code, passed, failed, errors,
                              stdout_tail, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (workspace, story_id, session_id, phase, exit_code, passed,
         failed, errors, stdout_tail, _utcnow_iso()),
    )
    conn.commit()


def link_file(
    conn: sqlite3.Connection,
    workspace: str,
    story_key: str,
    path: str,
    kind: str = "code",
    batch_id: Optional[int] = None,
) -> None:
    """Record that ``story_key`` (in ``workspace``) touched ``path``.

    When ``batch_id`` is provided, also stamps which batch did the
    most recent touch — that's what per-batch repair attribution
    joins against to map a compile error back to the owning story.
    A subsequent call with a different (non-None) ``batch_id`` updates
    the stamp; ``None`` is a no-op against the stored value.
    """
    row = conn.execute(
        "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
        (workspace, story_key),
    ).fetchone()
    if row is None:
        return
    if batch_id is None:
        conn.execute(
            "INSERT OR IGNORE INTO file_links(workspace, story_id, path, kind) "
            "VALUES(?, ?, ?, ?)",
            (workspace, row[0], path, kind),
        )
    else:
        conn.execute(
            "INSERT INTO file_links(workspace, story_id, path, kind, batch_id) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(story_id, path, kind) DO UPDATE SET "
            "batch_id = excluded.batch_id, workspace = excluded.workspace",
            (workspace, row[0], path, kind, batch_id),
        )
    conn.commit()


def files_for_batch(
    conn: sqlite3.Connection, batch_id: int,
) -> list[tuple[str, str, str]]:
    """Return ``[(story_key, path, kind), ...]`` for all files stamped
    with this batch_id. Used by code review / test gen / security scan
    to scope their inputs to "what this batch touched"."""
    rows = conn.execute(
        "SELECT stories.story_key, file_links.path, file_links.kind "
        "FROM file_links "
        "JOIN stories ON stories.id = file_links.story_id "
        "WHERE file_links.batch_id = ? "
        "ORDER BY stories.story_key, file_links.path",
        (batch_id,),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def set_batch_committed_sha(
    conn: sqlite3.Connection, batch_id: int, sha: str,
) -> None:
    """Record the git SHA of the BATCH-N commit. No-op when batch_id
    is unknown."""
    conn.execute(
        "UPDATE batches SET committed_sha = ? WHERE id = ?",
        (sha, batch_id),
    )
    conn.commit()


def record_commit(
    conn: sqlite3.Connection,
    *,
    workspace: str,
    sha: str,
    story_key: Optional[str],
    session_id: str,
    message: str,
) -> None:
    story_id = None
    if story_key:
        row = conn.execute(
            "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
            (workspace, story_key),
        ).fetchone()
        story_id = row[0] if row else None
    conn.execute(
        "INSERT OR REPLACE INTO commits"
        "(sha, workspace, story_id, session_id, message, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (sha, workspace, story_id, session_id, message, _utcnow_iso()),
    )
    conn.commit()


def seal_batch_atomically(
    conn: sqlite3.Connection,
    *,
    workspace: str,
    batch_id: int,
    stories_in_batch: list[tuple[str, str, str]],
    blocked_count: int,
    committed_sha: Optional[str],
    batch_commit_message: Optional[str],
    session_id: str,
) -> list[str]:
    """Seal a batch in a single SQLite transaction.

    Crash-mid-commit safety: previously batch_commit_node called
    mark_done / resolve_defects / complete_batch / set_batch_committed_sha
    / record_commit in sequence, each with its own ``conn.commit()``. A
    crash between calls left the batch row ``running`` with some stories
    marked ``done`` — on resume, ``batch_planner_node`` saw inconsistent
    state. This helper does all those mutations in one transaction so the
    seal either lands fully or rolls back.

    The git commit (``_commit_for_batch``) is intentionally OUTSIDE this
    helper — it can't be rolled back, so it happens first; the resulting
    SHA is then stamped here together with everything else.

    Returns the list of ``story_key`` values that this call transitioned
    from a non-terminal status to ``done``.
    """
    done_keys: list[str] = []
    now = _utcnow_iso()
    batch_status = "complete" if blocked_count == 0 else "complete_with_blocks"
    try:
        conn.execute("BEGIN IMMEDIATE")
        for key, _title, status in stories_in_batch:
            if status in ("done", "blocked"):
                continue
            row = conn.execute(
                "SELECT id FROM stories WHERE workspace = ? AND story_key = ?",
                (workspace, key),
            ).fetchone()
            if row is None:
                continue
            sid = row[0]
            conn.execute(
                "UPDATE stories SET status = 'done', completed_at = ? "
                "WHERE workspace = ? AND story_key = ?",
                (now, workspace, key),
            )
            conn.execute(
                "UPDATE defects SET status = 'resolved', resolved_at = ? "
                "WHERE story_id = ? AND status = 'open'",
                (now, sid),
            )
            done_keys.append(key)

        if committed_sha:
            conn.execute(
                "UPDATE batches SET status = ?, completed_at = ?, "
                "committed_sha = ? WHERE id = ?",
                (batch_status, now, committed_sha, batch_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO commits"
                "(sha, workspace, story_id, session_id, message, created_at) "
                "VALUES(?, ?, NULL, ?, ?, ?)",
                (committed_sha, workspace, session_id,
                 batch_commit_message or "", now),
            )
        else:
            conn.execute(
                "UPDATE batches SET status = ?, completed_at = ? WHERE id = ?",
                (batch_status, now, batch_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return done_keys


# ---------------------------------------------------------------------------
# Markdown view regeneration
# ---------------------------------------------------------------------------

_STATUS_BADGE = {
    "planned": "⏳ planned",
    "in_progress": "🔧 in_progress",
    "done": "✅ done",
    "blocked": "⛔ blocked",
}


def _status_label(status: str) -> str:
    return _STATUS_BADGE.get(status, status)


def _render_stories_md(stories: list[dict[str, Any]]) -> str:
    lines = [
        "# Stories",
        "",
        "_Auto-generated from teane's state.db. Do not edit by hand._",
        "",
    ]
    if not stories:
        lines.append("_No stories yet._")
        lines.append("")
        return "\n".join(lines)

    # Group by feature, preserving first-seen order so the doc reads
    # in the same order decomposition produced (stories are returned
    # in insertion order by list_stories, which inherits feature order
    # since features are created before stories in decomposition_node).
    by_feature: dict[str, list[dict[str, Any]]] = {}
    feature_labels: dict[str, str] = {}
    for s in stories:
        fkey = s.get("feature_key") or "(unassigned)"
        if fkey not in by_feature:
            by_feature[fkey] = []
            feature_labels[fkey] = s.get("feature_name") or fkey
        by_feature[fkey].append(s)

    for fkey, fstories in by_feature.items():
        heading = feature_labels.get(fkey, fkey)
        if heading != fkey:
            lines.append(f"## {heading} ({fkey})")
        else:
            lines.append(f"## {heading}")
        lines.append("")
        lines.append("| Key | Status | Title | Depends on | External |")
        lines.append("| --- | --- | --- | --- | --- |")
        for s in fstories:
            deps = ", ".join(s["depends_on"]) or "—"
            ext = s.get("external_ref") or "—"
            lines.append(
                f"| {s['story_key']} | {_status_label(s['status'])} | "
                f"{s['title']} | {deps} | {ext} |"
            )
        lines.append("")
        for s in fstories:
            lines.append(f"### {s['story_key']} — {s['title']}")
            lines.append("")
            if s.get("description"):
                lines.append(s["description"])
                lines.append("")
            ac = s["acceptance_criteria"]
            if ac:
                lines.append("**Acceptance criteria:**")
                lines.append("")
                for item in ac:
                    lines.append(f"- {item}")
                lines.append("")
    return "\n".join(lines)


def _render_arch_coverage(
    stories: list[dict[str, Any]],
    arch_summary: Optional[dict[str, Any]],
) -> list[str]:
    """Render the "Architecture coverage" section appended to
    ``docs/TRACEABILITY.md``.

    Cross-references each §11 endpoint and component back to the
    story / stories that cite it (via ``rsd_story_ids``), surfacing
    the live story status next to the architectural artifact. A row
    with no matching story is flagged as a **gap** — the
    architecture defined the endpoint / component but no story
    captures the work.

    Returns an empty list (so the caller can ``extend`` without a
    conditional) when:
      - ``arch_summary`` is ``None`` or not a dict
      - the summary has neither endpoints nor components

    The story status carried into the table is the live DB status —
    a row reading ``STORY-3 (in_progress)`` flags incomplete work
    against a resolved arch decision better than the per-story drill-
    down does on its own.
    """
    if not arch_summary or not isinstance(arch_summary, dict):
        return []

    backend = arch_summary.get("backend") or {}
    endpoints = backend.get("endpoints") or []
    frontend = arch_summary.get("frontend") or "none"
    spec = arch_summary.get("frontend_spec")
    if not isinstance(spec, dict):
        legacy = arch_summary.get("frontend")
        spec = legacy if isinstance(legacy, dict) else {}
    components = spec.get("components") or []

    if not endpoints and not components:
        return []

    status_by_key: dict[str, str] = {
        s["story_key"]: s.get("status", "?") for s in stories
    }

    def _stories_cell(ids: list[str]) -> tuple[str, str]:
        """Return (story_cell, status_cell) — '—' when the architecture
        artifact has no story link at all, ``GAP`` when it lists IDs
        that the DB doesn't recognise (story removed / never created)."""
        if not ids:
            return "— (gap)", "—"
        bits: list[str] = []
        statuses: list[str] = []
        for sid in ids:
            if not isinstance(sid, str):
                continue
            if sid in status_by_key:
                bits.append(sid)
                statuses.append(_status_label(status_by_key[sid]))
            else:
                bits.append(f"{sid} (missing)")
                statuses.append("—")
        if not bits:
            return "— (gap)", "—"
        return ", ".join(bits), ", ".join(statuses)

    lines: list[str] = [
        "## Architecture coverage",
        "",
        "_Cross-references `docs/SPEC_ARCHITECTURE.md` §11 against the "
        "stories table. A `gap` row is an arch artifact with no "
        "story implementing it._",
        "",
    ]

    if endpoints:
        lines.extend([
            "### Endpoints",
            "",
            "| EP | Method | Path | Stories | Status |",
            "| --- | --- | --- | --- | --- |",
        ])
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            ids: list[str] = []
            for key in ("rsd_story_ids", "rsd_feature_ids"):
                # Only story IDs map to the stories table; FEAT-N rolls
                # up its constituent stories. We surface STORY-N when
                # present and skip FEAT-N here to keep the cell tight.
                if key == "rsd_story_ids":
                    ids.extend(ep.get(key) or [])
            stories_cell, status_cell = _stories_cell(ids)
            lines.append(
                f"| {ep.get('id', '?')} | {(ep.get('method') or '').upper()} | "
                f"`{ep.get('path', '')}` | {stories_cell} | {status_cell} |"
            )
        lines.append("")

    if components and frontend != "none":
        lines.extend([
            "### Components",
            "",
            "| Component | Path | Stories | Status |",
            "| --- | --- | --- | --- |",
        ])
        for cp in components:
            if not isinstance(cp, dict):
                continue
            ids = list(cp.get("rsd_story_ids") or [])
            stories_cell, status_cell = _stories_cell(ids)
            lines.append(
                f"| {cp.get('name', '?')} | `{cp.get('path', '')}` | "
                f"{stories_cell} | {status_cell} |"
            )
        lines.append("")

    return lines


def _render_requirements_coverage(
    requirements: list[dict[str, Any]],
    stories_by_req_key: dict[str, list[tuple[str, str]]],
) -> list[str]:
    """Render the v5 Requirements coverage table.

    ``stories_by_req_key`` maps each ``req_key`` to a list of
    ``(story_key, status)`` tuples. An empty list flags the
    requirement as **untraced** (gap) — no story satisfies it.

    Returns an empty list when ``requirements`` is empty so legacy
    workspaces with no v5 ingest produce byte-identical TRACEABILITY.md
    output to the previous version (the caller ``extend()``s).
    """
    if not requirements:
        return []
    lines: list[str] = [
        "## Requirements coverage",
        "",
        "_Cross-references ``docs/SPEC_REQUIREMENTS.md`` requirements "
        "against the stories table. A `gap` row is a declared "
        "requirement with no story satisfying it — the planner "
        "missed it or the spec needs revision._",
        "",
        "| Requirement | Kind | Title | Stories | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in requirements:
        title = (r.get("title") or "").strip() or "—"
        pairs = stories_by_req_key.get(r["req_key"], [])
        if not pairs:
            story_cell = "— (gap)"
            status_cell = "—"
        else:
            story_cell = ", ".join(p[0] for p in pairs)
            status_cell = ", ".join(_status_label(p[1]) for p in pairs)
        lines.append(
            f"| `{r['req_key']}` | {r.get('kind', '?')} | {title} | "
            f"{story_cell} | {status_cell} |"
        )
    lines.append("")
    return lines


def _render_ac_coverage(
    acs: list[dict[str, Any]],
    tests_by_ac_key: dict[str, list[str]],
) -> list[str]:
    """Render the v5 Acceptance-criteria coverage table.

    ``tests_by_ac_key`` maps each ``ac_key`` to the list of
    ``test_path`` strings that verify it (from ``test_verifies_ac``).
    Empty list = untested AC (gap).

    Returns an empty list when ``acs`` is empty so legacy workspaces
    produce no AC-coverage section.
    """
    if not acs:
        return []
    lines: list[str] = [
        "## Acceptance-criteria coverage",
        "",
        "_Each acceptance criterion paired with the tests that cite it "
        "via a ``# @verifies: STORY-N.AC-N`` marker. A `gap` row means "
        "no passing test claims to verify the criterion._",
        "",
        "| Story | AC | Text | Tests |",
        "| --- | --- | --- | --- |",
    ]
    for ac in acs:
        text = (ac.get("text") or "").strip()
        # Markdown tables don't render embedded pipes well; soften
        # them to a unicode bar so the cell stays visually distinct.
        text_cell = text.replace("|", "│")[:160] or "—"
        tests = tests_by_ac_key.get(ac["ac_key"], [])
        if not tests:
            tests_cell = "— (gap)"
        else:
            tests_cell = ", ".join(f"`{t}`" for t in tests)
        lines.append(
            f"| {ac['story_key']} | `{ac['ac_key']}` | {text_cell} | "
            f"{tests_cell} |"
        )
    lines.append("")
    return lines


def _render_traceability_md(
    stories: list[dict[str, Any]],
    files_by_story: dict[str, list[tuple[str, str]]],
    defects_by_story: dict[str, list[dict[str, Any]]],
    commits_by_story: dict[str, list[dict[str, Any]]],
    *,
    arch_summary: Optional[dict[str, Any]] = None,
    requirements: Optional[list[dict[str, Any]]] = None,
    stories_by_req_key: Optional[dict[str, list[tuple[str, str]]]] = None,
    acs: Optional[list[dict[str, Any]]] = None,
    tests_by_ac_key: Optional[dict[str, list[str]]] = None,
) -> str:
    lines = [
        "# Traceability matrix",
        "",
        "_Auto-generated from teane's state.db. Do not edit by hand._",
        "",
        "| Story | Status | Files | Tests | Defects | Commits |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if not stories:
        lines.append("| _none_ | | | | | |")
        return "\n".join(lines) + "\n"
    for s in stories:
        files = files_by_story.get(s["story_key"], [])
        code = sum(1 for _, k in files if k == "code")
        tests = sum(1 for _, k in files if k == "test")
        defects = defects_by_story.get(s["story_key"], [])
        open_defects = sum(1 for d in defects if d["status"] == "open")
        commits = commits_by_story.get(s["story_key"], [])
        commit_str = ", ".join(c["sha"][:7] for c in commits) or "—"
        defect_str = f"{open_defects} open / {len(defects)}" if defects else "—"
        lines.append(
            f"| {s['story_key']} | {_status_label(s['status'])} | "
            f"{code} code | {tests} tests | {defect_str} | {commit_str} |"
        )
    lines.append("")
    for s in stories:
        files = files_by_story.get(s["story_key"], [])
        defects = defects_by_story.get(s["story_key"], [])
        commits = commits_by_story.get(s["story_key"], [])
        if not (files or defects or commits):
            continue
        lines.append(f"## {s['story_key']} — {s['title']}")
        lines.append("")
        if files:
            lines.append("**Files:**")
            lines.append("")
            for path, kind in sorted(files):
                lines.append(f"- `{path}` _({kind})_")
            lines.append("")
        if defects:
            lines.append("**Defects:**")
            lines.append("")
            for d in defects:
                lines.append(
                    f"- [{d['status']}] {d['severity']}: {d['summary']}"
                )
            lines.append("")
        if commits:
            lines.append("**Commits:**")
            lines.append("")
            for c in commits:
                lines.append(f"- `{c['sha'][:7]}` — {c['message']}")
            lines.append("")
    # v5 Requirements + AC coverage sections — appended only when the
    # v5 ingest populated these collections (empty list inputs render
    # nothing, preserving byte-identical TRACEABILITY.md output for
    # legacy workspaces that never ran a v5 decomposition pass).
    lines.extend(_render_requirements_coverage(
        requirements or [], stories_by_req_key or {},
    ))
    lines.extend(_render_ac_coverage(
        acs or [], tests_by_ac_key or {},
    ))
    # Architecture coverage matrix, appended only when SPEC_ARCHITECTURE.md's
    # §11 summary is available. Bytes-identical TRACEABILITY.md output when
    # no summary is passed in — keeps backward compatibility with projects
    # that don't ship a machine-readable arch doc.
    lines.extend(_render_arch_coverage(stories, arch_summary))
    return "\n".join(lines)


def _collect_view_inputs(
    conn: sqlite3.Connection, workspace: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[tuple[str, str]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, list[tuple[str, str]]],
    list[dict[str, Any]],
    dict[str, list[str]],
]:
    """Collect everything the markdown renderers need in one pass.

    Returns the legacy 4-tuple (stories, files, defects, commits)
    extended with v5 traceability inputs:

      - ``requirements`` — full list for ``_render_requirements_coverage``.
      - ``stories_by_req_key`` — req_key → [(story_key, status), ...]
        from the ``story_satisfies_req`` join.
      - ``acs`` — flat AC list joined with story_key for ``_render_ac_coverage``.
      - ``tests_by_ac_key`` — ac_key → [test_path, ...] from
        ``test_verifies_ac``.

    Empty v5 inputs (no requirements/ACs/links rows) render to
    empty sections, so the markdown stays backward-compatible with
    pre-v5 workspaces.
    """
    stories = list_stories(conn, workspace)
    by_key = {s["story_key"]: s for s in stories}
    # O(1) story_id → story_key lookup so the FK resolution below is
    # linear in row count rather than n×m for n rows × m stories.
    id_to_key: dict[int, str] = {s["id"]: s["story_key"] for s in stories}

    files_by_story: dict[str, list[tuple[str, str]]] = {k: [] for k in by_key}
    for sid, path, kind in conn.execute(
        "SELECT story_id, path, kind FROM file_links WHERE workspace = ?",
        (workspace,),
    ):
        key = id_to_key.get(sid)
        if key is not None:
            files_by_story[key].append((path, kind))

    defects_by_story: dict[str, list[dict[str, Any]]] = {k: [] for k in by_key}
    for sid, severity, summary, status in conn.execute(
        "SELECT story_id, severity, summary, status FROM defects "
        "WHERE workspace = ? ORDER BY id",
        (workspace,),
    ):
        key = id_to_key.get(sid)
        if key is not None:
            defects_by_story[key].append(
                {"severity": severity, "summary": summary, "status": status}
            )

    commits_by_story: dict[str, list[dict[str, Any]]] = {k: [] for k in by_key}
    for sid, sha, message in conn.execute(
        "SELECT story_id, sha, message FROM commits "
        "WHERE workspace = ? ORDER BY created_at",
        (workspace,),
    ):
        key = id_to_key.get(sid)
        if key is not None:
            commits_by_story[key].append({"sha": sha, "message": message})

    # v5 traceability inputs ---------------------------------------------------
    requirements = list_requirements(conn, workspace)
    stories_by_req_key: dict[str, list[tuple[str, str]]] = {
        r["req_key"]: [] for r in requirements
    }
    if requirements:
        # LEFT JOIN so requirements with no satisfying story still appear
        # in the dict (with an empty list — rendered as a gap row).
        for req_key, story_key, status in conn.execute(
            "SELECT r.req_key, s.story_key, s.status "
            "FROM requirements r "
            "LEFT JOIN story_satisfies_req ssr ON ssr.requirement_id = r.id "
            "LEFT JOIN stories s ON s.id = ssr.story_id "
            "WHERE r.workspace = ? "
            "ORDER BY r.id, s.id",
            (workspace,),
        ):
            if story_key is not None:
                stories_by_req_key[req_key].append(
                    (story_key, status or "planned"),
                )

    acs: list[dict[str, Any]] = []
    tests_by_ac_key: dict[str, list[str]] = {}
    for ac_key, story_key, text, ordinal, story_id in conn.execute(
        "SELECT ac.ac_key, s.story_key, ac.text, ac.ordinal, ac.story_id "
        "FROM acceptance_criteria ac "
        "JOIN stories s ON s.id = ac.story_id "
        "WHERE ac.workspace = ? "
        "ORDER BY s.id, ac.ordinal",
        (workspace,),
    ):
        acs.append({
            "ac_key": ac_key,
            "story_key": story_key,
            "text": text,
            "ordinal": ordinal,
            "story_id": story_id,
        })
        tests_by_ac_key.setdefault(ac_key, [])

    if acs:
        for ac_key, test_path in conn.execute(
            "SELECT ac.ac_key, tva.test_path "
            "FROM test_verifies_ac tva "
            "JOIN acceptance_criteria ac ON ac.id = tva.ac_id "
            "WHERE tva.workspace = ? "
            "ORDER BY ac.story_id, ac.ordinal, tva.test_path",
            (workspace,),
        ):
            tests_by_ac_key.setdefault(ac_key, []).append(test_path)

    return (
        stories, files_by_story, defects_by_story, commits_by_story,
        requirements, stories_by_req_key, acs, tests_by_ac_key,
    )


def regenerate_markdown_views(
    conn: sqlite3.Connection,
    workspace_path: str,
    *,
    arch_summary: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """Rebuild ``docs/STORIES.md`` and ``docs/TRACEABILITY.md`` from the DB.

    Scoped to the calling workspace's rows only — the global state.db
    holds every app's rows, but each workspace's docs only reflect its
    own slice. Returns ``(stories_path, traceability_path)``. The agent
    calls this after every batch — the markdown files are derived
    state, never a write surface for the LLM.

    When ``arch_summary`` is supplied (the parsed §11 jsonc block from
    ``docs/SPEC_ARCHITECTURE.md``), TRACEABILITY.md picks up an
    "Architecture coverage" section that cross-references each
    endpoint and component against the stories table. The kwarg is
    optional and defaults to ``None`` so existing call sites
    (decomposition mid-run regenerations) keep their pre-existing
    output byte-for-byte until a caller explicitly opts in.
    """
    workspace = app_name_for_workspace(workspace_path)
    (
        stories, files_by_story, defects_by_story, commits_by_story,
        requirements, stories_by_req_key, acs, tests_by_ac_key,
    ) = _collect_view_inputs(conn, workspace)
    stories_md = _render_stories_md(stories)
    trace_md = _render_traceability_md(
        stories, files_by_story, defects_by_story, commits_by_story,
        arch_summary=arch_summary,
        requirements=requirements,
        stories_by_req_key=stories_by_req_key,
        acs=acs,
        tests_by_ac_key=tests_by_ac_key,
    )

    docs_dir = os.path.join(os.path.expanduser(workspace_path), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    stories_path = os.path.join(docs_dir, "STORIES.md")
    trace_path = os.path.join(docs_dir, "TRACEABILITY.md")
    with open(stories_path, "w", encoding="utf-8") as fh:
        fh.write(stories_md)
        if not stories_md.endswith("\n"):
            fh.write("\n")
    with open(trace_path, "w", encoding="utf-8") as fh:
        fh.write(trace_md)
        if not trace_md.endswith("\n"):
            fh.write("\n")
    return stories_path, trace_path
