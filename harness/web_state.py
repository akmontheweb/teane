"""Shared in-process state for the interactive dashboard.

This module is the data layer the web UI sits on top of. It owns:

- The **process registry** — the set of ``harness run`` subprocesses
  the dashboard has spawned, keyed by session id. The UI's
  "currently running" view, the cancel button, and the live event
  stream all consult this.
- The **HITL queue** — pending HITL prompts the harness has POSTed
  to the dashboard's ``/hitl/webhook`` endpoint. The dashboard's
  HTTP handler blocks the harness's POST while the UI displays the
  prompt; clicking an option signals the held handler with the
  operator's answer.
- The **web.db SQLite store** — for state that needs to survive a
  dashboard restart: audit log of operator writes, saved run
  presets ("nightly retest"), one-shot scheduled jobs the UI
  enqueued for the schedule daemon, and per-session chat notes
  the operator queued for the next HITL gate.

Nothing in this module knows about HTTP. ``harness/dashboard.py``
is the consumer; tests in ``tests/test_web_state.py`` exercise the
contracts directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_WEB_DB = "~/.harness/web.db"

# Keep a terminated WebProcess in the registry for this long so the
# "currently running" view can still show "exit 0 — completed 30s ago"
# instead of dropping the entry the second the subprocess exits.
_TERMINATED_TTL_SECONDS = 5 * 60


# ---------------------------------------------------------------------------
# 1. SQLite schema (web.db)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (ts DESC);

CREATE TABLE IF NOT EXISTS run_presets (
    name TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    prompt TEXT,
    harness_args TEXT,         -- JSON list
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS web_oneshot_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    fire_at_utc TEXT NOT NULL, -- ISO8601 UTC
    workspace TEXT NOT NULL,
    prompt TEXT,
    harness_args TEXT,         -- JSON list
    created_at TEXT NOT NULL,
    consumed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_oneshot_pending
    ON web_oneshot_jobs (consumed_at, fire_at_utc);

CREATE TABLE IF NOT EXISTS chat_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    note TEXT NOT NULL,
    consumed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_session_pending
    ON chat_notes (session_id, consumed_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_web_db(path: str = _DEFAULT_WEB_DB) -> sqlite3.Connection:
    """Open (creating + migrating if needed) the web.db SQLite store.

    Caller closes; per-request open/close keeps the contention story
    simple at the cost of a few extra fopens per page load — fine for a
    single-operator dashboard.
    """
    expanded = os.path.expanduser(path)
    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(expanded)
    conn.executescript(_SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# 2. Process registry — in-memory, thread-safe
# ---------------------------------------------------------------------------

@dataclass
class WebProcess:
    """A single ``harness run`` subprocess spawned by the dashboard.

    ``pid`` is the PID inside the registry; ``popen`` carries the
    asyncio / subprocess handle that the cancel handler signals.
    ``log_path`` is where the per-session JSONL lands so the SSE
    stream knows what to tail.
    """

    session_id: str
    pid: int
    argv: list[str]
    started_at: float = field(default_factory=time.time)
    log_path: str = ""
    workspace_path: str = ""
    prompt: str = ""
    exit_code: Optional[int] = None
    terminated_at: Optional[float] = None
    popen: Any = None  # subprocess.Popen or asyncio.subprocess.Process

    @property
    def is_running(self) -> bool:
        return self.exit_code is None

    def to_view(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "pid": self.pid,
            "argv": list(self.argv),
            "started_at": self.started_at,
            "log_path": self.log_path,
            "workspace_path": self.workspace_path,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
            "terminated_at": self.terminated_at,
            "is_running": self.is_running,
        }


class ProcessRegistry:
    """Tracks live + recently-terminated dashboard-spawned subprocesses.

    Thread-safe via a coarse lock — the dashboard HTTP handler is
    threaded but the registry is touched per request, not per byte,
    so contention is negligible.
    """

    def __init__(self, *, terminated_ttl_seconds: float = _TERMINATED_TTL_SECONDS):
        self._procs: dict[str, WebProcess] = {}
        self._lock = threading.RLock()
        self._terminated_ttl_seconds = terminated_ttl_seconds

    def register(self, proc: WebProcess) -> None:
        with self._lock:
            self._procs[proc.session_id] = proc

    def mark_terminated(self, session_id: str, exit_code: int) -> None:
        with self._lock:
            entry = self._procs.get(session_id)
            if entry is None:
                return
            entry.exit_code = exit_code
            entry.terminated_at = time.time()

    def get(self, session_id: str) -> Optional[WebProcess]:
        with self._lock:
            return self._procs.get(session_id)

    def list_all(self) -> list[WebProcess]:
        """All known processes, freshest first. Terminated entries are
        retained for ``terminated_ttl_seconds`` so the UI can still show
        the last outcome before they vanish."""
        with self._lock:
            self._prune_expired_locked()
            return sorted(
                self._procs.values(),
                key=lambda p: -(p.terminated_at or p.started_at),
            )

    def list_running(self) -> list[WebProcess]:
        with self._lock:
            return [p for p in self._procs.values() if p.is_running]

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._procs.pop(session_id, None)

    def _prune_expired_locked(self) -> None:
        if not self._procs:
            return
        cutoff = time.time() - self._terminated_ttl_seconds
        to_drop = [
            sid
            for sid, p in self._procs.items()
            if p.terminated_at is not None and p.terminated_at < cutoff
        ]
        for sid in to_drop:
            self._procs.pop(sid, None)


# ---------------------------------------------------------------------------
# 3. HITL queue — bridges the harness's blocking webhook to the UI
# ---------------------------------------------------------------------------

@dataclass
class PendingHitl:
    """A HITL prompt the harness has POSTed to the dashboard, waiting
    for the operator to answer.

    The dashboard's HTTP handler builds one of these, stores it under
    its ``request_id``, then **blocks on ``event``** (the operator
    pushes the answer via the UI; the held handler returns that
    answer body to the harness).
    """

    request_id: str
    session_id: str
    prompt: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class HitlQueue:
    """Pending HITL prompts keyed by request id.

    Two consumers:
    - The webhook handler (``register_pending(...)`` followed by
      blocking on the event, then ``pop_response(...)``).
    - The UI (``list_pending_for_session(...)`` to render the prompts;
      ``answer(...)`` to push the operator's decision and release the
      held handler).
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingHitl] = {}
        self._lock = threading.RLock()

    def register_pending(
        self, *, request_id: str, session_id: str, prompt: dict[str, Any],
    ) -> PendingHitl:
        entry = PendingHitl(
            request_id=request_id, session_id=session_id, prompt=prompt,
        )
        with self._lock:
            self._pending[request_id] = entry
        return entry

    def list_pending_for_session(self, session_id: str) -> list[PendingHitl]:
        with self._lock:
            return [
                p for p in self._pending.values() if p.session_id == session_id
            ]

    def get(self, request_id: str) -> Optional[PendingHitl]:
        with self._lock:
            return self._pending.get(request_id)

    def answer(
        self, request_id: str, response: dict[str, Any],
    ) -> bool:
        """Push the operator's response and release the held handler.

        Returns ``True`` if the prompt was found and signalled,
        ``False`` if no such request_id is pending (already answered,
        or expired)."""
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None or entry.event.is_set():
                return False
            entry.response = dict(response)
            entry.event.set()
            return True

    def pop_response(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return {}
        return dict(entry.response)


# ---------------------------------------------------------------------------
# 4. Chat-note queue — per-session free-text notes that ride the next HITL
# ---------------------------------------------------------------------------

def queue_chat_note(
    *, db_path: str, session_id: str, note: str,
) -> int:
    """Persist a free-text note the operator typed in the dashboard.
    Returns the row id of the inserted note. The webhook handler will
    consume + prepend pending notes to the next HITL response's
    ``extra_notes`` field for this session."""
    note = (note or "").strip()
    if not note:
        return 0
    conn = open_web_db(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO chat_notes (session_id, ts, note) VALUES (?, ?, ?)",
                (session_id, _utcnow_iso(), note),
            )
            return int(cur.lastrowid or 0)
    finally:
        conn.close()


def consume_chat_notes(*, db_path: str, session_id: str) -> list[str]:
    """Atomically claim all pending chat notes for ``session_id``,
    returning their bodies in order. Subsequent calls return ``[]``
    until new notes are queued."""
    conn = open_web_db(db_path)
    try:
        with conn:
            rows = conn.execute(
                "SELECT id, note FROM chat_notes "
                "WHERE session_id = ? AND consumed_at IS NULL "
                "ORDER BY id",
                (session_id,),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE chat_notes SET consumed_at = ? WHERE id IN ({placeholders})",
                (_utcnow_iso(), *ids),
            )
            return [str(r[1]) for r in rows]
    finally:
        conn.close()


def pending_chat_notes(*, db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Read-only — the UI uses this for the "queued notes" panel."""
    conn = open_web_db(db_path)
    try:
        rows = conn.execute(
            "SELECT id, ts, note FROM chat_notes "
            "WHERE session_id = ? AND consumed_at IS NULL "
            "ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {"id": int(r[0]), "ts": str(r[1]), "note": str(r[2])} for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. One-shot scheduled jobs — bridge to harness/schedule.py
# ---------------------------------------------------------------------------

def add_oneshot_job(
    *,
    db_path: str,
    name: str,
    fire_at_utc: datetime,
    workspace: str,
    prompt: str = "",
    harness_args: Optional[list[str]] = None,
) -> int:
    """Insert a one-shot job the schedule daemon will pick up at or
    after ``fire_at_utc``. Returns the row id."""
    import json
    if fire_at_utc.tzinfo is None:
        raise ValueError("fire_at_utc must be tz-aware (UTC)")
    conn = open_web_db(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO web_oneshot_jobs "
                "(name, fire_at_utc, workspace, prompt, harness_args, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(name).strip() or "oneshot",
                    fire_at_utc.isoformat(timespec="seconds"),
                    str(workspace),
                    str(prompt or ""),
                    json.dumps(list(harness_args or [])),
                    _utcnow_iso(),
                ),
            )
            return int(cur.lastrowid or 0)
    finally:
        conn.close()


def list_pending_oneshot_jobs(
    *, db_path: str, now: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """All jobs whose ``fire_at_utc <= now`` and which have not been
    consumed yet. Order: oldest fire time first."""
    import json
    if now is None:
        now = datetime.now(timezone.utc)
    conn = open_web_db(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, fire_at_utc, workspace, prompt, harness_args "
            "FROM web_oneshot_jobs "
            "WHERE consumed_at IS NULL AND fire_at_utc <= ? "
            "ORDER BY fire_at_utc",
            (now.isoformat(timespec="seconds"),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                args = json.loads(r[5]) if r[5] else []
            except (TypeError, ValueError):
                args = []
            out.append({
                "id": int(r[0]),
                "name": str(r[1]),
                "fire_at_utc": str(r[2]),
                "workspace": str(r[3]),
                "prompt": str(r[4] or ""),
                "harness_args": args,
            })
        return out
    finally:
        conn.close()


def mark_oneshot_consumed(*, db_path: str, job_id: int) -> None:
    conn = open_web_db(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE web_oneshot_jobs SET consumed_at = ? WHERE id = ?",
                (_utcnow_iso(), int(job_id)),
            )
    finally:
        conn.close()


def list_all_oneshot_jobs(
    *, db_path: str, limit: int = 100, include_consumed: bool = True,
) -> list[dict[str, Any]]:
    """For the dashboard's schedule view: every one-shot we know about,
    most recent first."""
    import json
    where = "" if include_consumed else "WHERE consumed_at IS NULL"
    conn = open_web_db(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, name, fire_at_utc, workspace, prompt, harness_args, "
            f"created_at, consumed_at FROM web_oneshot_jobs {where} "
            f"ORDER BY id DESC LIMIT ?",
            (max(1, min(1000, int(limit))),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                args = json.loads(r[5]) if r[5] else []
            except (TypeError, ValueError):
                args = []
            out.append({
                "id": int(r[0]), "name": str(r[1]),
                "fire_at_utc": str(r[2]), "workspace": str(r[3]),
                "prompt": str(r[4] or ""), "harness_args": args,
                "created_at": str(r[6]),
                "consumed_at": str(r[7]) if r[7] else None,
            })
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. Audit log — record who saved what config
# ---------------------------------------------------------------------------

def append_audit(
    *, db_path: str, action: str, target: str, detail: str = "",
) -> int:
    conn = open_web_db(db_path)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO audit_log (ts, action, target, detail) VALUES (?, ?, ?, ?)",
                (_utcnow_iso(), str(action), str(target), str(detail)),
            )
            return int(cur.lastrowid or 0)
    finally:
        conn.close()


def list_audit(*, db_path: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = open_web_db(db_path)
    try:
        rows = conn.execute(
            "SELECT ts, action, target, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?",
            (max(1, min(1000, int(limit))),),
        ).fetchall()
        return [
            {"ts": str(r[0]), "action": str(r[1]),
             "target": str(r[2]), "detail": str(r[3] or "")}
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 7. Run presets — saved (workspace, prompt, args) tuples for one-click launch
# ---------------------------------------------------------------------------

def save_run_preset(
    *,
    db_path: str,
    name: str,
    workspace: str,
    prompt: str = "",
    harness_args: Optional[list[str]] = None,
) -> None:
    import json
    name = str(name).strip()
    if not name:
        raise ValueError("preset name must be non-empty")
    conn = open_web_db(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_presets "
                "(name, workspace, prompt, harness_args, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    name, str(workspace), str(prompt or ""),
                    json.dumps(list(harness_args or [])),
                    _utcnow_iso(),
                ),
            )
    finally:
        conn.close()


def list_run_presets(*, db_path: str) -> list[dict[str, Any]]:
    import json
    conn = open_web_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name, workspace, prompt, harness_args, created_at "
            "FROM run_presets ORDER BY created_at DESC"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                args = json.loads(r[3]) if r[3] else []
            except (TypeError, ValueError):
                args = []
            out.append({
                "name": str(r[0]), "workspace": str(r[1]),
                "prompt": str(r[2] or ""), "harness_args": args,
                "created_at": str(r[4]),
            })
        return out
    finally:
        conn.close()


def delete_run_preset(*, db_path: str, name: str) -> None:
    conn = open_web_db(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM run_presets WHERE name = ?", (name,))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 8. Subprocess lifecycle helpers used by the "Run now" endpoint
# ---------------------------------------------------------------------------

async def _watch_subprocess(
    proc: WebProcess,
    registry: ProcessRegistry,
    *,
    audit_db_path: Optional[str] = None,
) -> None:
    """Background coroutine: await the asyncio subprocess, then mark
    the registry entry terminated. ``audit_db_path`` is optional —
    when supplied we record the exit code under ``action='run_exit'``
    so the audit log carries it."""
    popen = proc.popen
    try:
        if hasattr(popen, "wait") and asyncio.iscoroutinefunction(popen.wait):
            exit_code = await popen.wait()
        else:
            # Sync subprocess.Popen fallback (used by tests).
            loop = asyncio.get_running_loop()
            exit_code = await loop.run_in_executor(None, popen.wait)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[web_state] subprocess wait error for %s: %s",
                       proc.session_id, exc)
        exit_code = -1
    registry.mark_terminated(proc.session_id, int(exit_code or 0))
    if audit_db_path:
        try:
            append_audit(
                db_path=audit_db_path, action="run_exit",
                target=proc.session_id, detail=f"exit_code={exit_code}",
            )
        except Exception as exc:  # noqa: BLE001 — audit never blocks the loop
            logger.debug("[web_state] audit_log write failed: %s", exc)
