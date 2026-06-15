"""``harness web`` — read-only web UI (#14).

Surfaces the data the harness already emits: session history (from
``~/.harness/logs/*.jsonl``), cost burn-down (the same events),
scheduled-job state (from ``harness/schedule.py``'s SQLite store),
repo-index status (from ``harness/repo_index.py``'s SQLite store), and
the per-repo memory files (from ``harness/repo_memory.py``).

Design choices
==============
- **Stdlib HTTP server** (``http.server.ThreadingHTTPServer``) — keeps
  the dependency budget at zero. The dashboard is single-user and
  low-traffic; a threaded sync server is plenty.
- **Localhost-only by default** (``dashboard.host: 127.0.0.1``) so
  accidental public exposure isn't possible without an explicit
  config change.
- **Optional bearer-token auth** wired from day one for operators
  who flip the bind address. Tokens are constant-time-compared via
  :func:`hmac.compare_digest`. Never logged. When
  ``dashboard.token_env`` is set but the named env var is empty, the
  server refuses to start — fail-closed.
- **Chart.js via CDN** (one ``<script>`` tag, no build step). Air-
  gapped operators drop a local ``chart.js`` into
  ``~/.harness/dashboard_static/`` and point ``dashboard.static_dir``
  at it — the handler prefers any matching local file over the CDN.
- **Read-only** — no buttons that mutate state in v1. Killing /
  retrying sessions is a follow-up; it would need careful design of
  the auth + audit story.
"""

from __future__ import annotations

import hmac
import html
import http.server
import json
import logging
import os
import re
import socketserver
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8729
_DEFAULT_LOG_DIR = "~/.harness/logs"
_DEFAULT_METRICS_DIR = "~/.harness/metrics"
_DEFAULT_MEMORY_DIR = "~/.harness/memory"
_DEFAULT_INDEX_DIR = "~/.harness/repo_index"
_DEFAULT_SCHEDULE_DB = "~/.harness/schedule.db"
_DEFAULT_STATIC_DIR = "~/.harness/dashboard_static"

# Carbon Design System assets. v10 is the last vanilla (non-React) line —
# v11 is React-only and would break this SSR layout. The pin is deliberate;
# air-gapped operators can point dashboard.carbon_css_url at their own
# mirror via the same static_dir + config knob pattern Chart.js uses.
_DEFAULT_CARBON_CSS_URL = "https://unpkg.com/carbon-components@10.58.12/css/carbon-components.min.css"
_DEFAULT_CARBON_JS_URL = "https://unpkg.com/carbon-components@10.58.12/scripts/carbon-components.min.js"
_DEFAULT_DOCS_DIR = ""  # empty → resolve to <repo_root>/docs at request time


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """All knobs operators can tune from config.json."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = _DEFAULT_PORT
    token_env: str = ""            # env var holding the bearer token; empty = no auth
    log_dir: str = _DEFAULT_LOG_DIR
    metrics_dir: str = _DEFAULT_METRICS_DIR
    memory_dir: str = _DEFAULT_MEMORY_DIR
    repo_index_dir: str = _DEFAULT_INDEX_DIR
    schedule_db: str = _DEFAULT_SCHEDULE_DB
    static_dir: str = _DEFAULT_STATIC_DIR
    chart_js_url: str = "https://cdn.jsdelivr.net/npm/chart.js"
    sessions_max: int = 200        # don't enumerate forever
    # Tier B/C knobs. Writes-on is the DEFAULT — `harness web` ships
    # the full UI without ceremony. Operators who need a read-only
    # deployment can flip ``dashboard.writes_enabled: false`` in
    # config.json; that gate still rejects POSTs and renders the
    # informational "writes are disabled" panels on /run + /config-ui.
    writes_enabled: bool = True
    csrf_token_env: str = ""       # CSRF token env var; auto-generated when empty + writes_enabled
    hitl_webhook_secret: str = ""  # shared secret the harness POSTs with
    web_db_path: str = "~/.harness/web.db"
    config_path: str = ""          # canonical config.json path; empty → use discover_config
    carbon_css_url: str = _DEFAULT_CARBON_CSS_URL
    carbon_js_url: str = _DEFAULT_CARBON_JS_URL
    docs_dir: str = _DEFAULT_DOCS_DIR  # empty → <repo_root>/docs at request time

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "DashboardConfig":
        section = ((config or {}).get("dashboard") or {})
        # Pick up the harness's existing log / metrics dirs when the
        # operator hasn't customised dashboard.*.
        logging_section = (config or {}).get("logging") or {}
        metrics_section = (config or {}).get("metrics") or {}
        return cls(
            enabled=bool(section.get("enabled", False)),
            host=str(section.get("host", "127.0.0.1")),
            port=max(1, min(65535, int(section.get("port", _DEFAULT_PORT)))),
            token_env=str(section.get("token_env", "")),
            log_dir=str(
                section.get("log_dir")
                or logging_section.get("log_dir")
                or _DEFAULT_LOG_DIR
            ),
            metrics_dir=str(
                section.get("metrics_dir")
                or metrics_section.get("metrics_dir")
                or _DEFAULT_METRICS_DIR
            ),
            memory_dir=str(section.get("memory_dir", _DEFAULT_MEMORY_DIR)),
            repo_index_dir=str(section.get("repo_index_dir", _DEFAULT_INDEX_DIR)),
            schedule_db=str(section.get("schedule_db", _DEFAULT_SCHEDULE_DB)),
            static_dir=str(section.get("static_dir", _DEFAULT_STATIC_DIR)),
            chart_js_url=str(section.get("chart_js_url", "https://cdn.jsdelivr.net/npm/chart.js")),
            sessions_max=max(1, min(10000, int(section.get("sessions_max", 200)))),
            writes_enabled=bool(section.get("writes_enabled", True)),
            csrf_token_env=str(section.get("csrf_token_env", "")),
            hitl_webhook_secret=str(section.get("hitl_webhook_secret", "")),
            web_db_path=str(section.get("web_db_path", "~/.harness/web.db")),
            config_path=str(section.get("config_path", "")),
            carbon_css_url=str(section.get("carbon_css_url", _DEFAULT_CARBON_CSS_URL)),
            carbon_js_url=str(section.get("carbon_js_url", _DEFAULT_CARBON_JS_URL)),
            docs_dir=str(section.get("docs_dir", _DEFAULT_DOCS_DIR)),
        )


# ---------------------------------------------------------------------------
# 2. Data adapters — read-only over the harness's existing on-disk state
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    session_id: str
    started_at: str = ""
    ended_at: str = ""
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_calls: int = 0
    exit_code: Optional[int] = None
    workspace_path: str = ""
    log_path: str = ""


def list_sessions(cfg: DashboardConfig) -> list[SessionSummary]:
    """Walk the log dir and parse one summary per session JSONL file.

    The JSONL files are NOT a stable contract; we read them
    defensively (any line that fails to parse is skipped, any field
    we don't recognise is ignored). When the log dir is missing we
    return an empty list — the dashboard renders "no sessions yet"
    cleanly.
    """
    log_dir = os.path.expanduser(cfg.log_dir)
    if not os.path.isdir(log_dir):
        return []
    out: list[SessionSummary] = []
    try:
        entries = [
            (e, os.path.getmtime(os.path.join(log_dir, e)))
            for e in os.listdir(log_dir)
            if e.endswith(".jsonl")
        ]
    except OSError:
        return []
    entries.sort(key=lambda t: t[1], reverse=True)
    for filename, _mtime in entries[: cfg.sessions_max]:
        session_id = filename[: -len(".jsonl")]
        path = os.path.join(log_dir, filename)
        summary = _parse_session_log(session_id, path)
        out.append(summary)
    return out


def _parse_session_log(session_id: str, path: str) -> SessionSummary:
    summary = SessionSummary(session_id=session_id, log_path=path)
    first_event: Optional[dict[str, Any]] = None
    last_event: Optional[dict[str, Any]] = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                event = _safe_json(line)
                if not event:
                    continue
                if first_event is None:
                    first_event = event
                last_event = event
                # Lightweight accumulation across the full log.
                if event.get("event") == "llm_call":
                    summary.llm_calls += 1
                    summary.total_cost_usd += float(event.get("cost_usd", 0.0) or 0.0)
                    summary.total_input_tokens += int(event.get("tokens_in", 0) or 0)
                    summary.total_output_tokens += int(event.get("tokens_out", 0) or 0)
                if event.get("event") == "session_end":
                    summary.ended_at = str(event.get("timestamp") or summary.ended_at)
                    if event.get("exit_code") is not None:
                        summary.exit_code = int(event["exit_code"])
                if event.get("event") == "session_start":
                    summary.started_at = str(event.get("timestamp") or summary.started_at)
                    summary.workspace_path = str(event.get("workspace_path") or summary.workspace_path)
        # Fallback timestamps when explicit session_start/end events
        # weren't emitted.
        if not summary.started_at and first_event:
            summary.started_at = str(first_event.get("timestamp") or "")
        if not summary.ended_at and last_event:
            summary.ended_at = str(last_event.get("timestamp") or "")
    except OSError:
        pass
    return summary


def _parse_iso_utc(value: str) -> Optional[datetime]:
    """Best-effort ISO 8601 → aware UTC datetime. Returns None if the
    value isn't parseable. Mirrors the harness's observability format
    (``timestamp`` field in JSONL events)."""
    from datetime import timezone
    if not value:
        return None
    try:
        # `fromisoformat` handles both naive and offset-bearing strings.
        # Trailing 'Z' is not accepted pre-3.11 — normalise it.
        normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(normalised)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def summarize_sessions_window(
    sessions: list[SessionSummary],
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, Any]:
    """Aggregate completed/succeeded/failed counts, cumulative duration
    (seconds) and cumulative cost over the [start, end] window. A
    session counts towards the window if its ``started_at`` falls inside
    it. Sessions with un-parseable timestamps are skipped silently.
    """
    completed = 0
    succeeded = 0
    failed = 0
    duration_seconds = 0.0
    cost_usd = 0.0
    for s in sessions:
        started = _parse_iso_utc(s.started_at)
        if started is None or not (start_utc <= started <= end_utc):
            continue
        # Only an explicit ``exit_code`` (from a ``session_end`` event)
        # counts as completed — `_parse_session_log` fills ``ended_at``
        # with the last event timestamp for in-flight sessions, so we
        # can't trust ``ended_at`` alone.
        if s.exit_code is not None:
            completed += 1
            ended = _parse_iso_utc(s.ended_at)
            if ended is not None:
                duration_seconds += max(0.0, (ended - started).total_seconds())
            if s.exit_code == 0:
                succeeded += 1
            else:
                failed += 1
        cost_usd += float(s.total_cost_usd or 0.0)
    return {
        "completed": completed,
        "succeeded": succeeded,
        "failed": failed,
        "duration_seconds": duration_seconds,
        "cost_usd": cost_usd,
    }


def _last_event_name(log_path: str, *, tail_bytes: int = 4096) -> str:
    """Read the tail of a JSONL log and return the ``event`` value of
    the last fully-parseable line. Empty string when the file is
    missing, empty, or unparseable."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.read(1)  # drop the partial first line
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    last_event = ""
    for line in tail.splitlines():
        evt = _safe_json(line)
        if isinstance(evt, dict) and evt.get("event"):
            last_event = str(evt["event"])
    return last_event


def list_running_sessions(cfg: DashboardConfig) -> list[dict[str, Any]]:
    """Sessions that are *currently in flight*. Union of:

    1. The web-spawned :class:`ProcessRegistry` (covers runs started
       from the Run Harness page).
    2. JSONL logs whose tail does NOT contain a ``session_end`` event
       and whose mtime is within the last 24 hours (covers runs
       started from the CLI).

    Web entries win on duplicate ``session_id`` because they carry
    workspace + prompt context the log scan can't recover cheaply.
    """
    from datetime import datetime, timezone, timedelta
    rows: dict[str, dict[str, Any]] = {}

    log_dir = os.path.expanduser(cfg.log_dir)
    if os.path.isdir(log_dir):
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
        try:
            entries = os.listdir(log_dir)
        except OSError:
            entries = []
        for filename in entries:
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(log_dir, filename)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < cutoff:
                continue
            if _last_event_name(path) == "session_end":
                continue
            sid = filename[:-len(".jsonl")]
            # Pull the start timestamp + workspace from the first line.
            started_at = ""
            workspace_path = ""
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        evt = _safe_json(line)
                        if not evt:
                            continue
                        if evt.get("event") == "session_start":
                            started_at = str(evt.get("timestamp") or "")
                            workspace_path = str(evt.get("workspace_path") or "")
                            break
                        if not started_at:
                            started_at = str(evt.get("timestamp") or "")
            except OSError:
                pass
            rows[sid] = {
                "session_id": sid,
                "started_at": started_at,
                "workspace_path": workspace_path,
                "prompt": "",
                "source": "cli",
            }

    try:
        for proc in get_process_registry().list_running():
            from datetime import datetime as _dt, timezone as _tz
            iso = _dt.fromtimestamp(proc.started_at, tz=_tz.utc).isoformat()
            rows[proc.session_id] = {
                "session_id": proc.session_id,
                "started_at": iso,
                "workspace_path": proc.workspace_path,
                "prompt": proc.prompt,
                "source": "web",
            }
    except Exception:  # noqa: BLE001
        # Registry may not be initialised in tests; tolerate.
        pass

    return sorted(rows.values(), key=lambda r: r["started_at"], reverse=True)


def session_events(path: str, *, max_events: int = 1000) -> list[dict[str, Any]]:
    """Read the per-session JSONL log and return a capped list of
    decoded events for the detail view."""
    if not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                event = _safe_json(line)
                if event is None:
                    continue
                out.append(event)
                if len(out) >= max_events:
                    break
    except OSError:
        return []
    return out


def cost_burn_series(cfg: DashboardConfig) -> dict[str, Any]:
    """Aggregate ``cost_usd`` events across every session into a single
    time-ordered series the cost-burn Chart.js view consumes.

    Output shape::

        {
          "labels": [iso8601, iso8601, ...],
          "datasets": [
            {"label": "Cumulative spend", "data": [0.0, 0.04, 0.12, ...]},
            {"label": "Per-call cost",    "data": [0.0, 0.04, 0.08, ...]},
          ],
        }
    """
    sessions = list_sessions(cfg)
    rows: list[tuple[str, float]] = []
    for s in sessions:
        for evt in session_events(s.log_path, max_events=5000):
            if evt.get("event") == "llm_call":
                ts = str(evt.get("timestamp") or "")
                cost = float(evt.get("cost_usd", 0.0) or 0.0)
                if ts:
                    rows.append((ts, cost))
    rows.sort()
    labels: list[str] = []
    per_call: list[float] = []
    cumulative: list[float] = []
    running = 0.0
    for ts, cost in rows:
        labels.append(ts)
        per_call.append(cost)
        running += cost
        cumulative.append(round(running, 6))
    return {
        "labels": labels,
        "datasets": [
            {"label": "Cumulative spend ($)", "data": cumulative},
            {"label": "Per-call cost ($)", "data": per_call},
        ],
    }


def list_memory_files(cfg: DashboardConfig) -> list[dict[str, Any]]:
    mem_dir = os.path.expanduser(cfg.memory_dir)
    if not os.path.isdir(mem_dir):
        return []
    out: list[dict[str, Any]] = []
    try:
        for name in sorted(os.listdir(mem_dir)):
            if not name.endswith(".md"):
                continue
            path = os.path.join(mem_dir, name)
            try:
                size = os.path.getsize(path)
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            out.append({"name": name, "size": size, "mtime": mtime})
    except OSError:
        return []
    return out


def read_memory_file(cfg: DashboardConfig, name: str) -> Optional[str]:
    """Read a memory file by basename. Refuses any name containing a
    path separator or a parent traversal — defence-in-depth even though
    the route only passes URL-decoded basenames in."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    if not name.endswith(".md"):
        return None
    path = os.path.join(os.path.expanduser(cfg.memory_dir), name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def repo_index_status(cfg: DashboardConfig) -> list[dict[str, Any]]:
    """Pull every workspace meta row from the repo_index DB. Returns an
    empty list when the DB doesn't exist yet."""
    db_path = os.path.join(os.path.expanduser(cfg.repo_index_dir), "repo_index.db")
    if not os.path.isfile(db_path):
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT workspace_id, backend, built_at, chunk_count FROM repo_meta"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return []
    return [
        {
            "workspace_id": r[0], "backend": r[1],
            "built_at": r[2], "chunk_count": int(r[3] or 0),
        }
        for r in rows
    ]


def list_schedule_runs(cfg: DashboardConfig, *, limit: int = 100) -> list[dict[str, Any]]:
    db_path = os.path.expanduser(cfg.schedule_db)
    if not os.path.isfile(db_path):
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT job_name, started_at, ended_at, exit_code, duration_sec, log_path "
                "FROM schedule_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, min(1000, int(limit))),),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return []
    return [
        {
            "job_name": r[0], "started_at": r[1], "ended_at": r[2],
            "exit_code": r[3], "duration_sec": r[4], "log_path": r[5],
        }
        for r in rows
    ]


def _safe_json(text: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# 3. Auth
# ---------------------------------------------------------------------------

@dataclass
class AuthOutcome:
    ok: bool
    detail: str = ""


def resolve_expected_token(cfg: DashboardConfig) -> Optional[str]:
    """Return the token the dashboard expects, or ``None`` when auth
    is disabled. Raises :class:`RuntimeError` when ``token_env`` is
    set but the env var is empty — fail-closed at startup."""
    if not cfg.token_env:
        return None
    value = os.environ.get(cfg.token_env, "")
    if not value:
        raise RuntimeError(
            f"dashboard.token_env={cfg.token_env!r} is set but the "
            f"environment variable is empty. Export it before starting "
            f"the dashboard, or remove the token_env config key to "
            f"disable auth."
        )
    return value


def check_auth(
    expected_token: Optional[str], header_value: Optional[str],
) -> AuthOutcome:
    if expected_token is None:
        return AuthOutcome(ok=True, detail="auth disabled")
    if not header_value:
        return AuthOutcome(ok=False, detail="missing Authorization header")
    if not header_value.startswith("Bearer "):
        return AuthOutcome(ok=False, detail="Authorization header must start with 'Bearer '")
    provided = header_value[len("Bearer "):]
    if not hmac.compare_digest(provided, expected_token):
        return AuthOutcome(ok=False, detail="token mismatch")
    return AuthOutcome(ok=True, detail="ok")


# ---------------------------------------------------------------------------
# 4. HTTP handler
# ---------------------------------------------------------------------------

# myharness overrides on top of Carbon. Carbon ships its own type scale,
# spacing tokens, table styles, etc. — we only add things Carbon doesn't:
# the side-nav offset for main content, a couple of status-color helpers,
# and the legacy .card/.ok/.fail classes some existing renderers still use.
_BASE_CSS = """\
body { margin: 0; }
main.bx--content { margin-left: 16rem; padding: 2rem; min-height: calc(100vh - 3rem); background: #f4f4f4; }
main.bx--content h2 { font-weight: 400; margin-top: 0; }
.bx--side-nav__link--current, .bx--side-nav__link--current span { color: #fff !important; background: #393939; }
.muted { color: #6f6f6f; }
.ok { color: #198038; font-weight: 600; }
.fail { color: #da1e28; font-weight: 600; }
.card { background: #fff; border: 1px solid #e0e0e0; padding: 1rem; margin-bottom: 1rem; }
.card h2 { margin-top: 0; font-size: 1rem; font-weight: 600; }
.card pre { background: #f4f4f4; padding: 0.75rem; overflow: auto; font-size: 0.85rem; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e0e0e0; text-align: left; font-size: 0.875rem; }
th { background: #f4f4f4; font-weight: 600; }
.tile-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr)); gap: 1rem; }
.status-tile { background: #fff; border: 1px solid #e0e0e0; padding: 1rem; }
.status-tile h3 { margin: 0 0 0.75rem 0; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; color: #525252; }
.status-tile dl { margin: 0; display: grid; grid-template-columns: 1fr auto; row-gap: 0.25rem; column-gap: 1rem; font-size: 0.875rem; }
.status-tile dt { color: #525252; }
.status-tile dd { margin: 0; font-variant-numeric: tabular-nums; }
.dash-tile { background: #fff; border: 1px solid #e0e0e0; padding: 1rem; transition: background 0.1s; }
.dash-tile:hover { background: #e8e8e8; }
.dash-tile a { color: #0f62fe; text-decoration: none; display: block; }
.dash-tile h3 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; }
.dash-tile p { margin: 0; font-size: 0.875rem; color: #525252; }
.tag { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 0.75rem; font-size: 0.75rem; font-weight: 500; }
.tag-green { background: #defbe6; color: #0e6027; }
.tag-red { background: #ffd7d9; color: #a2191f; }
.tag-gray { background: #e0e0e0; color: #393939; }
"""


_NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    # (slug, label, href)
    ("status", "View Status", "/status"),
    ("run", "Run Harness", "/run"),
    ("config", "Configure Harness", "/config-ui"),
    ("dashboards", "View Dashboards", "/dashboards"),
    ("docs", "View Documents", "/docs"),
)


def _render_side_nav(active: str) -> str:
    items = []
    for slug, label, href in _NAV_ITEMS:
        cls = "bx--side-nav__link bx--side-nav__link--current" if slug == active else "bx--side-nav__link"
        items.append(
            f'<li class="bx--side-nav__item">'
            f'<a class="{cls}" href="{href}">'
            f'<span class="bx--side-nav__link-text">{html.escape(label)}</span>'
            f'</a></li>'
        )
    return (
        '<nav class="bx--side-nav bx--side-nav--expanded" aria-label="Side navigation">'
        '<ul class="bx--side-nav__items">' + "".join(items) + '</ul></nav>'
    )


def _layout(title: str, body: str, cfg: DashboardConfig, active: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — myharness</title>
<link rel="stylesheet" href="{html.escape(cfg.carbon_css_url)}">
<style>{_BASE_CSS}</style>
<script src="{html.escape(cfg.chart_js_url)}"></script>
</head>
<body class="bx--body">
<header class="bx--header" role="banner">
  <a class="bx--header__name" href="/status">
    <span class="bx--header__name--prefix">myharness</span>
  </a>
</header>
{_render_side_nav(active)}
<main class="bx--content">
<h2>{html.escape(title)}</h2>
{body}
</main>
</body>
</html>
"""


def _fmt_cost(value: float) -> str:
    return f"${value:.4f}" if value > 0 else "—"


def _fmt_int(value: int) -> str:
    return str(value) if value else "—"


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _render_sessions(cfg: DashboardConfig) -> str:
    sessions = list_sessions(cfg)
    if not sessions:
        return (
            "<p class='muted'>No sessions yet. Run "
            "<code>harness run -r ...</code> to populate this view.</p>"
        )
    rows = []
    for s in sessions:
        status = "—"
        cls = "muted"
        if s.exit_code == 0:
            status, cls = "success", "ok"
        elif s.exit_code is not None:
            status, cls = f"exit {s.exit_code}", "fail"
        rows.append(
            f"<tr>"
            f"<td><a href='/sessions/{_esc(s.session_id)}'>{_esc(s.session_id)}</a></td>"
            f"<td>{_esc(s.started_at)}</td>"
            f"<td>{_esc(s.ended_at)}</td>"
            f"<td class='{cls}'>{_esc(status)}</td>"
            f"<td>{_fmt_int(s.llm_calls)}</td>"
            f"<td>{_fmt_cost(s.total_cost_usd)}</td>"
            f"<td>{_fmt_int(s.total_input_tokens)}</td>"
            f"<td>{_fmt_int(s.total_output_tokens)}</td>"
            f"<td><span class='muted'>{_esc(s.workspace_path)}</span></td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<tr><th>session</th><th>started</th><th>ended</th><th>status</th>"
        "<th>calls</th><th>cost</th><th>tokens in</th><th>tokens out</th>"
        "<th>workspace</th></tr>"
        + "".join(rows) + "</table>"
    )


def _render_session_detail(cfg: DashboardConfig, session_id: str) -> str:
    log_path = os.path.join(os.path.expanduser(cfg.log_dir), f"{session_id}.jsonl")
    if not os.path.isfile(log_path):
        return f"<p class='fail'>No log file for session {_esc(session_id)}.</p>"
    events = session_events(log_path, max_events=2000)
    if not events:
        return f"<p class='muted'>Session {_esc(session_id)} log is empty or unparseable.</p>"
    body = []
    body.append(f"<div class='card'><h2>Events ({len(events)})</h2>")
    rows = []
    for evt in events[:200]:
        ts = _esc(evt.get("timestamp") or "")
        name = _esc(evt.get("event") or "?")
        details = {k: v for k, v in evt.items() if k not in ("event", "timestamp")}
        rows.append(
            f"<tr><td>{ts}</td><td>{name}</td>"
            f"<td><pre>{html.escape(json.dumps(details, indent=2, default=str))}</pre></td></tr>"
        )
    body.append("<table><tr><th>time</th><th>event</th><th>details</th></tr>"
                + "".join(rows) + "</table>")
    if len(events) > 200:
        body.append(f"<p class='muted'>(showing first 200 of {len(events)})</p>")
    body.append("</div>")
    return "".join(body)


def _render_cost(cfg: DashboardConfig) -> str:
    return f"""\
<div class='card'>
  <h2>Cumulative cost burn</h2>
  <canvas id='costChart' width='800' height='320'></canvas>
</div>
<script>
fetch('/api/cost-burn').then(r => r.json()).then(data => {{
  const ctx = document.getElementById('costChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: data,
    options: {{
      responsive: true,
      scales: {{ y: {{ beginAtZero: true }} }},
      plugins: {{ legend: {{ position: 'top' }} }},
    }},
  }});
}});
</script>
<p class='muted'>Source: ``cost_usd`` events from every session JSONL log under {html.escape(cfg.log_dir)}.</p>
"""


def _render_schedule(cfg: DashboardConfig) -> str:
    runs = list_schedule_runs(cfg, limit=200)
    if not runs:
        return (
            "<p class='muted'>No scheduled-job runs recorded yet. "
            "Configure jobs under <code>schedule.jobs</code> in "
            "<code>config.json</code> and start <code>harness schedule run</code>.</p>"
        )
    rows = []
    for r in runs:
        ec = r["exit_code"]
        if ec == 0:
            cls, status = "ok", "success"
        elif ec is None:
            cls, status = "muted", "running"
        else:
            cls, status = "fail", f"exit {ec}"
        duration = r["duration_sec"]
        rows.append(
            f"<tr>"
            f"<td>{_esc(r['job_name'])}</td>"
            f"<td>{_esc(r['started_at'])}</td>"
            f"<td>{_esc(r['ended_at'])}</td>"
            f"<td class='{cls}'>{status}</td>"
            f"<td>{(f'{duration:.1f}s' if duration is not None else '—')}</td>"
            f"<td><span class='muted'>{_esc(r['log_path'])}</span></td>"
            f"</tr>"
        )
    return ("<table><tr><th>job</th><th>started</th><th>ended</th>"
            "<th>status</th><th>duration</th><th>log</th></tr>"
            + "".join(rows) + "</table>")


def _render_index(cfg: DashboardConfig) -> str:
    rows = repo_index_status(cfg)
    if not rows:
        return (
            "<p class='muted'>No repo index built yet. Run "
            "<code>harness index build -r WORKSPACE</code> to populate this view.</p>"
        )
    items = []
    for r in rows:
        items.append(
            f"<tr><td><code>{_esc(r['workspace_id'])}</code></td>"
            f"<td>{_esc(r['backend'])}</td>"
            f"<td>{_esc(r['chunk_count'])}</td>"
            f"<td>{_esc(r['built_at'])}</td></tr>"
        )
    return ("<table><tr><th>workspace id</th><th>backend</th>"
            "<th>chunks</th><th>built at</th></tr>"
            + "".join(items) + "</table>")


def _render_memory(cfg: DashboardConfig) -> str:
    files = list_memory_files(cfg)
    if not files:
        return (
            "<p class='muted'>No per-repo memory files yet. The harness "
            "writes them when a session ends with "
            "<code>memory.enabled=true</code>.</p>"
        )
    rows = []
    for f in files:
        rows.append(
            f"<tr>"
            f"<td><a href='/memory/{urllib.parse.quote(f['name'])}'>{_esc(f['name'])}</a></td>"
            f"<td>{f['size']:,} bytes</td>"
            f"</tr>"
        )
    return ("<table><tr><th>file</th><th>size</th></tr>"
            + "".join(rows) + "</table>")


def _render_memory_file(cfg: DashboardConfig, name: str) -> tuple[int, str]:
    content = read_memory_file(cfg, name)
    if content is None:
        return 404, "<p class='fail'>Memory file not found.</p>"
    return 200, f"<div class='card'><h2>{_esc(name)}</h2><pre>{html.escape(content)}</pre></div>"


def _render_overview(cfg: DashboardConfig) -> str:
    sessions = list_sessions(cfg)
    total = sum(s.total_cost_usd for s in sessions)
    calls = sum(s.llm_calls for s in sessions)
    success = sum(1 for s in sessions if s.exit_code == 0)
    fail = sum(1 for s in sessions if s.exit_code not in (0, None))
    return f"""\
<div class='card'>
  <h2>At a glance</h2>
  <table>
    <tr><th>Sessions on disk</th><td>{len(sessions)}</td></tr>
    <tr><th>Successful runs</th><td class='ok'>{success}</td></tr>
    <tr><th>Failed runs</th><td class='fail'>{fail}</td></tr>
    <tr><th>Total LLM calls</th><td>{calls}</td></tr>
    <tr><th>Cumulative spend</th><td>{_fmt_cost(total)}</td></tr>
  </table>
</div>
<div class='card'>
  <h2>Where this data comes from</h2>
  <table>
    <tr><th>Sessions</th><td><code>{_esc(cfg.log_dir)}/*.jsonl</code></td></tr>
    <tr><th>Scheduled runs</th><td><code>{_esc(cfg.schedule_db)}</code></td></tr>
    <tr><th>Repo index</th><td><code>{_esc(cfg.repo_index_dir)}/repo_index.db</code></td></tr>
    <tr><th>Per-repo memory</th><td><code>{_esc(cfg.memory_dir)}/*.md</code></td></tr>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# 5. Router  ---  one entry per (regex, handler) pair
# ---------------------------------------------------------------------------

Route = tuple[re.Pattern[str], Callable[[DashboardConfig, dict[str, str]], tuple[int, str, str]]]


def _route_overview(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Overview", _render_overview(cfg), cfg)


def _route_sessions(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Sessions", _render_sessions(cfg), cfg, active="dashboards")


def _route_session_detail(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        f"Session {params['sid']}",
        _render_session_detail(cfg, params["sid"]),
        cfg, active="dashboards",
    )


def _route_cost(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Cost burn", _render_cost(cfg), cfg, active="dashboards")


def _route_api_cost_burn(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "application/json; charset=utf-8", json.dumps(cost_burn_series(cfg))


def _route_schedule(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Scheduled runs", _render_schedule(cfg), cfg, active="dashboards")


def _route_index(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Repo index", _render_index(cfg), cfg, active="dashboards")


def _route_memory(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Per-repo memory", _render_memory(cfg), cfg, active="dashboards")


def _route_memory_file(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    status, body = _render_memory_file(cfg, params["name"])
    return status, "text/html; charset=utf-8", _layout(
        f"Memory · {params['name']}", body, cfg, active="dashboards",
    )


# ---------------------------------------------------------------------------
# 5. Carbon shell — 5 top-level pages
# ---------------------------------------------------------------------------
# Body sentinel for HTTP 302. dispatch() returns one tuple; the request
# handler recognises this prefix and emits a real redirect.
_REDIRECT_SENTINEL = "__REDIRECT__"


def _route_root(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 302, "text/html; charset=utf-8", f"{_REDIRECT_SENTINEL}/status"


def _route_status(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        "View Status", _render_status(cfg), cfg, active="status",
    )


def _route_run(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        "Run Harness", _render_run_harness(cfg), cfg, active="run",
    )


def _route_configure_harness(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        "Configure Harness", _render_configure_harness(cfg), cfg, active="config",
    )


def _route_dashboards(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        "View Dashboards", _render_dashboards_landing(cfg), cfg, active="dashboards",
    )


def _route_docs(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        "View Documents", _render_docs_landing(cfg), cfg, active="docs",
    )


def _route_docs_file(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    relpath = urllib.parse.unquote(params["relpath"])
    status, body = _render_docs_file(cfg, relpath)
    title = f"Document · {relpath}" if status == 200 else "Document · not found"
    return status, "text/html; charset=utf-8", _layout(title, body, cfg, active="docs")


def _stub_panel(label: str) -> str:
    return (
        f"<div class='card'><p class='muted'>{html.escape(label)} — "
        f"this page is the Phase 1 shell stub. Content lands in a "
        f"subsequent phase.</p></div>"
    )


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _status_tile(label: str, summary: dict[str, Any]) -> str:
    return (
        f"<div class='status-tile'>"
        f"<h3>{html.escape(label)}</h3>"
        f"<dl>"
        f"<dt>Completed</dt><dd>{summary['completed']}</dd>"
        f"<dt>Succeeded</dt><dd class='ok'>{summary['succeeded']}</dd>"
        f"<dt>Failed</dt><dd class='fail'>{summary['failed']}</dd>"
        f"<dt>Cumulative time</dt><dd>{_fmt_duration(summary['duration_seconds'])}</dd>"
        f"<dt>Cumulative cost</dt><dd>{_fmt_cost(summary['cost_usd'])}</dd>"
        f"</dl></div>"
    )


def _render_status(cfg: DashboardConfig) -> str:
    from datetime import datetime, timezone, timedelta

    sessions = list_sessions(cfg)
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    section_a = (
        "<div class='card'>"
        "<h2>Summary</h2>"
        "<div class='tile-grid'>"
        + _status_tile("Today", summarize_sessions_window(sessions, day_start, now))
        + _status_tile("Past 7 days", summarize_sessions_window(sessions, week_start, now))
        + _status_tile("Past 30 days", summarize_sessions_window(sessions, month_start, now))
        + "</div></div>"
    )

    running = list_running_sessions(cfg)
    if running:
        rows = []
        for r in running:
            rows.append(
                f"<tr>"
                f"<td><a href='/sessions/{_esc(r['session_id'])}'>{_esc(r['session_id'])}</a></td>"
                f"<td>{_esc(r['started_at'])}</td>"
                f"<td>{_esc(r['workspace_path'])}</td>"
                f"<td><span class='tag tag-gray'>{_esc(r['source'])}</span></td>"
                f"<td><a href='/sessions/{_esc(r['session_id'])}'>Open dashboard</a></td>"
                f"</tr>"
            )
        section_b_body = (
            "<table><tr><th>Session</th><th>Started</th><th>Workspace</th>"
            "<th>Source</th><th></th></tr>" + "".join(rows) + "</table>"
        )
    else:
        section_b_body = "<p class='muted'>No sessions are running right now.</p>"
    section_b = f"<div class='card'><h2>Running now</h2>{section_b_body}</div>"

    today_sessions = [
        s for s in sessions
        if (d := _parse_iso_utc(s.started_at)) is not None and d >= day_start
    ]
    if today_sessions:
        rows = []
        for s in today_sessions:
            started = _parse_iso_utc(s.started_at)
            ended = _parse_iso_utc(s.ended_at)
            if s.exit_code is None:
                status_html = "<span class='tag tag-gray'>running</span>"
                duration = "—"
            elif s.exit_code == 0:
                status_html = "<span class='tag tag-green'>succeeded</span>"
                duration = _fmt_duration((ended - started).total_seconds()) if started and ended else "—"
            else:
                status_html = f"<span class='tag tag-red'>exit {s.exit_code}</span>"
                duration = _fmt_duration((ended - started).total_seconds()) if started and ended else "—"
            rows.append(
                f"<tr>"
                f"<td><a href='/sessions/{_esc(s.session_id)}'>{_esc(s.session_id)}</a></td>"
                f"<td>{status_html}</td>"
                f"<td>{duration}</td>"
                f"<td>{_fmt_cost(s.total_cost_usd)}</td>"
                f"<td><a href='/sessions/{_esc(s.session_id)}'>Open dashboard</a></td>"
                f"</tr>"
            )
        section_c_body = (
            "<table><tr><th>Session</th><th>Status</th><th>Duration</th>"
            "<th>Cost</th><th></th></tr>" + "".join(rows) + "</table>"
        )
    else:
        section_c_body = "<p class='muted'>No sessions started today.</p>"
    section_c = f"<div class='card'><h2>Today's runs</h2>{section_c_body}</div>"

    return section_a + section_b + section_c


def _render_run_harness(cfg: DashboardConfig) -> str:
    if not cfg.writes_enabled:
        return (
            "<div class='card'><p class='muted'>Writes are disabled "
            "by <code>dashboard.writes_enabled: false</code> in "
            "<code>config.json</code>. Flip it to <code>true</code> "
            "(or remove the override — writes are on by default) to "
            "run or schedule from this page.</p></div>"
        )
    csrf_token = resolve_csrf_token(cfg) or ""

    # Pending one-shot jobs table. The schedule helper filters to
    # "due now" by default — pass a far-future cutoff to surface every
    # not-yet-consumed row, including ones scheduled for tomorrow.
    from datetime import datetime, timezone
    far_future = datetime(2999, 1, 1, tzinfo=timezone.utc)
    try:
        pending = list_pending_oneshot_jobs(db_path=cfg.web_db_path, now=far_future)
    except Exception:  # noqa: BLE001
        pending = []
    if pending:
        rows = []
        for job in pending:
            args_pretty = " ".join(html.escape(str(a)) for a in job["harness_args"])
            rows.append(
                f"<tr>"
                f"<td>{_esc(job['fire_at_utc'])}</td>"
                f"<td>{_esc(job['name'])}</td>"
                f"<td>{_esc(job['workspace'])}</td>"
                f"<td>{_esc(job['prompt'][:80])}</td>"
                f"<td><span class='tag tag-gray'>scheduled</span></td>"
                f"<td><code>{args_pretty}</code></td>"
                f"</tr>"
            )
        pending_html = (
            "<table><tr><th>Fire at (UTC)</th><th>Name</th><th>Workspace</th>"
            "<th>Prompt</th><th>Status</th><th>Args</th></tr>"
            + "".join(rows) + "</table>"
        )
    else:
        pending_html = "<p class='muted'>No one-shot jobs scheduled.</p>"

    form = f"""
<div class='card'>
  <h2>Start a harness run</h2>
  <form id='run-form' method='post' action='/run/now'>
    <input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>
    <input type='hidden' id='fire-at-utc' name='fire_at_utc' value=''>
    <div style='margin-bottom:1rem'>
      <label class='bx--label' for='workspace'>Workspace path</label>
      <input class='bx--text-input' id='workspace' name='workspace' type='text' required>
    </div>
    <div style='margin-bottom:1rem'>
      <label class='bx--label' for='prompt'>Prompt</label>
      <textarea class='bx--text-area' id='prompt' name='prompt' rows='4' required></textarea>
    </div>
    <div style='margin-bottom:1rem'>
      <label class='bx--label' for='extra_args'>Extra harness args (space-separated)</label>
      <input class='bx--text-input' id='extra_args' name='extra_args' type='text'
             placeholder='--new_build=false --allow-network'>
    </div>
    <fieldset id='schedule-fields' style='display:none; border:none; padding:0; margin:1rem 0;'>
      <legend class='bx--label'>Scheduled run</legend>
      <div style='margin-bottom:1rem'>
        <label class='bx--label' for='job_name'>Job name</label>
        <input class='bx--text-input' id='job_name' name='name' type='text' placeholder='nightly retest'>
      </div>
      <div style='display:flex; gap:1rem; margin-bottom:1rem'>
        <div style='flex:1'>
          <label class='bx--label' for='fire_date'>Date (UTC)</label>
          <input class='bx--date-picker__input' id='fire_date' type='date'>
        </div>
        <div style='flex:1'>
          <label class='bx--label' for='fire_time'>Time (UTC)</label>
          <input class='bx--time-picker__input' id='fire_time' type='time' step='60'>
        </div>
      </div>
    </fieldset>
    <div style='margin-top:1.5rem'>
      <button class='bx--btn bx--btn--primary' type='submit'
              formaction='/run/now' id='run-now-btn'>Run Now</button>
      <button class='bx--btn bx--btn--secondary' type='button' id='reveal-schedule-btn'
              style='margin-left:0.5rem'>Schedule A Run</button>
      <button class='bx--btn bx--btn--primary' type='submit'
              formaction='/run/schedule' id='confirm-schedule-btn'
              style='margin-left:0.5rem; display:none'>Confirm Schedule</button>
    </div>
  </form>
</div>
<script>
(function() {{
  var reveal = document.getElementById('reveal-schedule-btn');
  var confirm = document.getElementById('confirm-schedule-btn');
  var runNow = document.getElementById('run-now-btn');
  var fields = document.getElementById('schedule-fields');
  var form = document.getElementById('run-form');
  var fireDate = document.getElementById('fire_date');
  var fireTime = document.getElementById('fire_time');
  var fireAt = document.getElementById('fire-at-utc');
  reveal.addEventListener('click', function() {{
    fields.style.display = 'block';
    confirm.style.display = 'inline-block';
    runNow.style.display = 'none';
    reveal.style.display = 'none';
  }});
  form.addEventListener('submit', function(e) {{
    if (e.submitter && e.submitter.id === 'confirm-schedule-btn') {{
      if (!fireDate.value || !fireTime.value) {{
        e.preventDefault();
        alert('Pick both a date and a time.');
        return;
      }}
      fireAt.value = fireDate.value + 'T' + fireTime.value + ':00Z';
    }}
  }});
}})();
</script>
"""

    return f"""{form}
<div class='card'>
  <h2>Scheduled runs</h2>
  {pending_html}
</div>"""


def _render_field_input_new(field) -> str:
    """Render one form field as a Carbon-classed input. Mirrors the
    legacy ``_render_field_input`` but reads the new ``choices`` and
    ``description`` attributes and tags inputs with ``bx--*`` classes.
    """
    from harness.web_forms import (
        FORM_KIND_CHECKBOX, FORM_KIND_JSON_DICT, FORM_KIND_JSON_LIST,
        FORM_KIND_NUMBER_FLOAT, FORM_KIND_NUMBER_INT, FORM_KIND_SELECT,
        FORM_KIND_TEXTAREA,
    )
    name_attr = html.escape(field.dotted_key)
    current = field.current_value
    if field.kind == FORM_KIND_CHECKBOX:
        checked = "checked" if current else ""
        return (
            f"<input type='checkbox' name='{name_attr}' value='1' {checked}> "
            f"<span class='muted'>(toggle)</span>"
        )
    if field.kind == FORM_KIND_SELECT:
        opts = []
        for choice in (field.choices or ()):
            sel = "selected" if str(current) == choice else ""
            opts.append(f"<option value='{html.escape(choice)}' {sel}>{html.escape(choice)}</option>")
        return f"<select class='bx--select-input' name='{name_attr}'>" + "".join(opts) + "</select>"
    if field.kind == FORM_KIND_NUMBER_INT:
        val = "" if current is None else html.escape(str(current))
        return f"<input class='bx--text-input' type='number' step='1' name='{name_attr}' value='{val}'>"
    if field.kind == FORM_KIND_NUMBER_FLOAT:
        val = "" if current is None else html.escape(str(current))
        return f"<input class='bx--text-input' type='number' step='any' name='{name_attr}' value='{val}'>"
    if field.kind in (FORM_KIND_JSON_LIST, FORM_KIND_JSON_DICT, FORM_KIND_TEXTAREA):
        if current is None:
            val = ""
        elif isinstance(current, str):
            val = current
        else:
            try:
                val = json.dumps(current, indent=2)
            except (TypeError, ValueError):
                val = str(current)
        return (
            f"<textarea class='bx--text-area' name='{name_attr}' rows='4' "
            f"style='font-family:monospace; width:100%'>{html.escape(val)}</textarea>"
        )
    # text / fallback.
    val = "" if current is None else html.escape(str(current))
    return f"<input class='bx--text-input' type='text' name='{name_attr}' value='{val}'>"


def _render_configure_harness(cfg: DashboardConfig) -> str:
    if not cfg.writes_enabled:
        return (
            "<div class='card'><p class='muted'>Writes are disabled "
            "by <code>dashboard.writes_enabled: false</code> in "
            "<code>config.json</code>. Current values are still "
            "viewable read-only at <a href='/config'>Configuration (raw)</a>.</p></div>"
        )

    from harness.web_forms import all_sections
    # Load the live config so the form pre-populates with current values.
    current_config: dict[str, Any] = {}
    try:
        config_path = cfg.config_path or _config_file_path(cfg)
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        current_config = {}

    csrf_token = resolve_csrf_token(cfg) or ""
    sections = all_sections(current_config=current_config)
    panels = []
    for sec in sections:
        if not sec.fields:
            # Section without renderable fields — show a one-line note so
            # the operator knows it exists but isn't web-editable.
            panels.append(
                f"<li class='bx--accordion__item'>"
                f"<button class='bx--accordion__heading' aria-expanded='false'>"
                f"<span class='bx--accordion__title'>{html.escape(sec.section)}</span>"
                f"</button>"
                f"<div class='bx--accordion__content'>"
                f"<p class='muted'>No web-editable fields registered for this section. "
                f"Edit <code>config.json</code> directly.</p></div></li>"
            )
            continue
        rows = []
        for f in sec.fields:
            description = html.escape(f.description) if f.description else "<span class='muted'>—</span>"
            input_html = _render_field_input_new(f)
            rows.append(
                f"<tr>"
                f"<td style='width:25%'><label class='bx--label'>{html.escape(f.name)}</label></td>"
                f"<td style='width:45%' class='muted'>{description}</td>"
                f"<td style='width:30%'>{input_html}</td>"
                f"</tr>"
            )
        panel = (
            f"<li class='bx--accordion__item'>"
            f"<button class='bx--accordion__heading' aria-expanded='false'>"
            f"<span class='bx--accordion__title'>{html.escape(sec.section)} "
            f"<span class='muted'>({len(sec.fields)} field{'s' if len(sec.fields) != 1 else ''})</span>"
            f"</span></button>"
            f"<div class='bx--accordion__content'>"
            f"<form method='post' action='/config/{html.escape(sec.section)}'>"
            f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
            f"<table>"
            f"<tr><th>Key</th><th>Meaning</th><th>Value</th></tr>"
            + "".join(rows) +
            f"</table>"
            f"<p><button class='bx--btn bx--btn--primary' type='submit'>Save {html.escape(sec.section)}</button></p>"
            f"</form></div></li>"
        )
        panels.append(panel)

    return (
        "<div class='card'>"
        "<h2>Configuration sections</h2>"
        "<p class='muted'>Each section is one top-level key in <code>config.json</code>. "
        "Save commits atomically and re-validates through the strict validator before landing.</p>"
        f"<ul class='bx--accordion'>{''.join(panels)}</ul>"
        "</div>"
        "<div class='card'>"
        "<p class='muted'>Deployment defaults live in <code>config/deployment.json</code> — "
        "edit directly until web editing lands for that file.</p>"
        "</div>"
        # Minimal accordion JS — toggles aria-expanded + a CSS class. Carbon's
        # bundled JS would also work but this is one inline handler.
        "<script>"
        "(function(){"
        "var btns = document.querySelectorAll('.bx--accordion__heading');"
        "btns.forEach(function(b){"
        "  b.addEventListener('click', function(){"
        "    var open = b.getAttribute('aria-expanded') === 'true';"
        "    b.setAttribute('aria-expanded', open ? 'false' : 'true');"
        "    var content = b.nextElementSibling;"
        "    if (content) content.style.display = open ? 'none' : 'block';"
        "  });"
        "  var content = b.nextElementSibling;"
        "  if (content) content.style.display = 'none';"
        "});"
        "})();"
        "</script>"
    )


_DASHBOARD_TILES: tuple[tuple[str, str, str], ...] = (
    # (title, description, href)
    ("View Status", "Day / week / month summary plus what's running right now.", "/status"),
    ("Cost burn-down", "Cumulative spend and per-call cost across every session.", "/cost"),
    ("Sessions list", "Every harness session on disk with exit code and token totals.", "/sessions"),
    ("Schedule history", "Past runs from the cron-driven scheduled-job daemon.", "/schedule"),
    ("Repo index", "Status of the semantic retrieval index per workspace.", "/index"),
    ("Memory list", "Per-repo memory files appended after each session.", "/memory"),
    ("Live runs", "Currently-running processes spawned from this dashboard.", "/live"),
    ("Configuration (raw)", "Section-by-section view of the legacy config form.", "/config"),
)


def _render_dashboards_landing(cfg: DashboardConfig) -> str:
    tiles = []
    for title, desc, href in _DASHBOARD_TILES:
        tiles.append(
            f"<div class='dash-tile'>"
            f"<a href='{html.escape(href)}'>"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(desc)}</p>"
            f"</a></div>"
        )
    return (
        "<div class='card'>"
        "<h2>Available dashboards</h2>"
        f"<div class='tile-grid'>{''.join(tiles)}</div>"
        "</div>"
    )


_DOC_ALLOWED_EXTENSIONS = (".md", ".txt")


def _resolve_docs_dir(cfg: DashboardConfig) -> str:
    """Resolve ``cfg.docs_dir`` to an absolute path, falling back to
    ``<repo_root>/docs`` (the harness package's own docs folder) when
    the operator hasn't customised it. The fallback lets a freshly
    installed harness show its shipped docs without configuration.
    """
    raw = cfg.docs_dir.strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    # Package install lives at .../harness/dashboard.py — the repo's
    # docs/ folder sits one level above.
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(package_dir, os.pardir, "docs"))


def list_docs(cfg: DashboardConfig) -> list[dict[str, Any]]:
    """Files in the configured ``docs_dir`` (recursive, one level deep
    of subdirs at most). Filters to the allowed extensions. Returns
    list of ``{name, size, mtime, relpath}`` sorted by name."""
    docs_dir = _resolve_docs_dir(cfg)
    if not os.path.isdir(docs_dir):
        return []
    out: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(docs_dir):
        for filename in files:
            if not filename.lower().endswith(_DOC_ALLOWED_EXTENSIONS):
                continue
            full = os.path.join(root, filename)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, docs_dir)
            out.append({
                "name": filename,
                "relpath": rel,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    out.sort(key=lambda r: r["relpath"].lower())
    return out


def read_doc_file(cfg: DashboardConfig, relpath: str) -> Optional[tuple[str, str]]:
    """Read a doc file inside ``docs_dir`` with strict path-traversal
    guards. Returns ``(content, extension)`` or ``None`` if the file
    is missing, outside the docs root, or has a disallowed extension.
    The caller decides how to render based on the extension.
    """
    if not relpath or "\x00" in relpath:
        return None
    docs_dir = os.path.realpath(_resolve_docs_dir(cfg))
    candidate = os.path.realpath(os.path.join(docs_dir, relpath))
    # Containment check — refuse anything that escapes docs_dir, even
    # via symlinks. ``commonpath`` would also work; ``startswith`` with
    # a trailing separator is enough here.
    if not (candidate == docs_dir or candidate.startswith(docs_dir + os.sep)):
        return None
    ext = os.path.splitext(candidate)[1].lower()
    if ext not in _DOC_ALLOWED_EXTENSIONS:
        return None
    if not os.path.isfile(candidate):
        return None
    try:
        with open(candidate, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), ext
    except OSError:
        return None


def render_markdown_minimal(text: str) -> str:
    """A small, stdlib-only Markdown → HTML converter for the docs
    viewer. Handles ATX headings (``# h1`` … ``###### h6``), fenced
    code blocks (```` ``` ````), unordered/ordered lists, inline
    ``code``, ``**bold**``, ``*italic*``, ``[link](url)``, blockquotes,
    horizontal rules, and paragraphs. Escapes all user content before
    inline rendering. Not CommonMark-strict; sufficient for the docs
    we ship.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    code_buf: list[str] = []
    code_lang = ""
    list_kind: Optional[str] = None  # "ul" | "ol" | None
    para_buf: list[str] = []

    def flush_para() -> None:
        if para_buf:
            out.append("<p>" + _md_inline(" ".join(para_buf)) + "</p>")
            para_buf.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if list_kind is not None:
            out.append(f"</{list_kind}>")
            list_kind = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if in_code:
            if line.strip().startswith("```"):
                escaped = html.escape("\n".join(code_buf))
                lang = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                out.append(f"<pre><code{lang}>{escaped}</code></pre>")
                code_buf.clear()
                code_lang = ""
                in_code = False
            else:
                code_buf.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            flush_list()
            in_code = True
            code_lang = stripped[3:].strip()
            continue
        if not stripped:
            flush_para()
            flush_list()
            continue
        if stripped in ("---", "***", "___"):
            flush_para()
            flush_list()
            out.append("<hr>")
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_para()
            flush_list()
            level = len(heading_match.group(1))
            out.append(f"<h{level}>{_md_inline(heading_match.group(2))}</h{level}>")
            continue
        ul_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        ol_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ul_match:
            flush_para()
            if list_kind != "ul":
                flush_list()
                out.append("<ul>")
                list_kind = "ul"
            out.append(f"<li>{_md_inline(ul_match.group(1))}</li>")
            continue
        if ol_match:
            flush_para()
            if list_kind != "ol":
                flush_list()
                out.append("<ol>")
                list_kind = "ol"
            out.append(f"<li>{_md_inline(ol_match.group(2))}</li>")
            continue
        if stripped.startswith(">"):
            flush_para()
            flush_list()
            out.append(f"<blockquote>{_md_inline(stripped.lstrip('>').strip())}</blockquote>")
            continue
        flush_list()
        para_buf.append(stripped)
    if in_code:  # unterminated fence — render what we have
        escaped = html.escape("\n".join(code_buf))
        out.append(f"<pre><code>{escaped}</code></pre>")
    flush_para()
    flush_list()
    return "\n".join(out)


def _md_inline(text: str) -> str:
    """Inline markdown: escape first, then apply code/bold/italic/link
    on the escaped string. Order matters — code spans first so backtick
    content isn't re-interpreted."""
    escaped = html.escape(text)
    # Inline `code`. Capture is greedy-safe because we limit to backtick
    # pairs on the same line.
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    # Links [text](url) — only allow http(s), relative, and anchor URLs
    # to avoid javascript: payloads.
    def _link(m: "re.Match[str]") -> str:
        text_part, url = m.group(1), m.group(2)
        if not re.match(r"^(https?:|/|#|[A-Za-z0-9_./\-]+)", url):
            return m.group(0)
        return f'<a href="{url}">{text_part}</a>'
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, escaped)
    # Bold **x** before italic *x* so ** doesn't get consumed as italic.
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def _fmt_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} GiB"


def _render_docs_landing(cfg: DashboardConfig) -> str:
    docs = list_docs(cfg)
    docs_dir = _resolve_docs_dir(cfg)
    if not docs:
        return (
            f"<div class='card'>"
            f"<h2>Documents</h2>"
            f"<p class='muted'>No documents found in <code>{_esc(docs_dir)}</code>. "
            f"Set <code>dashboard.docs_dir</code> in config.json to point at a folder of "
            f"<code>.md</code> or <code>.txt</code> files.</p>"
            f"</div>"
        )
    from datetime import datetime, timezone
    rows = []
    for doc in docs:
        modified = datetime.fromtimestamp(doc["mtime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        href = "/docs/" + urllib.parse.quote(doc["relpath"])
        rows.append(
            f"<tr>"
            f"<td><a href='{href}'>{_esc(doc['relpath'])}</a></td>"
            f"<td>{_fmt_size(int(doc['size']))}</td>"
            f"<td>{modified}</td>"
            f"</tr>"
        )
    return (
        f"<div class='card'>"
        f"<h2>Documents</h2>"
        f"<p class='muted'>Source: <code>{_esc(docs_dir)}</code></p>"
        "<table><tr><th>Document</th><th>Size</th><th>Modified (UTC)</th></tr>"
        + "".join(rows) +
        "</table></div>"
    )


def _render_docs_file(cfg: DashboardConfig, relpath: str) -> tuple[int, str]:
    result = read_doc_file(cfg, relpath)
    if result is None:
        return 404, "<p class='fail'>Document not found.</p>"
    content, ext = result
    if ext == ".md":
        rendered = render_markdown_minimal(content)
        body = f"<div class='card markdown-body'>{rendered}</div>"
    else:
        body = f"<div class='card'><pre>{html.escape(content)}</pre></div>"
    crumb = (
        f"<p class='muted'><a href='/docs'>← All documents</a> · "
        f"<code>{_esc(relpath)}</code></p>"
    )
    return 200, crumb + body


_ROUTES: list[Route] = [
    (re.compile(r"^/?$"), _route_root),
    (re.compile(r"^/sessions/?$"), _route_sessions),
    (re.compile(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/?$"), _route_session_detail),
    (re.compile(r"^/cost/?$"), _route_cost),
    (re.compile(r"^/api/cost-burn/?$"), _route_api_cost_burn),
    (re.compile(r"^/schedule/?$"), _route_schedule),
    (re.compile(r"^/index/?$"), _route_index),
    (re.compile(r"^/memory/?$"), _route_memory),
    (re.compile(r"^/memory/(?P<name>[A-Za-z0-9_.\-]+\.md)$"), _route_memory_file),
    # Carbon shell — 5 new top-level pages. Each renders a stub in Phase 1.
    (re.compile(r"^/status/?$"), _route_status),
    (re.compile(r"^/run/?$"), _route_run),
    (re.compile(r"^/config-ui/?$"), _route_configure_harness),
    (re.compile(r"^/dashboards/?$"), _route_dashboards),
    (re.compile(r"^/docs/?$"), _route_docs),
    # Doc detail — relpath allows subdirs (e.g. notes/a.md) but the
    # read_doc_file containment check rejects traversal regardless.
    (re.compile(r"^/docs/(?P<relpath>[A-Za-z0-9_./\-]+\.(?:md|txt))$"), _route_docs_file),
]


def dispatch(
    cfg: DashboardConfig, path: str,
) -> tuple[int, str, str]:
    """Pure function: given a URL path, return (status, content_type, body).

    Exposed so tests can exercise routes without standing up an actual
    HTTP server.
    """
    parsed = urllib.parse.urlparse(path)
    decoded = urllib.parse.unquote(parsed.path)
    for pattern, handler in _ROUTES:
        match = pattern.match(decoded)
        if match:
            return handler(cfg, match.groupdict())
    return 404, "text/html; charset=utf-8", _layout(
        "Not found",
        "<p class='fail'>404 — no route matches this path.</p>",
        cfg,
    )


# ---------------------------------------------------------------------------
# 6. Server
# ---------------------------------------------------------------------------

def make_request_handler(
    cfg: DashboardConfig, expected_token: Optional[str],
    *,
    csrf_token: Optional[str] = None,
) -> type[http.server.BaseHTTPRequestHandler]:
    """Construct a request handler class closed over the live config
    and tokens. Factory rather than a class attribute so tests can
    spin up multiple handlers with different settings."""

    if csrf_token is None:
        csrf_token = resolve_csrf_token(cfg)

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Quieter access logs — the harness's logging subsystem owns
        # the noisy channel.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("[dashboard] %s - %s", self.client_address[0], format % args)

        def _send(self, status: int, content_type: str, body: str,
                   *, extra_headers: Optional[dict[str, str]] = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ---- Auth helpers -------------------------------------------------

        def _is_authed(self) -> tuple[bool, str]:
            auth = check_auth(expected_token, self.headers.get("Authorization"))
            return auth.ok, auth.detail

        def _is_csrf_ok(self) -> tuple[bool, str]:
            cookie = None
            for raw in (self.headers.get("Cookie") or "").split(";"):
                if "=" not in raw:
                    continue
                k, v = raw.split("=", 1)
                if k.strip() == "csrf_token":
                    cookie = v.strip()
                    break
            outcome = check_csrf(csrf_token, self.headers.get("X-CSRF-Token"), cookie)
            return outcome.ok, outcome.detail

        def _csrf_set_cookie_header(self) -> Optional[str]:
            if csrf_token is None:
                return None
            # SameSite=Strict + HttpOnly off (JS needs to read it for the
            # double-submit pattern). For host-only localhost this is fine.
            return f"csrf_token={csrf_token}; Path=/; SameSite=Strict"

        # ---- GET ----------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            ok, detail = self._is_authed()
            if not ok:
                self._send(401, "text/plain; charset=utf-8", f"401 unauthorized: {detail}\n")
                return
            try:
                status, ctype, body = dispatch(cfg, self.path)
            except Exception:  # noqa: BLE001
                logger.exception("[dashboard] handler error")
                status, ctype, body = (500, "text/plain; charset=utf-8",
                                       "500 internal error\n")
            # SSE sentinel — stream instead of buffering.
            if ctype == "text/event-stream" and body.startswith("__SSE__"):
                session_id = body[len("__SSE__"):]
                self._stream_sse(session_id)
                return
            # 302 redirect sentinel (used by `/` → `/status`).
            if status == 302 and body.startswith(_REDIRECT_SENTINEL):
                target = body[len(_REDIRECT_SENTINEL):] or "/status"
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            extra = {}
            cookie = self._csrf_set_cookie_header()
            if cookie:
                extra["Set-Cookie"] = cookie
            self._send(status, ctype, body, extra_headers=extra)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        # ---- SSE ----------------------------------------------------------

        def _stream_sse(self, session_id: str) -> None:
            log_path = os.path.join(os.path.expanduser(cfg.log_dir), f"{session_id}.jsonl")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for evt in tail_session_events(log_path, follow=True, max_lines=2000):
                    line = "data: " + json.dumps(evt, default=str) + "\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                # Close marker.
                self.wfile.write(b"event: close\ndata:\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # noqa: BLE001
                logger.exception("[dashboard] SSE stream error")

        # ---- POST ---------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802 — stdlib API
            # /hitl/webhook is the only POST that uses a shared secret
            # rather than CSRF (the harness's HttpChannel POSTs to it).
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            if path == "/hitl/webhook" or path.startswith("/hitl/webhook?"):
                self._handle_hitl_webhook(parsed)
                return

            # All other writes: bearer + CSRF.
            ok, detail = self._is_authed()
            if not ok:
                self._send(401, "text/plain", f"401: {detail}\n")
                return
            ok, detail = self._is_csrf_ok()
            if not ok:
                self._send(403, "text/plain", f"403: {detail}\n")
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            form = _parse_form_body(raw)

            # Route the POST to the right handler.
            self._dispatch_write(path, form)

        def _dispatch_write(self, path: str, form: dict[str, Any]) -> None:
            # /config/<section>
            m = re.match(r"^/config/(?P<section>[A-Za-z0-9_]+)/?$", path)
            if m:
                self._handle_config_save(m.group("section"), form)
                return
            # /memory/<name>
            m = re.match(r"^/memory/(?P<name>[A-Za-z0-9_.\-]+\.md)$", path)
            if m:
                self._handle_memory_save(m.group("name"), form)
                return
            # /run/now
            if path == "/run/now":
                self._handle_run_now(form)
                return
            # /run/schedule
            if path == "/run/schedule":
                self._handle_run_schedule(form)
                return
            # /sessions/<sid>/cancel
            m = re.match(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/cancel/?$", path)
            if m:
                self._handle_cancel(m.group("sid"))
                return
            # /sessions/<sid>/note
            m = re.match(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/note/?$", path)
            if m:
                self._handle_note(m.group("sid"), form)
                return
            # /sessions/<sid>/hitl/answer
            m = re.match(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/hitl/answer/?$", path)
            if m:
                self._handle_hitl_answer(m.group("sid"), form)
                return
            # /schedule/jobs
            if path in ("/schedule/jobs", "/schedule/jobs/"):
                self._handle_schedule_add(form)
                return
            self._send(404, "text/plain", "404 not found\n")

        # ---- Write handlers ----------------------------------------------

        def _handle_config_save(self, section_name: str, form: dict[str, Any]) -> None:
            from harness.web_forms import build_section, parse_section_post
            current = read_config_file(cfg)
            section = build_section(section_name, current_config=current)
            parsed, errors = parse_section_post(section, form)
            if errors:
                err_map = {e.dotted_key: e.message for e in errors}
                body = _render_config_section(
                    cfg, section_name, csrf_token=csrf_token, errors=err_map,
                )
                self._send(400, "text/html; charset=utf-8",
                           _layout(f"Config · {section_name}", body, cfg))
                return
            ok, msg = write_config_section_atomic(cfg, section_name, parsed)
            if not ok:
                err_map = {section.fields[0].dotted_key: msg} if section.fields else {}
                body = _render_config_section(
                    cfg, section_name, csrf_token=csrf_token,
                    errors=err_map, flash=f"save failed: {msg}",
                )
                self._send(400, "text/html; charset=utf-8",
                           _layout(f"Config · {section_name}", body, cfg))
                return
            try:
                append_audit(db_path=cfg.web_db_path, action="config_save",
                             target=section_name, detail=json.dumps(parsed, default=str))
            except Exception:  # noqa: BLE001
                pass
            body = _render_config_section(
                cfg, section_name, csrf_token=csrf_token,
                flash="Saved.",
            )
            self._send(200, "text/html; charset=utf-8",
                       _layout(f"Config · {section_name}", body, cfg))

        def _handle_memory_save(self, name: str, form: dict[str, Any]) -> None:
            content = form.get("content")
            if isinstance(content, list):
                content = content[-1] if content else ""
            ok, msg = write_memory_file(cfg, name, str(content or ""))
            status = 200 if ok else 400
            flash = "Saved." if ok else f"Save failed: {msg}"
            body_status, body = _render_memory_edit(cfg, name, csrf_token, flash=flash)
            self._send(status if body_status == 200 else body_status,
                       "text/html; charset=utf-8",
                       _layout(f"Memory · {name}", body, cfg))

        def _handle_run_now(self, form: dict[str, Any]) -> None:
            workspace = str(form.get("workspace") or "").strip()
            prompt = str(form.get("prompt") or "").strip()
            extra = str(form.get("extra_args") or "").strip()
            if not workspace or not prompt:
                self._send(400, "text/plain", "workspace and prompt required\n")
                return
            extra_args = extra.split() if extra else []
            try:
                wp = spawn_harness_run(
                    cfg, workspace=workspace, prompt=prompt,
                    extra_args=extra_args,
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, "text/plain", f"spawn failed: {exc}\n")
                return
            self.send_response(303)
            self.send_header("Location", f"/sessions/{wp.session_id}")
            self.end_headers()

        def _handle_run_schedule(self, form: dict[str, Any]) -> None:
            from datetime import datetime as _dt
            workspace = str(form.get("workspace") or "").strip()
            prompt = str(form.get("prompt") or "").strip()
            fire_raw = str(form.get("fire_at_utc") or "").strip()
            name = str(form.get("name") or "web-oneshot").strip() or "web-oneshot"
            extra = str(form.get("extra_args") or "").strip()
            if not workspace or not prompt or not fire_raw:
                self._send(400, "text/plain", "workspace, prompt, fire_at_utc required\n")
                return
            try:
                fire_at = _dt.fromisoformat(fire_raw.replace("Z", "+00:00"))
                if fire_at.tzinfo is None:
                    from datetime import timezone as _tz
                    fire_at = fire_at.replace(tzinfo=_tz.utc)
            except ValueError as exc:
                self._send(400, "text/plain", f"fire_at_utc invalid: {exc}\n")
                return
            extra_args = extra.split() if extra else []
            try:
                row_id = add_oneshot_job(
                    db_path=cfg.web_db_path, name=name,
                    fire_at_utc=fire_at, workspace=workspace,
                    prompt=prompt, harness_args=extra_args,
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, "text/plain", f"enqueue failed: {exc}\n")
                return
            try:
                append_audit(db_path=cfg.web_db_path, action="run_schedule",
                             target=str(row_id), detail=f"fire_at={fire_at.isoformat()}")
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/schedule")
            self.end_headers()

        def _handle_cancel(self, session_id: str) -> None:
            ok = cancel_session(session_id)
            if not ok:
                self._send(404, "text/plain", "no live process for that session\n")
                return
            try:
                append_audit(db_path=cfg.web_db_path, action="cancel",
                             target=session_id, detail="SIGTERM")
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/live")
            self.end_headers()

        def _handle_note(self, session_id: str, form: dict[str, Any]) -> None:
            note = str(form.get("note") or "").strip()
            if not note:
                self._send(400, "text/plain", "note required\n")
                return
            try:
                queue_chat_note(db_path=cfg.web_db_path, session_id=session_id, note=note)
            except Exception as exc:  # noqa: BLE001
                self._send(500, "text/plain", f"queue failed: {exc}\n")
                return
            self.send_response(303)
            self.send_header("Location", f"/sessions/{session_id}")
            self.end_headers()

        def _handle_hitl_answer(self, session_id: str, form: dict[str, Any]) -> None:
            request_id = str(form.get("request_id") or "").strip()
            choice = str(form.get("choice") or "").strip()
            extra_notes = str(form.get("extra_notes") or "").strip()
            if not request_id or not choice:
                self._send(400, "text/plain", "request_id + choice required\n")
                return
            response = {"choice": choice, "extra_notes": extra_notes}
            # Also drain any queued chat notes for this session and prepend.
            try:
                pending = consume_chat_notes(db_path=cfg.web_db_path, session_id=session_id)
                if pending:
                    note_block = "\n".join(pending)
                    response["extra_notes"] = (
                        (extra_notes + "\n\n" + note_block).strip()
                        if extra_notes else note_block
                    )
            except Exception:  # noqa: BLE001
                pass
            ok = get_hitl_queue().answer(request_id, response)
            if not ok:
                self._send(404, "text/plain", "no pending HITL with that request_id\n")
                return
            try:
                append_audit(db_path=cfg.web_db_path, action="hitl_answer",
                             target=session_id, detail=f"request_id={request_id} choice={choice}")
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", f"/sessions/{session_id}")
            self.end_headers()

        def _handle_schedule_add(self, form: dict[str, Any]) -> None:
            job = {
                "name": str(form.get("name") or "").strip(),
                "schedule": str(form.get("schedule") or "").strip(),
                "workspace": str(form.get("workspace") or "").strip(),
                "prompt": str(form.get("prompt") or ""),
                "enabled": form.get("enabled") in ("on", "true", "1", True),
            }
            ok, msg = add_schedule_job_to_config(cfg, job=job)
            if not ok:
                self._send(400, "text/plain", msg + "\n")
                return
            try:
                append_audit(db_path=cfg.web_db_path, action="schedule_add",
                             target=job["name"], detail=msg)
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/schedule")
            self.end_headers()

        # ---- HITL webhook (the harness POSTs here; the UI is on the
        # other side) ------------------------------------------------------

        def _handle_hitl_webhook(self, parsed_url: urllib.parse.ParseResult) -> None:
            # Optional shared-secret check via query param or header.
            secret_required = cfg.hitl_webhook_secret
            if secret_required:
                qs = urllib.parse.parse_qs(parsed_url.query)
                provided = (
                    self.headers.get("X-HITL-Secret")
                    or (qs.get("secret") or [None])[0]
                )
                if not provided or not hmac.compare_digest(provided, secret_required):
                    self._send(401, "text/plain", "401 hitl webhook secret mismatch\n")
                    return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                prompt = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError):
                prompt = {"raw": raw.decode("utf-8", errors="replace")}
            qs = urllib.parse.parse_qs(parsed_url.query)
            session_id = (qs.get("session") or [""])[0] or str(prompt.get("session_id") or "unknown")
            import uuid as _uuid
            request_id = (
                str(prompt.get("request_id"))
                if prompt.get("request_id")
                else _uuid.uuid4().hex
            )
            entry = get_hitl_queue().register_pending(
                request_id=request_id, session_id=session_id, prompt=prompt,
            )
            # Block on the operator's answer. Default 10 minute cap so
            # an abandoned UI session doesn't keep the harness pinned
            # forever.
            timeout = float((cfg.csrf_token_env and 1) or 1) * 600.0
            if not entry.event.wait(timeout=timeout):
                self._send(504, "text/plain", "504 operator did not respond in time\n")
                return
            answer = get_hitl_queue().pop_response(request_id)
            data = json.dumps(answer).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return _Handler


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass
class _ServerHandle:
    """Tiny wrapper so callers can shut the server down from another
    thread without poking the stdlib types directly.

    ``csrf_token`` is the value the server expects on every write
    request when ``DashboardConfig.writes_enabled`` is True; ``None``
    otherwise. Tests use this to drive write paths without round-
    tripping through a real GET first."""

    server: _ThreadingServer
    thread: threading.Thread
    host: str
    port: int
    csrf_token: Optional[str] = None

    def shutdown(self) -> None:
        try:
            self.server.shutdown()
        finally:
            self.server.server_close()
        self.thread.join(timeout=5.0)


def start_server(
    cfg: DashboardConfig,
    *,
    blocking: bool = True,
) -> Optional[_ServerHandle]:
    """Start the dashboard's HTTP server. When ``blocking=True`` (the
    CLI's default), this runs ``serve_forever`` on the current thread
    and returns ``None`` when the operator Ctrl-C's it. When
    ``blocking=False``, the server starts on a background thread and a
    handle is returned for tests to shut down."""
    expected_token = resolve_expected_token(cfg)
    csrf = resolve_csrf_token(cfg)
    handler_class = make_request_handler(cfg, expected_token, csrf_token=csrf)
    server = _ThreadingServer((cfg.host, cfg.port), handler_class)
    if blocking:
        try:
            logger.info("[dashboard] listening on http://%s:%d/", cfg.host, cfg.port)
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("[dashboard] Ctrl-C received; shutting down.")
        finally:
            server.server_close()
        return None
    thread = threading.Thread(
        target=server.serve_forever, name="harness-dashboard", daemon=True,
    )
    thread.start()
    return _ServerHandle(
        server=server, thread=thread, host=cfg.host, port=cfg.port,
        csrf_token=csrf,
    )


# Tests reach into a few internals — re-export them so test files don't
# have to depend on underscore names.
ServerHandle = _ServerHandle


# ===========================================================================
# Tier B + C extensions: writes, control plane, HITL bridge
# ===========================================================================

# Module-level shared state. Lives for the lifetime of the dashboard
# process. The dashboard server constructor seeds these and the request
# handlers consult them.

from harness.web_state import (  # noqa: E402  (intentional late import)
    HitlQueue,
    ProcessRegistry,
    WebProcess,
    add_oneshot_job,
    append_audit,
    consume_chat_notes,
    list_pending_oneshot_jobs,
    pending_chat_notes,
    queue_chat_note,
)

_process_registry: Optional[ProcessRegistry] = None
_hitl_queue: Optional[HitlQueue] = None


def get_process_registry() -> ProcessRegistry:
    global _process_registry
    if _process_registry is None:
        _process_registry = ProcessRegistry()
    return _process_registry


def get_hitl_queue() -> HitlQueue:
    global _hitl_queue
    if _hitl_queue is None:
        _hitl_queue = HitlQueue()
    return _hitl_queue


def reset_shared_state() -> None:
    """Test hook — drop the shared registry + queue so each test gets
    isolated state."""
    global _process_registry, _hitl_queue
    _process_registry = None
    _hitl_queue = None


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------

import secrets as _secrets  # noqa: E402


def resolve_csrf_token(cfg: DashboardConfig) -> Optional[str]:
    """Resolve the CSRF token the server expects on write requests.

    - When ``writes_enabled=False``: no CSRF; the value returned is
      ``None`` and write routes return 403 regardless.
    - When ``writes_enabled=True``:
        - If ``csrf_token_env`` names an env var with a non-empty
          value, use it (operator can pin the token across restarts).
        - Otherwise generate a fresh 32-byte hex token per server boot.
      Either way the value is exposed to the UI in a ``Set-Cookie``
      header on the first authed GET so JS can echo it back as
      ``X-CSRF-Token`` on subsequent POSTs.
    """
    if not cfg.writes_enabled:
        return None
    if cfg.csrf_token_env:
        value = os.environ.get(cfg.csrf_token_env, "")
        if value:
            return value
    return _secrets.token_hex(32)


def check_csrf(
    expected_token: Optional[str], header_value: Optional[str],
    cookie_value: Optional[str],
) -> AuthOutcome:
    """Double-submit cookie pattern: the value in the cookie must
    equal the value in the X-CSRF-Token header, and both must equal
    the server's expected token."""
    if expected_token is None:
        return AuthOutcome(ok=False, detail="writes disabled (dashboard.writes_enabled=false)")
    if not header_value:
        return AuthOutcome(ok=False, detail="missing X-CSRF-Token header")
    if not cookie_value:
        return AuthOutcome(ok=False, detail="missing csrf cookie")
    if not (hmac.compare_digest(header_value, expected_token)
            and hmac.compare_digest(cookie_value, expected_token)):
        return AuthOutcome(ok=False, detail="csrf token mismatch")
    return AuthOutcome(ok=True, detail="ok")


# ---------------------------------------------------------------------------
# Config file read/write (Tier B)
# ---------------------------------------------------------------------------

def _config_file_path(cfg: DashboardConfig) -> str:
    if cfg.config_path:
        return os.path.expanduser(cfg.config_path)
    # Match the harness's canonical path resolution.
    try:
        from harness.cli import _get_global_config_path
        return _get_global_config_path()
    except Exception:  # noqa: BLE001
        return os.path.expanduser("~/.harness/config.json")


def read_config_file(cfg: DashboardConfig) -> dict[str, Any]:
    path = _config_file_path(cfg)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("[dashboard] config read failed: %s", exc)
        return {}
    # Strip comment-only keys (the harness uses '_'-prefixed keys for docs).
    return {k: v for k, v in data.items() if not k.startswith("_")}


def write_config_section_atomic(
    cfg: DashboardConfig, section: str, new_section_value: Any,
) -> tuple[bool, str]:
    """Read the current config, replace ``section``, validate the
    merged result through the strict validator, write atomically.
    Returns ``(success, message)``. On validation failure the disk
    file is untouched."""
    path = _config_file_path(cfg)
    if not os.path.isfile(path):
        return False, f"config file not found at {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            full = json.load(f)
    except (OSError, ValueError) as exc:
        return False, f"could not read existing config: {exc}"
    full[section] = new_section_value
    # Strict validation BEFORE writing.
    try:
        from harness.cli import validate_config_strict, _strip_comments
        validate_config_strict(_strip_comments(dict(full)), source=path)
    except Exception as exc:  # noqa: BLE001
        return False, f"validation failed: {exc}"
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=4)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, f"write failed: {exc}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Memory file write (Tier B)
# ---------------------------------------------------------------------------

def write_memory_file(
    cfg: DashboardConfig, name: str, content: str,
) -> tuple[bool, str]:
    if not name or "/" in name or "\\" in name or ".." in name:
        return False, "invalid file name"
    if not name.endswith(".md"):
        return False, "memory files must end in .md"
    mem_dir = os.path.expanduser(cfg.memory_dir)
    os.makedirs(mem_dir, exist_ok=True)
    path = os.path.join(mem_dir, name)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, f"write failed: {exc}"
    return True, path


# ---------------------------------------------------------------------------
# Schedule-job CRUD against config.json (Tier B)
# ---------------------------------------------------------------------------

def add_schedule_job_to_config(
    cfg: DashboardConfig, *, job: dict[str, Any],
) -> tuple[bool, str]:
    """Append (or replace by name) a job in config.schedule.jobs, with
    validation through harness/schedule.py's parser. Atomic write."""
    from harness.schedule import parse_schedule
    try:
        parse_schedule(str(job.get("schedule") or ""))
    except ValueError as exc:
        return False, f"schedule rejected: {exc}"
    if not job.get("name"):
        return False, "job name required"
    if not job.get("workspace"):
        return False, "workspace required"
    full = read_config_file(cfg)
    schedule_section = dict(full.get("schedule") or {})
    jobs = list(schedule_section.get("jobs") or [])
    # Replace by name; preserves ordering for unchanged jobs.
    jobs = [j for j in jobs if isinstance(j, dict) and j.get("name") != job["name"]]
    jobs.append(job)
    schedule_section["jobs"] = jobs
    return write_config_section_atomic(cfg, "schedule", schedule_section)


def remove_schedule_job_from_config(
    cfg: DashboardConfig, name: str,
) -> tuple[bool, str]:
    full = read_config_file(cfg)
    schedule_section = dict(full.get("schedule") or {})
    jobs = [
        j for j in (schedule_section.get("jobs") or [])
        if isinstance(j, dict) and j.get("name") != name
    ]
    schedule_section["jobs"] = jobs
    return write_config_section_atomic(cfg, "schedule", schedule_section)


# ---------------------------------------------------------------------------
# Subprocess management (Tier C — Run now)
# ---------------------------------------------------------------------------

def spawn_harness_run(
    cfg: DashboardConfig,
    *,
    workspace: str,
    prompt: str,
    extra_args: Optional[list[str]] = None,
    harness_binary: str = "harness",
) -> WebProcess:
    """Spawn a `harness run` subprocess, register it, and return the
    :class:`WebProcess` handle. Sets ``HARNESS_HITL_WEBHOOK_URL`` so the
    harness's HttpChannel POSTs HITL prompts back to this dashboard.
    """
    import subprocess as _sub
    import uuid as _uuid

    session_id = f"web-{_uuid.uuid4().hex[:12]}"
    log_dir = os.path.expanduser(cfg.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session_id}.jsonl")
    # Pre-create the log file so the SSE stream's tail can start
    # before the harness has written anything.
    open(log_path, "a", encoding="utf-8").close()

    argv = [
        harness_binary, "run",
        "-r", workspace,
        "-p", prompt,
        "--session-id", session_id,
    ]
    argv += list(extra_args or [])

    env = dict(os.environ)
    env["HARNESS_HITL_WEBHOOK_URL"] = (
        f"http://{cfg.host}:{cfg.port}/hitl/webhook?session={session_id}"
    )
    if cfg.hitl_webhook_secret:
        env["HARNESS_HITL_WEBHOOK_SECRET"] = cfg.hitl_webhook_secret

    proc = _sub.Popen(
        argv,
        stdout=open(log_path + ".stdout", "ab"),
        stderr=_sub.STDOUT,
        env=env,
        start_new_session=True,
    )
    wp = WebProcess(
        session_id=session_id, pid=proc.pid, argv=argv,
        log_path=log_path, workspace_path=workspace, prompt=prompt,
        popen=proc,
    )
    get_process_registry().register(wp)

    # Background thread: wait on the process and mark terminated.
    def _watch():
        try:
            ec = proc.wait()
        except Exception:  # noqa: BLE001
            ec = -1
        get_process_registry().mark_terminated(session_id, int(ec or 0))
        try:
            append_audit(
                db_path=cfg.web_db_path, action="run_exit",
                target=session_id, detail=f"exit_code={ec}",
            )
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_watch, daemon=True, name=f"web-run-{session_id}").start()
    try:
        append_audit(
            db_path=cfg.web_db_path, action="run_now",
            target=session_id, detail=f"argv={' '.join(argv)}",
        )
    except Exception:  # noqa: BLE001
        pass
    return wp


def cancel_session(session_id: str) -> bool:
    """SIGTERM the process group for ``session_id``. Returns True when
    a signal was sent, False when no live process matches."""
    import signal as _signal
    reg = get_process_registry()
    entry = reg.get(session_id)
    if entry is None or not entry.is_running:
        return False
    try:
        os.killpg(os.getpgid(entry.pid), _signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# SSE event stream (Tier C)
# ---------------------------------------------------------------------------

def tail_session_events(
    log_path: str, *,
    max_lines: int = 5000,
    poll_interval: float = 0.5,
    follow: bool = True,
):
    """Generator that yields decoded JSON event lines as they appear in
    the log file. Stops when ``follow`` is False and we hit EOF, or
    when the caller breaks out."""
    pos = 0
    yielded = 0
    while True:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                line = f.readline()
                while line:
                    pos = f.tell()
                    if line.strip():
                        evt = _safe_json(line)
                        if evt is not None:
                            yield evt
                            yielded += 1
                            if yielded >= max_lines:
                                return
                    line = f.readline()
        except OSError:
            pass
        if not follow:
            return
        time.sleep(poll_interval)
        # Heuristic shutdown: if the process for this log is gone +
        # terminated, stop following after one more pass.
        try:
            reg = get_process_registry()
            for proc in reg.list_all():
                if proc.log_path == log_path and not proc.is_running:
                    # Drain remaining and exit.
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(pos)
                            for tail_line in f:
                                evt = _safe_json(tail_line)
                                if evt is not None:
                                    yield evt
                    except OSError:
                        pass
                    return
        except Exception:  # noqa: BLE001
            pass


import time  # noqa: E402  # imported here to keep section local


# ---------------------------------------------------------------------------
# Render — config view + edit form (Tier A + B)
# ---------------------------------------------------------------------------

def _render_config_index(cfg: DashboardConfig) -> str:
    from harness.web_forms import all_sections
    current = read_config_file(cfg)
    sections = all_sections(current_config=current)
    rows = []
    for s in sections:
        link = f"<a href='/config/{_esc(s.section)}'>{_esc(s.section)}</a>"
        count = len(s.fields)
        rows.append(
            f"<tr><td>{link}</td><td>{count} field(s)</td>"
            f"<td>{('editable' if cfg.writes_enabled else 'view only')}</td></tr>"
        )
    write_state = (
        "Writes are <strong>enabled</strong>. Forms can save changes."
        if cfg.writes_enabled else
        "Writes are disabled by <code>dashboard.writes_enabled: false</code> "
        "in <code>config.json</code>."
    )
    return (
        f"<div class='card'><p>{write_state}</p></div>"
        "<table><tr><th>section</th><th>fields</th><th>mode</th></tr>"
        + "".join(rows) + "</table>"
    )


def _render_field_input(f: Any, error: str = "") -> str:
    from harness.web_forms import (
        FORM_KIND_CHECKBOX,
        FORM_KIND_JSON_DICT,
        FORM_KIND_JSON_LIST,
        FORM_KIND_NUMBER_FLOAT,
        FORM_KIND_NUMBER_INT,
    )
    name = f.dotted_key
    cv = f.current_value
    err = f"<div class='fail'>{html.escape(error)}</div>" if error else ""
    if f.kind == FORM_KIND_CHECKBOX:
        checked = "checked" if cv else ""
        return (
            f"<label><input type='checkbox' name='{html.escape(name)}' {checked}> "
            f"{html.escape(f.name)}</label>{err}"
        )
    if f.kind == FORM_KIND_NUMBER_INT:
        val = "" if cv is None else str(cv)
        return (
            f"<label>{html.escape(f.name)}<br>"
            f"<input type='number' step='1' name='{html.escape(name)}' value='{html.escape(val)}'>"
            f"</label>{err}"
        )
    if f.kind == FORM_KIND_NUMBER_FLOAT:
        val = "" if cv is None else str(cv)
        return (
            f"<label>{html.escape(f.name)}<br>"
            f"<input type='number' step='any' name='{html.escape(name)}' value='{html.escape(val)}'>"
            f"</label>{err}"
        )
    if f.kind in (FORM_KIND_JSON_LIST, FORM_KIND_JSON_DICT):
        val = "" if cv is None else json.dumps(cv, indent=2)
        return (
            f"<label>{html.escape(f.name)} <em>(JSON)</em><br>"
            f"<textarea rows='4' cols='60' name='{html.escape(name)}'>"
            f"{html.escape(val)}</textarea></label>{err}"
        )
    # text fallback
    val = "" if cv is None else str(cv)
    return (
        f"<label>{html.escape(f.name)}<br>"
        f"<input type='text' name='{html.escape(name)}' value='{html.escape(val)}' style='width:60%'>"
        f"</label>{err}"
    )


def _render_config_section(
    cfg: DashboardConfig, section_name: str,
    *,
    csrf_token: Optional[str] = None,
    errors: Optional[dict[str, str]] = None,
    flash: str = "",
) -> str:
    from harness.web_forms import build_section
    current = read_config_file(cfg)
    section = build_section(section_name, current_config=current)
    if not section.fields:
        return (
            f"<p class='muted'>No editable fields registered for "
            f"<code>{_esc(section_name)}</code>. Either it has no nested "
            f"keys in the validator, or all entries are dynamic "
            f"(e.g. <code>models.*</code>).</p>"
        )
    errors = errors or {}
    inputs = []
    for f in section.fields:
        err = errors.get(f.dotted_key, "")
        inputs.append(f"<div class='field'>{_render_field_input(f, err)}</div>")
    if cfg.writes_enabled and csrf_token is not None:
        save = (
            f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
            "<p><button type='submit'>Save changes</button></p>"
        )
        form_open = f"<form method='post' action='/config/{_esc(section_name)}'>"
        form_close = "</form>"
    else:
        save = "<p class='muted'>Read-only (writes disabled).</p>"
        form_open = "<div>"
        form_close = "</div>"
    flash_html = f"<div class='card ok'>{html.escape(flash)}</div>" if flash else ""
    return (
        f"{flash_html}"
        f"{form_open}"
        f"<div class='card'><h3>{_esc(section_name)}</h3>"
        + "".join(inputs) + save + "</div>"
        + form_close
    )


# ---------------------------------------------------------------------------
# Render — Live + memory edit + new run form
# ---------------------------------------------------------------------------

def _render_live(cfg: DashboardConfig) -> str:
    reg = get_process_registry()
    live = reg.list_running()
    recent = [p for p in reg.list_all() if not p.is_running][:10]
    rows = []
    for p in live:
        rows.append(
            f"<tr><td><a href='/sessions/{_esc(p.session_id)}'>{_esc(p.session_id)}</a></td>"
            f"<td>{_esc(p.workspace_path)}</td>"
            f"<td>{_esc(p.prompt[:80])}</td>"
            f"<td><span class='ok'>running</span></td>"
            f"<td>"
            f"<form method='post' action='/sessions/{_esc(p.session_id)}/cancel' style='display:inline'>"
            f"<input type='hidden' name='csrf_token' value=''>"
            f"<button>Cancel</button></form></td></tr>"
        )
    for p in recent:
        cls = "ok" if p.exit_code == 0 else "fail"
        status = "success" if p.exit_code == 0 else f"exit {p.exit_code}"
        rows.append(
            f"<tr><td><a href='/sessions/{_esc(p.session_id)}'>{_esc(p.session_id)}</a></td>"
            f"<td>{_esc(p.workspace_path)}</td>"
            f"<td>{_esc(p.prompt[:80])}</td>"
            f"<td class='{cls}'>{status}</td><td>—</td></tr>"
        )
    table = (
        "<table><tr><th>session</th><th>workspace</th><th>prompt</th>"
        "<th>status</th><th>actions</th></tr>"
        + "".join(rows) + "</table>"
    ) if rows else "<p class='muted'>No runs in flight.</p>"
    new_run_button = (
        "<p><a href='/run/new'><button>+ New run</button></a></p>"
        if cfg.writes_enabled else ""
    )
    return new_run_button + table


def _render_memory_edit(cfg: DashboardConfig, name: str, csrf_token: Optional[str], flash: str = "") -> tuple[int, str]:
    content = read_memory_file(cfg, name)
    if content is None:
        return 404, "<p class='fail'>Memory file not found.</p>"
    flash_html = f"<div class='card ok'>{html.escape(flash)}</div>" if flash else ""
    if cfg.writes_enabled and csrf_token is not None:
        body = (
            f"{flash_html}"
            f"<form method='post' action='/memory/{html.escape(name)}'>"
            f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
            f"<div class='card'><h3>{_esc(name)}</h3>"
            f"<textarea name='content' rows='30' cols='100'>{html.escape(content)}</textarea>"
            "<p><button>Save</button></p></div></form>"
        )
    else:
        body = f"<div class='card'><h3>{_esc(name)}</h3><pre>{html.escape(content)}</pre></div>"
    return 200, body


def _render_run_new(cfg: DashboardConfig, csrf_token: Optional[str], flash: str = "") -> str:
    if not cfg.writes_enabled or csrf_token is None:
        return (
            "<p class='muted'>Writes are disabled by "
            "<code>dashboard.writes_enabled: false</code> in "
            "<code>config.json</code>; cannot start runs from the web.</p>"
        )
    flash_html = f"<div class='card ok'>{html.escape(flash)}</div>" if flash else ""
    return (
        f"{flash_html}"
        "<form method='post' action='/run/now'>"
        f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
        "<div class='card'><h3>Start a run</h3>"
        "<label>Workspace path<br><input type='text' name='workspace' style='width:80%' required></label><br><br>"
        "<label>Prompt<br><textarea name='prompt' rows='4' cols='80' required></textarea></label><br><br>"
        "<label>Extra args (space-separated)<br>"
        "<input type='text' name='extra_args' placeholder='--new_build=false --allow-network' style='width:80%'></label>"
        "<p><button type='submit'>Run now</button></p></div></form>"
        "<form method='post' action='/run/schedule'>"
        f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
        "<div class='card'><h3>Schedule for later</h3>"
        "<label>Workspace path<br><input type='text' name='workspace' style='width:80%' required></label><br><br>"
        "<label>Prompt<br><textarea name='prompt' rows='4' cols='80' required></textarea></label><br><br>"
        "<label>Fire at (UTC, ISO8601 e.g. 2026-06-20T03:00:00Z)<br>"
        "<input type='text' name='fire_at_utc' placeholder='2026-06-20T03:00:00Z' required></label><br><br>"
        "<label>Job name<br><input type='text' name='name' placeholder='nightly retest'></label><br><br>"
        "<label>Extra args (space-separated)<br>"
        "<input type='text' name='extra_args' style='width:80%'></label>"
        "<p><button type='submit'>Schedule</button></p></div></form>"
    )


def _render_session_with_hitl(cfg: DashboardConfig, session_id: str) -> str:
    """Detailed per-session view that includes pending HITL prompts +
    a queued-notes panel when writes are enabled."""
    log_path = os.path.join(os.path.expanduser(cfg.log_dir), f"{session_id}.jsonl")
    base = _render_session_detail(cfg, session_id)
    parts = [base]
    q = get_hitl_queue()
    pending = q.list_pending_for_session(session_id)
    if pending:
        rows = []
        for p in pending:
            prompt_text = html.escape(json.dumps(p.prompt, indent=2, default=str))
            rows.append(
                f"<div class='card'><h3>HITL pending — {_esc(p.request_id)}</h3>"
                f"<pre>{prompt_text}</pre>"
                f"<form method='post' action='/sessions/{_esc(session_id)}/hitl/answer'>"
                "<input type='hidden' name='csrf_token' value=''>"
                f"<input type='hidden' name='request_id' value='{html.escape(p.request_id)}'>"
                "<label>Choice (a/e/m/s)<br>"
                "<input type='text' name='choice' placeholder='a'></label><br><br>"
                "<label>Extra notes (optional)<br>"
                "<textarea name='extra_notes' rows='3' cols='80'></textarea></label>"
                "<p><button>Submit decision</button></p></form></div>"
            )
        parts.append("".join(rows))
    if cfg.writes_enabled:
        try:
            notes = pending_chat_notes(db_path=cfg.web_db_path, session_id=session_id)
        except Exception:  # noqa: BLE001
            notes = []
        note_rows = "".join(
            f"<li>{html.escape(n['note'])} <span class='muted'>(queued {n['ts']})</span></li>"
            for n in notes
        )
        parts.append(
            "<div class='card'><h3>Chat notes (queued for next HITL)</h3>"
            f"<ul>{note_rows or '<li class=muted>none queued</li>'}</ul>"
            f"<form method='post' action='/sessions/{_esc(session_id)}/note'>"
            "<input type='hidden' name='csrf_token' value=''>"
            "<textarea name='note' rows='3' cols='80' placeholder='Note to ride into the next HITL prompt'></textarea>"
            "<p><button>Queue note</button></p></form></div>"
        )
    if log_path:
        parts.append(
            "<div class='card'><h3>Live events</h3>"
            "<p class='muted'>EventSource streaming from "
            f"<code>/api/sessions/{_esc(session_id)}/events</code>.</p>"
            "<pre id='live-events' style='max-height:480px;overflow:auto'></pre>"
            "<script>"
            f"const es = new EventSource('/api/sessions/{_esc(session_id)}/events');"
            "const pre = document.getElementById('live-events');"
            "es.onmessage = (e) => { pre.textContent += e.data + '\\n'; };"
            "es.addEventListener('close', () => es.close());"
            "</script></div>"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Form parsing — read the POST body
# ---------------------------------------------------------------------------

def _parse_form_body(body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    parsed = urllib.parse.parse_qs(text, keep_blank_values=True)
    # Flatten single-element lists; preserve multi-valued as lists.
    out: dict[str, Any] = {}
    for k, v in parsed.items():
        if isinstance(v, list) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Route handlers — Tier A new views
# ---------------------------------------------------------------------------

def _route_live(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Live runs", _render_live(cfg), cfg, active="dashboards")


def _route_config_index(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Configuration", _render_config_index(cfg), cfg, active="dashboards")


def _route_config_section(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    csrf = resolve_csrf_token(cfg)
    body = _render_config_section(cfg, params["section"], csrf_token=csrf)
    return 200, "text/html; charset=utf-8", _layout(
        f"Config · {params['section']}", body, cfg, active="dashboards",
    )


def _route_run_new(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    csrf = resolve_csrf_token(cfg)
    return 200, "text/html; charset=utf-8", _layout(
        "New run", _render_run_new(cfg, csrf), cfg, active="dashboards",
    )


# ---------------------------------------------------------------------------
# Tier C: pending HITL listing (read endpoint; answer + cancel are POSTs)
# ---------------------------------------------------------------------------

def _route_hitl_pending(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    q = get_hitl_queue()
    items = q.list_pending_for_session(params["sid"])
    body = [
        {"request_id": p.request_id, "prompt": p.prompt}
        for p in items
    ]
    return 200, "application/json; charset=utf-8", json.dumps(body)


# ---------------------------------------------------------------------------
# Tier C: SSE stream endpoint — produces the body inline
# ---------------------------------------------------------------------------

def _route_session_events_sse_marker(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    """Dispatch returns a sentinel that the request handler recognises
    and replaces with the actual streaming response. We can't yield
    here because the route function signature returns one tuple."""
    return 200, "text/event-stream", f"__SSE__{params['sid']}"


# ---------------------------------------------------------------------------
# Tier C: webhook handler — returns sentinel; the do_POST handler blocks
# ---------------------------------------------------------------------------

def _route_hitl_webhook_marker(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "application/json", "__HITL_WEBHOOK__"


# (write handlers live in the do_POST path; see _Handler below.)


# ---------------------------------------------------------------------------
# Extend the route table
# ---------------------------------------------------------------------------

_ROUTES.extend([
    (re.compile(r"^/live/?$"), _route_live),
    (re.compile(r"^/config/?$"), _route_config_index),
    (re.compile(r"^/config/(?P<section>[A-Za-z0-9_]+)/?$"), _route_config_section),
    (re.compile(r"^/run/new/?$"), _route_run_new),
    (re.compile(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/hitl/pending/?$"), _route_hitl_pending),
    (re.compile(r"^/api/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/events/?$"), _route_session_events_sse_marker),
    (re.compile(r"^/hitl/webhook/?$"), _route_hitl_webhook_marker),
])
