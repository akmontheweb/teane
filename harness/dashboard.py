"""``harness dashboard`` — read-only web UI (#14).

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
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8729
_DEFAULT_LOG_DIR = "~/.harness/logs"
_DEFAULT_METRICS_DIR = "~/.harness/metrics"
_DEFAULT_MEMORY_DIR = "~/.harness/memory"
_DEFAULT_INDEX_DIR = "~/.harness/repo_index"
_DEFAULT_SCHEDULE_DB = "~/.harness/schedule.db"
_DEFAULT_STATIC_DIR = "~/.harness/dashboard_static"


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
    # Tier B/C knobs.
    writes_enabled: bool = False   # opt-in for editing forms + Run-from-web
    csrf_token_env: str = ""       # CSRF token env var; auto-generated when empty + writes_enabled
    hitl_webhook_secret: str = ""  # shared secret the harness POSTs with
    web_db_path: str = "~/.harness/web.db"
    config_path: str = ""          # canonical config.json path; empty → use discover_config

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
            writes_enabled=bool(section.get("writes_enabled", False)),
            csrf_token_env=str(section.get("csrf_token_env", "")),
            hitl_webhook_secret=str(section.get("hitl_webhook_secret", "")),
            web_db_path=str(section.get("web_db_path", "~/.harness/web.db")),
            config_path=str(section.get("config_path", "")),
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

_BASE_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 0; color: #1f2328; background: #f6f8fa; }
header { background: #24292f; color: #fff; padding: 14px 24px; }
header a { color: #d0d7de; text-decoration: none; margin-right: 14px; font-size: 13px; }
header a:hover { color: #fff; }
header h1 { display: inline; font-size: 18px; margin: 0 24px 0 0; }
main { max-width: 1180px; margin: 24px auto; padding: 0 16px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d7de;
        border-radius: 6px; overflow: hidden; }
th, td { padding: 8px 12px; border-bottom: 1px solid #d0d7de; text-align: left; font-size: 13px; }
th { background: #f6f8fa; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.muted { color: #6e7781; }
.ok { color: #1a7f37; font-weight: 600; }
.fail { color: #cf222e; font-weight: 600; }
pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
      padding: 12px; overflow: auto; font-size: 12.5px; line-height: 1.45; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
        padding: 16px; margin-bottom: 16px; }
.card h2 { margin-top: 0; font-size: 16px; }
"""


def _layout(title: str, body: str, cfg: DashboardConfig) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — myharness dashboard</title>
<style>{_BASE_CSS}</style>
<script src="{html.escape(cfg.chart_js_url)}"></script>
</head>
<body>
<header>
  <h1>myharness</h1>
  <a href="/">overview</a>
  <a href="/sessions">sessions</a>
  <a href="/cost">cost</a>
  <a href="/schedule">schedule</a>
  <a href="/index">repo index</a>
  <a href="/memory">memory</a>
</header>
<main>
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
    return 200, "text/html; charset=utf-8", _layout("Sessions", _render_sessions(cfg), cfg)


def _route_session_detail(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout(
        f"Session {params['sid']}",
        _render_session_detail(cfg, params["sid"]),
        cfg,
    )


def _route_cost(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Cost burn", _render_cost(cfg), cfg)


def _route_api_cost_burn(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "application/json; charset=utf-8", json.dumps(cost_burn_series(cfg))


def _route_schedule(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Scheduled runs", _render_schedule(cfg), cfg)


def _route_index(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Repo index", _render_index(cfg), cfg)


def _route_memory(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Per-repo memory", _render_memory(cfg), cfg)


def _route_memory_file(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    status, body = _render_memory_file(cfg, params["name"])
    return status, "text/html; charset=utf-8", _layout(
        f"Memory · {params['name']}", body, cfg,
    )


_ROUTES: list[Route] = [
    (re.compile(r"^/?$"), _route_overview),
    (re.compile(r"^/sessions/?$"), _route_sessions),
    (re.compile(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/?$"), _route_session_detail),
    (re.compile(r"^/cost/?$"), _route_cost),
    (re.compile(r"^/api/cost-burn/?$"), _route_api_cost_burn),
    (re.compile(r"^/schedule/?$"), _route_schedule),
    (re.compile(r"^/index/?$"), _route_index),
    (re.compile(r"^/memory/?$"), _route_memory),
    (re.compile(r"^/memory/(?P<name>[A-Za-z0-9_.\-]+\.md)$"), _route_memory_file),
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
        "Writes are disabled. Pass <code>--writes-enabled</code> "
        "to <code>harness dashboard</code> to enable editing."
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
            "<p class='muted'>Writes are disabled; cannot start runs from the web. "
            "Pass <code>--writes-enabled</code> to <code>harness dashboard</code>.</p>"
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
    return 200, "text/html; charset=utf-8", _layout("Live runs", _render_live(cfg), cfg)


def _route_config_index(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    return 200, "text/html; charset=utf-8", _layout("Configuration", _render_config_index(cfg), cfg)


def _route_config_section(cfg: DashboardConfig, params: dict[str, str]) -> tuple[int, str, str]:
    csrf = resolve_csrf_token(cfg)
    body = _render_config_section(cfg, params["section"], csrf_token=csrf)
    return 200, "text/html; charset=utf-8", _layout(
        f"Config · {params['section']}", body, cfg,
    )


def _route_run_new(cfg: DashboardConfig, _params: dict[str, str]) -> tuple[int, str, str]:
    csrf = resolve_csrf_token(cfg)
    return 200, "text/html; charset=utf-8", _layout(
        "New run", _render_run_new(cfg, csrf), cfg,
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
