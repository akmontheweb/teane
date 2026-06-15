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
) -> type[http.server.BaseHTTPRequestHandler]:
    """Construct a request handler class closed over the live config
    and token. We use a factory rather than a class attribute so the
    test suite can spin up multiple handlers with different settings."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Quieter access logs — the harness's logging subsystem owns
        # the noisy channel.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("[dashboard] %s - %s", self.client_address[0], format % args)

        def _send(self, status: int, content_type: str, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            auth = check_auth(expected_token, self.headers.get("Authorization"))
            if not auth.ok:
                self._send(
                    401, "text/plain; charset=utf-8",
                    f"401 unauthorized: {auth.detail}\n",
                )
                return
            try:
                status, ctype, body = dispatch(cfg, self.path)
            except Exception:  # noqa: BLE001
                logger.exception("[dashboard] handler error")
                status, ctype, body = (
                    500, "text/plain; charset=utf-8",
                    "500 internal error (see server log)\n",
                )
            self._send(status, ctype, body)

        def do_HEAD(self) -> None:  # noqa: N802
            # Convenience for ``curl -I``; HEAD doesn't return body.
            self.do_GET()

    return _Handler


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass
class _ServerHandle:
    """Tiny wrapper so callers can shut the server down from another
    thread without poking the stdlib types directly."""

    server: _ThreadingServer
    thread: threading.Thread
    host: str
    port: int

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
    handler_class = make_request_handler(cfg, expected_token)
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
    return _ServerHandle(server=server, thread=thread, host=cfg.host, port=cfg.port)


# Tests reach into a few internals — re-export them so test files don't
# have to depend on underscore names.
ServerHandle = _ServerHandle
