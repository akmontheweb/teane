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

# Cap on the size of prompts accepted by the /run/now and /run/resume web
# handlers. ~50 KB is roughly two long pages — enough for any realistic
# product-requirement paste, small enough that no single submission can
# DoS the spawn loop. Operators with legitimately larger inputs should
# upload a spec file instead of pasting.
_RUN_PROMPT_MAX_CHARS = 50_000


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
    hitl_webhook_timeout_seconds: float = 600.0  # max seconds the webhook handler blocks waiting for an operator answer
    audit_log_retention_days: int = 90  # web.db audit_log rows older than this are pruned at server start; 0 disables
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
            hitl_webhook_timeout_seconds=max(
                1.0,
                float(section.get("hitl_webhook_timeout_seconds", 600.0)),
            ),
            audit_log_retention_days=max(
                0,
                int(section.get("audit_log_retention_days", 90)),
            ),
            web_db_path=str(section.get("web_db_path", "~/.harness/web.db")),
            config_path=str(section.get("config_path", "")),
            carbon_css_url=str(section.get("carbon_css_url", _DEFAULT_CARBON_CSS_URL)),
            carbon_js_url=str(section.get("carbon_js_url", _DEFAULT_CARBON_JS_URL)),
            docs_dir=str(section.get("docs_dir", _DEFAULT_DOCS_DIR)),
        )


# ---------------------------------------------------------------------------
# 1b. Static asset serving — packaged CSS/JS/icons/fonts plus operator
#     overrides via ``dashboard.static_dir``. Keeps the dashboard self-
#     contained (no CDN required) and air-gap friendly: operators can
#     drop a replacement ``app.css`` or ``sprite.svg`` into static_dir
#     and it wins over the packaged copy without touching the wheel.
# ---------------------------------------------------------------------------

_PACKAGED_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Whitelist of extensions we will serve and the content-type each maps to.
# Everything else 404s — refusing to serve an unknown extension is the
# defense against operators dropping a .py or .sh into static_dir and
# having the browser try to execute it.
_STATIC_CONTENT_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".map": "application/json",
}

# Only allow safe path characters. The regex on the route enforces this
# too, but defense-in-depth: nothing in static_dir is ever named with
# spaces, colons, or backslashes, so refusing them outright is the
# simplest containment story.
_STATIC_SAFE_RELPATH = re.compile(r"^[A-Za-z0-9_./\-]+$")


def _serve_static(cfg: DashboardConfig, relpath: str) -> tuple[int, str, bytes]:
    """Serve a static asset for a `/static/<relpath>` (or `/favicon.ico`)
    request.

    Resolution order:
      1. Operator override at ``$(cfg.static_dir)/<relpath>``.
      2. Packaged file at ``harness/static/<relpath>``.

    Containment is enforced via ``realpath`` so symlink-escape and
    ``..`` traversal both fail closed. Disallowed extensions 404. The
    returned body is ``bytes`` (binary-safe).
    """
    if not relpath or "\x00" in relpath or not _STATIC_SAFE_RELPATH.match(relpath):
        return 404, "text/plain; charset=utf-8", b"404 not found\n"
    if relpath.startswith("/") or ".." in relpath.split("/"):
        return 404, "text/plain; charset=utf-8", b"404 not found\n"

    ext = os.path.splitext(relpath)[1].lower()
    content_type = _STATIC_CONTENT_TYPES.get(ext)
    if content_type is None:
        return 404, "text/plain; charset=utf-8", b"404 not found\n"

    candidates: list[str] = []
    override_dir = (cfg.static_dir or "").strip()
    if override_dir:
        candidates.append(os.path.expanduser(override_dir))
    candidates.append(_PACKAGED_STATIC_DIR)

    for root in candidates:
        try:
            root_real = os.path.realpath(root)
            target = os.path.realpath(os.path.join(root, relpath))
        except OSError:
            continue
        if not os.path.isdir(root_real):
            continue
        # Strict containment: the resolved target must live under the
        # resolved root. Reject anything that climbs out via symlink or
        # ``..``.
        if not (target == root_real or target.startswith(root_real + os.sep)):
            continue
        if not os.path.isfile(target):
            continue
        try:
            with open(target, "rb") as f:
                data = f.read()
        except OSError:
            continue
        return 200, content_type, data

    return 404, "text/plain; charset=utf-8", b"404 not found\n"


# ---------------------------------------------------------------------------
# 1b. Filesystem helpers — folder picker + uploads
# ---------------------------------------------------------------------------

_PRODUCT_SPEC_DIR_NAME = "product_spec"
_SPEC_ALLOWED_EXTS = frozenset({".txt", ".md"})
_SKILL_ALLOWED_EXT = ".py"
_MAX_SKILL_BYTES = 256 * 1024  # individual skill source files stay small
_MEMORY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_FILENAME_MAX_LEN = 255


def _safe_basename(filename: str) -> str:
    """Strip directory components and reject empty / traversal names.

    Returns ``""`` for unsafe input so callers can surface a 400.
    Spaces and other printable characters are allowed; only path
    separators (already stripped by ``os.path.basename``), null bytes,
    and other control characters are rejected.
    """
    if not filename:
        return ""
    base = os.path.basename(filename.replace("\\", "/"))
    if not base or base in {".", ".."}:
        return ""
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in base):
        return ""
    if len(base.encode("utf-8")) > _FILENAME_MAX_LEN:
        return ""
    return base


def _list_directory_entries(target: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """List immediate child directories of ``target``. Returns
    ``(entries, error)``. Symlinks are followed for the is-dir check
    but the listing itself does not recurse."""
    if "\x00" in target:
        return [], "path contains null byte"
    abs_path = os.path.abspath(os.path.expanduser(target))
    if not os.path.isdir(abs_path):
        return [], f"not a directory: {abs_path}"
    try:
        names = sorted(os.listdir(abs_path), key=str.lower)
    except OSError as exc:
        return [], f"cannot list {abs_path}: {exc}"
    entries: list[dict[str, Any]] = []
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(abs_path, name)
        try:
            is_dir = os.path.isdir(full)
        except OSError:
            continue
        if not is_dir:
            continue
        entries.append({"name": name, "path": full})
    return entries, None


def _browse_response(query_path: str) -> tuple[int, str, str]:
    """Build the JSON response for ``GET /api/browse?path=...``."""
    raw = (query_path or "").strip() or os.path.expanduser("~")
    abs_path = os.path.abspath(os.path.expanduser(raw))
    entries, err = _list_directory_entries(abs_path)
    if err is not None:
        body = json.dumps({"ok": False, "error": err, "path": abs_path})
        return 400, "application/json; charset=utf-8", body
    parent = os.path.dirname(abs_path) if abs_path != os.path.dirname(abs_path) else ""
    payload = {
        "ok": True,
        "path": abs_path,
        "parent": parent,
        "entries": entries,
    }
    return 200, "application/json; charset=utf-8", json.dumps(payload)


def _persist_product_spec(
    workspace: str, filename: str, data: bytes,
) -> tuple[str, Optional[str]]:
    """Write an uploaded spec file to ``<workspace>/product_spec/`` and
    return ``(saved_path, error)``. The workspace must exist and the
    filename must end in ``.txt`` or ``.md``."""
    base = _safe_basename(filename)
    if not base:
        return "", "unsafe filename"
    ext = os.path.splitext(base)[1].lower()
    if ext not in _SPEC_ALLOWED_EXTS:
        allowed = ", ".join(sorted(_SPEC_ALLOWED_EXTS))
        return "", f"only {allowed} files are accepted (got {ext or '?'})"
    abs_workspace = os.path.abspath(os.path.expanduser(workspace or ""))
    if not os.path.isdir(abs_workspace):
        return "", f"workspace does not exist: {abs_workspace}"
    spec_dir = os.path.join(abs_workspace, _PRODUCT_SPEC_DIR_NAME)
    try:
        os.makedirs(spec_dir, exist_ok=True)
    except OSError as exc:
        return "", f"could not create {spec_dir}: {exc}"
    target = os.path.join(spec_dir, base)
    try:
        with open(target, "wb") as f:
            f.write(data)
    except OSError as exc:
        return "", f"write failed: {exc}"
    return target, None


def _write_web_input_sidecar(workspace: str, text: str) -> str:
    """Mirror an operator-entered product requirement into
    ``<workspace>/product_spec/web_input.md`` so the harness consolidator
    picks it up alongside any uploaded files. Raises ``OSError`` on
    failure."""
    abs_workspace = os.path.abspath(os.path.expanduser(workspace or ""))
    spec_dir = os.path.join(abs_workspace, _PRODUCT_SPEC_DIR_NAME)
    os.makedirs(spec_dir, exist_ok=True)
    target = os.path.join(spec_dir, "web_input.md")
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    return target


def _resolved_user_skills_dir(cfg: "DashboardConfig") -> str:
    """Resolve ``skills.user_skills_dir`` from the live config, falling
    back to the same ``~/.harness/skills`` default that
    ``harness.skills`` uses."""
    try:
        live = read_config_file(cfg) or {}
    except Exception:  # noqa: BLE001 — defaults are fine if the config is broken
        live = {}
    raw = ((live.get("skills") or {}).get("user_skills_dir")
           or "~/.harness/skills")
    return os.path.abspath(os.path.expanduser(str(raw)))


def _persist_user_skill(
    cfg: "DashboardConfig", filename: str, data: bytes,
) -> tuple[str, Optional[str]]:
    base = _safe_basename(filename)
    if not base:
        return "", "unsafe filename"
    if os.path.splitext(base)[1].lower() != _SKILL_ALLOWED_EXT:
        return "", "only .py files are accepted"
    if len(data) > _MAX_SKILL_BYTES:
        return "", f"file exceeds {_MAX_SKILL_BYTES} bytes"
    skills_dir = _resolved_user_skills_dir(cfg)
    try:
        os.makedirs(skills_dir, exist_ok=True)
    except OSError as exc:
        return "", f"could not create {skills_dir}: {exc}"
    target = os.path.join(skills_dir, base)
    try:
        with open(target, "wb") as f:
            f.write(data)
    except OSError as exc:
        return "", f"write failed: {exc}"
    return target, None


def _delete_user_skill(
    cfg: "DashboardConfig", filename: str,
) -> tuple[str, Optional[str]]:
    base = _safe_basename(filename)
    if not base:
        return "", "unsafe filename"
    if os.path.splitext(base)[1].lower() != _SKILL_ALLOWED_EXT:
        return "", "only .py files can be deleted"
    skills_dir = _resolved_user_skills_dir(cfg)
    target = os.path.join(skills_dir, base)
    # Containment — never let the form path climb out of skills_dir.
    if os.path.realpath(os.path.dirname(target)) != os.path.realpath(skills_dir):
        return "", "filename escapes skills directory"
    if not os.path.isfile(target):
        return "", f"no such skill: {base}"
    try:
        os.remove(target)
    except OSError as exc:
        return "", f"delete failed: {exc}"
    return target, None


def _list_user_skill_files(cfg: "DashboardConfig") -> list[str]:
    """Return the basenames of every ``*.py`` file in the user skills
    directory (sorted). The configure-page Skills card renders this list
    with delete buttons."""
    skills_dir = _resolved_user_skills_dir(cfg)
    if not os.path.isdir(skills_dir):
        return []
    try:
        return sorted(
            name for name in os.listdir(skills_dir)
            if name.endswith(_SKILL_ALLOWED_EXT)
            and not name.startswith("_")
            and os.path.isfile(os.path.join(skills_dir, name))
        )
    except OSError:
        return []


def _resolved_memory_dir(cfg: "DashboardConfig") -> str:
    """Resolve ``memory.dir`` from the live config (default
    ``~/.harness/memory``)."""
    try:
        live = read_config_file(cfg) or {}
    except Exception:  # noqa: BLE001
        live = {}
    raw = ((live.get("memory") or {}).get("dir") or "~/.harness/memory")
    return os.path.abspath(os.path.expanduser(str(raw)))


def _persist_new_memory(
    cfg: "DashboardConfig", name: str, content: str,
) -> tuple[str, Optional[str]]:
    name = (name or "").strip().lower()
    if not _MEMORY_NAME_RE.match(name):
        return "", (
            "memory name must match [a-z0-9][a-z0-9_-]{0,63} "
            "(lowercase letters, digits, hyphen, underscore)"
        )
    if not content.strip():
        return "", "memory content is empty"
    mem_dir = _resolved_memory_dir(cfg)
    try:
        os.makedirs(mem_dir, exist_ok=True)
    except OSError as exc:
        return "", f"could not create {mem_dir}: {exc}"
    target = os.path.join(mem_dir, f"{name}.md")
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return "", f"write failed: {exc}"
    return target, None


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
# Bare-minimum inline fallback. The full stylesheet lives at
# harness/static/css/app.css and is served via the /static/ route the
# layout pulls in. Keeping a tiny inline copy means pages still render
# something sensible (correct margins + tag colors) when an air-gap
# operator hasn't yet mirrored the assets. Remove this fallback after
# one release cycle.
_BASE_CSS = """\
body { margin: 0; font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
main.bx--content { margin-left: 16rem; padding: 2rem; background: #f4f4f4; min-height: calc(100vh - 3rem); }
.bx--side-nav__link--current, .bx--side-nav__link--current span { color: #fff !important; background: #393939; }
.muted { color: #6f6f6f; }
.ok { color: #198038; font-weight: 600; }
.fail { color: #da1e28; font-weight: 600; }
.tag { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 0.75rem; font-size: 0.75rem; font-weight: 500; }
.tag-green { background: #defbe6; color: #0e6027; }
.tag-red { background: #ffd7d9; color: #a2191f; }
.tag-gray { background: #e0e0e0; color: #393939; }
"""


def _icon(name: str, size: int = 16, klass: str = "") -> str:
    """Render an inline reference to an icon symbol in
    /static/icons/sprite.svg.

    ``name`` is the symbol id without the ``i-`` prefix (e.g. ``launch``,
    ``copy``, ``play``). Unknown names render as the empty string —
    no Carbon-style placeholder — so a typo doesn't blow up a page.
    """
    if not name or not re.fullmatch(r"[a-z0-9\-]+", name):
        return ""
    klass_attr = f" {klass}".rstrip() if klass else ""
    return (
        f"<svg class=\"icon{klass_attr}\" width=\"{size}\" height=\"{size}\" "
        f"aria-hidden=\"true\" focusable=\"false\">"
        f"<use href=\"/static/icons/sprite.svg#i-{name}\"/></svg>"
    )


def _copyable(value: str, label: str = "Copy") -> str:
    """Wrap a string in an inline ``<code>`` plus a copy-to-clipboard
    button. The button carries the value in ``data-copy``; the JS in
    dashboard.js delegates the click and shows a toast on success.

    Use this for any short value an operator might want to paste into
    a terminal: session ids, workspace paths, configuration keys.
    """
    val = html.escape(value or "")
    label_attr = html.escape(label)
    icon = _icon("copy", size=14)
    return (
        f"<span class='id-cell'>"
        f"<code>{val}</code>"
        f"<button class='copy-btn' type='button' data-copy='{val}' "
        f"aria-label='{label_attr}' title='{label_attr}'>{icon}</button>"
        f"</span>"
    )


def _breadcrumb(items: list[tuple[str, Optional[str]]]) -> str:
    """Render a breadcrumb trail.

    ``items`` is a list of ``(label, href)`` tuples. Pass ``href=None``
    for the current page (the last item) — it renders as plain text.

    Example::

        _breadcrumb([("Sessions", "/sessions"), ("abc123", None)])
    """
    parts: list[str] = []
    for label, href in items:
        text = html.escape(label)
        if href:
            parts.append(
                f"<li><a href='{html.escape(href)}'>{text}</a></li>"
            )
        else:
            parts.append(
                f"<li class='breadcrumb__current' aria-current='page'>{text}</li>"
            )
    return f"<ol class='breadcrumb' aria-label='Breadcrumb'>{''.join(parts)}</ol>"


def _empty_state(
    icon: str,
    title: str,
    body: str,
    cta_text: Optional[str] = None,
    cta_href: Optional[str] = None,
) -> str:
    """Render a centered empty-state card with an icon, headline,
    explanatory paragraph, and an optional call-to-action button.

    Use anywhere a renderer previously emitted a lone ``<p class='muted'>
    No foo yet…</p>``. The CTA replaces the previous "go run the CLI"
    hint with a one-click affordance.

    ``body`` is treated as plain text and HTML-escaped — callers that
    want a richer empty-state body should compose the markup themselves
    and use a different helper.
    """
    cta_html = ""
    if cta_text and cta_href:
        cta_html = (
            f"<p><a class='bx--btn bx--btn--primary' "
            f"href='{html.escape(cta_href)}'>{_icon('add')}{html.escape(cta_text)}</a></p>"
        )
    return (
        f"<div class='empty-state'>"
        f"<div class='empty-state__icon'>{_icon(icon, size=32, klass='icon--lg')}</div>"
        f"<h3 class='empty-state__title'>{html.escape(title)}</h3>"
        f"<p class='empty-state__body'>{html.escape(body)}</p>"
        f"{cta_html}"
        f"</div>"
    )


_NAV_ITEMS: tuple[tuple[str, str, str, str], ...] = (
    # (slug, label, href, icon symbol name from sprite.svg)
    ("status", "View Status", "/status", "chart-line"),
    ("run", "Run Harness", "/run", "play"),
    ("config", "Configure Harness", "/config-ui", "settings"),
    ("dashboards", "View Dashboards", "/dashboards", "dashboard"),
    ("docs", "View Documents", "/docs", "document"),
)


def _render_side_nav(active: str) -> str:
    items = []
    for slug, label, href, icon_name in _NAV_ITEMS:
        cls = "bx--side-nav__link bx--side-nav__link--current" if slug == active else "bx--side-nav__link"
        icon_svg = _icon(icon_name, size=16) if icon_name else ""
        items.append(
            f'<li class="bx--side-nav__item">'
            f'<a class="{cls}" href="{href}">'
            f'{icon_svg}'
            f'<span class="bx--side-nav__link-text">{html.escape(label)}</span>'
            f'</a></li>'
        )
    return (
        '<nav id="side-nav" class="bx--side-nav bx--side-nav--expanded" aria-label="Side navigation">'
        '<ul class="bx--side-nav__items">' + "".join(items) + '</ul></nav>'
    )


def _layout(title: str, body: str, cfg: DashboardConfig, active: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#161616">
<title>{html.escape(title)} — myharness</title>
<link rel="icon" href="/static/favicon.ico">
<link rel="stylesheet" href="{html.escape(cfg.carbon_css_url)}">
<link rel="stylesheet" href="/static/css/app.css">
<style>{_BASE_CSS}</style>
<script src="{html.escape(cfg.chart_js_url)}"></script>
<script defer src="/static/js/dashboard.js"></script>
</head>
<body class="bx--body">
<header class="bx--header" role="banner">
  <button id="nav-toggle" type="button" class="nav-toggle" aria-label="Toggle navigation"
          aria-controls="side-nav" aria-expanded="false">
    {_icon("menu", size=20)}
  </button>
  <a class="bx--header__name" href="/status">
    <span class="bx--header__name--prefix">myharness</span>
  </a>
  <div class="header-actions">
    <button id="auto-refresh-toggle" type="button" class="auto-refresh-btn"
            aria-pressed="false" title="Auto-refresh every 15s">
      {_icon("renew", size=16)}<span class="auto-refresh-btn__label">Auto-refresh: off</span>
    </button>
  </div>
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
        return _empty_state(
            icon="list",
            title="No sessions yet",
            body=(
                "Run the harness from the Run Harness page or the "
                "`harness run` CLI to populate this view."
            ),
            cta_text="Run harness",
            cta_href="/run",
        )
    rows = []
    for s in sessions:
        status = "—"
        cls = "muted"
        if s.exit_code == 0:
            status, cls = "success", "ok"
        elif s.exit_code is not None:
            status, cls = f"exit {s.exit_code}", "fail"
        sid = _esc(s.session_id)
        rows.append(
            f"<tr>"
            f"<td><a href='/sessions/{sid}'>{sid}</a> "
            f"<button class='copy-btn' type='button' data-copy='{sid}' "
            f"aria-label='Copy session id'>{_icon('copy', size=14)}</button></td>"
            f"<td>{_esc(s.started_at)}</td>"
            f"<td>{_esc(s.ended_at)}</td>"
            f"<td class='{cls}'>{_esc(status)}</td>"
            f"<td class='num'>{_fmt_int(s.llm_calls)}</td>"
            f"<td class='num'>{_fmt_cost(s.total_cost_usd)}</td>"
            f"<td class='num'>{_fmt_int(s.total_input_tokens)}</td>"
            f"<td class='num'>{_fmt_int(s.total_output_tokens)}</td>"
            f"<td><span class='muted'>{_esc(s.workspace_path)}</span></td>"
            f"</tr>"
        )
    return (
        "<div class='table-wrap'><table id='sessions-table'>"
        "<thead><tr>"
        "<th data-sort='str'>session</th>"
        "<th data-sort='date'>started</th>"
        "<th data-sort='date'>ended</th>"
        "<th data-sort='str'>status</th>"
        "<th class='num' data-sort='num'>calls</th>"
        "<th class='num' data-sort='num'>cost</th>"
        "<th class='num' data-sort='num'>tokens in</th>"
        "<th class='num' data-sort='num'>tokens out</th>"
        "<th data-sort='str'>workspace</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def _render_session_detail(cfg: DashboardConfig, session_id: str) -> str:
    log_path = os.path.join(os.path.expanduser(cfg.log_dir), f"{session_id}.jsonl")
    crumb = _breadcrumb([
        ("Sessions", "/sessions"),
        (session_id, None),
    ])
    if not os.path.isfile(log_path):
        return crumb + f"<p class='fail'>No log file for session {_esc(session_id)}.</p>"
    events = session_events(log_path, max_events=2000)
    if not events:
        return crumb + f"<p class='muted'>Session {_esc(session_id)} log is empty or unparseable.</p>"
    body = [crumb]
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
            f"<td class='num'>{(f'{duration:.1f}s' if duration is not None else '—')}</td>"
            f"<td><span class='muted'>{_esc(r['log_path'])}</span></td>"
            f"</tr>"
        )
    return (
        "<div class='table-wrap'><table id='schedule-table'>"
        "<thead><tr><th>job</th><th>started</th><th>ended</th>"
        "<th>status</th><th class='num'>duration</th><th>log</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


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
    crumb = _breadcrumb([("Memory", "/memory"), (name, None)])
    if content is None:
        return 404, crumb + "<p class='fail'>Memory file not found.</p>"
    return 200, (
        crumb
        + f"<div class='card'><h2>{_esc(name)}</h2><pre>{html.escape(content)}</pre></div>"
    )


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


def _route_api_config_mtime(
    cfg: DashboardConfig, _params: dict[str, str],
) -> tuple[int, str, str]:
    """Return the current ``config.json`` mtime as JSON. The Configure
    page polls this every few seconds and shows a "config changed
    externally — Reload" banner when the value differs from the
    snapshot stamped at page-render time."""
    mtime_ns = config_file_mtime_ns(cfg)
    payload = {"mtime_ns": mtime_ns}
    return 200, "application/json", json.dumps(payload)


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
            "<div class='table-wrap'><table id='running-now-table'>"
            "<thead><tr>"
            "<th data-sort='str'>Session</th>"
            "<th data-sort='date'>Started</th>"
            "<th data-sort='str'>Workspace</th>"
            "<th data-sort='str'>Source</th>"
            "<th></th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    else:
        section_b_body = (
            "<p class='muted mb-0'>No sessions are running right now. "
            "Use <a href='/run'>Run Harness</a> to start one.</p>"
        )
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
                f"<td class='num'>{duration}</td>"
                f"<td class='num'>{_fmt_cost(s.total_cost_usd)}</td>"
                f"<td><a href='/sessions/{_esc(s.session_id)}'>Open dashboard</a></td>"
                f"</tr>"
            )
        section_c_body = (
            "<div class='table-wrap'><table id='today-runs-table'>"
            "<thead><tr>"
            "<th data-sort='str'>Session</th>"
            "<th data-sort='str'>Status</th>"
            "<th class='num' data-sort='num'>Duration</th>"
            "<th class='num' data-sort='num'>Cost</th>"
            "<th></th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    else:
        section_c_body = "<p class='muted mb-0'>No sessions started today.</p>"
    section_c = f"<div class='card'><h2>Today's runs</h2>{section_c_body}</div>"

    return section_a + section_b + section_c


def _collect_run_argv(form: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Translate the Run Harness form's per-flag inputs into a CLI argv
    list. Thin wrapper over :func:`harness.web_forms.build_run_argv_from_form`
    so the request handler doesn't import the form module directly."""
    from harness.web_forms import build_run_argv_from_form
    return build_run_argv_from_form(form)


def _render_run_flag_input(flag) -> str:
    """Render one CLI flag's input element. Mirrors the Carbon styling of
    the config form so the two screens feel consistent."""
    from harness.web_forms import (
        FORM_KIND_NUMBER_INT, FORM_KIND_SELECT, FORM_KIND_YES_NO,
    )
    name = html.escape(flag.field_id)
    default = "" if flag.default is None else html.escape(str(flag.default))
    if flag.kind == FORM_KIND_YES_NO:
        opts = []
        for choice in ("no", "yes"):
            sel = "selected" if str(flag.default).lower() == choice else ""
            opts.append(f"<option value='{choice}' {sel}>{choice.capitalize()}</option>")
        return (
            f"<select class='bx--select-input w-100' name='{name}'>"
            + "".join(opts) + "</select>"
        )
    if flag.kind == FORM_KIND_SELECT:
        opts = []
        for choice in flag.choices:
            sel = "selected" if str(flag.default) == choice else ""
            opts.append(
                f"<option value='{html.escape(choice)}' {sel}>{html.escape(choice)}</option>"
            )
        return (
            f"<select class='bx--select-input w-100' name='{name}'>"
            + "".join(opts) + "</select>"
        )
    if flag.kind == FORM_KIND_NUMBER_INT:
        bounds = []
        if flag.min_value is not None:
            bounds.append(f"min='{flag.min_value}'")
        if flag.max_value is not None:
            bounds.append(f"max='{flag.max_value}'")
        bound_attrs = (" " + " ".join(bounds)) if bounds else ""
        return (
            f"<input class='bx--text-input w-100' type='number' step='1' name='{name}' "
            f"value='{default}'{bound_attrs}>"
        )
    # FORM_KIND_TEXT fallback.
    return (
        f"<input class='bx--text-input w-100' type='text' name='{name}' "
        f"value='{default}'>"
    )


def _render_run_flag_rows() -> str:
    """All CLI-flag rows for the Run Harness form table."""
    from harness.web_forms import run_flags
    rows = []
    for flag in run_flags():
        rows.append(
            f"<tr>"
            f"<td><label class='bx--label' for='{html.escape(flag.field_id)}'>"
            f"{html.escape(flag.label)} "
            f"<code class='muted fs-sm'>{html.escape(flag.flag)}</code>"
            f"</label></td>"
            f"<td class='muted'>{html.escape(flag.description)}</td>"
            f"<td>{_render_run_flag_input(flag)}</td>"
            f"</tr>"
        )
    return "".join(rows)


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
            "<div class='table-wrap'><table id='pending-jobs-table'>"
            "<thead><tr><th>Fire at (UTC)</th><th>Name</th><th>Workspace</th>"
            "<th>Prompt</th><th>Status</th><th>Args</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    else:
        pending_html = "<p class='muted'>No one-shot jobs scheduled.</p>"

    flag_rows = _render_run_flag_rows()
    session_picker_html = _render_session_picker(cfg)

    # The form carries TWO mutually-exclusive panels: the New-session
    # panel (workspace + prompt + flags + schedule) and the Resume panel
    # (session picker + optional workspace/prompt overrides). Switching
    # is pure CSS via the hidden radios at the top of the form — no JS
    # needed for the tab toggle itself.
    form = f"""
<div class='card'>
  <h2>Start a harness run</h2>
  <form id='run-form' method='post' action='/run/now' class='run-form'>
    <input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>
    <input type='hidden' id='fire-at-utc' name='fire_at_utc' value=''>

    <!-- Mode radios: hidden visually but drive the panel toggle below
         via CSS sibling selectors. -->
    <input type='radio' name='run_mode' id='mode-new' value='new' checked
           class='run-mode-radio'>
    <input type='radio' name='run_mode' id='mode-resume' value='resume'
           class='run-mode-radio'>

    <div class='run-tabs' role='tablist' aria-label='Session mode'>
      <label for='mode-new' class='run-tab' role='tab'>
        {_icon("add")}New session
      </label>
      <label for='mode-resume' class='run-tab' role='tab'>
        {_icon("renew")}Resume existing session
      </label>
    </div>

    <!-- NEW session panel (default) -->
    <fieldset class='run-panel run-panel--new'>
      <legend class='bx--label run-panel__legend'>New session</legend>
      <div class='field'>
        <label class='bx--label' for='workspace'>Workspace path</label>
        <div class='workspace-picker'>
          <input class='bx--text-input workspace-picker__input' id='workspace'
                 name='workspace' type='text' placeholder='/path/to/repo' required>
          <button type='button' class='bx--btn bx--btn--tertiary workspace-picker__btn'
                  id='workspace-browse-btn' aria-label='Browse for workspace folder'>
            {_icon("folder")}Browse
          </button>
        </div>
        <p class='muted fs-sm mt-2'>A valid workspace path is required.</p>
      </div>
      <input type='hidden' id='spec-file-path' name='spec_file_path' value=''>
      <div class='field'>
        <label class='bx--label' for='prompt'>Product Requirement</label>
        <textarea class='bx--text-area' id='prompt' name='prompt' rows='4'
                  placeholder='Enter your production specification here or upload a product specification document (in .txt or .md format)'></textarea>
        <div class='spec-upload mt-2'>
          <input type='file' id='spec-file' accept='.txt,.md' class='hidden'>
          <button type='button' class='bx--btn bx--btn--tertiary spec-upload__btn'
                  id='spec-upload-btn'>
            {_icon("upload")}Upload product specification
          </button>
          <span class='spec-upload__name muted' id='spec-upload-name'></span>
          <button type='button' class='spec-upload__clear bx--btn bx--btn--ghost hidden'
                  id='spec-upload-clear' aria-label='Clear uploaded file'>&times;</button>
        </div>
      </div>
      <fieldset class='field-group'>
        <legend class='bx--label'>Run options</legend>
        <p class='muted mb-3'>One field per CLI flag. Leave a text/number field blank to use the harness default. Yes/No flags emit the flag only on Yes.</p>
        <div class='table-wrap'>
          <table class='w-100 run-options-table'>
            <thead><tr><th>Flag</th><th>Meaning</th><th>Value</th></tr></thead>
            <tbody>{flag_rows}</tbody>
          </table>
        </div>
      </fieldset>
      <fieldset id='schedule-fields' class='hidden field-group'>
        <legend class='bx--label'>Scheduled run</legend>
        <div class='field'>
          <label class='bx--label' for='job_name'>Job name</label>
          <input class='bx--text-input' id='job_name' name='name' type='text' placeholder='nightly retest'>
        </div>
        <div class='field flex gap-4'>
          <div class='flex-1'>
            <label class='bx--label' for='fire_date'>Date (UTC)</label>
            <input class='bx--date-picker__input' id='fire_date' type='date'>
          </div>
          <div class='flex-1'>
            <label class='bx--label' for='fire_time'>Time (UTC)</label>
            <input class='bx--time-picker__input' id='fire_time' type='time' step='60'>
          </div>
        </div>
      </fieldset>
      <div class='actions'>
        <button class='bx--btn bx--btn--primary' type='submit'
                formaction='/run/now' id='run-now-btn'>{_icon("play")}Run Now</button>
        <button class='bx--btn bx--btn--secondary' type='button' id='reveal-schedule-btn'>{_icon("calendar")}Schedule A Run</button>
        <button class='bx--btn bx--btn--primary hidden' type='submit'
                formaction='/run/schedule' id='confirm-schedule-btn'>{_icon("checkmark-filled")}Confirm Schedule</button>
      </div>
    </fieldset>

    <!-- RESUME session panel -->
    <fieldset class='run-panel run-panel--resume'>
      <legend class='bx--label run-panel__legend'>Resume existing session</legend>
      <p class='muted mb-3'>Pick a session to continue from its last checkpoint.
      Workspace is auto-detected from the checkpoint; supply an override only if
      the path moved. The optional prompt is appended to the resumed conversation.</p>
      {session_picker_html}
      <details class='mt-4'>
        <summary class='mb-3'>Optional overrides</summary>
        <div class='field'>
          <label class='bx--label' for='resume-workspace'>Workspace override (rarely needed)</label>
          <input class='bx--text-input' id='resume-workspace' name='workspace' type='text'
                 placeholder='Auto-detected from checkpoint'>
        </div>
        <div class='field'>
          <label class='bx--label' for='resume-prompt'>Append prompt to resumed session</label>
          <textarea class='bx--text-area' id='resume-prompt' name='prompt' rows='3'
                    placeholder='Optional — leave blank to resume without a new prompt'></textarea>
        </div>
      </details>
      <div class='actions'>
        <button class='bx--btn bx--btn--primary' type='submit'
                formaction='/run/resume' id='resume-now-btn'>{_icon("renew")}Resume Now</button>
      </div>
    </fieldset>
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
    fields.classList.remove('hidden');
    confirm.classList.remove('hidden');
    runNow.classList.add('hidden');
    reveal.classList.add('hidden');
  }});
  form.addEventListener('submit', function(e) {{
    var submitter = e.submitter;
    if (submitter && submitter.id === 'confirm-schedule-btn') {{
      if (!fireDate.value || !fireTime.value) {{
        e.preventDefault();
        alert('Pick both a date and a time.');
        return;
      }}
      fireAt.value = fireDate.value + 'T' + fireTime.value + ':00Z';
    }}
    // Resume requires a session pick — block submission if none selected.
    if (submitter && submitter.id === 'resume-now-btn') {{
      var picked = form.querySelector("input[name='resume_session_id']:checked");
      if (!picked) {{
        e.preventDefault();
        alert('Pick a session from the list first.');
      }}
    }}
    // Run Now / Schedule both require workspace + (prompt OR uploaded
    // spec file) — surfaced inline (the inputs intentionally don't
    // carry the `required` attribute because we share the form with
    // the Resume panel which has neither field).
    if (submitter && (submitter.id === 'run-now-btn' || submitter.id === 'confirm-schedule-btn')) {{
      var ws = document.getElementById('workspace');
      var pr = document.getElementById('prompt');
      var sf = document.getElementById('spec-file-path');
      if (!ws.value.trim()) {{
        e.preventDefault();
        if (window.toast) {{ toast('Workspace path is required.', 'error'); }}
        else {{ alert('Workspace path is required.'); }}
        return;
      }}
      var hasText = pr.value.trim().length > 0;
      var hasFile = (sf && sf.value.trim().length > 0);
      if (!hasText && !hasFile) {{
        e.preventDefault();
        var msg = 'Enter a product requirement or upload a .txt/.md document.';
        if (window.toast) {{ toast(msg, 'error'); }}
        else {{ alert(msg); }}
      }}
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


def _render_session_picker(cfg: DashboardConfig) -> str:
    """Render the "pick a session to resume" table for the Resume panel
    on the Run Harness page.

    Each row carries a labelled radio input (``resume_session_id``) so
    submitting the form posts the chosen session id. The Delete column
    holds a per-row mini-form that POSTs to ``/sessions/{sid}/purge``
    and is OUTSIDE the surrounding Resume form (HTML forbids nested
    <form> elements) — the table renders raw in the document so the
    delete forms can be siblings of the picker, not children.

    The table opts into sort + filter via ``data-sort`` attributes
    (handled by dashboard.js' ``enhanceTable``)."""
    sessions = list_sessions(cfg)
    if not sessions:
        return (
            "<div class='card empty-state'>"
            "<p class='muted'>No sessions on disk yet. Start a new session "
            "via the <em>New session</em> tab — once it runs, it shows up "
            "here for resume.</p></div>"
        )
    rows: list[str] = []
    for s in sessions:
        sid = _esc(s.session_id)
        started = _esc(s.started_at) if s.started_at else "—"
        ended = _esc(s.ended_at) if s.ended_at else "<span class='muted'>(running)</span>"
        ws = s.workspace_path or ""
        # "App name" heuristic: basename of the workspace path. Operators
        # typically clone a repo into a directory named after the project,
        # so this works well as a quick label. Falls back to the session
        # id when no workspace is recorded.
        app = os.path.basename(ws.rstrip("/")) if ws else s.session_id
        if s.exit_code == 0:
            status = "<span class='tag tag-green'>succeeded</span>"
        elif s.exit_code is None:
            status = "<span class='tag tag-gray'>running</span>"
        else:
            status = f"<span class='tag tag-red'>exit {s.exit_code}</span>"
        radio_id = f"resume-row-{sid}"
        # Delete button: a plain <button type='button'> so it's NOT a
        # form submitter — the JS handler in dashboard.js
        # (wireSessionPurge) does a fetch POST to
        # /sessions/{sid}/purge with the CSRF header. This avoids
        # nesting a <form> inside the surrounding Resume <form> (the
        # HTML spec forbids that, and browsers handle it badly).
        delete_btn = (
            f"<button type='button' class='ct-remove session-row__delete' "
            f"data-purge-session='{sid}' "
            f"aria-label='Delete session {sid}' "
            f"title='Permanently delete this session — checkpoint + log'>"
            f"&times;</button>"
        )
        rows.append(
            f"<tr class='session-row' data-row-session='{sid}'>"
            f"<td class='session-row__pick'>"
            f"<input type='radio' name='resume_session_id' value='{sid}' "
            f"id='{radio_id}'>"
            f"</td>"
            f"<td><label for='{radio_id}' class='session-row__sid-label'>"
            f"<code>{sid}</code></label></td>"
            f"<td>{started}</td>"
            f"<td>{ended}</td>"
            f"<td><span class='muted'>{_esc(ws)}</span></td>"
            f"<td>{_esc(app)}</td>"
            f"<td>{status}</td>"
            f"<td class='session-row__delete-cell'>{delete_btn}</td>"
            f"</tr>"
        )
    return (
        "<div class='table-wrap session-picker'>"
        "<table id='resume-session-picker-table' class='session-picker__table'>"
        "<thead><tr>"
        "<th></th>"
        "<th data-sort='str'>Session ID</th>"
        "<th data-sort='date'>Created</th>"
        "<th data-sort='date'>Last update</th>"
        "<th data-sort='str'>Repo / workspace</th>"
        "<th data-sort='str'>App</th>"
        "<th data-sort='str'>Status</th>"
        "<th>Delete</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


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


# Display-name overrides keep config-key→form-label decoupled from the
# wire format. The "dashboard" section is shown as "Web Defaults" under
# the "Harness Web" group so the wording matches the rest of the app
# (the underlying key stays ``dashboard`` for back-compat with existing
# operator configs).
_SECTION_LABEL_OVERRIDES: dict[str, str] = {
    "dashboard": "Web Defaults",
}


def _render_section_extras(
    cfg: "DashboardConfig", section_key: str, csrf_token: str,
) -> str:
    """Per-section augmentations rendered below the generic tree form.

    Currently used by Skills (list + upload + delete of ``.py`` files
    in ``user_skills_dir``) and Memory (a "New memory" inline editor
    that writes ``<memory.dir>/<name>.md``). Returns empty string for
    sections without extras."""
    if section_key == "skills":
        return _render_skills_extras(cfg, csrf_token)
    if section_key == "memory":
        return _render_memory_extras(cfg, csrf_token)
    return ""


def _render_skills_extras(cfg: "DashboardConfig", csrf_token: str) -> str:
    files = _list_user_skill_files(cfg)
    skills_dir = _resolved_user_skills_dir(cfg)
    rows: list[str] = []
    if files:
        for name in files:
            rows.append(
                "<li class='skill-file-row'>"
                f"<span class='skill-file-row__name'>{html.escape(name)}</span>"
                "<form method='post' action='/api/skills/delete' class='inline-form'>"
                f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
                f"<input type='hidden' name='filename' value='{html.escape(name)}'>"
                "<button type='submit' class='bx--btn bx--btn--ghost skill-file-row__del' "
                "aria-label='Delete skill'>"
                f"{_icon('trash-can')}Delete</button>"
                "</form>"
                "</li>"
            )
    else:
        rows.append(
            "<li class='muted'>No user skill files yet. "
            "Upload a <code>.py</code> module that calls "
            "<code>harness.skills.register(...)</code> at import time.</li>"
        )
    return (
        "<div class='skill-files'>"
        "<h4 class='skill-files__heading'>Skill files in "
        f"<code>{html.escape(skills_dir)}</code></h4>"
        f"<ul class='skill-file-list'>{''.join(rows)}</ul>"
        "<form method='post' action='/api/skills/upload' "
        "enctype='multipart/form-data' class='skill-upload-form'>"
        f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
        "<label class='bx--label' for='skill-upload-input'>Upload a new skill (.py)</label>"
        "<div class='skill-upload-row'>"
        "<input type='file' id='skill-upload-input' name='file' accept='.py' required>"
        "<button class='bx--btn bx--btn--secondary' type='submit'>"
        f"{_icon('upload')}Upload skill</button>"
        "</div>"
        "</form>"
        "</div>"
    )


def _render_memory_extras(cfg: "DashboardConfig", csrf_token: str) -> str:
    mem_dir = _resolved_memory_dir(cfg)
    return (
        "<div class='memory-new'>"
        "<h4 class='memory-new__heading'>Add a new memory</h4>"
        "<p class='muted fs-sm'>Writes "
        f"<code>{html.escape(mem_dir)}/&lt;name&gt;.md</code> "
        "and shows up on the <a href='/memory'>Memory page</a> immediately.</p>"
        "<form method='post' action='/api/memory/new' class='memory-new__form'>"
        f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
        "<div class='field'>"
        "<label class='bx--label' for='memory-new-name'>Name</label>"
        "<input class='bx--text-input' id='memory-new-name' name='name' "
        "type='text' placeholder='my-memory' required>"
        "</div>"
        "<div class='field'>"
        "<label class='bx--label' for='memory-new-content'>Content (markdown)</label>"
        "<textarea class='bx--text-area' id='memory-new-content' name='content' "
        "rows='6' placeholder='Markdown memory content' required></textarea>"
        "</div>"
        "<div class='actions'>"
        "<button class='bx--btn bx--btn--primary' type='submit'>"
        f"{_icon('save')}Save memory</button>"
        "<button class='bx--btn bx--btn--secondary ct-section__cancel' "
        "type='button'>Cancel</button>"
        "</div>"
        "</form>"
        "</div>"
    )


def _render_configure_harness(cfg: DashboardConfig) -> str:
    if not cfg.writes_enabled:
        return (
            "<div class='card'><p class='muted'>Writes are disabled "
            "by <code>dashboard.writes_enabled: false</code> in "
            "<code>config.json</code>. Current values are still "
            "viewable read-only at <a href='/config'>Configuration (raw)</a>.</p></div>"
        )

    from harness.config_tree import render_model_routing, render_tree
    from harness.web_forms import _CONFIG_GROUPS

    # Load the live config so the form pre-populates with current values.
    current_config: dict[str, Any] = {}
    try:
        config_path = cfg.config_path or _config_file_path(cfg)
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        current_config = {}
    # Strip the comment-only keys (those starting with "_") — they're
    # documentation only and the harness loader drops them at startup.
    current_config = {k: v for k, v in current_config.items() if not k.startswith("_")}

    csrf_token = resolve_csrf_token(cfg) or ""
    # Snapshot the file mtime so the save handler can detect concurrent
    # external edits and the JS poller can detect them while the page
    # is open. Empty string when the file is missing — the page is
    # creating a config from scratch, so there's nothing to be stale of.
    base_mtime_ns = config_file_mtime_ns(cfg)
    base_mtime_attr = "" if base_mtime_ns is None else str(base_mtime_ns)

    # Sections inside each group that have a closed-shape schema — for
    # these we render with allow_add_keys=False so the operator doesn't
    # see misleading + Add affordances at the top level.
    closed_shape_sections: set[str] = set()
    try:
        from harness.cli import _KNOWN_NESTED_KEYS
        closed_shape_sections = set(_KNOWN_NESTED_KEYS.keys())
    except Exception:  # noqa: BLE001
        pass

    # Sorted list of model keys from the live config — used by the
    # custom model_routing renderer to populate "Model name" dropdowns.
    available_models: list[str] = sorted(
        (current_config.get("models") or {}).keys()
    )

    def _render_section_editor(section_key: str) -> str:
        """Render one top-level section as a tree-shaped form posting
        to /config-tree/<section>. Sections missing from the live
        config get an empty editor (operator can fill in defaults)."""
        live_value = current_config.get(section_key)
        # Default empty containers for known sections so a missing
        # section still renders an editable shell.
        if live_value is None:
            live_value = {} if section_key in closed_shape_sections else ""
        # Sections whose existing UI complaint was "Save button has no
        # field to save" — pre-populate the known keys so each gets an
        # editable row even on a fresh install. Strings only; booleans
        # and ints in these sections aren't affected.
        if section_key == "github" and isinstance(live_value, dict):
            for known_key in ("gh_path", "default_owner", "default_repo"):
                live_value.setdefault(known_key, "")
        if section_key == "skills" and isinstance(live_value, dict):
            live_value.setdefault("user_skills_dir", "~/.harness/skills")
        allow_add = section_key not in closed_shape_sections
        # model_routing gets a custom grouped renderer — Role → Primary/
        # Fallback → Model + Thinking — instead of the generic flat tree.
        # The form fields it emits still use the same __path[] flat
        # schema, so parse_tree round-trips identically.
        if section_key == "model_routing" and isinstance(live_value, dict):
            tree_html = render_model_routing(
                live_value, path=section_key,
                available_models=available_models,
            )
        else:
            tree_html = render_tree(
                live_value, path=section_key, depth=0,
                allow_add_keys=allow_add,
            )
        # Summary line: how many top-level keys/entries.
        if isinstance(live_value, dict):
            count_text = f"({len(live_value)} entr{'y' if len(live_value) == 1 else 'ies'})"
        elif isinstance(live_value, list):
            count_text = f"({len(live_value)} item{'s' if len(live_value) != 1 else ''})"
        else:
            count_text = "(scalar)"
        display_name = _SECTION_LABEL_OVERRIDES.get(section_key, section_key)
        # Extras render below the standard tree form — used by sections
        # that surface filesystem-backed assets (Skills, Memory) alongside
        # their config knobs.
        extras_html = _render_section_extras(cfg, section_key, csrf_token)
        return (
            f"<details class='ct-section' data-section='{html.escape(section_key)}'"
            f" id='section-{html.escape(section_key)}'>"
            f"<summary class='ct-section__head'>"
            f"<span class='ct-section__toggle' aria-hidden='true'>+</span>"
            f"<span class='ct-section__name'>{html.escape(display_name)}</span>"
            f"<span class='muted ml-2 fs-sm'>{count_text}</span>"
            f"</summary>"
            f"<div class='ct-section__body'>"
            f"<form method='post' action='/config-tree/{html.escape(section_key)}' "
            f"class='ct-form'>"
            f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
            f"<input type='hidden' name='__base_mtime_ns' value='{html.escape(base_mtime_attr)}'>"
            f"<div class='ct-tree'>{tree_html}</div>"
            f"<div class='actions'>"
            f"<button class='bx--btn bx--btn--primary' type='submit'>"
            f"{_icon('save')}Save {html.escape(display_name)}</button>"
            f"<button class='bx--btn bx--btn--secondary ct-section__cancel' "
            f"type='button'>Cancel</button>"
            f"</div>"
            f"</form>"
            f"{extras_html}"
            f"</div></details>"
        )

    # Render each group from the existing _CONFIG_GROUPS so the page
    # stays organised. Any top-level key in the live config that's not
    # in any group lands in a synthesized "Other" group at the end.
    placed_sections: set[str] = set()
    group_blocks: list[str] = []
    for slug, title, section_names in _CONFIG_GROUPS:
        rendered_sections: list[str] = []
        for section_key in section_names:
            if section_key not in current_config and section_key not in closed_shape_sections \
                    and not isinstance(current_config.get(section_key), (dict, list)):
                # Section truly absent — skip in the UI (avoids noise).
                continue
            rendered_sections.append(_render_section_editor(section_key))
            placed_sections.add(section_key)
        if not rendered_sections:
            continue
        group_blocks.append(
            f"<details class='config-group' data-group='{html.escape(slug)}'>"
            f"<summary class='config-group__heading'>"
            f"<span class='config-group__toggle' aria-hidden='true'>+</span>"
            f"<span class='config-group__title'>{html.escape(title)}</span>"
            f"<span class='muted ml-2 fs-sm config-group__count'>"
            f"({len(rendered_sections)} section{'s' if len(rendered_sections) != 1 else ''})</span>"
            f"</summary>"
            f"<div class='config-group__body'>"
            + "".join(rendered_sections) +
            "</div></details>"
        )

    # Catch-all: any top-level key NOT in any group becomes "Other".
    other_sections: list[str] = []
    for section_key in sorted(current_config.keys()):
        if section_key in placed_sections:
            continue
        other_sections.append(_render_section_editor(section_key))
    if other_sections:
        group_blocks.append(
            f"<details class='config-group' data-group='other'>"
            f"<summary class='config-group__heading'>"
            f"<span class='config-group__toggle' aria-hidden='true'>+</span>"
            f"<span class='config-group__title'>Other</span>"
            f"<span class='muted ml-2 fs-sm config-group__count'>"
            f"({len(other_sections)} section{'s' if len(other_sections) != 1 else ''})</span>"
            f"</summary>"
            f"<div class='config-group__body'>"
            + "".join(other_sections) +
            "</div></details>"
        )

    # Banner placeholder driven by dashboard.js — when the poller sees
    # config.json's mtime change on disk it un-hides this card so the
    # operator knows their form values are stale.
    stale_banner = (
        "<div class='card fail config-stale-banner' "
        "id='config-stale-banner' hidden role='alert'>"
        "<p><strong>config.json was modified outside this tab.</strong> "
        "Your form values may be stale. "
        "<a href='/config-ui'>Reload the page</a> to see the latest values "
        "before saving.</p>"
        "</div>"
    )
    return (
        f"<div class='configure-page' "
        f"data-config-mtime-ns='{html.escape(base_mtime_attr)}' "
        f"data-config-mtime-poll-url='/api/config-mtime'>"
        f"{stale_banner}"
        "<div class='card'>"
        "<h2>Configuration sections</h2>"
        "<p class='muted'>Edit any value in <code>config.json</code> right here. "
        "The structure mirrors the JSON file. Collections that can grow show a "
        "<strong>+ Add</strong> button — for example, expand <em>LLM Registry → models</em> "
        "to register a new model. Save commits atomically and re-validates through "
        "the strict validator before landing.</p>"
        f"<div class='config-groups'>{''.join(group_blocks)}</div>"
        "</div>"
        "<div class='card'>"
        "<p class='muted'>Deployment defaults live in <code>config/deployment.json</code> — "
        "edit directly until web editing lands for that file.</p>"
        "</div>"
        "</div>"
    )


_DASHBOARD_TILES: tuple[tuple[str, str, str, str], ...] = (
    # (title, description, href, icon)
    ("View Status", "Day / week / month summary plus what's running right now.", "/status", "chart-line"),
    ("Cost burn-down", "Cumulative spend and per-call cost across every session.", "/cost", "chart-line"),
    ("Sessions list", "Every harness session on disk with exit code and token totals.", "/sessions", "list"),
    ("Schedule history", "Past runs from the cron-driven scheduled-job daemon.", "/schedule", "calendar"),
    ("Repo index", "Status of the semantic retrieval index per workspace.", "/index", "search"),
    ("Memory list", "Per-repo memory files appended after each session.", "/memory", "document"),
    ("Live runs", "Currently-running processes spawned from this dashboard.", "/live", "terminal"),
    ("Configuration (raw)", "Section-by-section view of the legacy config form.", "/config", "settings"),
)


def _render_dashboards_landing(cfg: DashboardConfig) -> str:
    tiles = []
    for title, desc, href, icon_name in _DASHBOARD_TILES:
        icon_svg = _icon(icon_name, size=24, klass="icon--lg dash-tile__icon")
        tiles.append(
            f"<div class='dash-tile'>"
            f"<a href='{html.escape(href)}'>"
            f"{icon_svg}"
            f"<div>"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(desc)}</p>"
            f"</div>"
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
            f"<td><a href='{href}'>{_icon('document')}{_esc(doc['relpath'])}</a></td>"
            f"<td class='num'>{_fmt_size(int(doc['size']))}</td>"
            f"<td>{modified}</td>"
            f"</tr>"
        )
    return (
        f"<div class='card'>"
        f"<h2>Documents</h2>"
        f"<p class='muted'>Source: <code>{_esc(docs_dir)}</code></p>"
        "<div class='table-wrap'><table id='docs-table'>"
        "<thead><tr><th>Document</th><th class='num'>Size</th><th>Modified (UTC)</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
        "</div>"
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
    crumb = _breadcrumb([("Documents", "/docs"), (relpath, None)])
    return 200, crumb + body


_ROUTES: list[Route] = [
    (re.compile(r"^/?$"), _route_root),
    (re.compile(r"^/sessions/?$"), _route_sessions),
    (re.compile(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/?$"), _route_session_detail),
    (re.compile(r"^/cost/?$"), _route_cost),
    (re.compile(r"^/api/cost-burn/?$"), _route_api_cost_burn),
    (re.compile(r"^/api/config-mtime/?$"), _route_api_config_mtime),
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

        def _send(self, status: int, content_type: str, body: Any,
                   *, extra_headers: Optional[dict[str, str]] = None,
                   cache_control: str = "no-store") -> None:
            # Accept str (UTF-8 encoded) or bytes (binary-safe). Static
            # assets need bytes for fonts/icons/favicons; the rest of
            # the app keeps emitting strings.
            data = body if isinstance(body, (bytes, bytearray)) else str(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache_control)
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

        # ---- Static assets -----------------------------------------------

        def _maybe_serve_static(self) -> bool:
            """If the request is for a static asset, serve it and return
            True. Otherwise return False so do_GET falls through to the
            normal routing."""
            parsed = urllib.parse.urlparse(self.path)
            decoded = urllib.parse.unquote(parsed.path)
            if decoded == "/favicon.ico":
                relpath = "favicon.ico"
            elif decoded.startswith("/static/"):
                relpath = decoded[len("/static/"):]
            else:
                return False
            status, ctype, data = _serve_static(cfg, relpath)
            # Long cache for assets so reloads are cheap; ETag/version
            # busting can come later if we start mutating files in place.
            cache = "public, max-age=3600" if status == 200 else "no-store"
            self._send(status, ctype, data, cache_control=cache)
            return True

        # ---- GET ----------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            # Static assets (css/js/icons/fonts/favicon) ship outside the
            # auth gate so they render correctly on the 401 page itself
            # and so air-gap mirrors don't need a token. They're public
            # by design: nothing in static_dir is sensitive.
            if self._maybe_serve_static():
                return
            ok, detail = self._is_authed()
            if not ok:
                self._send(401, "text/plain; charset=utf-8", f"401 unauthorized: {detail}\n")
                return
            # /api/browse needs query-string access that the route table
            # doesn't propagate, so it's handled inline here.
            parsed_url = urllib.parse.urlparse(self.path)
            if urllib.parse.unquote(parsed_url.path) == "/api/browse":
                q = urllib.parse.parse_qs(parsed_url.query)
                requested = (q.get("path", [""])[0] or "").strip()
                status, ctype, body = _browse_response(requested)
                self._send(status, ctype, body)
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
            content_type = self.headers.get("Content-Type") or ""

            # Multipart uploads go through the dedicated dispatch path so
            # the file bytes survive parsing. Everything else stays on
            # the legacy urlencoded path.
            if "multipart/form-data" in content_type.lower():
                try:
                    fields, files = _parse_multipart_body(raw, content_type)
                except ValueError as exc:
                    self._send(400, "text/plain", f"upload rejected: {exc}\n")
                    return
                self._dispatch_multipart(path, fields, files)
                return

            form = _parse_form_body(raw)

            # Route the POST to the right handler.
            self._dispatch_write(path, form)

        def _dispatch_write(self, path: str, form: dict[str, Any]) -> None:
            # /config-tree/<section> — the new tree-shaped editor.
            m = re.match(r"^/config-tree/(?P<section>[A-Za-z0-9_]+)/?$", path)
            if m:
                self._handle_config_tree_save(m.group("section"), form)
                return
            # /config/<section> — legacy curated editor (kept for the
            # /config raw page that still wires it).
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
            # /run/resume
            if path == "/run/resume":
                self._handle_run_resume(form)
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
            # /sessions/<sid>/purge — wipe everything: checkpoint rows
            # + JSONL log. Mirrors `harness purge --session-id`.
            m = re.match(r"^/sessions/(?P<sid>[A-Za-z0-9_.\-]+)/purge/?$", path)
            if m:
                self._handle_session_purge(m.group("sid"))
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
            # /api/skills/delete — urlencoded form with filename field.
            if path == "/api/skills/delete":
                self._handle_api_skills_delete(form)
                return
            # /api/memory/new — urlencoded form with name + content.
            if path == "/api/memory/new":
                self._handle_api_memory_new(form)
                return
            self._send(404, "text/plain", "404 not found\n")

        def _dispatch_multipart(
            self,
            path: str,
            fields: dict[str, str],
            files: dict[str, tuple[str, bytes]],
        ) -> None:
            """Route multipart POSTs introduced by the configure-page +
            run-page overhaul (product-spec upload, user-skills upload).
            Returns 404 for any path not explicitly recognised."""
            if path == "/api/upload-spec":
                self._handle_api_upload_spec(fields, files)
                return
            if path == "/api/skills/upload":
                self._handle_api_skills_upload(fields, files)
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
            base_mtime_ns = _extract_base_mtime_ns(form)
            ok, msg = write_config_section_atomic(
                cfg, section_name, parsed,
                expected_base_mtime_ns=base_mtime_ns,
            )
            if not ok:
                if msg.startswith(CONFIG_STALE_MARKER):
                    flash = msg[len(CONFIG_STALE_MARKER):].strip()
                    body = _render_config_section(
                        cfg, section_name, csrf_token=csrf_token,
                        flash=flash,
                    )
                    self._send(409, "text/html; charset=utf-8",
                               _layout(f"Config · {section_name}", body, cfg))
                    return
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

        def _handle_config_tree_save(
            self, section_name: str, form: dict[str, Any],
        ) -> None:
            """Save a top-level config section submitted by the tree
            editor. The form payload carries ``__path[]`` / ``__type[]``
            / ``__value[]`` arrays that :func:`parse_tree` rebuilds into
            the original nested shape. The reconstructed root is itself
            a dict whose only top-level key is ``section_name`` — we
            extract that value and hand it to ``write_config_section_atomic``."""
            from harness.config_tree import TreeParseError, parse_tree
            try:
                rebuilt = parse_tree(form)
            except TreeParseError as exc:
                self._send(
                    400, "text/html; charset=utf-8",
                    _layout(
                        f"Config · {section_name}",
                        f"<div class='card'><p class='fail'>"
                        f"Could not parse form: {html.escape(str(exc))}</p>"
                        f"<p><a href='/config-ui'>Back to Configure Harness</a></p>"
                        f"</div>",
                        cfg, active="config",
                    ),
                )
                return
            # ``rebuilt`` is the section dict (or list/scalar); the
            # paths the renderer emits are all prefixed with the section
            # name, so rebuilt looks like ``{section_name: <value>}``.
            section_value = rebuilt.get(section_name) if isinstance(rebuilt, dict) else rebuilt
            if section_value is None:
                # Either the operator cleared every field, or the section
                # was never present. Save an empty container so the validator
                # sees an explicit shape rather than vanishing.
                section_value = {}
            base_mtime_ns = _extract_base_mtime_ns(form)
            ok, msg = write_config_section_atomic(
                cfg, section_name, section_value,
                expected_base_mtime_ns=base_mtime_ns,
            )
            if not ok:
                # Re-render the page with the failure surfaced as a flash.
                # Stale-write conflicts get a 409 + plain message so the
                # operator immediately understands "reload, don't retry".
                body = _render_configure_harness(cfg)
                if msg.startswith(CONFIG_STALE_MARKER):
                    plain = msg[len(CONFIG_STALE_MARKER):].strip()
                    fail_card = (
                        f"<div class='card fail'><p>"
                        f"Save of <code>{html.escape(section_name)}</code> rejected: "
                        f"{html.escape(plain)}</p></div>"
                    )
                    status_code = 409
                else:
                    fail_card = (
                        f"<div class='card'><p class='fail'>"
                        f"Save of <code>{html.escape(section_name)}</code> failed: "
                        f"{html.escape(msg)}</p></div>"
                    )
                    status_code = 400
                self._send(
                    status_code, "text/html; charset=utf-8",
                    _layout("Configure Harness", fail_card + body, cfg, active="config"),
                )
                return
            try:
                append_audit(
                    db_path=cfg.web_db_path, action="config_tree_save",
                    target=section_name,
                    detail=json.dumps(section_value, default=str)[:4096],
                )
            except Exception:  # noqa: BLE001
                pass
            # PRG: redirect with ?saved=<section> so the toast surfaces
            # on the next render (wireToastFromQuery in dashboard.js).
            self.send_response(303)
            self.send_header(
                "Location",
                f"/config-ui?saved={urllib.parse.quote(section_name)}",
            )
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

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
            spec_file_path = str(form.get("spec_file_path") or "").strip()
            if not workspace:
                self._send(400, "text/plain", "workspace required\n")
                return
            workspace_resolved = os.path.expanduser(workspace)
            if not os.path.isdir(workspace_resolved):
                self._send(
                    400, "text/plain",
                    f"workspace not found: {workspace}\n",
                )
                return
            # The Product Requirement input accepts either pasted text or
            # an uploaded .txt/.md document (whose path lands in the
            # hidden ``spec_file_path`` field via /api/upload-spec). At
            # least one must be supplied so the harness has something to
            # work from.
            if not prompt and not spec_file_path:
                self._send(
                    400, "text/plain",
                    "Provide a product requirement: either enter text or "
                    "upload a .txt/.md document.\n",
                )
                return
            if len(prompt) > _RUN_PROMPT_MAX_CHARS:
                self._send(
                    400, "text/plain",
                    f"prompt too long ({len(prompt)} chars; "
                    f"max {_RUN_PROMPT_MAX_CHARS})\n",
                )
                return
            # Block parallel runs against the same workspace — the
            # build/patch pipeline isn't designed to interleave two
            # concurrent sessions safely.
            if get_process_registry().has_running_for_workspace(workspace):
                self._send(
                    409, "text/plain",
                    f"A run is already in progress for {workspace}. "
                    f"Try again once that build/patch cycle completes.\n",
                )
                return
            extra_args, flag_errors = _collect_run_argv(form)
            if flag_errors:
                self._send(400, "text/plain",
                           "invalid run options:\n  - " + "\n  - ".join(flag_errors) + "\n")
                return
            # When the operator both uploads a spec AND enters text, the
            # text rides along as a sibling ``web_input.md`` inside the
            # same ``product_spec/`` folder so the harness consolidator
            # picks up both inputs.
            if spec_file_path and prompt:
                try:
                    _write_web_input_sidecar(workspace, prompt)
                except OSError as exc:
                    logger.warning("[run] failed to write web_input.md: %s", exc)
            try:
                # Pass an empty prompt when only a spec file was uploaded
                # — the harness reads ``product_spec/*`` itself.
                wp = spawn_harness_run(
                    cfg, workspace=workspace,
                    prompt=prompt or "(see product_spec/)",
                    extra_args=extra_args,
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, "text/plain", f"spawn failed: {exc}\n")
                return
            self.send_response(303)
            self.send_header("Location", f"/sessions/{wp.session_id}")
            self.end_headers()

        def _handle_run_resume(self, form: dict[str, Any]) -> None:
            """Resume an existing checkpointed session. The form carries
            ``resume_session_id`` (required), plus optional ``workspace``
            override and ``prompt`` appendix. We spawn the same way
            ``cmd_resume`` would, and the dashboard tracks it via the
            existing process registry."""
            session_id = str(form.get("resume_session_id") or "").strip()
            if not session_id:
                self._send(400, "text/plain", "resume_session_id required\n")
                return
            # Workspace/prompt are optional; the resume CLI auto-detects
            # workspace from the checkpoint if omitted.
            workspace = str(form.get("workspace") or "").strip() or None
            prompt = str(form.get("prompt") or "").strip() or None
            if workspace:
                workspace_resolved = os.path.expanduser(workspace)
                if not os.path.isdir(workspace_resolved):
                    self._send(
                        400, "text/plain",
                        f"workspace not found: {workspace}\n",
                    )
                    return
            if prompt and len(prompt) > _RUN_PROMPT_MAX_CHARS:
                self._send(
                    400, "text/plain",
                    f"prompt too long ({len(prompt)} chars; "
                    f"max {_RUN_PROMPT_MAX_CHARS})\n",
                )
                return
            try:
                wp = spawn_harness_resume(
                    cfg, session_id=session_id,
                    workspace=workspace, prompt=prompt,
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, "text/plain", f"resume spawn failed: {exc}\n")
                return
            self.send_response(303)
            self.send_header("Location", f"/sessions/{wp.session_id}")
            self.end_headers()

        def _handle_run_schedule(self, form: dict[str, Any]) -> None:
            from datetime import datetime as _dt, timedelta as _td
            workspace = str(form.get("workspace") or "").strip()
            prompt = str(form.get("prompt") or "").strip()
            spec_file_path = str(form.get("spec_file_path") or "").strip()
            fire_raw = str(form.get("fire_at_utc") or "").strip()
            name = str(form.get("name") or "web-oneshot").strip() or "web-oneshot"
            if not workspace or not fire_raw:
                self._send(400, "text/plain", "workspace and fire_at_utc required\n")
                return
            if not prompt and not spec_file_path:
                self._send(
                    400, "text/plain",
                    "Provide a product requirement: either enter text or "
                    "upload a .txt/.md document.\n",
                )
                return
            try:
                fire_at = _dt.fromisoformat(fire_raw.replace("Z", "+00:00"))
                if fire_at.tzinfo is None:
                    from datetime import timezone as _tz
                    fire_at = fire_at.replace(tzinfo=_tz.utc)
            except ValueError as exc:
                self._send(400, "text/plain", f"fire_at_utc invalid: {exc}\n")
                return
            # Block back-to-back schedules — two jobs landing on the same
            # daemon tick can race on the same workspace or saturate the
            # subprocess pool. The window matches the 10-minute spacing
            # the configure-page overhaul surfaces in the UI hint.
            try:
                conflicts = find_oneshot_jobs_near(
                    db_path=cfg.web_db_path,
                    fire_at_utc=fire_at, window_minutes=10,
                )
            except Exception:  # noqa: BLE001 — DB issues shouldn't take the form down
                conflicts = []
            if conflicts:
                suggested = (fire_at + _td(minutes=10)).isoformat(timespec="seconds")
                clash = conflicts[0].get("fire_at_utc", "")
                self._send(
                    409, "text/plain",
                    f"A run is already scheduled at {clash} (within 10 "
                    f"minutes of {fire_at.isoformat(timespec='seconds')}). "
                    f"Pick a time at least 10 minutes apart, e.g. {suggested}.\n",
                )
                return
            extra_args, flag_errors = _collect_run_argv(form)
            if flag_errors:
                self._send(400, "text/plain",
                           "invalid run options:\n  - " + "\n  - ".join(flag_errors) + "\n")
                return
            if spec_file_path and prompt:
                try:
                    _write_web_input_sidecar(workspace, prompt)
                except OSError as exc:
                    logger.warning("[schedule] failed to write web_input.md: %s", exc)
            try:
                row_id = add_oneshot_job(
                    db_path=cfg.web_db_path, name=name,
                    fire_at_utc=fire_at, workspace=workspace,
                    prompt=prompt or "(see product_spec/)",
                    harness_args=extra_args,
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

        # ---- New file-upload + browser endpoints --------------------------

        def _handle_api_upload_spec(
            self,
            fields: dict[str, str],
            files: dict[str, tuple[str, bytes]],
        ) -> None:
            """Persist an uploaded product-spec document under
            ``<workspace>/product_spec/``. The form carries the
            workspace path (mandatory) plus the file part. Only ``.txt``
            and ``.md`` filenames are accepted — anything else returns
            400 so the JS surfaces a toast."""
            workspace = (fields.get("workspace") or "").strip()
            if not workspace:
                self._send(400, "text/plain", "workspace required\n")
                return
            file_part = files.get("file")
            if file_part is None:
                self._send(400, "text/plain", "missing file part\n")
                return
            filename, data = file_part
            saved_path, err = _persist_product_spec(workspace, filename, data)
            if err is not None:
                self._send(400, "text/plain", f"{err}\n")
                return
            try:
                append_audit(
                    db_path=cfg.web_db_path, action="spec_upload",
                    target=workspace,
                    detail=f"path={saved_path} bytes={len(data)}",
                )
            except Exception:  # noqa: BLE001
                pass
            payload = json.dumps({"ok": True, "saved_as": saved_path})
            self._send(200, "application/json; charset=utf-8", payload)

        def _handle_api_skills_upload(
            self,
            fields: dict[str, str],
            files: dict[str, tuple[str, bytes]],
        ) -> None:
            """Drop an uploaded ``.py`` skill into the configured user
            skills directory. The directory is created if missing."""
            file_part = files.get("file")
            if file_part is None:
                self._send(400, "text/plain", "missing file part\n")
                return
            filename, data = file_part
            saved_path, err = _persist_user_skill(cfg, filename, data)
            if err is not None:
                self._send(400, "text/plain", f"{err}\n")
                return
            try:
                append_audit(
                    db_path=cfg.web_db_path, action="skill_upload",
                    target=saved_path, detail=f"bytes={len(data)}",
                )
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/config-ui#skills")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _handle_api_skills_delete(self, form: dict[str, Any]) -> None:
            filename = str(form.get("filename") or "").strip()
            removed, err = _delete_user_skill(cfg, filename)
            if err is not None:
                self._send(400, "text/plain", f"{err}\n")
                return
            try:
                append_audit(
                    db_path=cfg.web_db_path, action="skill_delete",
                    target=removed, detail="",
                )
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/config-ui#skills")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _handle_api_memory_new(self, form: dict[str, Any]) -> None:
            name = str(form.get("name") or "").strip()
            content = str(form.get("content") or "")
            saved_path, err = _persist_new_memory(cfg, name, content)
            if err is not None:
                self._send(400, "text/plain", f"{err}\n")
                return
            try:
                append_audit(
                    db_path=cfg.web_db_path, action="memory_new",
                    target=saved_path, detail=f"bytes={len(content.encode('utf-8'))}",
                )
            except Exception:  # noqa: BLE001
                pass
            self.send_response(303)
            self.send_header("Location", "/config-ui#memory")
            self.send_header("Content-Length", "0")
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

        def _handle_session_purge(self, session_id: str) -> None:
            """Wipe ``session_id`` completely from harness memory:
            checkpoint rows in the SQLite store + the JSONL log file
            on disk. Mirrors ``harness purge --session-id <id>``.

            If the session is currently running, SIGTERM the process
            first and wait briefly for it to exit before purging — a
            running session would re-create the log file we just
            deleted.
            """
            import shlex as _shlex
            import subprocess as _sub
            import time as _time

            # 1. Stop the live process if any. cancel_session is a no-op
            # when the session isn't running, so this is safe either way.
            cancelled = cancel_session(session_id)
            if cancelled:
                # Give the process a brief grace period to flush + exit.
                for _ in range(20):  # up to 2s
                    entry = get_process_registry().get(session_id)
                    if entry is None or not entry.is_running:
                        break
                    _time.sleep(0.1)

            # 2. Spawn `harness purge --session-id <id>` synchronously
            # and pass --yes so it doesn't try to read from stdin. We
            # block until the subprocess returns so the redirect lands
            # an up-to-date list back at the operator.
            argv = ["harness", "purge", "--session-id", session_id]
            try:
                proc = _sub.run(
                    argv, capture_output=True, text=True, timeout=30,
                )
            except _sub.TimeoutExpired:
                self._send(
                    504, "text/plain",
                    f"purge timed out (>30s) for session {session_id}\n",
                )
                return
            except FileNotFoundError:
                self._send(
                    500, "text/plain",
                    "harness CLI not on PATH — cannot run purge\n",
                )
                return

            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                self._send(
                    500, "text/plain",
                    f"purge failed (exit {proc.returncode}): {detail or 'no detail'}\n",
                )
                return

            try:
                append_audit(
                    db_path=cfg.web_db_path, action="session_purge",
                    target=session_id,
                    detail=f"argv={_shlex.join(argv)} exit=0",
                )
            except Exception:  # noqa: BLE001
                pass

            # PRG: redirect to the Run page with the Resume tab
            # pre-selected and a toast telling the operator what just
            # happened. dashboard.js' wireToastFromQuery surfaces it.
            saved_msg = f"deleted+session+{session_id}"
            self.send_response(303)
            self.send_header(
                "Location",
                f"/run?mode=resume&saved={urllib.parse.quote(saved_msg)}",
            )
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
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
            # forever. Tunable via dashboard.hitl_webhook_timeout_seconds.
            timeout = float(cfg.hitl_webhook_timeout_seconds or 600.0)
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
    # One-shot audit-log retention sweep at server start. Otherwise the
    # table grows forever; even on a low-traffic dashboard that's
    # eventually a long-tail problem.
    try:
        from harness.web_state import prune_audit_log
        removed = prune_audit_log(
            db_path=cfg.web_db_path,
            days=int(cfg.audit_log_retention_days),
        )
        if removed:
            logger.info("[dashboard] pruned %d audit_log row(s) > %d days old",
                        removed, cfg.audit_log_retention_days)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[dashboard] audit_log prune skipped: %s", exc)
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
    find_oneshot_jobs_near,
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


def config_file_mtime_ns(cfg: DashboardConfig) -> Optional[int]:
    """Return the canonical ``config.json`` mtime in nanoseconds, or
    ``None`` when the file is missing / unreadable. Used to detect
    out-of-band edits between page render and save (Tier-B optimistic
    concurrency) and to drive the live-poll banner on the Configure
    page. Integer ns is exact across the JSON / form-data round trip —
    floats would lose precision."""
    path = _config_file_path(cfg)
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


CONFIG_STALE_MARKER = "__config_stale__"


def write_config_section_atomic(
    cfg: DashboardConfig, section: str, new_section_value: Any,
    *,
    expected_base_mtime_ns: Optional[int] = None,
) -> tuple[bool, str]:
    """Read the current config, replace ``section``, validate the
    merged result through the strict validator, write atomically.
    Returns ``(success, message)``. On validation failure the disk
    file is untouched.

    When ``expected_base_mtime_ns`` is supplied, the disk file's
    current mtime must match it — otherwise an external process has
    rewritten the file since the operator opened the page, and we
    refuse the save with a message prefixed by :data:`CONFIG_STALE_MARKER`
    so the caller can surface a "reload to see latest values" banner
    instead of a generic validation error."""
    path = _config_file_path(cfg)
    if not os.path.isfile(path):
        return False, f"config file not found at {path}"
    if expected_base_mtime_ns is not None:
        try:
            current_mtime_ns = os.stat(path).st_mtime_ns
        except OSError as exc:
            return False, f"could not stat config: {exc}"
        if current_mtime_ns != expected_base_mtime_ns:
            return False, (
                f"{CONFIG_STALE_MARKER} config.json was modified outside "
                f"this browser tab since the page was loaded. Reload to "
                f"see the current values, then re-apply your edits."
            )
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

    # Open the stdout sink, hand the FD to Popen (which dup's it into the
    # child), then close the parent's copy so the dashboard process doesn't
    # leak one FD per spawned run.
    stdout_fh = open(log_path + ".stdout", "ab")
    try:
        proc = _sub.Popen(
            argv,
            stdout=stdout_fh,
            stderr=_sub.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        stdout_fh.close()
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


def spawn_harness_resume(
    cfg: DashboardConfig,
    *,
    session_id: str,
    workspace: Optional[str] = None,
    prompt: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    harness_binary: str = "harness",
) -> WebProcess:
    """Spawn a ``harness resume --session-id <id>`` subprocess for an
    existing checkpointed session. Mirrors :func:`spawn_harness_run`'s
    registry / log-tail / HITL webhook wiring so the dashboard tracks
    the resumed session the same way it tracks a fresh one.

    ``workspace`` is optional — the resume CLI auto-detects from the
    checkpoint if omitted. ``prompt`` is optional and, when provided,
    appended to the resumed conversation.
    """
    import subprocess as _sub

    if not session_id:
        raise ValueError("session_id is required for resume")

    log_dir = os.path.expanduser(cfg.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Resume keeps the SAME session_id so subsequent log writes append
    # to the existing JSONL — that's what makes the session look "live"
    # again in the dashboard's session list.
    log_path = os.path.join(log_dir, f"{session_id}.jsonl")
    # Ensure the log file exists so the SSE tail can attach immediately.
    open(log_path, "a", encoding="utf-8").close()

    argv: list[str] = [harness_binary, "resume", "--session-id", session_id]
    if workspace:
        argv.extend(["--workspace", workspace])
    if prompt:
        argv.extend(["--prompt", prompt])
    argv += list(extra_args or [])

    env = dict(os.environ)
    env["HARNESS_HITL_WEBHOOK_URL"] = (
        f"http://{cfg.host}:{cfg.port}/hitl/webhook?session={session_id}"
    )
    if cfg.hitl_webhook_secret:
        env["HARNESS_HITL_WEBHOOK_SECRET"] = cfg.hitl_webhook_secret

    # See spawn_harness_run for why we close the parent FD after Popen.
    stdout_fh = open(log_path + ".stdout", "ab")
    try:
        proc = _sub.Popen(
            argv,
            stdout=stdout_fh,
            stderr=_sub.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        stdout_fh.close()
    wp = WebProcess(
        session_id=session_id, pid=proc.pid, argv=argv,
        log_path=log_path,
        workspace_path=workspace or "",
        prompt=prompt or "",
        popen=proc,
    )
    get_process_registry().register(wp)

    def _watch():
        try:
            ec = proc.wait()
        except Exception:  # noqa: BLE001
            ec = -1
        get_process_registry().mark_terminated(session_id, int(ec or 0))
        try:
            append_audit(
                db_path=cfg.web_db_path, action="resume_exit",
                target=session_id, detail=f"exit_code={ec}",
            )
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_watch, daemon=True, name=f"web-resume-{session_id}").start()
    try:
        append_audit(
            db_path=cfg.web_db_path, action="run_resume",
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
        base_mtime_ns = config_file_mtime_ns(cfg)
        base_mtime_attr = "" if base_mtime_ns is None else str(base_mtime_ns)
        save = (
            f"<input type='hidden' name='csrf_token' value='{html.escape(csrf_token)}'>"
            f"<input type='hidden' name='__base_mtime_ns' value='{html.escape(base_mtime_attr)}'>"
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
        sse_url = f"/api/sessions/{_esc(session_id)}/events"
        parts.append(
            "<div class='card'><h3>Live events</h3>"
            "<p class='muted'>Streaming from "
            f"<code>{sse_url}</code>. Click a chip to hide an event type.</p>"
            "<div class='event-stream-filters' role='group' "
            "aria-label='Filter events by type'></div>"
            "<ul id='event-stream' class='event-stream' "
            f"data-sse-url='{sse_url}'></ul>"
            "</div>"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Form parsing — read the POST body
# ---------------------------------------------------------------------------

def _extract_base_mtime_ns(form: dict[str, Any]) -> Optional[int]:
    """Pull the hidden ``__base_mtime_ns`` field out of a POSTed form
    and return it as an int. Returns ``None`` when the field is missing
    or empty (no baseline → skip the stale check; legitimate when the
    config file didn't exist at render time)."""
    raw = form.get("__base_mtime_ns")
    if isinstance(raw, list):
        raw = raw[-1] if raw else None
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


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


_MAX_MULTIPART_BYTES = 5 * 1024 * 1024  # 5 MiB — guards against huge uploads
_MULTIPART_DISPOSITION_RE = re.compile(
    rb'form-data;\s*(?:name="(?P<name>[^"]*)")?'
    rb'(?:[^;]*?;\s*filename="(?P<filename>[^"]*)")?',
    re.IGNORECASE,
)


def _parse_multipart_body(
    body: bytes, content_type: str,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse a ``multipart/form-data`` body.

    Returns ``(fields, files)`` where ``fields`` maps field name → text
    value (UTF-8 decoded) and ``files`` maps field name → (filename, bytes).

    Deliberately minimal — supports the cases the dashboard's file
    upload endpoints need (one or two text fields plus one file part).
    Multiple parts sharing the same name keep only the last; that matches
    how the form-encoded path already collapses single-value lists.

    Raises ``ValueError`` when the body is malformed or oversized — callers
    surface a 400.
    """
    if len(body) > _MAX_MULTIPART_BYTES:
        raise ValueError(
            f"upload exceeds {_MAX_MULTIPART_BYTES} bytes "
            f"(got {len(body)})"
        )
    ct = content_type or ""
    if "multipart/form-data" not in ct.lower():
        raise ValueError("content-type is not multipart/form-data")
    # Extract the boundary marker. RFC 2046 allows the value to be
    # quoted; strip optional whitespace + quotes either way.
    boundary = ""
    for part in ct.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise ValueError("missing multipart boundary")
    sep = b"--" + boundary.encode("ascii", errors="replace")
    # Strip the leading separator + the trailing close marker.
    chunks = body.split(sep)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for chunk in chunks:
        chunk = chunk.lstrip(b"\r\n")
        if not chunk or chunk == b"--" or chunk.startswith(b"--"):
            continue
        # Header / body delimiter is the empty line.
        hb_split = chunk.split(b"\r\n\r\n", 1)
        if len(hb_split) != 2:
            continue
        headers_blob, value = hb_split
        # Drop the trailing CRLF that precedes the next boundary.
        if value.endswith(b"\r\n"):
            value = value[:-2]
        name: Optional[str] = None
        filename: Optional[str] = None
        for header_line in headers_blob.split(b"\r\n"):
            if header_line.lower().startswith(b"content-disposition:"):
                disp = header_line.split(b":", 1)[1].strip()
                m = _MULTIPART_DISPOSITION_RE.match(disp)
                if m:
                    nb = m.group("name")
                    fb = m.group("filename")
                    if nb is not None:
                        name = nb.decode("utf-8", errors="replace")
                    if fb is not None:
                        filename = fb.decode("utf-8", errors="replace")
                break
        if not name:
            continue
        if filename is not None:
            files[name] = (filename, value)
        else:
            fields[name] = value.decode("utf-8", errors="replace")
    return fields, files


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
