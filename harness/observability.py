"""
Observability — structured logging, per-session log files, and LangSmith tracing.

This module establishes the harness's observability layer on top of Python's
stdlib logging. It is intentionally free of heavy dependencies: structured
JSON logs use a custom Formatter, and LangSmith tracing is opt-in behind an
import guard.

Integration points:
  - cli.py: call configure_logging() once after config discovery, before
    graph execution.
  - gateway.py: call emit_event("llm_call", ...) after each dispatch.
  - sandbox.py: call emit_event("build_start"/"build_end", ...).
  - graph.py: call emit_event("node_transition", ...) from route functions.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 0. Active-session ContextVar
# ---------------------------------------------------------------------------
#
# Propagates the running session_id to downstream code (most importantly
# ``Gateway.dispatch``) without threading it through every call. ContextVar
# is asyncio-aware: concurrent dispatches from speculative variants each see
# the right session because LangGraph fans out under the same context.
#
# Set once at graph entry via ``set_active_session_id`` and reset at exit
# (use ``with active_session_scope(sid):`` for safety). Defaults to
# "unknown" when the harness is invoked outside a graph runner (e.g. unit
# tests) so dump filenames stay sortable even then.

_active_session_id: ContextVar[str] = ContextVar(
    "harness_active_session_id", default="unknown",
)


def set_active_session_id(session_id: str) -> Any:
    """Bind ``session_id`` to the current asyncio context.

    Returns the ``ContextVar.Token`` so callers can ``reset`` later. Prefer
    ``active_session_scope`` for context-manager semantics.
    """
    return _active_session_id.set(session_id or "unknown")


def get_active_session_id() -> str:
    """Read the current session_id, or ``"unknown"`` when unset."""
    return _active_session_id.get()


class active_session_scope:  # noqa: N801 — context-manager naming convention
    """``with`` block that binds and unbinds the active session id.

    Usage::

        with active_session_scope(state["session_id"]):
            await compiled_graph.ainvoke(...)
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._token: Optional[Any] = None

    def __enter__(self) -> "active_session_scope":
        self._token = _active_session_id.set(self._session_id or "unknown")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._token is not None:
            _active_session_id.reset(self._token)
            self._token = None


# ---------------------------------------------------------------------------
# 1. JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Emit one compact JSON object per log line.

    Standard fields included on every record:
      ts     — ISO 8601 UTC timestamp
      level  — DEBUG / INFO / WARNING / ERROR / CRITICAL
      logger — module name (logging.getLogger(__name__) convention)
      msg    — the formatted message string

    Any ``extra`` keyword arguments passed to the logger call are merged
    into the JSON object at the top level, making them first-class query
    targets in log-analysis tools.
    """

    # Stdlib LogRecord attributes that we handle explicitly; all other
    # attributes from extra= are passed through to the output.
    _STDLIB_ATTRS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "taskName",
        "message",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        obj: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        # Merge any extra= fields supplied by the caller
        for key, value in record.__dict__.items():
            if key not in self._STDLIB_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # only include JSON-serialisable values
                    obj[key] = value
                except (TypeError, ValueError):
                    obj[key] = repr(value)

        return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2. Structured event helper
# ---------------------------------------------------------------------------

_event_logger = logging.getLogger("harness.events")


def emit_event(name: str, **fields: Any) -> None:
    """
    Emit a structured event record (INFO level — successful / observational).

    Events are logged to the ``harness.events`` logger with an ``event``
    field, so they are trivially grep-able in the per-session JSONL file:

        jq 'select(.event == "llm_call")' ~/.harness/logs/<session>.jsonl

    Args:
        name:   Short snake_case event name (e.g. "llm_call", "build_end").
        fields: Arbitrary key-value payload merged into the JSON record.
    """
    _event_logger.info("", extra={"event": name, **fields})


# Process-start reference for incident wall-clock. Captured at import so a
# build run measures its own elapsed time; a resumed run (fresh process)
# measures time-since-resume rather than spanning the human pause, which is
# the number an incident-cost analysis actually wants.
_PROCESS_START_MONOTONIC = time.monotonic()


def process_uptime_s() -> float:
    """Seconds since this harness process started (monotonic, resume-safe)."""
    return max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC)


# Normalized incident-cause vocabulary. Maps the proximate HITL trigger
# string (produced by graph._infer_hitl_trigger) to a stable category so a
# post-hoc "where does the pain go" analysis groups by cause without parsing
# free-form trigger text — the exact archaeology this telemetry replaces.
# Ordered most-specific first; the first predicate that matches wins.
def classify_incident_cause(trigger: str) -> str:
    """Bucket a HITL/long-loop trigger into a stable cause category.

    Categories (grep targets): ``test_unsatisfiable``, ``test_generation``,
    ``test_traceability``, ``patching_zero_patch``, ``patching_stuck_target``,
    ``patching_allowlist``, ``repair_distraction``, ``no_progress``,
    ``budget_exhausted``, ``spec_decomposition``, ``security``,
    ``env_infra``, ``build_failure``, ``llm_behavior``, ``other``.

    Note the proximate trigger is not always the root cause — a
    ``patching_zero_patch`` loop is frequently a *test* problem underneath
    (repair stuck on a test it may not edit). The incident event carries a
    separate ``on_test_file`` flag to disambiguate those; this function only
    classifies the trigger surface.
    """
    tl = (trigger or "").strip().lower()
    if not tl:
        return "other"
    if tl.startswith("unsatisfiable_test"):
        return "test_unsatisfiable"
    if tl.startswith("llm_behavior:test_generation") or "test_generation" in tl:
        return "test_generation"
    if tl.startswith("traceability"):
        return "test_traceability"
    if tl.startswith("zero_patch_loop"):
        return "patching_zero_patch"
    if tl.startswith("replace_block_stuck"):
        return "patching_stuck_target"
    if tl.startswith("all_allowlist_rejected"):
        return "patching_allowlist"
    if "distraction" in tl:
        return "repair_distraction"
    if tl.startswith("repair_loop") or tl.startswith("repair_limit"):
        return "repair_limit"
    if tl.startswith("no_progress"):
        return "no_progress"
    if tl.startswith("budget"):
        return "budget_exhausted"
    if tl.startswith("decomposition"):
        return "spec_decomposition"
    if tl.startswith("security"):
        return "security"
    if tl.startswith("env_misconfig") or tl.startswith("build_command"):
        return "env_infra"
    if tl.startswith("persistent_build_failure") or tl.startswith("build_failure"):
        return "build_failure"
    if tl.startswith("llm_behavior"):
        return "llm_behavior"
    return "other"


def emit_incident(
    *,
    trigger: str,
    session_id: str = "",
    usd_spent: Optional[float] = None,
    wall_clock_s: Optional[float] = None,
    on_test_file: Optional[bool] = None,
    rounds: Optional[int] = None,
    node: str = "",
    story_id: str = "",
    modified_files: int = 0,
    cause: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit a structured ``incident`` event for a HITL or long-loop stall.

    One record per expensive juncture, carrying the three things the
    monthly "do we need the LLM for tests" decision needs and that the
    pre-existing ``hitl_fired`` event lacked: a normalized **cause**, the
    **$ spent** this run, and the **wall-clock** consumed — plus
    ``on_test_file`` to separate test-caused stalls from prod-code ones.

    Query::

        jq 'select(.event=="incident")' ~/.harness/logs/<session>.jsonl

    ``usd_spent``/``wall_clock_s`` default to None → the caller didn't
    supply them; ``wall_clock_s`` falls back to process uptime. Emitted at
    WARNING so incidents surface above the INFO event stream.
    """
    payload: dict[str, Any] = {
        "cause": cause or classify_incident_cause(trigger),
        "trigger": trigger,
        "session_id": session_id,
        "usd_spent": (
            round(float(usd_spent), 6) if usd_spent is not None else None
        ),
        "wall_clock_s": round(
            wall_clock_s if wall_clock_s is not None else process_uptime_s(), 1
        ),
        "on_test_file": on_test_file,
        "rounds": rounds,
        "node": node,
        "story_id": story_id,
        "modified_files": modified_files,
    }
    payload.update(extra)
    # Drop None values so the JSON stays lean and queries don't trip on nulls.
    clean = {k: v for k, v in payload.items() if v is not None}
    _event_logger.warning("", extra={"event": "incident", **clean})


def summarize_incidents(paths: "list[str]") -> dict[str, Any]:
    """Aggregate ``incident`` events across session logs into the
    cause × cost distribution the "where does the pain go" decision needs.

    Reads each JSONL path, keeps ``event == "incident"`` records, and
    buckets by ``cause`` with count, summed ``usd_spent``, summed
    ``wall_clock_s``, and how many were flagged ``on_test_file``. Also rolls
    up a ``test_share`` (fraction of incidents whose cause is a test bucket
    OR which were flagged on a test file) — the single number that answers
    "are tests the problem". Malformed lines and unreadable files are
    skipped, not fatal. Pure over its inputs; no globbing or clock reads.
    """
    _TEST_CAUSES = {
        "test_unsatisfiable", "test_generation", "test_traceability",
    }
    by_cause: dict[str, dict[str, float]] = {}
    total = 0
    test_related = 0
    total_usd = 0.0
    total_wall = 0.0
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for ln in lines:
            if '"incident"' not in ln:
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("event") != "incident":
                continue
            cause = str(rec.get("cause") or "other")
            usd = float(rec.get("usd_spent") or 0.0)
            wall = float(rec.get("wall_clock_s") or 0.0)
            on_test = bool(rec.get("on_test_file"))
            b = by_cause.setdefault(
                cause, {"count": 0, "usd": 0.0, "wall_s": 0.0, "on_test": 0},
            )
            b["count"] += 1
            b["usd"] += usd
            b["wall_s"] += wall
            b["on_test"] += int(on_test)
            total += 1
            total_usd += usd
            total_wall += wall
            if cause in _TEST_CAUSES or on_test:
                test_related += 1
    return {
        "total_incidents": total,
        "total_usd": round(total_usd, 4),
        "total_wall_s": round(total_wall, 1),
        "test_related": test_related,
        "test_share": round(test_related / total, 3) if total else 0.0,
        "by_cause": {
            c: {
                "count": int(v["count"]),
                "usd": round(v["usd"], 4),
                "wall_s": round(v["wall_s"], 1),
                "on_test": int(v["on_test"]),
            }
            for c, v in sorted(
                by_cause.items(), key=lambda kv: -kv[1]["count"],
            )
        },
    }


def log_failure(name: str, **fields: Any) -> None:
    """
    Emit a structured failure event (ERROR level).

    Mirror of ``emit_event`` for failure paths. Pre-existing failures used
    ad-hoc ``logger.error("...")`` strings, which meant grepping for
    failures required scanning by message fragment instead of by an
    event name. Standardising on ``log_failure(name, **fields)`` makes
    failure modes a first-class catalogue:

        jq 'select(.event == "sandbox_start_failed")' ~/.harness/logs/<session>.jsonl

    Canonical failure names currently emitted:
      - sandbox_start_failed     — sandbox auto-detect could not find a backend
      - token_budget_exhausted   — gateway refused dispatch (hard_cap_usd hit)
      - hitl_gate_blocked        — developer chose abandon at a HITL gate

    Args:
        name:   Short snake_case event name. Use the suffix ``_failed``,
                ``_exhausted``, or ``_blocked`` so the catalogue stays
                scannable. Add new names to the list above when wiring
                a new failure site.
        fields: Arbitrary key-value payload merged into the JSON record.
    """
    _event_logger.error("", extra={"event": name, **fields})


# ---------------------------------------------------------------------------
# 3. Logging configuration
# ---------------------------------------------------------------------------

# Defaults for the rotating file handler. A typical session writes a few
# hundred KB of JSONL; 10 MB × 5 backups gives ~50 MB per session before
# the oldest backup is dropped — enough for post-mortem of any single run
# without unbounded disk growth over weeks of operation.
_DEFAULT_LOG_MAX_BYTES = 10_000_000
_DEFAULT_LOG_BACKUP_COUNT = 5


def configure_logging(
    session_id: str,
    log_dir: str = "~/.harness/logs",
    level: str = "INFO",
    langsmith_enabled: bool = False,
    json_stderr: bool = False,
    max_bytes: int = _DEFAULT_LOG_MAX_BYTES,
    backup_count: int = _DEFAULT_LOG_BACKUP_COUNT,
    console_level: Optional[str] = None,
) -> Optional[str]:
    """
    Configure the harness logging stack for a session.

    Call once at CLI startup, after config discovery, before graph execution.

    Sets up:
    1. Stderr handler — human-readable (default) or JSON (when json_stderr=True).
    2. File handler — one JSONL file per session at <log_dir>/<session_id>.jsonl.
    3. LangSmith client — when langsmith_enabled=True AND LANGCHAIN_API_KEY is set.

    Args:
        session_id:        The harness session ID (used as the log filename).
        log_dir:           Directory for JSONL session log files.
        level:             Root log level string ("DEBUG", "INFO", etc.).
        langsmith_enabled: Opt-in to LangSmith tracing.
        json_stderr:       Emit JSON to stderr instead of the default human format.
        max_bytes:         Rotate the session log file when it grows past this
                           many bytes (0 disables rotation; falls back to a
                           plain FileHandler for parity with legacy behavior).
        backup_count:      Number of rotated backups to keep alongside the
                           live file (oldest is deleted on the next rotation).
        console_level:     Threshold for the live stderr (console) handler,
                           independent of the file handler. None → mirror
                           ``level`` (legacy behavior). "WARNING" keeps the
                           console quiet while the file still captures the
                           full INFO stream to ``tail -f``. "OFF" / "NONE"
                           silences the console entirely.

    Returns:
        Absolute path of the created session log file, or None on failure.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers the root logger already has so we own the config.
    root.handlers.clear()

    # --- stderr (console) handler ---
    # console_level governs what the operator sees live on the terminal,
    # independently of the file handler (which always captures the full
    # stream). Set console_level to "OFF"/"NONE"/"SILENT" to suppress the
    # console entirely and rely solely on `tail -f <log_file>`.
    console_level_str = (console_level or level).strip().upper()
    if console_level_str not in {"OFF", "NONE", "SILENT"}:
        console_numeric = getattr(logging, console_level_str, numeric_level)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(console_numeric)
        if json_stderr:
            stderr_handler.setFormatter(JSONFormatter())
        else:
            stderr_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            ))
        root.addHandler(stderr_handler)

    # --- per-session JSONL file handler ---
    # Rotating by default so a long pilot session can't silently fill the
    # operator's disk. When max_bytes=0 we drop back to plain FileHandler
    # (used by tests that assert on a single non-rotating .jsonl file).
    log_file_path: Optional[str] = None
    try:
        expanded_dir = os.path.expanduser(log_dir)
        os.makedirs(expanded_dir, exist_ok=True)
        candidate_path = os.path.join(expanded_dir, f"{session_id}.jsonl")
        # Audit §5.12: if the canonical session log already exists AND
        # is being actively appended (different process), suffix this
        # writer's filename with our PID so the RotatingFileHandler's
        # doRollover doesn't race another process for the same files.
        # Metrics aggregation globs <session_id>*.jsonl so multi-PID
        # variants are still discovered and summed.
        if os.path.exists(candidate_path):
            try:
                # Heuristic: file modified in the last 30s → assume a
                # live writer. Otherwise it's just our prior crash;
                # reuse the canonical name.
                age = time.time() - os.stat(candidate_path).st_mtime
            except OSError:
                age = 1e9
            if age < 30.0:
                candidate_path = os.path.join(
                    expanded_dir, f"{session_id}.{os.getpid()}.jsonl",
                )
        file_handler: logging.Handler
        if max_bytes and max_bytes > 0:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                candidate_path,
                maxBytes=max_bytes,
                backupCount=max(0, backup_count),
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(candidate_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)
        log_file_path = candidate_path  # only set after handler is live
        # Maintain a stable <log_dir>/latest.jsonl symlink → current session
        # so `tail -f ~/.harness/logs/latest.jsonl` works without knowing the
        # session id in advance. Best-effort; ignore platforms/FS that refuse.
        try:
            latest_link = os.path.join(expanded_dir, "latest.jsonl")
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.unlink(latest_link)
            os.symlink(os.path.basename(candidate_path), latest_link)
        except OSError:
            pass
        logging.getLogger("harness").info(
            "Session log file: %s", log_file_path, extra={"session_id": session_id}
        )
        # Print the resolved path unconditionally so it is visible even when
        # the console handler is quieted (console_level=WARNING/OFF).
        print(f"[teane] session log → {log_file_path}", file=sys.stderr, flush=True)
    except OSError as e:
        logging.getLogger("harness").warning(
            "Could not create session log file in %s: %s", log_dir, e
        )

    # --- optional LangSmith tracing ---
    if langsmith_enabled:
        _init_langsmith(session_id)

    return log_file_path


def _init_langsmith(session_id: str) -> None:
    """Initialise LangSmith tracing if the SDK and API key are available."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        logging.getLogger("harness.observability").info(
            "LangSmith tracing requested but LANGCHAIN_API_KEY is not set. Skipping."
        )
        return
    try:
        import langsmith  # noqa: F401 — the import itself registers the callback
        os.environ.setdefault("LANGSMITH_PROJECT", f"harness-{session_id[:8]}")
        os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
        logging.getLogger("harness.observability").info(
            "LangSmith tracing enabled (project=%s).",
            os.environ.get("LANGSMITH_PROJECT"),
        )
    except ImportError:
        logging.getLogger("harness.observability").warning(
            "langsmith package not installed — LangSmith tracing disabled. "
            "Install with: pip install langsmith"
        )
