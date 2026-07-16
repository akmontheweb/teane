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
