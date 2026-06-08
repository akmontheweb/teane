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
from datetime import datetime, timezone
from typing import Any, Optional


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
    Emit a structured event record.

    Events are logged at INFO level to the ``harness.events`` logger.
    Because they carry a ``event`` field in the JSON output they are
    trivially grep-able in the per-session JSONL file:

        jq 'select(.event == "llm_call")' ~/.harness/logs/<session>.jsonl

    Args:
        name:   Short snake_case event name (e.g. "llm_call", "build_end").
        fields: Arbitrary key-value payload merged into the JSON record.
    """
    _event_logger.info("", extra={"event": name, **fields})


# ---------------------------------------------------------------------------
# 3. Logging configuration
# ---------------------------------------------------------------------------

def configure_logging(
    session_id: str,
    log_dir: str = "~/.harness/logs",
    level: str = "INFO",
    langsmith_enabled: bool = False,
    json_stderr: bool = False,
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

    Returns:
        Absolute path of the created session log file, or None on failure.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers the root logger already has so we own the config.
    root.handlers.clear()

    # --- stderr handler ---
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(numeric_level)
    if json_stderr:
        stderr_handler.setFormatter(JSONFormatter())
    else:
        stderr_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
    root.addHandler(stderr_handler)

    # --- per-session JSONL file handler ---
    log_file_path: Optional[str] = None
    try:
        expanded_dir = os.path.expanduser(log_dir)
        os.makedirs(expanded_dir, exist_ok=True)
        candidate_path = os.path.join(expanded_dir, f"{session_id}.jsonl")
        file_handler = logging.FileHandler(candidate_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)
        log_file_path = candidate_path  # only set after handler is live
        logging.getLogger("harness").info(
            "Session log file: %s", log_file_path, extra={"session_id": session_id}
        )
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
