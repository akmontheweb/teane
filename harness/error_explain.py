"""Plain-English explanations for session end conditions.

Phase 8 of the consumer-grade UI overhaul. Pure function: takes a
session's tail of events + optional stderr tail + exit code, returns
an :class:`ErrorExplanation` the dashboard can render alongside a
"Resume" button.

Design notes:
* The rule table is intentionally short. A giant lookup table becomes
  its own maintenance burden; the fallback (raw last line) is fine
  when nothing matches — the operator still gets *something*.
* Every rule returns a ``suggested_action`` string, because "here's
  what to do" is more actionable than "here's what broke."
* The status vocabulary mirrors the existing exit-code categorisation
  used elsewhere in the dashboard: ok / crashed / killed / budget /
  running.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


STATUS_OK       = "ok"
STATUS_CRASHED  = "crashed"
STATUS_KILLED   = "killed"
STATUS_BUDGET   = "budget"
STATUS_RUNNING  = "running"


@dataclass(frozen=True)
class ErrorExplanation:
    """What the session-end card renders."""

    status: str
    headline: str
    cause: str
    suggested_action: str


# Ordered pattern → explanation. First hit wins; put the more-specific
# patterns first. Every entry documents WHY it's here — a stray "429
# rate limited" rule sitting next to a truthy "network error" rule can
# hide the more actionable message otherwise.
_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    # LLM provider errors — the most common actionable failure. Each
    # gets a specific remediation line rather than "check your keys."
    (re.compile(r"529\s+overloaded", re.I),
     "The LLM provider is overloaded",
     "Retry in a few minutes; providers usually recover on their own."),
    (re.compile(r"\bAPI(?:key)?\b.*(?:invalid|expired|revoked)", re.I),
     "API key rejected",
     "Set a fresh key in ~/.teane/env.sh and re-source your shell, "
     "then Resume."),
    (re.compile(r"\brate.?limit(ed|ing)?\b", re.I),
     "Rate-limited by the LLM provider",
     "Wait until the provider's quota window resets or lower this "
     "run's concurrency (speculative.num_variants), then Resume."),
    (re.compile(r"\bbudget.*(exceed|exhaust)", re.I),
     "Token budget exhausted",
     "Raise token_budget.hard_cap_usd in config.json (or apply the "
     "Balanced preset for a moderate cap), then Resume."),
    # Sandbox / build failures.
    (re.compile(r"docker.*not\s+(installed|found|running)", re.I),
     "Docker isn't running",
     "Start Docker Desktop (or the docker daemon) and Resume, or "
     "switch sandbox.backend to 'unshare' if you're on Linux."),
    (re.compile(r"disk\s+full|no\s+space\s+left", re.I),
     "The disk filled up mid-run",
     "Free some space (temp files, docker images) and Resume."),
    # Process-level.
    (re.compile(r"SIG(TERM|INT|KILL)|received\s+signal|process\s+killed", re.I),
     "The run was killed",
     "Restart with Resume to pick up from the last checkpoint."),
    (re.compile(r"connection\s+(reset|refused|aborted)", re.I),
     "Network connection dropped",
     "Check network connectivity, then Resume."),
)


def _last_event(events: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    last = None
    for evt in events:
        last = evt
    return last


def _first_failure_after(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The most-recent failure-shaped event. Used when session_end
    itself is clean but a preceding failure explains the exit."""
    for evt in reversed(events):
        name = str(evt.get("event") or "")
        if name.endswith("_failed") or name == "log_failure" or evt.get("error"):
            return evt
    return None


def _match_rules(text: str) -> Optional[ErrorExplanation]:
    for pattern, headline, action in _RULES:
        if pattern.search(text):
            snippet = text.strip().splitlines()[-1] if text.strip() else ""
            return ErrorExplanation(
                status=STATUS_CRASHED,
                headline=headline,
                cause=snippet[:240],
                suggested_action=action,
            )
    return None


def explain_session_end(
    events: list[dict[str, Any]],
    *,
    stderr_tail: str = "",
    exit_code: Optional[int] = None,
    process_still_running: bool = False,
) -> ErrorExplanation:
    """Produce a plain-English explanation for the session's state.

    Precedence:
    1. Still running → status="running", no explanation.
    2. session_end event present → use its exit_code and inspect
       preceding failures for the cause.
    3. Otherwise, treat as "process gone unexpectedly" (killed).
    """
    if process_still_running:
        return ErrorExplanation(
            status=STATUS_RUNNING,
            headline="Session is still running",
            cause="",
            suggested_action="",
        )

    last = _last_event(events)
    last_name = str((last or {}).get("event") or "")

    if last_name == "session_end":
        effective_exit = last.get("exit_code")
        if effective_exit is None:
            effective_exit = exit_code
        try:
            code_int = int(effective_exit) if effective_exit is not None else None
        except (TypeError, ValueError):
            code_int = None
        if code_int == 0:
            return ErrorExplanation(
                status=STATUS_OK,
                headline="Session finished cleanly",
                cause="Exit code 0",
                suggested_action="",
            )
        # Non-zero exit: try to pin down why.
        failure = _first_failure_after(events)
        text_sources = []
        if failure:
            text_sources.append(str(failure.get("error") or failure.get("message") or ""))
            text_sources.append(str(failure.get("event") or ""))
        text_sources.append(stderr_tail)
        combined = "\n".join(t for t in text_sources if t)
        matched = _match_rules(combined)
        if matched:
            # A rule match already carries status="crashed"; preserve
            # its headline / action.
            return matched
        # No rule matched: fall back to the raw stderr / failure text.
        raw = combined.strip().splitlines()[-1] if combined.strip() else ""
        return ErrorExplanation(
            status=STATUS_CRASHED,
            headline=f"Session exited with code {code_int}",
            cause=raw[:240] or "No detail available in the log tail.",
            suggested_action="Read the log for context, then Resume "
                             "when the underlying issue is fixed.",
        )

    # Log tail is not session_end and process is not running.
    # Interpret as an external kill / process gone.
    return ErrorExplanation(
        status=STATUS_KILLED,
        headline="Process was killed externally",
        cause="Session log has no session_end event; the harness "
              "process is no longer registered.",
        suggested_action="Resume to pick up from the last checkpoint.",
    )
