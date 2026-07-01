"""Phase 8 regression: plain-English session-end explanations."""

from __future__ import annotations

from harness.error_explain import (
    STATUS_CRASHED,
    STATUS_KILLED,
    STATUS_OK,
    STATUS_RUNNING,
    explain_session_end,
)


def test_running_session_reports_running_status_and_empty_cause():
    exp = explain_session_end([], stderr_tail="", process_still_running=True)
    assert exp.status == STATUS_RUNNING
    assert exp.headline
    assert exp.suggested_action == ""


def test_clean_session_end_reports_ok():
    events = [{"event": "session_start"}, {"event": "session_end", "exit_code": 0}]
    exp = explain_session_end(events)
    assert exp.status == STATUS_OK
    assert "clean" in exp.headline.lower()


def test_missing_session_end_reports_killed():
    """No session_end event + process gone → killed externally."""
    events = [{"event": "session_start"}, {"event": "llm_call"}]
    exp = explain_session_end(events)
    assert exp.status == STATUS_KILLED
    assert "kill" in exp.headline.lower()
    assert "Resume" in exp.suggested_action


def test_non_zero_exit_with_provider_overload():
    events = [
        {"event": "session_start"},
        {"event": "tool_call_failed", "error": "HTTPError 529 overloaded_error"},
        {"event": "session_end", "exit_code": 1},
    ]
    exp = explain_session_end(events)
    assert exp.status == STATUS_CRASHED
    assert "overloaded" in exp.headline.lower()
    assert "retry" in exp.suggested_action.lower()


def test_non_zero_exit_with_rate_limit():
    events = [
        {"event": "session_start"},
        {"event": "tool_call_failed", "error": "Rate limited by provider"},
        {"event": "session_end", "exit_code": 1},
    ]
    exp = explain_session_end(events)
    assert exp.status == STATUS_CRASHED
    assert "rate" in exp.headline.lower() or "rate" in exp.cause.lower()


def test_non_zero_exit_with_budget_exhaustion():
    events = [
        {"event": "session_start"},
        {"event": "token_budget_exhausted", "error": "budget exceeded"},
        {"event": "session_end", "exit_code": 2},
    ]
    exp = explain_session_end(events)
    assert "budget" in exp.headline.lower()
    assert "hard_cap_usd" in exp.suggested_action


def test_non_zero_exit_with_docker_missing_via_stderr():
    events = [
        {"event": "session_start"},
        {"event": "session_end", "exit_code": 4},
    ]
    stderr = (
        "Traceback (most recent call last):\n"
        "  File 'sandbox.py' ... RuntimeError: docker not installed on this machine\n"
    )
    exp = explain_session_end(events, stderr_tail=stderr)
    assert "docker" in exp.headline.lower()
    assert "unshare" in exp.suggested_action.lower()


def test_no_rule_match_falls_back_to_last_line_of_stderr():
    events = [
        {"event": "session_start"},
        {"event": "session_end", "exit_code": 7},
    ]
    stderr = "Line one\nLine two — the interesting one\n"
    exp = explain_session_end(events, stderr_tail=stderr)
    assert exp.status == STATUS_CRASHED
    assert "code 7" in exp.headline
    assert "interesting one" in exp.cause


def test_no_signal_at_all_uses_a_neutral_message():
    events = [
        {"event": "session_start"},
        {"event": "session_end", "exit_code": 5},
    ]
    exp = explain_session_end(events, stderr_tail="")
    assert exp.status == STATUS_CRASHED
    assert exp.cause  # never empty — either the raw tail or a fallback
