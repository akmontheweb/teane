"""Phase 4 regression: HITL card polish + plain-English headline +
inline context (command/file/cwd/diff).

Desktop notifications are wired in JS; not exercised here — the
regression test for those lives in a browser-headless integration
run outside this suite. This file focuses on the server-rendered
HTML contract Phase 4 owns.
"""

from __future__ import annotations

from harness.dashboard import (
    DashboardConfig,
    _hitl_prompt_headline,
    _render_inline_context,
    _render_pending_hitl_rows,
)


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "enabled": True,
                "writes_enabled": True,
            }
        }
    )


# ---------------------------------------------------------------------------
# _hitl_prompt_headline
# ---------------------------------------------------------------------------

def test_headline_prefers_metadata_override():
    prompt = {"type": "prompt", "metadata": {"headline": "Approve the fix?"}}
    assert _hitl_prompt_headline(prompt) == "Approve the fix?"


def test_headline_falls_back_to_type_table():
    assert _hitl_prompt_headline({"type": "confirm"}) == "Please confirm"
    assert _hitl_prompt_headline({"type": "prompt"}) == "Choose how to continue"
    assert _hitl_prompt_headline({"type": "notes"}) == "Please add notes"
    assert _hitl_prompt_headline({"type": "wait_for_edit"}) == "Waiting for your edit"


def test_headline_falls_back_to_generic_for_unknown_type():
    assert _hitl_prompt_headline({"type": "future-type-not-yet-supported"}) == \
        "Human input needed"
    assert _hitl_prompt_headline({}) == "Human input needed"
    assert _hitl_prompt_headline("not-a-dict") == "Human input needed"


# ---------------------------------------------------------------------------
# _render_inline_context
# ---------------------------------------------------------------------------

def test_inline_context_empty_when_no_known_fields():
    assert _render_inline_context({}) == ""
    assert _render_inline_context({"other_field": "value"}) == ""
    assert _render_inline_context("not-a-dict") == ""


def test_inline_context_renders_command_file_cwd_rows():
    body = _render_inline_context({
        "command": "pytest -k auth",
        "file": "/workspace/tests/test_auth.py",
        "cwd": "/workspace",
    })
    assert "Command" in body
    assert "pytest -k auth" in body
    assert "File" in body
    assert "/workspace/tests/test_auth.py" in body
    assert "Working dir" in body


def test_inline_context_truncates_long_diff_with_marker():
    long_diff = "\n".join(f"line-{i}" for i in range(50))
    body = _render_inline_context({"diff": long_diff})
    assert "line-0" in body
    assert "line-11" in body  # first 12 lines shown
    assert "line-49" not in body
    assert "more line" in body  # truncation marker


def test_inline_context_escapes_html_in_values():
    body = _render_inline_context({"command": "<script>alert(1)</script>"})
    assert "&lt;script&gt;" in body
    assert "<script>" not in body


# ---------------------------------------------------------------------------
# _render_pending_hitl_rows — end-to-end card structure
# ---------------------------------------------------------------------------

def _register_pending(session_id: str, prompt: dict, request_id: str = "req-x"):
    from harness.web_state import get_hitl_queue
    q = get_hitl_queue()
    # Clear any leaked pending from earlier tests.
    q.clear_pending_for_session(session_id)
    q.register_pending(request_id=request_id, session_id=session_id, prompt=prompt)
    return q


def test_hitl_card_uses_new_hitl_card_class_and_shows_headline(tmp_path):
    session_id = "sess-hitl-headline"
    _register_pending(session_id, {
        "type": "confirm",
        "message": "Overwrite existing README?",
        "default": False,
    })
    try:
        body = _render_pending_hitl_rows(_cfg(tmp_path), session_id)
    finally:
        from harness.web_state import get_hitl_queue
        get_hitl_queue().clear_pending_for_session(session_id)
    assert "hitl-card" in body
    # Old Carbon-classed alert div is gone.
    assert "card hitl-alert" not in body
    # Plain-English headline from the type table.
    assert "Please confirm" in body
    # Original message is still there for detail.
    assert "Overwrite existing README?" in body


def test_hitl_card_renders_inline_metadata_before_input(tmp_path):
    session_id = "sess-hitl-context"
    _register_pending(session_id, {
        "type": "confirm",
        "message": "Run this command?",
        "metadata": {
            "command": "npm install",
            "cwd": "/workspace/frontend",
        },
    })
    try:
        body = _render_pending_hitl_rows(_cfg(tmp_path), session_id)
    finally:
        from harness.web_state import get_hitl_queue
        get_hitl_queue().clear_pending_for_session(session_id)
    # Command block shows up.
    assert "npm install" in body
    assert "Working dir" in body
    # Inline context appears BEFORE the form action (input/button block).
    ctx_idx = body.find("Working dir")
    form_idx = body.find("<form")
    assert ctx_idx != -1
    assert form_idx != -1
    assert ctx_idx < form_idx, (
        "Inline context (command/cwd) must render above the input so the "
        "operator sees what's being asked before choosing."
    )


def test_hitl_card_uses_primary_button_class(tmp_path):
    session_id = "sess-hitl-btn"
    _register_pending(session_id, {"type": "confirm", "message": "OK?"})
    try:
        body = _render_pending_hitl_rows(_cfg(tmp_path), session_id)
    finally:
        from harness.web_state import get_hitl_queue
        get_hitl_queue().clear_pending_for_session(session_id)
    # New Linear-style primary button class from tokens.css.
    assert "btn btn--primary" in body
