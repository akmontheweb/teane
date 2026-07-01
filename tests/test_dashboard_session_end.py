"""Phase 8 regression: session-end card renders above HITL slot with a
plain-English cause and a working Resume form."""

from __future__ import annotations

import json

from harness.dashboard import (
    DashboardConfig,
    _render_session_end_card,
    _render_session_with_hitl,
)


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "repo_index_dir": str(tmp_path / "idx"),
                "schedule_db": str(tmp_path / "schedule.db"),
                "enabled": True,
                "writes_enabled": True,
            }
        }
    )


def _write_log(tmp_path, session_id, events):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return str(path)


def test_session_end_card_hidden_while_running(tmp_path, monkeypatch):
    from harness import dashboard as _d
    monkeypatch.setattr(_d, "_process_is_running", lambda _sid: True)
    _write_log(tmp_path, "sess-live", [{"event": "session_start"}])
    body = _render_session_end_card(_cfg(tmp_path), "sess-live",
                                    str(tmp_path / "logs" / "sess-live.jsonl"))
    assert body == ""


def test_session_end_card_renders_ok_on_clean_finish(tmp_path, monkeypatch):
    from harness import dashboard as _d
    monkeypatch.setattr(_d, "_process_is_running", lambda _sid: False)
    _write_log(tmp_path, "sess-ok", [
        {"event": "session_start"},
        {"event": "session_end", "exit_code": 0},
    ])
    body = _render_session_end_card(_cfg(tmp_path), "sess-ok",
                                    str(tmp_path / "logs" / "sess-ok.jsonl"))
    assert "session-end-card--ok" in body
    assert "cleanly" in body.lower()
    # No Resume button for clean exits.
    assert "action='/run/resume'" not in body


def test_session_end_card_offers_resume_on_crash(tmp_path, monkeypatch):
    from harness import dashboard as _d
    monkeypatch.setattr(_d, "_process_is_running", lambda _sid: False)
    _write_log(tmp_path, "sess-crash", [
        {"event": "session_start", "workspace_path": "/tmp/workspace"},
        {"event": "tool_call_failed", "error": "529 overloaded_error"},
        {"event": "session_end", "exit_code": 1},
    ])
    body = _render_session_end_card(_cfg(tmp_path), "sess-crash",
                                    str(tmp_path / "logs" / "sess-crash.jsonl"))
    assert "session-end-card--crashed" in body
    assert "overloaded" in body.lower()
    assert "action='/run/resume'" in body
    # Workspace pre-filled so the operator doesn't have to type it again.
    assert "value='/tmp/workspace'" in body
    assert "resume_session_id" in body


def test_session_end_card_shows_cost_so_far(tmp_path, monkeypatch):
    from harness import dashboard as _d
    monkeypatch.setattr(_d, "_process_is_running", lambda _sid: False)
    _write_log(tmp_path, "sess-cost", [
        {"event": "session_start"},
        {"event": "llm_call", "ts": "t", "cost_usd": 1.234,
         "tokens_in": 100, "tokens_out": 50},
        {"event": "session_end", "exit_code": 1},
    ])
    body = _render_session_end_card(_cfg(tmp_path), "sess-cost",
                                    str(tmp_path / "logs" / "sess-cost.jsonl"))
    assert "$1.2340" in body
    assert "1 LLM call" in body


def test_session_detail_includes_end_card_above_hitl_slot(tmp_path, monkeypatch):
    """Integration: `_render_session_with_hitl` places the end card
    ABOVE the HITL slot so a crash cause is the first thing seen."""
    from harness import dashboard as _d
    monkeypatch.setattr(_d, "_process_is_running", lambda _sid: False)
    _write_log(tmp_path, "sess-integ", [
        {"event": "session_start"},
        {"event": "session_end", "exit_code": 1},
    ])
    body = _render_session_with_hitl(_cfg(tmp_path), "sess-integ")
    end_idx = body.find("session-end-card")
    hitl_idx = body.find("hitl-pending-slot")
    assert end_idx != -1
    assert hitl_idx != -1
    assert end_idx < hitl_idx, (
        "session-end-card must render above hitl-pending-slot so "
        "the crash cause reads before any residual HITL banner."
    )
