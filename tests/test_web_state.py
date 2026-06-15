"""Regression tests for the web-app data layer (harness/web_state.py).

Covers the four state owners:
    - ProcessRegistry (in-memory; thread-safety smoke + TTL semantics)
    - HitlQueue (in-memory; register/answer/release flow)
    - chat_notes table (queue / consume / pending)
    - web_oneshot_jobs table (insert / list pending / mark consumed)
    - audit_log + run_presets helpers
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from harness.web_state import (
    HitlQueue,
    ProcessRegistry,
    WebProcess,
    add_oneshot_job,
    append_audit,
    consume_chat_notes,
    delete_run_preset,
    list_all_oneshot_jobs,
    list_audit,
    list_pending_oneshot_jobs,
    list_run_presets,
    mark_oneshot_consumed,
    open_web_db,
    pending_chat_notes,
    queue_chat_note,
    save_run_preset,
)


UTC = timezone.utc


# ---------------------------------------------------------------------------
# ProcessRegistry
# ---------------------------------------------------------------------------

def _proc(session_id: str, *, pid: int = 100) -> WebProcess:
    return WebProcess(
        session_id=session_id, pid=pid, argv=["harness", "run"],
        log_path=f"/tmp/{session_id}.jsonl",
    )


def test_registry_round_trip():
    reg = ProcessRegistry()
    p = _proc("s1")
    reg.register(p)
    assert reg.get("s1") is p
    assert reg.list_running() == [p]
    assert reg.list_all() == [p]


def test_registry_marks_terminated_and_keeps_within_ttl():
    reg = ProcessRegistry(terminated_ttl_seconds=60)
    reg.register(_proc("s1"))
    reg.mark_terminated("s1", exit_code=0)
    entry = reg.get("s1")
    assert entry is not None
    assert entry.is_running is False
    assert entry.exit_code == 0
    # Still listed because TTL is generous.
    assert any(p.session_id == "s1" for p in reg.list_all())
    assert reg.list_running() == []


def test_registry_evicts_terminated_after_ttl():
    reg = ProcessRegistry(terminated_ttl_seconds=0)
    reg.register(_proc("s1"))
    reg.mark_terminated("s1", exit_code=1)
    # Past the TTL → list_all prunes.
    time.sleep(0.01)
    listed = reg.list_all()
    assert all(p.session_id != "s1" for p in listed)


def test_registry_remove_drops_entry():
    reg = ProcessRegistry()
    reg.register(_proc("s1"))
    reg.remove("s1")
    assert reg.get("s1") is None


def test_registry_thread_safety_smoke():
    reg = ProcessRegistry()

    def _writer():
        for i in range(200):
            reg.register(_proc(f"sess-{i:04d}", pid=10_000 + i))

    def _terminator():
        for i in range(200):
            reg.mark_terminated(f"sess-{i:04d}", exit_code=0)

    t1 = threading.Thread(target=_writer)
    t2 = threading.Thread(target=_terminator)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Just asserting no crash; the count is bounded by writer pace.
    assert isinstance(reg.list_all(), list)


# ---------------------------------------------------------------------------
# HitlQueue
# ---------------------------------------------------------------------------

def test_hitl_queue_register_then_answer_releases_held_handler():
    q = HitlQueue()
    pending = q.register_pending(
        request_id="req-1", session_id="s1",
        prompt={"gate": "REQUIREMENTS", "options": ["a", "e", "m", "s"]},
    )
    assert pending.event.is_set() is False
    ok = q.answer("req-1", {"choice": "a", "extra_notes": ""})
    assert ok is True
    assert pending.event.is_set() is True
    assert q.pop_response("req-1") == {"choice": "a", "extra_notes": ""}


def test_hitl_queue_double_answer_returns_false():
    q = HitlQueue()
    q.register_pending(request_id="req-2", session_id="s", prompt={})
    assert q.answer("req-2", {"choice": "a"}) is True
    assert q.answer("req-2", {"choice": "e"}) is False


def test_hitl_queue_answer_unknown_returns_false():
    q = HitlQueue()
    assert q.answer("never-registered", {"choice": "a"}) is False


def test_hitl_queue_list_pending_filters_by_session():
    q = HitlQueue()
    q.register_pending(request_id="r1", session_id="A", prompt={})
    q.register_pending(request_id="r2", session_id="A", prompt={})
    q.register_pending(request_id="r3", session_id="B", prompt={})
    assert {p.request_id for p in q.list_pending_for_session("A")} == {"r1", "r2"}
    assert {p.request_id for p in q.list_pending_for_session("B")} == {"r3"}


def test_hitl_queue_holds_handler_until_event_set():
    """Cross-thread coordination smoke. The 'webhook' thread waits on
    the event; the 'UI' thread answers; the webhook thread proceeds."""
    q = HitlQueue()
    pending = q.register_pending(request_id="r", session_id="s", prompt={})
    received: list[dict] = []

    def _webhook():
        pending.event.wait(timeout=2.0)
        received.append(q.pop_response("r"))

    t = threading.Thread(target=_webhook)
    t.start()
    time.sleep(0.05)
    assert q.answer("r", {"choice": "a"}) is True
    t.join(timeout=2.0)
    assert received == [{"choice": "a"}]


# ---------------------------------------------------------------------------
# chat_notes table
# ---------------------------------------------------------------------------

def test_chat_notes_round_trip(tmp_path):
    db = str(tmp_path / "web.db")
    queue_chat_note(db_path=db, session_id="s1", note="first")
    queue_chat_note(db_path=db, session_id="s1", note="second")
    queue_chat_note(db_path=db, session_id="other", note="other")
    pending = pending_chat_notes(db_path=db, session_id="s1")
    assert [p["note"] for p in pending] == ["first", "second"]
    consumed = consume_chat_notes(db_path=db, session_id="s1")
    assert consumed == ["first", "second"]
    # Second consume returns nothing.
    assert consume_chat_notes(db_path=db, session_id="s1") == []
    # Other session's note is untouched.
    assert [p["note"] for p in pending_chat_notes(db_path=db, session_id="other")] == ["other"]


def test_chat_notes_ignores_empty(tmp_path):
    db = str(tmp_path / "web.db")
    assert queue_chat_note(db_path=db, session_id="s", note="") == 0
    assert queue_chat_note(db_path=db, session_id="s", note="   ") == 0
    assert pending_chat_notes(db_path=db, session_id="s") == []


# ---------------------------------------------------------------------------
# web_oneshot_jobs
# ---------------------------------------------------------------------------

def test_oneshot_jobs_insert_then_picked_up_when_due(tmp_path):
    db = str(tmp_path / "web.db")
    past = datetime.now(UTC) - timedelta(minutes=1)
    future = datetime.now(UTC) + timedelta(hours=1)
    add_oneshot_job(
        db_path=db, name="late-night",
        fire_at_utc=past, workspace="/repo",
        prompt="run the thing", harness_args=["-v"],
    )
    add_oneshot_job(
        db_path=db, name="tomorrow",
        fire_at_utc=future, workspace="/repo",
    )
    pending = list_pending_oneshot_jobs(db_path=db)
    # Only the past one is due.
    assert [j["name"] for j in pending] == ["late-night"]
    assert pending[0]["harness_args"] == ["-v"]
    # Mark consumed → next call returns nothing.
    mark_oneshot_consumed(db_path=db, job_id=pending[0]["id"])
    assert list_pending_oneshot_jobs(db_path=db) == []


def test_oneshot_jobs_rejects_naive_datetime(tmp_path):
    db = str(tmp_path / "web.db")
    with pytest.raises(ValueError, match="tz-aware"):
        add_oneshot_job(
            db_path=db, name="bad",
            fire_at_utc=datetime(2026, 1, 1),  # naive
            workspace="/repo",
        )


def test_list_all_oneshot_jobs_returns_consumed_too(tmp_path):
    db = str(tmp_path / "web.db")
    past = datetime.now(UTC) - timedelta(seconds=10)
    add_oneshot_job(db_path=db, name="a", fire_at_utc=past, workspace="/a")
    add_oneshot_job(db_path=db, name="b", fire_at_utc=past, workspace="/b")
    pending = list_pending_oneshot_jobs(db_path=db)
    mark_oneshot_consumed(db_path=db, job_id=pending[0]["id"])
    all_jobs = list_all_oneshot_jobs(db_path=db)
    assert len(all_jobs) == 2
    consumed_states = [j["consumed_at"] is not None for j in all_jobs]
    assert any(consumed_states)


# ---------------------------------------------------------------------------
# audit_log + run_presets
# ---------------------------------------------------------------------------

def test_audit_log_records_actions(tmp_path):
    db = str(tmp_path / "web.db")
    append_audit(db_path=db, action="config_save",
                 target="token_budget", detail="hard_cap_usd=5.0")
    append_audit(db_path=db, action="run_now",
                 target="sess-001", detail="argv=...")
    rows = list_audit(db_path=db, limit=10)
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["action"] == "run_now"
    assert rows[1]["target"] == "token_budget"


def test_run_presets_round_trip(tmp_path):
    db = str(tmp_path / "web.db")
    save_run_preset(
        db_path=db, name="nightly-retest",
        workspace="/repo", prompt="Regenerate failing tests",
        harness_args=["--new_build=false"],
    )
    save_run_preset(
        db_path=db, name="full-rebuild",
        workspace="/repo2", prompt="Greenfield",
    )
    presets = list_run_presets(db_path=db)
    names = [p["name"] for p in presets]
    assert "nightly-retest" in names and "full-rebuild" in names
    nightly = next(p for p in presets if p["name"] == "nightly-retest")
    assert nightly["harness_args"] == ["--new_build=false"]
    delete_run_preset(db_path=db, name="nightly-retest")
    after = [p["name"] for p in list_run_presets(db_path=db)]
    assert "nightly-retest" not in after


def test_run_presets_rejects_empty_name(tmp_path):
    db = str(tmp_path / "web.db")
    with pytest.raises(ValueError):
        save_run_preset(db_path=db, name="", workspace="/repo")


# ---------------------------------------------------------------------------
# open_web_db is idempotent + creates parent dir
# ---------------------------------------------------------------------------

def test_open_web_db_creates_parent_dir(tmp_path):
    db = str(tmp_path / "nested" / "subdir" / "web.db")
    conn = open_web_db(db)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {r[0] for r in rows}
        assert {"audit_log", "run_presets", "web_oneshot_jobs", "chat_notes"} <= names
    finally:
        conn.close()
