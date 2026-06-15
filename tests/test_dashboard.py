"""Regression tests for the read-only dashboard (#14).

These tests exercise the pure ``dispatch`` function + data adapters
directly so we don't need to stand up a real HTTP server. One
end-to-end test does spin up the threaded server on localhost on an
ephemeral port to confirm the request/response loop wires up cleanly.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.request

import pytest

from harness.dashboard import (
    DashboardConfig,
    check_auth,
    cost_burn_series,
    dispatch,
    list_memory_files,
    list_schedule_runs,
    list_sessions,
    read_memory_file,
    repo_index_status,
    resolve_expected_token,
    start_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_session_log(log_dir, session_id, events):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{session_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return path


def _make_cfg(tmp_path, **overrides):
    base = {
        "log_dir": str(tmp_path / "logs"),
        "metrics_dir": str(tmp_path / "metrics"),
        "memory_dir": str(tmp_path / "memory"),
        "repo_index_dir": str(tmp_path / "idx"),
        "schedule_db": str(tmp_path / "schedule.db"),
        "static_dir": str(tmp_path / "static"),
        "enabled": True,
    }
    base.update(overrides)
    return DashboardConfig.from_config({"dashboard": base})


# ---------------------------------------------------------------------------
# 1. DashboardConfig
# ---------------------------------------------------------------------------

def test_dashboard_config_defaults_to_disabled_and_localhost():
    cfg = DashboardConfig.from_config({})
    assert cfg.enabled is False
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8729
    assert cfg.token_env == ""


def test_dashboard_config_inherits_log_dir_from_logging_section():
    cfg = DashboardConfig.from_config({
        "logging": {"log_dir": "/custom/log/dir"},
    })
    assert cfg.log_dir == "/custom/log/dir"


def test_dashboard_config_port_clamped_to_valid_range():
    cfg = DashboardConfig.from_config({"dashboard": {"port": 0}})
    assert cfg.port == 1
    cfg2 = DashboardConfig.from_config({"dashboard": {"port": 99999}})
    assert cfg2.port == 65535


# ---------------------------------------------------------------------------
# 2. Auth
# ---------------------------------------------------------------------------

def test_resolve_expected_token_returns_none_when_disabled():
    cfg = DashboardConfig()  # token_env=""
    assert resolve_expected_token(cfg) is None


def test_resolve_expected_token_reads_env(monkeypatch):
    monkeypatch.setenv("DASH_TOKEN", "s3cret")
    cfg = DashboardConfig(token_env="DASH_TOKEN")
    assert resolve_expected_token(cfg) == "s3cret"


def test_resolve_expected_token_fails_closed_when_env_missing(monkeypatch):
    monkeypatch.delenv("DASH_TOKEN", raising=False)
    cfg = DashboardConfig(token_env="DASH_TOKEN")
    with pytest.raises(RuntimeError, match="empty"):
        resolve_expected_token(cfg)


def test_check_auth_disabled_allows_everything():
    out = check_auth(None, None)
    assert out.ok is True


def test_check_auth_rejects_missing_header():
    out = check_auth("xyz", None)
    assert out.ok is False
    assert "missing" in out.detail


def test_check_auth_rejects_wrong_prefix():
    out = check_auth("xyz", "Token xyz")
    assert out.ok is False
    assert "Bearer" in out.detail


def test_check_auth_rejects_mismatched_token():
    out = check_auth("expected", "Bearer different")
    assert out.ok is False
    assert "mismatch" in out.detail


def test_check_auth_accepts_correct_token():
    out = check_auth("expected", "Bearer expected")
    assert out.ok is True


# ---------------------------------------------------------------------------
# 3. Data adapters
# ---------------------------------------------------------------------------

def test_list_sessions_empty_when_log_dir_missing(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert list_sessions(cfg) == []


def test_list_sessions_parses_events(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "sess-001", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z",
         "workspace_path": "/tmp/repo"},
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:01Z",
         "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.02},
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:02Z",
         "tokens_in": 80, "tokens_out": 40, "cost_usd": 0.015},
        {"event": "session_end", "timestamp": "2026-06-15T10:00:30Z",
         "exit_code": 0},
    ])
    sessions = list_sessions(cfg)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "sess-001"
    assert s.exit_code == 0
    assert s.llm_calls == 2
    assert abs(s.total_cost_usd - 0.035) < 1e-9
    assert s.total_input_tokens == 180
    assert s.total_output_tokens == 90
    assert s.workspace_path == "/tmp/repo"


def test_cost_burn_series_aggregates_and_orders(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "a", [
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:00Z", "cost_usd": 0.1},
        {"event": "llm_call", "timestamp": "2026-06-15T10:01:00Z", "cost_usd": 0.2},
    ])
    _write_session_log(cfg.log_dir, "b", [
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:30Z", "cost_usd": 0.05},
    ])
    series = cost_burn_series(cfg)
    assert series["labels"] == sorted(series["labels"])  # time-ordered
    cum = series["datasets"][0]["data"]
    # Cumulative ends at 0.35
    assert abs(cum[-1] - 0.35) < 1e-9
    assert cum == sorted(cum)  # monotonic increasing


def test_repo_index_status_empty_when_db_missing(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert repo_index_status(cfg) == []


def test_repo_index_status_reads_meta(tmp_path):
    cfg = _make_cfg(tmp_path)
    os.makedirs(cfg.repo_index_dir, exist_ok=True)
    db = os.path.join(cfg.repo_index_dir, "repo_index.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE repo_meta (workspace_id TEXT PRIMARY KEY, "
            "backend TEXT NOT NULL, idf_json TEXT, built_at TEXT NOT NULL, "
            "chunk_count INTEGER NOT NULL DEFAULT 0);"
        )
        conn.execute(
            "INSERT INTO repo_meta VALUES "
            "('ws-1', 'tfidf', '{}', '2026-06-15T10:00:00Z', 42);"
        )
        conn.commit()
    finally:
        conn.close()
    rows = repo_index_status(cfg)
    assert rows == [{
        "workspace_id": "ws-1", "backend": "tfidf",
        "built_at": "2026-06-15T10:00:00Z", "chunk_count": 42,
    }]


def test_list_schedule_runs_empty_when_db_missing(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert list_schedule_runs(cfg) == []


def test_list_schedule_runs_reads_rows(tmp_path):
    cfg = _make_cfg(tmp_path)
    conn = sqlite3.connect(cfg.schedule_db)
    try:
        conn.execute(
            "CREATE TABLE schedule_runs (job_name TEXT NOT NULL, "
            "started_at TEXT NOT NULL, ended_at TEXT, exit_code INTEGER, "
            "duration_sec REAL, log_path TEXT, "
            "PRIMARY KEY (job_name, started_at));"
        )
        conn.execute(
            "INSERT INTO schedule_runs VALUES "
            "('nightly', '2026-06-15T02:00:00Z', '2026-06-15T02:05:00Z', "
            "0, 300.0, '/tmp/log');"
        )
        conn.commit()
    finally:
        conn.close()
    rows = list_schedule_runs(cfg)
    assert rows == [{
        "job_name": "nightly",
        "started_at": "2026-06-15T02:00:00Z",
        "ended_at": "2026-06-15T02:05:00Z",
        "exit_code": 0,
        "duration_sec": 300.0,
        "log_path": "/tmp/log",
    }]


def test_read_memory_file_rejects_traversal(tmp_path):
    cfg = _make_cfg(tmp_path)
    os.makedirs(cfg.memory_dir, exist_ok=True)
    (tmp_path / "secret.txt").write_text("not for the dashboard")
    # Refused outright.
    assert read_memory_file(cfg, "../secret.txt") is None
    assert read_memory_file(cfg, "subdir/file.md") is None
    assert read_memory_file(cfg, "no-extension") is None
    assert read_memory_file(cfg, "") is None


def test_read_memory_file_reads_valid_file(tmp_path):
    cfg = _make_cfg(tmp_path)
    os.makedirs(cfg.memory_dir, exist_ok=True)
    (tmp_path / "memory" / "abc1234.md").write_text("# notes\n\nhello\n")
    out = read_memory_file(cfg, "abc1234.md")
    assert out is not None
    assert "hello" in out


def test_list_memory_files_returns_metadata(tmp_path):
    cfg = _make_cfg(tmp_path)
    os.makedirs(cfg.memory_dir, exist_ok=True)
    (tmp_path / "memory" / "a.md").write_text("x")
    (tmp_path / "memory" / "b.txt").write_text("y")  # ignored
    files = list_memory_files(cfg)
    assert [f["name"] for f in files] == ["a.md"]


# ---------------------------------------------------------------------------
# 4. dispatch — routes return the right shape
# ---------------------------------------------------------------------------

def test_dispatch_overview(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, ctype, body = dispatch(cfg, "/")
    assert status == 200
    assert "text/html" in ctype
    assert "Overview" in body
    assert "Sessions on disk" in body


def test_dispatch_404_for_unknown(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _, body = dispatch(cfg, "/does/not/exist")
    assert status == 404
    assert "404" in body


def test_dispatch_session_detail_404_when_missing(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _, body = dispatch(cfg, "/sessions/nope")
    assert status == 200  # rendered page; inner body reports the missing log
    assert "No log file" in body


def test_dispatch_session_detail_renders_events(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "good", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"},
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:01Z",
         "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.001},
    ])
    status, _, body = dispatch(cfg, "/sessions/good")
    assert status == 200
    assert "llm_call" in body


def test_dispatch_api_cost_burn_returns_json(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "s", [
        {"event": "llm_call", "timestamp": "2026-06-15T10:00:00Z", "cost_usd": 0.05},
    ])
    status, ctype, body = dispatch(cfg, "/api/cost-burn")
    assert status == 200
    assert "application/json" in ctype
    data = json.loads(body)
    assert "datasets" in data and "labels" in data


def test_dispatch_memory_routes(tmp_path):
    cfg = _make_cfg(tmp_path)
    os.makedirs(cfg.memory_dir, exist_ok=True)
    (tmp_path / "memory" / "abc.md").write_text("# memo")
    status, _, body = dispatch(cfg, "/memory")
    assert status == 200
    assert "abc.md" in body
    status2, _, body2 = dispatch(cfg, "/memory/abc.md")
    assert status2 == 200
    assert "memo" in body2


def test_dispatch_schedule_renders_table_or_placeholder(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _, body = dispatch(cfg, "/schedule")
    assert status == 200
    assert ("schedule" in body.lower())


def test_dispatch_index_renders_table_or_placeholder(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _, body = dispatch(cfg, "/index")
    assert status == 200
    assert "repo index" in body.lower()


# ---------------------------------------------------------------------------
# 5. End-to-end — real server, real socket
# ---------------------------------------------------------------------------

def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_server_round_trip_returns_overview(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    handle = start_server(cfg, blocking=False)
    assert handle is not None
    try:
        url = f"http://{cfg.host}:{cfg.port}/"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "Overview" in body
    finally:
        handle.shutdown()


def test_server_enforces_auth_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("DASH_AUTH_TOKEN", "swordfish")
    cfg = _make_cfg(tmp_path, token_env="DASH_AUTH_TOKEN")
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    handle = start_server(cfg, blocking=False)
    assert handle is not None
    try:
        url = f"http://{cfg.host}:{cfg.port}/sessions"
        # No header → 401.
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url, timeout=2.0)
        assert exc.value.code == 401
        # Correct token → 200.
        req = urllib.request.Request(url, headers={"Authorization": "Bearer swordfish"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
        # Wrong token → 401.
        req2 = urllib.request.Request(url, headers={"Authorization": "Bearer wrong"})
        with pytest.raises(urllib.error.HTTPError) as exc2:
            urllib.request.urlopen(req2, timeout=2.0)
        assert exc2.value.code == 401
    finally:
        handle.shutdown()


def test_server_refuses_to_start_when_token_env_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    cfg = _make_cfg(tmp_path, token_env="MISSING_TOKEN")
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    with pytest.raises(RuntimeError, match="empty"):
        start_server(cfg, blocking=False)
