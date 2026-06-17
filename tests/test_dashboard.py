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
    _serve_static,
    check_auth,
    cost_burn_series,
    dispatch,
    list_memory_files,
    list_running_sessions,
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


def test_dashboard_config_hitl_webhook_timeout_default_and_override():
    cfg = DashboardConfig.from_config({})
    assert cfg.hitl_webhook_timeout_seconds == 600.0
    cfg2 = DashboardConfig.from_config({
        "dashboard": {"hitl_webhook_timeout_seconds": 1200.0},
    })
    assert cfg2.hitl_webhook_timeout_seconds == 1200.0


def test_dashboard_config_audit_log_retention_default_and_override():
    cfg = DashboardConfig.from_config({})
    assert cfg.audit_log_retention_days == 90
    cfg2 = DashboardConfig.from_config({
        "dashboard": {"audit_log_retention_days": 30},
    })
    assert cfg2.audit_log_retention_days == 30


def test_empty_state_escapes_body():
    """_empty_state's ``body`` parameter is rendered as plain text. A
    caller that passes user-derived data with HTML in it must not
    produce live markup."""
    from harness.dashboard import _empty_state
    html_out = _empty_state(
        icon="list",
        title="No sessions yet",
        body="<script>alert(1)</script>",
    )
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out


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


def test_list_running_sessions_excludes_logs_without_session_start(tmp_path):
    """Dashboard-boot tombstones — JSONL files that hold only
    ``init_observability`` startup chatter (no ``event`` field
    anywhere) — must NOT appear as running sessions. Without this
    filter, every ``harness web start`` left behind a phantom row
    that piled up across restarts (17 phantoms after a few cycles
    in one operator session)."""
    cfg = _make_cfg(tmp_path)
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # Tombstone: plain logger output, no event field at all.
    tombstone = os.path.join(log_dir, "web-deadbeef0001.jsonl")
    with open(tombstone, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": "2026-06-17T05:14:16Z", "level": "INFO",
            "logger": "harness",
            "msg": "Session log file: ...",
        }) + "\n")
        f.write(json.dumps({
            "ts": "2026-06-17T05:14:16Z", "level": "INFO",
            "logger": "harness.gateway",
            "msg": "[gateway] Registered model 'openai:gpt-4o'.",
        }) + "\n")
    rows = list_running_sessions(cfg)
    assert rows == [], (
        f"expected dashboard-boot tombstone to be filtered, got: {rows}"
    )


def test_list_running_sessions_includes_logs_with_session_start_but_no_session_end(tmp_path):
    """An in-flight harness run emits ``session_start`` at the top of
    its log and won't write ``session_end`` until it finishes — the
    only signal that a real run is currently live. That signature
    must keep the row visible."""
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "live-sess", [
        {"event": "session_start", "timestamp": "2026-06-17T05:20:00Z",
         "workspace_path": "/tmp/ciod"},
        {"event": "llm_call", "timestamp": "2026-06-17T05:20:10Z",
         "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.02},
    ])
    rows = list_running_sessions(cfg)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "live-sess"
    assert rows[0]["workspace_path"] == "/tmp/ciod"
    assert rows[0]["source"] == "cli"


def test_list_running_sessions_still_excludes_session_end_logs(tmp_path):
    """Regression guard for the original behaviour: a log whose tail
    event is ``session_end`` is a completed run, never a running
    one — independent of the new ``session_start`` requirement."""
    cfg = _make_cfg(tmp_path)
    _write_session_log(cfg.log_dir, "done-sess", [
        {"event": "session_start", "timestamp": "2026-06-17T05:15:00Z"},
        {"event": "session_end", "timestamp": "2026-06-17T05:16:00Z",
         "exit_code": 0},
    ])
    assert list_running_sessions(cfg) == []


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

def test_dispatch_root_redirects_to_status(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _ctype, body = dispatch(cfg, "/")
    assert status == 302
    assert body.endswith("/status")


def test_dispatch_status_renders_with_side_nav(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, ctype, body = dispatch(cfg, "/status")
    assert status == 200
    assert "text/html" in ctype
    assert "View Status" in body
    assert "bx--side-nav" in body
    assert "bx--side-nav__link--current" in body


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
# 4b. Static asset pipeline — packaged + operator-override + traversal
# ---------------------------------------------------------------------------

def test_serve_static_returns_packaged_css(tmp_path):
    """The packaged harness/static/css/app.css ships with every wheel
    and must be served as text/css when no operator override matches."""
    cfg = _make_cfg(tmp_path)  # static_dir points at empty tmp dir
    status, ctype, data = _serve_static(cfg, "css/app.css")
    assert status == 200
    assert ctype.startswith("text/css")
    assert isinstance(data, (bytes, bytearray))


def test_serve_static_returns_packaged_favicon(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, ctype, data = _serve_static(cfg, "favicon.ico")
    assert status == 200
    assert "icon" in ctype
    # Real ICO files start with the 6-byte header 00 00 01 00 NN 00.
    assert data[:4] == b"\x00\x00\x01\x00"


def test_serve_static_returns_packaged_sprite_and_js(tmp_path):
    cfg = _make_cfg(tmp_path)
    s1, c1, _ = _serve_static(cfg, "icons/sprite.svg")
    assert s1 == 200 and "svg" in c1
    s2, c2, _ = _serve_static(cfg, "js/dashboard.js")
    assert s2 == 200 and "javascript" in c2


def test_serve_static_operator_override_wins_over_packaged(tmp_path):
    """An operator file at static_dir/css/app.css must shadow the
    packaged copy. This is the air-gap escape hatch — mirror the
    sheet into your own dir without rebuilding the wheel."""
    override_root = tmp_path / "static-override"
    (override_root / "css").mkdir(parents=True)
    (override_root / "css" / "app.css").write_bytes(b"/* operator override */\n")
    cfg = _make_cfg(tmp_path, static_dir=str(override_root))
    status, ctype, data = _serve_static(cfg, "css/app.css")
    assert status == 200
    assert ctype.startswith("text/css")
    assert data == b"/* operator override */\n"


def test_serve_static_rejects_path_traversal(tmp_path):
    """Even when an override dir exists, `..` segments must not escape
    its containment. Defense-in-depth alongside the route regex."""
    override_root = tmp_path / "static-override"
    override_root.mkdir()
    (tmp_path / "secret.css").write_text("/* not yours */")
    cfg = _make_cfg(tmp_path, static_dir=str(override_root))
    status, _, _ = _serve_static(cfg, "../secret.css")
    assert status == 404


def test_serve_static_rejects_disallowed_extension(tmp_path):
    """Only the whitelisted extensions render. Otherwise an operator
    could drop a .py / .sh / .html into static_dir and confuse a
    browser that auto-executes."""
    override_root = tmp_path / "static-override"
    override_root.mkdir()
    (override_root / "evil.html").write_bytes(b"<script>1</script>")
    cfg = _make_cfg(tmp_path, static_dir=str(override_root))
    status, _, _ = _serve_static(cfg, "evil.html")
    assert status == 404


def test_serve_static_rejects_missing_file(tmp_path):
    cfg = _make_cfg(tmp_path)
    status, _, _ = _serve_static(cfg, "does/not/exist.css")
    assert status == 404


def test_layout_links_static_stylesheet_and_favicon_and_js(tmp_path):
    """Every rendered page must pull in the external app.css, the
    favicon, and the dashboard.js bundle. Defends against accidental
    removal during future refactors."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    # Asset URLs carry a ``?v=<mtime-hex>`` cache-buster, so match by prefix.
    assert 'href="/static/css/app.css?v=' in body or 'href="/static/css/app.css"' in body
    assert 'href="/static/favicon.ico"' in body
    assert 'src="/static/js/dashboard.js?v=' in body or 'src="/static/js/dashboard.js"' in body
    assert 'name="viewport"' in body  # mobile-ready


def test_side_nav_renders_icons_per_item(tmp_path):
    """Each of the five top-level nav items shows an icon via the
    /static/icons/sprite.svg sprite. Icons stay in sync via _NAV_ITEMS."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    # All five nav items must reference a sprite symbol.
    for symbol in ("chart-line", "play", "settings", "dashboard", "document"):
        assert f"#i-{symbol}" in body, f"missing nav icon {symbol!r}"


def test_dashboards_landing_tiles_render_icons(tmp_path):
    """Each tile on the /dashboards landing has an icon prefix so the
    page reads as a real launcher, not a wall of links."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/dashboards")
    # At least one icon symbol used, and the icon wrapper class present.
    assert "dash-tile__icon" in body
    assert "#i-" in body


def test_run_now_button_has_play_icon_prefix(tmp_path, monkeypatch):
    """The primary action button on /run uses the play icon for the
    universal recognisable affordance."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path, csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # The button still ends with "Run Now</button>" (test contract from
    # earlier PRs) but now has a #i-play sprite reference immediately
    # before that text.
    assert "#i-play" in body
    assert ">Run Now</button>" in body


def test_sessions_table_uses_thead_tbody(tmp_path):
    """Modern table markup so Carbon's CSS bindings and our zebra/
    sticky-header styling lock in. Add thead/tbody when adding new
    tables — they're cheap and fix a lot of visual bugs."""
    _write_session_log(tmp_path / "logs", "demo", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"},
        {"event": "session_end", "timestamp": "2026-06-15T10:00:01Z", "exit_code": 0},
    ])
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/sessions")
    assert "<thead>" in body and "<tbody>" in body
    assert "id='sessions-table'" in body
    assert "table-wrap" in body


def test_sessions_table_columns_have_sort_metadata(tmp_path):
    """Every <th> on the sessions table opts into client-side sorting
    via data-sort, so dashboard.js can wire up header-click sort
    without server changes."""
    _write_session_log(tmp_path / "logs", "demo", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"},
        {"event": "session_end", "timestamp": "2026-06-15T10:00:01Z", "exit_code": 0},
    ])
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/sessions")
    # str / date / num kinds all present in the header row.
    assert "data-sort='str'" in body
    assert "data-sort='date'" in body
    assert "data-sort='num'" in body


def test_session_row_has_copy_button(tmp_path):
    """Session IDs in the sessions table are copyable via the click-
    delegated copy button. Operators paste these into terminals all
    the time — saves a triple-click + Ctrl+C."""
    _write_session_log(tmp_path / "logs", "abc123", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"},
        {"event": "session_end", "timestamp": "2026-06-15T10:00:01Z", "exit_code": 0},
    ])
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/sessions")
    assert "data-copy='abc123'" in body
    assert "copy-btn" in body


def test_empty_sessions_renders_cta_to_run(tmp_path):
    """Empty state replaces the lone muted paragraph with a real
    component that includes an icon, a headline, and a one-click
    button to start a run."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/sessions")
    assert "empty-state" in body
    assert "No sessions yet" in body
    # CTA points at /run (the most common next action).
    assert "href='/run'" in body
    # Old "harness run -r ..." literal is no longer the only hint.
    assert "empty-state__title" in body


def test_session_detail_renders_breadcrumb(tmp_path):
    """Detail pages show a breadcrumb trail so operators always know
    where they are. The new helper replaces ad-hoc `← All foo` links."""
    _write_session_log(tmp_path / "logs", "feed", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"},
    ])
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/sessions/feed")
    assert "class='breadcrumb'" in body
    assert "aria-label='Breadcrumb'" in body
    # Parent link + current page.
    assert "href='/sessions'>Sessions</a>" in body
    assert "breadcrumb__current" in body


def test_docs_file_breadcrumb_replaces_old_links(tmp_path):
    """The old `← All documents · {path}` markup is gone; the new
    helper renders a real breadcrumb component instead."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# heading\n")
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    _, _, body = dispatch(cfg, "/docs/guide.md")
    assert "class='breadcrumb'" in body
    assert "← All documents" not in body  # legacy markup removed


def test_layout_includes_auto_refresh_toggle(tmp_path):
    """Every page surfaces the auto-refresh toggle in the header. The
    JS in dashboard.js wires the click handler + localStorage state."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    assert 'id="auto-refresh-toggle"' in body
    assert 'aria-pressed="false"' in body  # off by default
    assert "Auto-refresh: off" in body


def test_layout_includes_toast_surface_via_dashboard_js(tmp_path):
    """The /static/js/dashboard.js bundle pulls in the toast helper
    (ensureToastHost creates #toast-host on boot). Save handlers will
    be wired to append ?saved=… in a follow-up; the surface is here."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    # The script tag references dashboard.js (toast helper lives in it).
    # The URL carries a ``?v=<mtime-hex>`` cache-buster appended in _layout.
    assert (
        'src="/static/js/dashboard.js?v=' in body
        or 'src="/static/js/dashboard.js"' in body
    )


def test_run_page_has_new_and_resume_tabs(tmp_path, monkeypatch):
    """Run Harness page exposes two mutually-exclusive modes via CSS-
    driven radio tabs at the top of the form: New session (default,
    checked) and Resume existing session."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path, csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # Both mode radios present, New is default.
    assert "id='mode-new'" in body and "value='new' checked" in body
    assert "id='mode-resume'" in body
    # Both tab labels reachable.
    assert ">New session" in body
    assert ">Resume existing session" in body
    # Both panels render.
    assert "run-panel--new" in body
    assert "run-panel--resume" in body
    # Resume button posts to the new /run/resume handler.
    assert "formaction='/run/resume'" in body
    assert ">Resume Now</button>" in body


def test_run_page_resume_panel_shows_session_picker_columns(tmp_path, monkeypatch):
    """The Resume panel renders a session picker with all the columns
    the operator needs: session id, created, last update, repo /
    workspace, app name, status."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    # Seed two sessions: one finished, one running.
    _write_session_log(tmp_path / "logs", "done-abc", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z",
         "workspace_path": "/home/op/projects/myharness"},
        {"event": "session_end", "timestamp": "2026-06-15T10:30:00Z", "exit_code": 0},
    ])
    _write_session_log(tmp_path / "logs", "live-xyz", [
        {"event": "session_start", "timestamp": "2026-06-16T08:00:00Z",
         "workspace_path": "/home/op/projects/widget"},
    ])
    cfg = _make_cfg(
        tmp_path, csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # Column headers expected by the user.
    for header in ("Session ID", "Created", "Last update", "Repo / workspace",
                   "App", "Status"):
        assert header in body, f"missing picker column {header!r}"
    # Both sessions show up as rows.
    assert "value='done-abc'" in body
    assert "value='live-xyz'" in body
    # "App" cell uses workspace basename.
    assert ">myharness<" in body
    assert ">widget<" in body
    # The radio name is what the server reads on POST.
    assert "name='resume_session_id'" in body
    # The picker table opts into sort.
    assert "id='resume-session-picker-table'" in body
    assert "data-sort='date'" in body


def test_resume_picker_has_delete_column_per_row(tmp_path, monkeypatch):
    """Each row in the resume picker gets a Delete column with a
    data-purge-session button so JS can POST to /sessions/{sid}/purge."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    _write_session_log(tmp_path / "logs", "del-target", [
        {"event": "session_start", "timestamp": "2026-06-15T10:00:00Z",
         "workspace_path": "/home/op/projects/myharness"},
        {"event": "session_end", "timestamp": "2026-06-15T10:01:00Z", "exit_code": 0},
    ])
    cfg = _make_cfg(
        tmp_path, csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # Delete column header exists.
    assert ">Delete</th>" in body
    # Each row carries a data-purge-session button keyed to its sid.
    assert "data-purge-session='del-target'" in body
    # No nested <form> inside the resume form (HTML invalid). The
    # delete is a plain button — JS handles the POST.
    assert "action='/sessions/del-target/purge'" not in body
    # The session row carries the sid for the JS click handler.
    assert "data-row-session='del-target'" in body


def test_run_page_resume_picker_empty_state(tmp_path, monkeypatch):
    """When no sessions exist on disk, the resume panel shows a clear
    empty-state pointer back to the New session tab."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path, csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # The picker shows the empty hint instead of a table.
    assert "No sessions on disk yet" in body
    assert "run-panel--resume" in body


def test_layout_includes_mobile_nav_toggle(tmp_path):
    """The hamburger button lives in the header and controls the side
    nav (via aria-controls). CSS hides it above 768px; JS toggles
    body[data-nav-open] on click."""
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    assert 'id="nav-toggle"' in body
    assert 'aria-controls="side-nav"' in body
    # The side-nav has the matching id so aria-controls resolves.
    assert 'id="side-nav"' in body


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


def test_server_root_redirects_to_status_page(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    handle = start_server(cfg, blocking=False)
    assert handle is not None
    try:
        url = f"http://{cfg.host}:{cfg.port}/"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200  # urlopen follows 302
            body = resp.read().decode("utf-8")
        assert "View Status" in body
        assert "bx--side-nav" in body
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


def test_server_serves_static_assets_without_auth(tmp_path, monkeypatch):
    """Static assets must render even when the bearer-token gate is on
    so the 401 page itself can be styled and so air-gap mirrors don't
    need a token. Nothing in static/ is sensitive."""
    monkeypatch.setenv("DASH_AUTH_TOKEN", "swordfish")
    cfg = _make_cfg(tmp_path, token_env="DASH_AUTH_TOKEN")
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    handle = start_server(cfg, blocking=False)
    assert handle is not None
    try:
        # No auth header — protected route is 401.
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"http://{cfg.host}:{cfg.port}/sessions", timeout=2.0,
            )
        assert exc.value.code == 401
        # No auth header — static asset is 200.
        with urllib.request.urlopen(
            f"http://{cfg.host}:{cfg.port}/static/css/app.css", timeout=2.0,
        ) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith("text/css")
        # Favicon shorthand also works.
        with urllib.request.urlopen(
            f"http://{cfg.host}:{cfg.port}/favicon.ico", timeout=2.0,
        ) as resp:
            assert resp.status == 200
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# Carbon shell & new top-level pages (#15)
# ---------------------------------------------------------------------------

def _utc_iso(minutes_ago=0):
    from datetime import datetime, timezone, timedelta
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def test_layout_renders_carbon_link_and_side_nav(tmp_path):
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    assert "carbon-components" in body  # CSS link
    assert "bx--side-nav" in body
    # All five top-level items present
    for label in ("View Status", "Run Harness", "Configure Harness", "View Dashboards", "View Documents"):
        assert label in body


def test_status_page_highlights_status_nav_item(tmp_path):
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/status")
    # The current item gets a bx--side-nav__link--current class
    assert 'bx--side-nav__link bx--side-nav__link--current" href="/status"' in body


def test_docs_page_highlights_docs_nav_item(tmp_path):
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/docs")
    assert 'bx--side-nav__link bx--side-nav__link--current" href="/docs"' in body


# --- View Status ------------------------------------------------------------

def test_status_summarises_today_succeeded_and_failed(tmp_path):
    cfg = _make_cfg(tmp_path)
    # One succeeded today
    _write_session_log(cfg.log_dir, "ok", [
        {"event": "session_start", "timestamp": _utc_iso(60), "workspace_path": "/w"},
        {"event": "llm_call", "timestamp": _utc_iso(60), "cost_usd": 0.05, "tokens_in": 10, "tokens_out": 5},
        {"event": "session_end", "timestamp": _utc_iso(55), "exit_code": 0},
    ])
    # One failed today
    _write_session_log(cfg.log_dir, "bad", [
        {"event": "session_start", "timestamp": _utc_iso(40), "workspace_path": "/w"},
        {"event": "session_end", "timestamp": _utc_iso(35), "exit_code": 1},
    ])
    _, _, body = dispatch(cfg, "/status")
    assert "Today" in body
    # 1 succeeded, 1 failed, 2 completed in the today tile
    assert "Succeeded" in body and "Failed" in body
    # Row for the succeeded session shows succeeded tag
    assert "tag-green" in body
    assert "tag-red" in body
    # Open dashboard links exist for both
    assert "/sessions/ok" in body
    assert "/sessions/bad" in body


def test_status_lists_running_session_with_no_session_end(tmp_path):
    cfg = _make_cfg(tmp_path)
    # In-flight CLI session — last event is not session_end
    _write_session_log(cfg.log_dir, "inflight", [
        {"event": "session_start", "timestamp": _utc_iso(5), "workspace_path": "/w"},
        {"event": "llm_call", "timestamp": _utc_iso(3), "cost_usd": 0.01},
    ])
    _, _, body = dispatch(cfg, "/status")
    assert "Running now" in body
    assert "inflight" in body
    assert "tag-gray" in body  # source=cli tag


# --- View Dashboards --------------------------------------------------------

def test_dashboards_landing_lists_all_tile_links(tmp_path):
    cfg = _make_cfg(tmp_path)
    _, _, body = dispatch(cfg, "/dashboards")
    for href in ("/status", "/cost", "/sessions", "/schedule", "/index", "/memory", "/live", "/config"):
        assert f"href='{href}'" in body


# --- View Documents ---------------------------------------------------------

def test_docs_index_lists_md_files_only(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.md").write_text("# Alpha\nhello\n")
    (docs_dir / "notes.txt").write_text("just text\n")
    (docs_dir / "skip.bin").write_text("binary\n")
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    _, _, body = dispatch(cfg, "/docs")
    assert "alpha.md" in body
    assert "notes.txt" in body
    assert "skip.bin" not in body


def test_docs_file_renders_markdown_to_html(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text(
        "# Heading\n\nA **bold** word and *italic*.\n\n"
        "- item one\n- item two\n\n"
        "```python\nprint('hi')\n```\n"
    )
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    status, _, body = dispatch(cfg, "/docs/guide.md")
    assert status == 200
    assert "<h1>Heading</h1>" in body
    assert "<strong>bold</strong>" in body
    assert "<em>italic</em>" in body
    assert "<ul>" in body and "<li>item one</li>" in body
    assert '<code class="language-python">' in body or "<code>print" in body
    # No download disposition
    assert "attachment" not in body


def test_docs_file_renders_txt_as_escaped_pre(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "raw.txt").write_text("<script>alert(1)</script>\n")
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    status, _, body = dispatch(cfg, "/docs/raw.txt")
    assert status == 200
    assert "<pre>" in body
    # < and > escaped
    assert "&lt;script&gt;" in body
    assert "<script>alert" not in body


def test_docs_file_rejects_path_traversal(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("# secret\n")
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    # The regex route allows the chars; the containment check rejects.
    # urllib-encoded ../ also rejected.
    status, _, _ = dispatch(cfg, "/docs/..%2Fsecret.md")
    # Either the route doesn't match (404 from default) or the read returns 404.
    assert status == 404


def test_docs_file_rejects_disallowed_extension(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "thing.bin").write_text("nope\n")
    cfg = _make_cfg(tmp_path, docs_dir=str(docs_dir))
    # The route regex itself only allows .md/.txt so we expect a 404 from dispatch.
    status, _, _ = dispatch(cfg, "/docs/thing.bin")
    assert status == 404


# --- Run Harness ------------------------------------------------------------

def test_run_page_renders_form_with_default_writes_on(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CSRF", "tok")
    # Default cfg has writes_enabled=True — the form should render with
    # no extra flag.
    cfg = _make_cfg(
        tmp_path,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    assert cfg.writes_enabled is True
    _, _, body = dispatch(cfg, "/run")
    assert ">Run Now</button>" in body


def test_run_page_says_writes_disabled_when_explicitly_off(tmp_path):
    cfg = _make_cfg(tmp_path, writes_enabled=False)
    _, _, body = dispatch(cfg, "/run")
    assert "Writes are disabled" in body
    # Hint points at the config knob, not a CLI flag.
    assert "dashboard.writes_enabled" in body


def test_run_page_renders_form_when_writes_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # Primary buttons
    assert ">Run Now</button>" in body
    assert ">Schedule A Run</button>" in body
    # Hidden CSRF + fire-at-utc inputs
    assert "name='csrf_token' value='tok'" in body
    assert "id='fire-at-utc'" in body
    # Scheduled-runs table heading
    assert "Scheduled runs" in body


def test_run_page_renders_per_flag_inputs(tmp_path, monkeypatch):
    """The Run Harness page mirrors the interactive CLI wizard: workspace
    + prompt have dedicated inputs at the top, and the Run options table
    surfaces exactly the three wizard fields (git mode, new build,
    discover). Other CLI flags stay on the terminal."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run")
    # The legacy "Extra harness args" combined textbox is gone.
    assert "Extra harness args" not in body
    assert "name='extra_args'" not in body
    # --git enable/disable select.
    assert "name='flag.git'" in body
    assert "<option value='enable'" in body and "<option value='disable'" in body
    # --new-build true/false select.
    assert "name='flag.new_build'" in body
    assert "<option value='true'" in body and "<option value='false'" in body
    # --discover yes/no select.
    assert "name='flag.discover'" in body
    assert "<option value='yes'" in body and "<option value='no'" in body
    # Flags NOT in the wizard stay off the web page — the operator gets
    # them on the terminal. Catching their absence here is the drift
    # detector for "did someone add another input?".
    for absent in (
        "flag.build_cmd", "flag.output_dir", "flag.session_id",
        "flag.thread_id", "flag.allow_network", "flag.verbose",
        "flag.dev_deployment", "flag.force_lock", "flag.assume_yes",
        "flag.spec_review_cycles", "flag.code_review_cycles",
    ):
        assert absent not in body, f"unexpected flag input {absent!r} on Run page"
    # CLI flag names are echoed so operators learn the vocabulary.
    assert "--git" in body
    assert "--new-build" in body
    assert "--discover" in body


# --- Configure Harness ------------------------------------------------------

def test_config_ui_off_when_writes_explicitly_disabled(tmp_path):
    cfg = _make_cfg(tmp_path, writes_enabled=False)
    _, _, body = dispatch(cfg, "/config-ui")
    assert "Writes are disabled" in body
    assert "dashboard.writes_enabled" in body


def test_config_ui_renders_sandbox_section_via_tree_editor(tmp_path, monkeypatch):
    """The sandbox section renders through the tree editor: each
    sub-key gets a __path[]/__type[]/__value[] triple and the section
    form posts to /config-tree/sandbox. Replaces the old curated
    accordion + per-field dropdown."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"sandbox": {"backend": "docker"}}))
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
        config_path=str(config_path),
    )
    _, _, body = dispatch(cfg, "/config-ui")
    # Section renders with the new ct-section wrapper, not bx--accordion.
    assert "data-section='sandbox'" in body
    # The form posts to the new /config-tree/<section> handler.
    assert "action='/config-tree/sandbox'" in body
    # The current backend value is in a __path/__type/__value triple.
    assert "name='__path[]' value='sandbox/backend'" in body
    assert "name='__type[]' value='str'" in body
    assert "value='docker'" in body
    # Per-section save button preserves "Save sandbox" wording.
    assert "Save sandbox" in body
    # Footer "deployment.json" notice still present.
    assert "deployment.json" in body


def test_config_ui_groups_related_sections_with_collapsible_headers(tmp_path, monkeypatch):
    """Related config sections are bundled under a group header (LLM
    Registry, LLM Routing, etc.) that the operator can expand/collapse."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "sandbox": {"backend": "docker"},
        "model_routing": {"planning_primary": "gpt-4"},
    }))
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
        config_path=str(config_path),
    )
    _, _, body = dispatch(cfg, "/config-ui")
    # Outer group is a native <details> with the legacy class names
    # preserved so operator CSS overrides keep working.
    assert "<details class='config-group'" in body
    assert "<summary class='config-group__heading'" in body
    # Toggle glyph: "+" rendered in markup (CSS swaps to "−" when [open]).
    assert "config-group__toggle" in body and ">+<" in body
    # Group titles operators expect to see.
    for title in (
        "General",
        "LLM Registry",
        "LLM Routing",
        "Sandbox &amp; Security",
        "Budget &amp; Throttling",
        "Logging &amp; Debug",
        "Skills &amp; Tools",
        "Patching &amp; Speculation",
        "Storage &amp; Memory",
        "Scheduling",
        "Dashboard",
    ):
        assert title in body, f"missing group header {title!r}"
    # Groups start collapsed — <details> is closed by default.
    assert "<details open" not in body  # no group is force-expanded
    # Inside, the new tree editor renders one editor per section with
    # path-encoded form fields targeting /config-tree/<section>.
    assert "data-section='sandbox'" in body
    assert "data-section='model_routing'" in body
    assert "name='__path[]' value='sandbox/backend'" in body
    assert "name='__path[]' value='model_routing/planning_primary'" in body


# ---------------------------------------------------------------------------
# Configure page — external-edit detection (mtime stamp + /api/config-mtime)
# ---------------------------------------------------------------------------

def _minimal_valid_config_dict() -> dict:
    """Smallest config shape ``validate_config_strict`` accepts. Used
    by the stale-write tests so the writer's atomic-write path actually
    reaches the mtime comparison instead of bouncing off validation."""
    return {
        "build_command": "make",
        "allow_network": False,
        "product_spec_dir": "p",
        "sandbox": {"backend": "auto", "docker_image": "x:1",
                    "docker_memory_limit": "512m", "docker_cpu_limit": "1.0",
                    "docker_pids_limit": 100, "readonly_cache_mounts": [],
                    "timeout_seconds": 300, "pgid_kill_on_timeout": True,
                    "restore_workspace_ownership": True},
        "token_budget": {"hard_cap_usd": 1.0,
                         "context_window_threshold_pct": 0.85},
        "models": {"m": {
            "provider": "ollama", "model_id": "m",
            "context_window": 4096, "input_cost_per_1m": 0.0,
            "output_cost_per_1m": 0.0,
            "api_base_url": "http://x", "api_key": "",
            "supports_thinking": False, "supports_cache": False,
        }},
        "model_routing": {"planning_primary": "m",
                          "patching_primary": "m",
                          "repair_primary": "m"},
        "persistence": {"db_path": "~/.harness/x.db", "ttl_days": 30},
    }

def test_config_ui_stamps_base_mtime_ns_on_every_section_form(tmp_path, monkeypatch):
    """Every per-section form on /config-ui carries a hidden
    ``__base_mtime_ns`` field that snapshots the on-disk mtime so the
    save handler can detect concurrent external rewrites. The outer
    ``.configure-page`` wrapper exposes the same value as a data-attr
    for the live-poll banner JS."""
    from harness.dashboard import config_file_mtime_ns
    monkeypatch.setenv("FAKE_CSRF", "tok")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"sandbox": {"backend": "docker"}}))
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
        config_path=str(config_path),
    )
    expected_mtime = config_file_mtime_ns(cfg)
    assert expected_mtime is not None
    _, _, body = dispatch(cfg, "/config-ui")
    # Outer wrapper carries the snapshot + poll URL for the JS.
    assert f"data-config-mtime-ns='{expected_mtime}'" in body
    assert "data-config-mtime-poll-url='/api/config-mtime'" in body
    # Stale banner placeholder is rendered (hidden until JS unhides it).
    assert "id='config-stale-banner'" in body and "hidden" in body
    # Each section form also carries the hidden field — both for the
    # tree editor and (defensively) the legacy curated form, so a save
    # POST always has the baseline available.
    assert f"name='__base_mtime_ns' value='{expected_mtime}'" in body


def test_config_ui_empty_base_mtime_when_config_file_missing(tmp_path, monkeypatch):
    """When config.json doesn't exist yet (fresh install / new
    deployment), the hidden field renders an empty value — the save
    handler interprets that as "no baseline, skip stale check"."""
    monkeypatch.setenv("FAKE_CSRF", "tok")
    cfg = _make_cfg(
        tmp_path,
        writes_enabled=True,
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
        config_path=str(tmp_path / "missing.json"),
    )
    _, _, body = dispatch(cfg, "/config-ui")
    assert "data-config-mtime-ns=''" in body
    # No baseline → no banner trigger, but the wrapper still renders so
    # the poller no-ops cleanly.
    assert "configure-page" in body


def test_api_config_mtime_returns_current_ns(tmp_path):
    """``GET /api/config-mtime`` reports the file's current mtime in
    nanoseconds. The Configure page polls this and compares against
    its render-time baseline to detect external edits."""
    import os as _os
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"sandbox": {"backend": "docker"}}))
    cfg = _make_cfg(tmp_path, config_path=str(config_path))
    status, ctype, body = dispatch(cfg, "/api/config-mtime")
    assert status == 200
    assert "application/json" in ctype
    payload = json.loads(body)
    # Serialized as a STRING so JavaScript's JSON.parse doesn't lose
    # precision on modern ns mtimes (~1.75e18, well past 2**53).
    assert payload["mtime_ns"] == str(_os.stat(str(config_path)).st_mtime_ns)


def test_api_config_mtime_returns_null_when_missing(tmp_path):
    """If the file is missing, the endpoint returns ``mtime_ns: null``.
    The poller treats that as "no change" — there's no baseline to
    diverge from anyway."""
    cfg = _make_cfg(tmp_path, config_path=str(tmp_path / "missing.json"))
    status, _, body = dispatch(cfg, "/api/config-mtime")
    assert status == 200
    assert json.loads(body) == {"mtime_ns": None}


def test_write_config_section_atomic_rejects_stale_base_mtime(tmp_path):
    """When the supplied base mtime doesn't match the file's current
    mtime, the writer refuses to save and returns the
    :data:`CONFIG_STALE_MARKER` sentinel so the handler can render a
    "reload to see latest values" banner instead of a validation error.
    The stale check runs BEFORE strict validation, so an external
    rewrite by an arbitrary process trips this even for a minimal
    config shape."""
    from harness.dashboard import (
        CONFIG_STALE_MARKER, config_file_mtime_ns,
        write_config_section_atomic,
    )

    config_path = tmp_path / "config.json"
    initial = _minimal_valid_config_dict()
    config_path.write_text(json.dumps(initial))
    cfg = _make_cfg(tmp_path, config_path=str(config_path), writes_enabled=True)
    baseline = config_file_mtime_ns(cfg)
    # Simulate an external rewrite by re-touching the file with a new mtime.
    # os.utime with ns=(now+1s) guarantees a strictly different mtime even
    # on coarse-resolution filesystems.
    new_ns = (baseline or 0) + 1_000_000_000  # +1s
    os.utime(str(config_path), ns=(new_ns, new_ns))
    # Save with the stale baseline → rejected, file untouched.
    new_sandbox = dict(initial["sandbox"], timeout_seconds=999)
    ok, msg = write_config_section_atomic(
        cfg, "sandbox", new_sandbox, expected_base_mtime_ns=baseline,
    )
    assert ok is False
    assert msg.startswith(CONFIG_STALE_MARKER)
    # File contents unchanged.
    with open(config_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["sandbox"]["timeout_seconds"] == 300


def test_write_config_section_atomic_succeeds_with_matching_mtime(tmp_path):
    """When the submitted baseline still matches disk, the save lands."""
    from harness.dashboard import (
        config_file_mtime_ns, write_config_section_atomic,
    )

    config_path = tmp_path / "config.json"
    initial = _minimal_valid_config_dict()
    config_path.write_text(json.dumps(initial))
    cfg = _make_cfg(tmp_path, config_path=str(config_path), writes_enabled=True)
    baseline = config_file_mtime_ns(cfg)
    new_sandbox = dict(initial["sandbox"], timeout_seconds=999)
    ok, msg = write_config_section_atomic(
        cfg, "sandbox", new_sandbox, expected_base_mtime_ns=baseline,
    )
    assert ok is True, msg
    with open(config_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["sandbox"]["timeout_seconds"] == 999


def test_write_config_section_atomic_no_baseline_skips_stale_check(tmp_path):
    """When the caller doesn't supply a baseline (legacy clients,
    file-missing-at-render case), the writer doesn't enforce the
    stale check at all — preserving the pre-change behavior."""
    from harness.dashboard import write_config_section_atomic

    config_path = tmp_path / "config.json"
    initial = _minimal_valid_config_dict()
    config_path.write_text(json.dumps(initial))
    cfg = _make_cfg(tmp_path, config_path=str(config_path), writes_enabled=True)
    # Touch the file to a totally different mtime — irrelevant when no
    # baseline is passed.
    os.utime(str(config_path), ns=(123_000_000_000, 123_000_000_000))
    new_sandbox = dict(initial["sandbox"], timeout_seconds=42)
    ok, msg = write_config_section_atomic(
        cfg, "sandbox", new_sandbox, expected_base_mtime_ns=None,
    )
    assert ok is True, msg
