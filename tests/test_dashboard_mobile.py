"""Phase 7 regression: mobile session route + one-shot bearer bootstrap."""

from __future__ import annotations

import http.client
import socket


from harness.dashboard import (
    DashboardConfig,
    _render_mobile_session,
    dispatch,
    start_server,
)


def _cfg(tmp_path, **overrides):
    base = {
        "log_dir": str(tmp_path / "logs"),
        "metrics_dir": str(tmp_path / "metrics"),
        "memory_dir": str(tmp_path / "memory"),
        "repo_index_dir": str(tmp_path / "idx"),
        "schedule_db": str(tmp_path / "schedule.db"),
        "web_db_path": str(tmp_path / "web.db"),
        "enabled": True,
        "writes_enabled": True,
    }
    base.update(overrides)
    return DashboardConfig.from_config({"dashboard": base})


# ---------------------------------------------------------------------------
# Body renderer
# ---------------------------------------------------------------------------

def test_mobile_body_contains_activity_feed_and_bottom_bar(tmp_path):
    body = _render_mobile_session(_cfg(tmp_path), "sess-x")
    # Activity feed lives inside — same Alpine component as desktop.
    assert "teaneActivityFeed" in body
    # HITL slot is present (empty when nothing pending).
    assert "hitl-pending-slot" in body
    # Fixed bottom bar with three thumb-targets.
    assert "mobile-bottom-bar" in body
    assert "Home" in body
    assert "HITL" in body
    assert "Full view" in body


def test_mobile_route_returns_200_when_auth_disabled(tmp_path):
    """When no bearer token is configured, /m/<sid> is directly
    reachable — mirrors how every other page behaves in that mode."""
    status, ctype, body = dispatch(_cfg(tmp_path), "/m/sess-x")
    assert status == 200
    assert ctype.startswith("text/html")
    assert "mobile-view" in body


# ---------------------------------------------------------------------------
# One-shot bearer-via-querystring bootstrap
# ---------------------------------------------------------------------------

def _start_server(cfg):
    """Start a real threaded server on the port already baked into
    ``cfg`` so we can verify the query-string cookie bootstrap
    end-to-end. ``blocking=False`` returns the handle."""
    handle = start_server(cfg, blocking=False)
    assert handle is not None
    return handle


def _pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mobile_bootstrap_exchanges_query_token_for_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("TEANE_TEST_TOKEN", "sekret-mobile-token")
    cfg = _cfg(
        tmp_path,
        token_env="TEANE_TEST_TOKEN",
    )
    cfg.port = _pick_free_port()
    handle = _start_server(cfg)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=4.0)
        conn.request("GET", "/m/sess-x?t=sekret-mobile-token")
        resp = conn.getresponse()
        # Body is empty on the 302 but stdlib requires we drain it
        # before reusing the connection or reading response headers
        # reliably in some Python versions.
        resp.read()
        assert resp.status == 302, f"expected 302 bootstrap redirect, got {resp.status}"
        # Landed on the token-less URL.
        assert resp.getheader("Location") == "/m/sess-x"
        # Cookie set for /m/* only, with the right attributes.
        set_cookie = resp.getheader("Set-Cookie") or ""
        assert "teane_mobile_auth=sekret-mobile-token" in set_cookie
        assert "Path=/m/" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite=Strict" in set_cookie
    finally:
        handle.shutdown()


def test_mobile_cookie_auths_subsequent_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("TEANE_TEST_TOKEN", "sekret-mobile-token")
    cfg = _cfg(tmp_path, token_env="TEANE_TEST_TOKEN")
    cfg.port = _pick_free_port()
    handle = _start_server(cfg)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=4.0)
        conn.request(
            "GET", "/m/sess-x",
            headers={"Cookie": "teane_mobile_auth=sekret-mobile-token"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        assert resp.status == 200
        assert "mobile-view" in body
    finally:
        handle.shutdown()


def test_mobile_query_bootstrap_rejects_bad_token(tmp_path, monkeypatch):
    monkeypatch.setenv("TEANE_TEST_TOKEN", "sekret-mobile-token")
    cfg = _cfg(tmp_path, token_env="TEANE_TEST_TOKEN")
    cfg.port = _pick_free_port()
    handle = _start_server(cfg)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=4.0)
        conn.request("GET", "/m/sess-x?t=wrong-token")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 401, (
            "wrong token must NOT bootstrap a cookie; 401 is the right "
            "response so the phone browser doesn't cache a broken URL."
        )
    finally:
        handle.shutdown()
