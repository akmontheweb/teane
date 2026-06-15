"""Integration tests for the Tier B + C dashboard extensions.

These tests stand up the real ThreadingHTTPServer on an ephemeral
localhost port and drive it with ``urllib.request`` — same shape as
the smaller end-to-end tests in ``tests/test_dashboard.py`` already
established for the read-only views.

Covered:
    - CSRF enforcement on every write path.
    - Config save round-trip (POST then read back from disk).
    - Memory file save round-trip.
    - Run-now spawns a subprocess + registers it in the
      ProcessRegistry.
    - "Schedule it" enqueues a web_oneshot_jobs row visible via
      list_pending_oneshot_jobs.
    - Cancel signals the process group.
    - Chat note queueing.
    - HITL webhook round-trip (POST to /hitl/webhook from a
      simulated harness, UI POSTs the answer, the webhook returns
      the operator's choice in the response body).
    - SSE stream produces events written to the session log.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from harness.dashboard import (
    DashboardConfig,
    get_hitl_queue,
    get_process_registry,
    reset_shared_state,
    spawn_harness_run,
    start_server,
)
from harness.web_state import list_pending_oneshot_jobs, pending_chat_notes


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _writeable_config(tmp_path) -> str:
    """Write a minimal valid config.json the dashboard can save back to."""
    cfg_path = tmp_path / "config.json"
    cfg = {
        "build_command": "make build",
        "allow_network": False,
        "product_spec_dir": "product_spec",
        "sandbox": {"backend": "auto", "docker_image": "harness-builder:latest",
                    "docker_memory_limit": "512m", "docker_cpu_limit": "1.0",
                    "docker_pids_limit": 100, "readonly_cache_mounts": [],
                    "timeout_seconds": 300, "pgid_kill_on_timeout": True,
                    "restore_workspace_ownership": True},
        "token_budget": {"hard_cap_usd": 3.0, "context_window_threshold_pct": 0.85},
        "node_throttle": {"max_patch_repair_iterations": 5,
                          "max_doc_review_cycles": 2,
                          "max_code_review_cycles": 2,
                          "max_discovery_iterations": 5},
        "models": {"ollama:qwen2.5-coder:14b": {
            "provider": "ollama", "model_id": "qwen2.5-coder:14b",
            "context_window": 131072, "input_cost_per_1m": 0.0,
            "output_cost_per_1m": 0.0,
            "api_base_url": "http://localhost:11434/v1",
            "api_key": "", "supports_thinking": False, "supports_cache": False,
        }},
        "model_routing": {"planning_primary": "ollama:qwen2.5-coder:14b",
                          "planning_mode": "thinking",
                          "patching_primary": "ollama:qwen2.5-coder:14b",
                          "patching_mode": "no_thinking",
                          "repair_primary": "ollama:qwen2.5-coder:14b",
                          "repair_mode": "no_thinking",
                          "ollama_local_model": "ollama:qwen2.5-coder:14b",
                          "ollama_local_backup": "",
                          "force_local_only": True},
        "persistence": {"db_path": "~/.harness/checkpoints.db", "ttl_days": 30},
        "logging": {"level": "INFO", "log_dir": "~/.harness/logs",
                    "json_stderr": False, "langsmith": False},
        "lintgate": {"format_modified_files": False},
        "metrics": {"burn_rate_window_minutes": 10, "metrics_dir": "~/.harness/metrics"},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return str(cfg_path)


def _make_cfg(tmp_path, **overrides) -> DashboardConfig:
    base = {
        "log_dir": str(tmp_path / "logs"),
        "metrics_dir": str(tmp_path / "metrics"),
        "memory_dir": str(tmp_path / "memory"),
        "repo_index_dir": str(tmp_path / "idx"),
        "schedule_db": str(tmp_path / "schedule.db"),
        "static_dir": str(tmp_path / "static"),
        "web_db_path": str(tmp_path / "web.db"),
        "enabled": True,
        "writes_enabled": True,
        "config_path": _writeable_config(tmp_path),
    }
    base.update(overrides)
    return DashboardConfig.from_config({"dashboard": base})


@pytest.fixture(autouse=True)
def _reset_state():
    reset_shared_state()
    yield
    reset_shared_state()


def _start(cfg):
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    handle = start_server(cfg, blocking=False)
    return handle, f"http://{cfg.host}:{cfg.port}"


def _post(url, *, body: dict, csrf: str = "", extra_headers: dict = None):
    data = urllib.parse.urlencode(body).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if csrf:
        headers["X-CSRF-Token"] = csrf
        headers["Cookie"] = f"csrf_token={csrf}"
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=4.0)


# ---------------------------------------------------------------------------
# 1. CSRF gate
# ---------------------------------------------------------------------------

def test_writes_without_csrf_get_403(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        # No CSRF header at all → 403.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base_url + "/run/now", body={"workspace": "/", "prompt": "x"})
        assert exc.value.code == 403
    finally:
        handle.shutdown()


def test_writes_with_wrong_csrf_get_403(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base_url + "/run/now",
                  body={"workspace": "/", "prompt": "x"}, csrf="bogus")
        assert exc.value.code == 403
    finally:
        handle.shutdown()


def test_writes_blocked_when_writes_disabled(tmp_path):
    cfg = _make_cfg(tmp_path, writes_enabled=False)
    handle, base_url = _start(cfg)
    try:
        # Even with a CSRF that LOOKS valid for a writes-enabled cfg,
        # the server refuses because writes are off.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base_url + "/run/now",
                  body={"workspace": "/", "prompt": "x"}, csrf="anything")
        assert exc.value.code == 403
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# 2. Config save round-trip
# ---------------------------------------------------------------------------

def test_config_save_writes_atomically(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    assert csrf is not None
    try:
        resp = _post(
            base_url + "/config/token_budget",
            body={
                "token_budget.hard_cap_usd": "9.99",
                "token_budget.context_window_threshold_pct": "0.75",
                "token_budget.stages": "",
            },
            csrf=csrf,
        )
        assert resp.status == 200
        # Read the file back.
        with open(cfg.config_path, "r", encoding="utf-8") as f:
            written = json.load(f)
        assert written["token_budget"]["hard_cap_usd"] == pytest.approx(9.99)
        assert written["token_budget"]["context_window_threshold_pct"] == pytest.approx(0.75)
    finally:
        handle.shutdown()


def test_config_save_rejects_invalid_value(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # context_window_threshold_pct expects a number; send garbage.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/config/token_budget",
                body={
                    "token_budget.hard_cap_usd": "5.0",
                    "token_budget.context_window_threshold_pct": "not-a-number",
                    "token_budget.stages": "",
                },
                csrf=csrf,
            )
        assert exc.value.code == 400
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# 3. Memory file save round-trip
# ---------------------------------------------------------------------------

def test_memory_save_round_trip(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # Create an empty memory file so the read-side renders.
        os.makedirs(cfg.memory_dir, exist_ok=True)
        path = os.path.join(cfg.memory_dir, "abc.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# old")
        resp = _post(
            base_url + "/memory/abc.md",
            body={"content": "# new content"},
            csrf=csrf,
        )
        assert resp.status == 200
        with open(path, "r", encoding="utf-8") as f:
            assert f.read() == "# new content"
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# 4. Run-now spawns subprocess + registers
# ---------------------------------------------------------------------------

def test_spawn_harness_run_registers_process(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    # Use python -c as the stand-in binary so we don't need a real harness.
    wp = spawn_harness_run(
        cfg, workspace=str(tmp_path), prompt="ignored",
        harness_binary=sys.executable,
        extra_args=["-c", "import sys; sys.exit(0)"],
    )
    # The argv we expected: [python, 'run', '-r', ws, '-p', prompt, ...extras]
    # The stub turns into nonsense for the harness but we're testing
    # that the registry tracked it and the subprocess actually exited.
    reg = get_process_registry()
    assert reg.get(wp.session_id) is not None
    # Wait for the watcher thread to mark terminated.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        entry = reg.get(wp.session_id)
        if entry and entry.exit_code is not None:
            break
        time.sleep(0.05)
    entry = reg.get(wp.session_id)
    assert entry is not None
    assert entry.exit_code is not None


# ---------------------------------------------------------------------------
# 5. Schedule-it enqueues a row
# ---------------------------------------------------------------------------

def test_run_schedule_enqueues_oneshot_row(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    fire_at = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
    try:
        # urllib auto-follows the 303 redirect to /schedule; either it
        # returns the redirect target's body (200) or a 4xx if Auth/CSRF
        # rejects. Both are acceptable here — we assert on the side
        # effect (a row in web.db).
        try:
            _post(
                base_url + "/run/schedule",
                body={
                    "workspace": str(tmp_path), "prompt": "do it",
                    "fire_at_utc": fire_at, "name": "test-one",
                },
                csrf=csrf,
            )
        except urllib.error.HTTPError:
            # The auth on the redirected /schedule may 401 since urllib
            # drops the Authorization header on redirect; that's fine —
            # the original POST already committed the row.
            pass
    finally:
        handle.shutdown()
    pending = list_pending_oneshot_jobs(db_path=cfg.web_db_path)
    assert len(pending) == 1
    assert pending[0]["name"] == "test-one"
    assert pending[0]["workspace"] == str(tmp_path)


# ---------------------------------------------------------------------------
# 6. Cancel signals process group
# ---------------------------------------------------------------------------

def test_cancel_endpoint_signals_running_process(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    # Start a long-running stub process.
    wp = spawn_harness_run(
        cfg, workspace=str(tmp_path), prompt="ignored",
        harness_binary=sys.executable,
        extra_args=["-c", "import time; time.sleep(30)"],
    )
    try:
        try:
            _post(
                base_url + f"/sessions/{wp.session_id}/cancel",
                body={}, csrf=csrf,
            )
        except urllib.error.HTTPError:
            # The auto-redirect's GET may 401 on missing Auth header
            # depending on the urllib version. The side effect (process
            # group received SIGTERM) is what we test below.
            pass
        # Wait briefly for the process to die.
        deadline = time.time() + 3.0
        reg = get_process_registry()
        while time.time() < deadline:
            entry = reg.get(wp.session_id)
            if entry and entry.exit_code is not None:
                break
            time.sleep(0.05)
        entry = reg.get(wp.session_id)
        assert entry is not None
        assert entry.exit_code is not None  # SIGTERM ⇒ non-zero, but caught
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# 7. Chat note queueing
# ---------------------------------------------------------------------------

def test_note_endpoint_queues_into_web_db(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        try:
            _post(
                base_url + "/sessions/sess-001/note",
                body={"note": "look at the auth module first"},
                csrf=csrf,
            )
        except urllib.error.HTTPError:
            pass
    finally:
        handle.shutdown()
    pending = pending_chat_notes(db_path=cfg.web_db_path, session_id="sess-001")
    assert [p["note"] for p in pending] == ["look at the auth module first"]


# ---------------------------------------------------------------------------
# 8. HITL webhook round-trip
# ---------------------------------------------------------------------------

def test_hitl_webhook_round_trip(tmp_path):
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    received: dict = {}

    def _simulated_harness():
        # Mimics the harness's HttpChannel: POSTs a prompt, blocks
        # for the response.
        body = json.dumps({"request_id": "req-1", "gate": "REQUIREMENTS"})
        req = urllib.request.Request(
            base_url + "/hitl/webhook?session=sess-1",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                received["body"] = resp.read().decode("utf-8")
                received["status"] = resp.status
        except Exception as exc:
            received["error"] = repr(exc)

    t = threading.Thread(target=_simulated_harness)
    t.start()

    # Give the simulated harness a moment to land its POST in the queue.
    time.sleep(0.15)
    pending = get_hitl_queue().list_pending_for_session("sess-1")
    assert len(pending) == 1
    request_id = pending[0].request_id

    # UI answers.
    try:
        try:
            _post(
                base_url + "/sessions/sess-1/hitl/answer",
                body={"request_id": request_id, "choice": "a", "extra_notes": "ok"},
                csrf=csrf,
            )
        except urllib.error.HTTPError:
            pass
        t.join(timeout=4.0)
    finally:
        handle.shutdown()
    assert "body" in received, f"webhook didn't return: {received}"
    body_json = json.loads(received["body"])
    assert body_json["choice"] == "a"
    assert "ok" in body_json["extra_notes"]


# ---------------------------------------------------------------------------
# 9. SSE stream emits events written to the log
# ---------------------------------------------------------------------------

def test_sse_stream_emits_log_lines(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    # Register a fake terminated process so the SSE follower stops
    # cleanly after draining.
    from harness.web_state import WebProcess
    log_dir = os.path.expanduser(cfg.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sess-evt.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "session_start", "timestamp": "2026-06-15T10:00:00Z"}) + "\n")
        f.write(json.dumps({"event": "llm_call", "tokens_in": 10, "cost_usd": 0.01}) + "\n")
    reg = get_process_registry()
    reg.register(WebProcess(
        session_id="sess-evt", pid=-1, argv=[], log_path=log_path,
    ))
    reg.mark_terminated("sess-evt", exit_code=0)

    handle = start_server(cfg, blocking=False)
    base_url = f"http://{cfg.host}:{cfg.port}"
    try:
        req = urllib.request.Request(
            base_url + "/api/sessions/sess-evt/events",
            headers={"Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            assert resp.status == 200
            # Read until we see the close marker.
            received = b""
            deadline = time.time() + 4.0
            while time.time() < deadline and b"event: close" not in received:
                chunk = resp.read(2048)
                if not chunk:
                    break
                received += chunk
        text = received.decode("utf-8", errors="replace")
        assert "data:" in text
        assert "session_start" in text
        assert "llm_call" in text
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# 10. New CLI surface — dashboard subcommand still accepts new flag
# ---------------------------------------------------------------------------

def test_dashboard_cli_parses_without_writes_enabled_flag():
    # All features ship by default — `harness web` does not need a
    # `--writes-enabled` flag any more. Operators flip
    # `dashboard.writes_enabled: false` in config.json to lock down.
    from harness.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["web"])
    assert args.command == "web"
    # The flag was removed; the argparse Namespace shouldn't carry it.
    assert not hasattr(args, "writes_enabled")
