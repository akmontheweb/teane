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


def test_config_save_rejected_when_disk_changed_under_us(tmp_path):
    """End-to-end: render-time stamps base mtime → external rewrite →
    save POST carrying the stale baseline gets 409 with the
    operator-facing "modified outside this browser tab" message. The
    file content is left untouched so the operator can reload and
    re-apply their edits against the current state."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # Snapshot the file's current mtime as the operator's "baseline".
        baseline_ns = os.stat(cfg.config_path).st_mtime_ns
        # Simulate a backend rewrite by bumping the mtime forward 1s.
        new_ns = baseline_ns + 1_000_000_000
        os.utime(cfg.config_path, ns=(new_ns, new_ns))
        # Save with the stale baseline → must be rejected with 409.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/config/token_budget",
                body={
                    "token_budget.hard_cap_usd": "9.99",
                    "token_budget.context_window_threshold_pct": "0.75",
                    "token_budget.stages": "",
                    "__base_mtime_ns": str(baseline_ns),
                },
                csrf=csrf,
            )
        assert exc.value.code == 409
        msg = exc.value.read().decode("utf-8", errors="replace")
        assert "modified outside this browser tab" in msg
        # File untouched: hard_cap_usd still its original value.
        with open(cfg.config_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["token_budget"]["hard_cap_usd"] == pytest.approx(3.0)
    finally:
        handle.shutdown()


def test_config_save_succeeds_with_fresh_base_mtime(tmp_path):
    """The stale-write check doesn't block legitimate saves — when the
    submitted baseline still matches disk, the write lands and the
    file's mtime advances (which is what the next render baselines
    against)."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        baseline_ns = os.stat(cfg.config_path).st_mtime_ns
        resp = _post(
            base_url + "/config/token_budget",
            body={
                "token_budget.hard_cap_usd": "12.5",
                "token_budget.context_window_threshold_pct": "0.9",
                "token_budget.stages": "",
                "__base_mtime_ns": str(baseline_ns),
            },
            csrf=csrf,
        )
        assert resp.status == 200
        with open(cfg.config_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["token_budget"]["hard_cap_usd"] == pytest.approx(12.5)
        assert os.stat(cfg.config_path).st_mtime_ns >= baseline_ns
    finally:
        handle.shutdown()


def test_api_config_mtime_endpoint_serves_current_mtime(tmp_path):
    """The poll endpoint is reachable through the real HTTP server and
    returns the on-disk mtime as JSON. Used by the live-poll banner
    JS to detect external edits while the Configure page is open."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        resp = urllib.request.urlopen(
            base_url + "/api/config-mtime", timeout=4.0,
        )
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))
        # mtime_ns is serialized as a string — see _route_api_config_mtime.
        assert payload["mtime_ns"] == str(os.stat(cfg.config_path).st_mtime_ns)
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

def test_run_now_blocks_when_workspace_already_running(tmp_path):
    """Configure-page overhaul: ``/run/now`` returns 409 when the
    workspace already has a live harness subprocess registered."""
    from harness.dashboard import get_process_registry
    from harness.web_state import WebProcess

    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # Pre-register a "running" process for this workspace. Use the
        # test process's own PID so the registry's orphan-prune sweep
        # (which flips entries with dead PIDs to terminated) doesn't
        # silently flip our fake busy entry behind the test's back.
        get_process_registry().register(WebProcess(
            session_id="busy", pid=os.getpid(), argv=["harness"],
            workspace_path=str(tmp_path),
        ))
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/run/now",
                body={"workspace": str(tmp_path), "prompt": "second"},
                csrf=csrf,
            )
        assert exc.value.code == 409
        assert b"already in progress" in exc.value.read()
    finally:
        handle.shutdown()


def test_run_now_rejects_nonexistent_workspace(tmp_path):
    """``/run/now`` validates the workspace path exists before spawning
    a subprocess. A typo'd path returns 400 instead of bouncing off a
    confusing harness-run error several seconds later."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/run/now",
                body={
                    "workspace": str(tmp_path / "does-not-exist"),
                    "prompt": "hi",
                },
                csrf=csrf,
            )
        assert exc.value.code == 400
        assert b"workspace not found" in exc.value.read()
    finally:
        handle.shutdown()


def test_run_now_rejects_oversize_prompt(tmp_path):
    """``/run/now`` caps prompt size at 50 KB so no single submission
    can DoS the spawn loop."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/run/now",
                body={
                    "workspace": str(tmp_path),
                    "prompt": "x" * 60_000,
                },
                csrf=csrf,
            )
        assert exc.value.code == 400
        assert b"prompt too long" in exc.value.read()
    finally:
        handle.shutdown()


def test_run_now_accepts_file_only_input(tmp_path):
    """When the operator uploads a spec file via /api/upload-spec, the
    Run-now form may submit with an empty ``prompt`` as long as
    ``spec_file_path`` is set."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # Empty prompt + no spec_file_path → 400.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/run/now",
                body={"workspace": str(tmp_path), "prompt": ""},
                csrf=csrf,
            )
        assert exc.value.code == 400
        # Empty prompt WITH a spec_file_path passes the form check
        # (the spawn itself may then fail with the python stub binary,
        # but we only care that the validation gate accepts it).
        spec_path = str(tmp_path / "product_spec" / "x.md")
        try:
            _post(
                base_url + "/run/now",
                body={
                    "workspace": str(tmp_path), "prompt": "",
                    "spec_file_path": spec_path,
                },
                csrf=csrf,
            )
        except urllib.error.HTTPError as exc:
            # 500 is acceptable — means the spawn ran but failed against
            # the non-harness binary. What matters is "not 400".
            assert exc.code != 400
    finally:
        handle.shutdown()


def test_run_schedule_blocks_when_within_10_minutes(tmp_path):
    """Two scheduled runs within 10 minutes of each other return 409."""
    from harness.web_state import add_oneshot_job
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    base_time = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
    add_oneshot_job(
        db_path=cfg.web_db_path, name="existing",
        fire_at_utc=base_time, workspace=str(tmp_path),
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/run/schedule",
                body={
                    "workspace": str(tmp_path), "prompt": "second",
                    "fire_at_utc": (base_time + timedelta(minutes=5)).isoformat(),
                    "name": "clash",
                },
                csrf=csrf,
            )
        assert exc.value.code == 409
        msg = exc.value.read()
        assert b"already scheduled" in msg
        assert b"Pick a time" in msg
    finally:
        handle.shutdown()


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
# 9b. Session detail page surfaces pending HITL prompts
# ---------------------------------------------------------------------------

def test_session_detail_renders_pending_hitl_form(tmp_path):
    """A pending HITL prompt for a session must show up on
    ``/sessions/<sid>`` as a card with a CSRF-wired answer form.
    Without this wiring, operators have no in-browser way to answer
    and the harness times out waiting for the webhook response."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    assert csrf  # writes_enabled fixture should always yield a token
    try:
        # Make sure the log file exists so _render_session_detail doesn't
        # short-circuit before the HITL panel is appended.
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "sess-hitl.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "session_start"}) + "\n")

        get_hitl_queue().register_pending(
            request_id="req-abc",
            session_id="sess-hitl",
            prompt={"gate": "REQUIREMENTS", "question": "approve?"},
        )

        req = urllib.request.Request(
            base_url + "/sessions/sess-hitl",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        resp = urllib.request.urlopen(req, timeout=4.0)
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "HITL pending" in body
        assert "req-abc" in body
        assert "/sessions/sess-hitl/hitl/answer" in body
        # The hidden csrf_token field must carry a non-empty value. The
        # functional CSRF check (cookie + X-CSRF-Token) is covered by
        # test_hitl_webhook_round_trip; what matters here is that the
        # earlier ``value=''`` placeholder is gone.
        assert "name='csrf_token' value=''" not in body
        import re as _re
        m = _re.search(r"name='csrf_token' value='([^']+)'", body)
        assert m is not None and len(m.group(1)) > 0
    finally:
        handle.shutdown()


def test_session_detail_renders_hitl_above_events(tmp_path):
    """When a HITL prompt is pending, its card must render BEFORE the
    session-detail / Events card so operators see the gate as soon as
    they land on the page. Reordered in the cockpit overhaul — without
    this assertion the order could silently regress."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    assert csrf
    try:
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "sess-order.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "session_start"}) + "\n")
        get_hitl_queue().register_pending(
            request_id="req-order",
            session_id="sess-order",
            prompt={"gate": "REQUIREMENTS", "question": "approve?"},
        )
        req = urllib.request.Request(
            base_url + "/sessions/sess-order",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        body = urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")
        hitl_idx = body.find("hitl-alert")
        events_idx = body.find("Events (")
        assert hitl_idx != -1, "expected hitl-alert wrapper class on pending card"
        assert events_idx != -1, "expected an Events header"
        assert hitl_idx < events_idx, (
            "HITL pending card must render before the Events section"
        )
        # autofocus on the choice input so operators answer without clicking.
        assert "name='choice'" in body and "autofocus" in body
    finally:
        handle.shutdown()


def test_session_detail_renders_stdout_stream_panel(tmp_path):
    """The session page must surface a stdout-stream container wired
    to /api/sessions/<sid>/stdout so the operator sees raw subprocess
    output alongside the JSONL events."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "sess-stdout.jsonl"), "w") as f:
            f.write(json.dumps({"event": "session_start"}) + "\n")
        req = urllib.request.Request(
            base_url + "/sessions/sess-stdout",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        body = urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")
        assert "id='stdout-stream'" in body
        assert "/api/sessions/sess-stdout/stdout" in body
        assert "stdout-stream" in body  # class hook for sticky-bottom JS
    finally:
        handle.shutdown()


def test_stdout_sse_endpoint_streams_lines(tmp_path):
    """GET /api/sessions/<sid>/stdout opens an SSE stream of lines from
    the harness subprocess's combined-output file. The endpoint must
    return text/event-stream and frame each line as a JSON ``{"text"}``
    payload so the dashboard's wireStdoutStream() can render it."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        stdout_path = os.path.join(log_dir, "sess-raw.jsonl.stdout")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("hello\nworld\n")
        # Register the process as "terminated" so the generator drains
        # and returns instead of polling forever.
        from harness.web_state import WebProcess
        reg = get_process_registry()
        reg.register(WebProcess(
            session_id="sess-raw",
            pid=-1,
            argv=[],
            log_path=os.path.join(log_dir, "sess-raw.jsonl"),
        ))
        reg.mark_terminated("sess-raw", 0)
        resp = urllib.request.urlopen(
            base_url + "/api/sessions/sess-raw/stdout", timeout=4.0,
        )
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/event-stream")
        body = resp.read().decode("utf-8")
        assert '"text": "hello"' in body
        assert '"text": "world"' in body
    finally:
        handle.shutdown()


def test_tail_session_stdout_yields_lines_then_exits(tmp_path):
    """Unit test for the generator powering the stdout SSE endpoint.
    With ``follow=False`` it should drain whatever's in the file and
    then return."""
    from harness.dashboard import tail_session_stdout

    p = tmp_path / "x.stdout"
    p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    events = list(tail_session_stdout(str(p), follow=False))
    assert events == [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]


# ---------------------------------------------------------------------------
# Operator console — Currently Running list on /run + /run/console/{sid}
# ---------------------------------------------------------------------------

def test_run_page_shows_currently_running_with_pid(tmp_path):
    """Operators need a way back into a live run from /run without
    fishing through the historical session list. The Currently Running
    card lists each registered subprocess with session id, started time,
    PID, workspace, and a Console link. Without the PID column an
    operator can't correlate the entry to ``ps`` / ``kill -9``."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        from harness.web_state import WebProcess
        # The registry prunes entries whose PID is no longer alive
        # before returning list_running() — use this test process's own
        # PID so the fake entry survives the pruning pass.
        live_pid = os.getpid()
        reg = get_process_registry()
        reg.register(WebProcess(
            session_id="web-livethere",
            pid=live_pid,
            argv=["harness", "run"],
            log_path=os.path.join(os.path.expanduser(cfg.log_dir),
                                  "web-livethere.jsonl"),
            workspace_path=str(tmp_path),
            prompt="example",
        ))
        req = urllib.request.Request(
            base_url + "/run",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        body = urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")
        # Currently Running card + table id
        assert "Currently running" in body
        assert "running-runs-table" in body
        # Session id, PID, and Console link must all be present
        assert "web-livethere" in body
        assert str(live_pid) in body
        assert "/run/console/web-livethere" in body
    finally:
        handle.shutdown()


def test_run_page_shows_no_runs_in_progress_when_empty(tmp_path):
    """When no registered processes are running, the Currently Running
    card still renders with a muted 'No runs in progress.' placeholder
    rather than vanishing — operators rely on its presence to know the
    feature exists."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        req = urllib.request.Request(
            base_url + "/run",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        body = urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")
        assert "Currently running" in body
        assert "No runs in progress." in body
    finally:
        handle.shutdown()


def test_run_console_returns_200_for_known_session(tmp_path):
    """/run/console/{sid} is the operator's live cockpit for an active
    run. The body is deliberately minimal: HITL slot, chat-notes slot,
    raw stdout/stderr stream — no events table, no JSONL events list.
    Operators asked for "just the chat window and the logs" so the
    console keeps the surface focused. The HITL slot still carries a
    hidden SSE channel URL so Phase 2's live banner works without the
    visible events list."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "sess-cn.jsonl"), "w") as f:
            f.write(json.dumps({"event": "session_start"}) + "\n")
        req = urllib.request.Request(
            base_url + "/run/console/sess-cn",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        body = urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")
        # Chrome + Close button back to /run
        assert "run-console-chrome" in body
        assert "Close console" in body
        assert "href='/run'" in body or 'href="/run"' in body
        # Kept panels: HITL slot, chat-notes slot, raw stdout stream.
        assert "id='hitl-pending-slot'" in body or 'id="hitl-pending-slot"' in body
        assert "id='chat-notes-slot'" in body or 'id="chat-notes-slot"' in body
        assert "id='stdout-stream'" in body or 'id="stdout-stream"' in body
        # Hidden SSE channel for live HITL surfacing — Phase 2 contract.
        assert "data-hitl-sse-url" in body
        assert "/api/sessions/sess-cn/events" in body
        # Removed panels: JSONL events list + the session-detail events
        # table. Asserting absence prevents an inadvertent re-add.
        assert "id='event-stream'" not in body and 'id="event-stream"' not in body
        assert "Live events" not in body
    finally:
        handle.shutdown()


def test_run_console_returns_404_for_unknown_session(tmp_path):
    """A stale Console link (process gone, no on-disk log) must 404 so
    the operator sees the broken URL instead of a confusing empty page."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                base_url + "/run/console/nope-not-a-session",
                timeout=4.0,
            )
        assert exc.value.code == 404
    finally:
        handle.shutdown()


def test_run_console_marks_exited_when_process_terminated(tmp_path):
    """For an exited run, the console keeps working (streams are
    scrollable) but shows an 'Exited' tag and applies the
    ``run-console--exited`` wrapper so CSS can dim the chat textarea."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        from harness.web_state import WebProcess
        reg = get_process_registry()
        reg.register(WebProcess(
            session_id="sess-done",
            pid=1,
            argv=[],
            log_path=os.path.join(os.path.expanduser(cfg.log_dir),
                                  "sess-done.jsonl"),
        ))
        reg.mark_terminated("sess-done", 0)
        log_dir = os.path.expanduser(cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "sess-done.jsonl"), "w") as f:
            f.write(json.dumps({"event": "session_end", "exit_code": 0}) + "\n")
        body = urllib.request.urlopen(
            base_url + "/run/console/sess-done", timeout=4.0,
        ).read().decode("utf-8")
        assert "run-console--exited" in body
        assert "Exited (code 0)" in body
    finally:
        handle.shutdown()


def test_run_now_redirects_to_console(tmp_path, monkeypatch):
    """After Run Now the operator must land on /run/console/{sid} so
    the cockpit (live logs + inline HITL) opens immediately — NOT on
    the historical session-detail page. The earlier UX bounced through
    /sessions/{sid} which sat under Dashboards; operators read that as
    'I got sent away from my run'."""
    import subprocess

    captured: dict = {}

    class _StubPopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            self.pid = 12345

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _StubPopen)
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        # Disable urllib auto-follow so we can read the 303 Location.
        opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler()  # default behaviour
        )

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_a, **_kw):  # noqa: D401
                return None

        opener = urllib.request.build_opener(_NoRedirect())
        data = urllib.parse.urlencode({
            "workspace": str(tmp_path),
            "prompt": "demo",
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url + "/run/now", data=data, method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": csrf,
                "Cookie": f"csrf_token={csrf}",
            },
        )
        try:
            resp = opener.open(req, timeout=4.0)
            assert resp.status == 303
            location = resp.headers.get("Location", "")
        except urllib.error.HTTPError as exc:
            assert exc.code == 303
            location = exc.headers.get("Location", "")
        assert location.startswith("/run/console/")
    finally:
        handle.shutdown()


# ---------------------------------------------------------------------------
# Operator console — AJAX slot endpoints + harness hitl_pending events
# ---------------------------------------------------------------------------

def test_hitl_pending_html_returns_card_html(tmp_path):
    """The /sessions/{sid}/hitl/pending.html fragment endpoint powers
    the SSE-driven live banner refresh — it returns just the HITL card
    markup so the JS can swap ``#hitl-pending-slot`` innerHTML in place
    without touching the rest of the page."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        get_hitl_queue().register_pending(
            request_id="req-ajax",
            session_id="sess-ajax",
            prompt={"gate": "REQUIREMENTS", "question": "approve?"},
        )
        req = urllib.request.Request(
            base_url + "/sessions/sess-ajax/hitl/pending.html",
            headers={"Cookie": f"csrf_token={csrf}"},
        )
        resp = urllib.request.urlopen(req, timeout=4.0)
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/html")
        body = resp.read().decode("utf-8")
        assert "hitl-alert" in body
        assert "req-ajax" in body
        assert "data-ajax='hitl-answer'" in body
    finally:
        handle.shutdown()


def test_hitl_pending_html_returns_empty_when_nothing_pending(tmp_path):
    """When no HITL is queued, the fragment endpoint returns an empty
    body — JS uses that to clear the slot without a special-case."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        resp = urllib.request.urlopen(
            base_url + "/sessions/sess-empty/hitl/pending.html",
            timeout=4.0,
        )
        assert resp.status == 200
        assert resp.read().decode("utf-8") == ""
    finally:
        handle.shutdown()


def _fetch_hitl_card(base_url: str, csrf: str, session_id: str) -> str:
    req = urllib.request.Request(
        base_url + f"/sessions/{session_id}/hitl/pending.html",
        headers={"Cookie": f"csrf_token={csrf}"},
    )
    return urllib.request.urlopen(req, timeout=4.0).read().decode("utf-8")


def test_hitl_render_prompt_kind_uses_labeled_dropdown(tmp_path):
    """A ``type: prompt`` HITL with ``option_labels`` must render as a
    select-dropdown whose option text spells out what each answer
    means — not a JSON dump with a single-letter free-text input.

    This is the core operator-facing fix: the question reads as prose
    and the dropdown labels remove the need to memorise cli.py's menu
    letters."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    assert csrf
    try:
        get_hitl_queue().register_pending(
            request_id="req-prompt",
            session_id="sess-typed",
            prompt={
                "type": "prompt",
                "message": "Select action",
                "options": ["v", "r", "q"],
                "default": "r",
                "option_labels": {
                    "v": "View active file diffs",
                    "r": "Resume graph execution",
                    "q": "Abandon session",
                },
            },
        )
        body = _fetch_hitl_card(base_url, csrf, "sess-typed")
        assert "<select name='choice'" in body
        # No free-text choice input on typed prompts.
        assert "<input type='text' name='choice'" not in body
        # Question rendered as prose (not raw JSON).
        assert "Select action" in body
        # Each option carries its human-readable label, prefixed with
        # the key so operators who know the cli shortcut still see it.
        assert "[v] View active file diffs" in body
        assert "[r] Resume graph execution" in body
        assert "[q] Abandon session" in body
        # `default: r` must be preselected.
        assert "value='r' selected" in body
    finally:
        handle.shutdown()


def test_hitl_render_prompt_without_labels_still_shows_dropdown(tmp_path):
    """When the harness sends a ``type: prompt`` payload but omits
    ``option_labels`` (legacy clients), the dashboard still renders a
    dropdown — option text falls back to the key alone — instead of
    dumping JSON."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        get_hitl_queue().register_pending(
            request_id="req-bare",
            session_id="sess-bare",
            prompt={
                "type": "prompt",
                "message": "Pick one",
                "options": ["a", "b"],
                "default": "a",
            },
        )
        body = _fetch_hitl_card(base_url, csrf, "sess-bare")
        assert "<select name='choice'" in body
        assert "[a] a" in body and "[b] b" in body
    finally:
        handle.shutdown()


def test_hitl_render_confirm_kind_renders_yes_no(tmp_path):
    """``type: confirm`` HITL (the abandon-confirmation follow-up
    when an operator picks [q]) must render as a Yes/No dropdown with
    the default preselected — not a JSON dump."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        get_hitl_queue().register_pending(
            request_id="req-confirm",
            session_id="sess-confirm",
            prompt={
                "type": "confirm",
                "message": "Confirm abandon?",
                "options": [],
                "default": "n",
            },
        )
        body = _fetch_hitl_card(base_url, csrf, "sess-confirm")
        assert "Confirm abandon?" in body
        assert "<select name='choice'" in body
        assert "value='y'>Yes" in body
        assert "value='n' selected>No" in body
    finally:
        handle.shutdown()


def test_hitl_render_notes_kind_renders_required_textarea(tmp_path):
    """``type: notes`` HITL (free-text hint injection) must render the
    textarea as the required answer field, with a hidden ``choice``
    placeholder so the existing answer handler still accepts the
    submission."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        get_hitl_queue().register_pending(
            request_id="req-notes",
            session_id="sess-notes",
            prompt={
                "type": "notes",
                "message": "Enter repair hint",
                "options": [],
                "default": "",
            },
        )
        body = _fetch_hitl_card(base_url, csrf, "sess-notes")
        assert "Enter repair hint" in body
        assert "<input type='hidden' name='choice' value='ok'>" in body
        assert "<textarea name='extra_notes'" in body and "required" in body
        assert "Submit notes" in body
    finally:
        handle.shutdown()


def test_hitl_render_wait_for_edit_renders_done_button(tmp_path):
    """``type: wait_for_edit`` HITL must surface the file path the
    harness is blocked on and a "Done editing" button. The hidden
    ``choice=done`` placeholder satisfies the existing answer handler
    contract — any 200 unblocks the harness."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    csrf = handle.csrf_token
    try:
        get_hitl_queue().register_pending(
            request_id="req-wait",
            session_id="sess-wait",
            prompt={
                "type": "wait_for_edit",
                "message": "/workspace/foo.py",
                "options": [],
                "default": "done",
            },
        )
        body = _fetch_hitl_card(base_url, csrf, "sess-wait")
        assert "/workspace/foo.py" in body
        assert "<input type='hidden' name='choice' value='done'>" in body
        assert "Done editing" in body
    finally:
        handle.shutdown()


def test_notes_html_returns_chat_panel(tmp_path):
    """/sessions/{sid}/notes.html powers the live chat-notes slot
    refresh after a note submit. Must return the queued-notes list +
    the textarea form."""
    cfg = _make_cfg(tmp_path)
    handle, base_url = _start(cfg)
    try:
        resp = urllib.request.urlopen(
            base_url + "/sessions/sess-notes/notes.html",
            timeout=4.0,
        )
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert "Chat notes" in body
        assert "data-ajax='chat-note'" in body
        # The CSRF field has to be present — JS sends X-CSRF-Token but
        # the server also re-validates the form body field defensively.
        assert "name='csrf_token'" in body
    finally:
        handle.shutdown()


def test_http_channel_emits_hitl_pending_and_resolved_events(tmp_path):
    """Phase 2 contract: HttpChannel._post emits ``hitl_pending`` to
    the session JSONL right before the webhook POST, and
    ``hitl_resolved`` immediately after the operator's answer comes
    back. The dashboard's SSE stream relays these so the banner appears
    / disappears in ~100ms — no polling, no page refresh."""
    import logging
    from harness.hitl import HttpChannel

    captured: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record)

    cap = _Cap()
    events_logger = logging.getLogger("harness.events")
    events_logger.addHandler(cap)
    events_logger.setLevel(logging.INFO)
    try:
        channel = HttpChannel("http://127.0.0.1:1/hitl/webhook", timeout=1.0,
                              max_retries=0)

        import urllib.request as _ur

        class _StubResp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def _stub_urlopen(req, timeout=None):
            return _StubResp(json.dumps({"answer": "a"}).encode("utf-8"))

        original = _ur.urlopen
        _ur.urlopen = _stub_urlopen  # type: ignore[assignment]
        try:
            out = channel.prompt("approve?", options=["a", "e"], default="e")
            assert out == "a"
        finally:
            _ur.urlopen = original  # type: ignore[assignment]

        kinds = [getattr(r, "event", None) for r in captured]
        assert "hitl_pending" in kinds
        assert "hitl_resolved" in kinds
        # hitl_pending should fire BEFORE hitl_resolved.
        assert kinds.index("hitl_pending") < kinds.index("hitl_resolved")
    finally:
        events_logger.removeHandler(cap)


# ---------------------------------------------------------------------------
# 9c. spawn_harness_run propagates the operator-wait timeout to the harness
# ---------------------------------------------------------------------------

def test_spawn_harness_run_propagates_hitl_webhook_timeout(tmp_path, monkeypatch):
    """The dashboard's hitl_webhook_timeout_seconds (default 600s) is
    how long the webhook handler will block waiting for an operator
    answer. The harness's HttpChannel default request timeout is 30s,
    so without this propagation the harness aborts ~20x faster than
    the dashboard. Spawn must export ``HARNESS_HITL_WEBHOOK_TIMEOUT``
    set to (dashboard wait + 30s buffer)."""
    import subprocess

    captured: dict = {}

    class _StubPopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            self.pid = -1
            self._argv = argv

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _StubPopen)
    cfg = _make_cfg(tmp_path, hitl_webhook_timeout_seconds=420.0)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    spawn_harness_run(
        cfg, workspace=str(tmp_path), prompt="x",
        harness_binary=sys.executable,
    )
    env = captured["env"]
    assert env is not None
    assert "HARNESS_HITL_WEBHOOK_TIMEOUT" in env
    assert float(env["HARNESS_HITL_WEBHOOK_TIMEOUT"]) == pytest.approx(450.0)


def test_spawn_harness_run_honors_preexisting_timeout_env(tmp_path, monkeypatch):
    """Operators who pin HARNESS_HITL_WEBHOOK_TIMEOUT in the dashboard's
    own environment win — the spawn uses ``env.setdefault`` so that
    explicit override survives."""
    import subprocess

    captured: dict = {}

    class _StubPopen:
        def __init__(self, argv, **kwargs):
            captured["env"] = kwargs.get("env")
            self.pid = -1

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _StubPopen)
    monkeypatch.setenv("HARNESS_HITL_WEBHOOK_TIMEOUT", "77")
    cfg = _make_cfg(tmp_path, hitl_webhook_timeout_seconds=600.0)
    cfg.host = "127.0.0.1"
    cfg.port = _free_port()
    spawn_harness_run(
        cfg, workspace=str(tmp_path), prompt="x",
        harness_binary=sys.executable,
    )
    assert captured["env"]["HARNESS_HITL_WEBHOOK_TIMEOUT"] == "77"


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
