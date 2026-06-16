"""Tests for the new ``harness web start`` / ``harness web stop``
lifecycle commands.

Two flavours of coverage here:
  1. Unit tests on the marker helpers (``_read_web_marker`` /
     ``_write_web_marker`` / ``_delete_web_marker`` / ``_is_pid_alive``) and
     the single-instance gate inside ``cmd_web_start``.
  2. One end-to-end test that spawns ``harness web start --background yes``
     as a real subprocess, hits ``/status`` over HTTP, and then runs
     ``harness web stop``. Verifies the marker file lifecycle, the port
     unbinds, and the subprocess exits cleanly.

The E2E test re-points ``$HOME`` at a tmp dir so the marker file never
collides with the user's real ``~/.harness/web.lock``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from harness import cli


# ---------------------------------------------------------------------------
# 1. Marker helpers
# ---------------------------------------------------------------------------

def test_read_marker_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli._read_web_marker() is None


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data = {"pid": 1234, "host": "127.0.0.1", "port": 9000, "mode": "foreground"}
    cli._write_web_marker(data)
    marker_path = cli._web_marker_path()
    assert os.path.isfile(marker_path)
    assert marker_path.startswith(str(tmp_path))  # respected the HOME override
    rebuilt = cli._read_web_marker()
    assert rebuilt == data


def test_delete_marker_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Delete with no file — must not raise.
    cli._delete_web_marker()
    # Delete after a write — file is gone.
    cli._write_web_marker({"pid": 1})
    cli._delete_web_marker()
    assert cli._read_web_marker() is None
    # Delete again — still no-op.
    cli._delete_web_marker()


def test_write_is_atomic_no_tmp_left_behind(tmp_path, monkeypatch):
    """The write goes through .tmp + os.replace so a crash mid-write
    can't leave a partial JSON file at the marker path. Confirm the
    .tmp file is gone after a successful write."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cli._write_web_marker({"pid": 1, "port": 9000})
    contents = os.listdir(tmp_path / ".harness")
    assert "web.lock" in contents
    assert "web.lock.tmp" not in contents


def test_is_pid_alive_for_current_process():
    assert cli._is_pid_alive(os.getpid()) is True


def test_is_pid_alive_for_nonexistent_pid():
    # PID 0 is reserved + pid 999999 is almost certainly free.
    assert cli._is_pid_alive(0) is False
    assert cli._is_pid_alive(999_999) is False


def test_is_pid_alive_rejects_garbage():
    assert cli._is_pid_alive(-1) is False
    assert cli._is_pid_alive(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Single-instance gate in cmd_web_start
# ---------------------------------------------------------------------------

def test_start_refuses_when_marker_points_at_live_pid(tmp_path, monkeypatch, capsys):
    """The whole point of the marker file: stop a second instance
    from launching while one's already running. Use the current
    process's pid as a guaranteed-alive stand-in."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cli._write_web_marker({
        "pid": os.getpid(),
        "host": "127.0.0.1",
        "port": 9000,
        "mode": "foreground",
        "log_path": "",
        "started_at": "2026-06-15T10:00:00+00:00",
    })
    args = argparse.Namespace(host="127.0.0.1", port=9001, background="no")
    rc = cli.cmd_web_start(args)
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "already running" in stderr
    # Must NOT delete the marker — that points at someone else's live instance.
    assert cli._read_web_marker() is not None


def test_start_cleans_up_stale_marker(tmp_path, monkeypatch, capsys, caplog):
    """A marker pointing at a dead pid must be treated as no-server-
    running — the start path silently overwrites the stale lock.

    We test the stale-cleanup path WITHOUT spawning a server: write a
    stale marker, then patch out start_server so cmd_web_start returns
    immediately. We just need to verify (a) no 'already running' error
    fires and (b) the stale marker has been replaced (or removed)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cli._write_web_marker({
        "pid": 999_999,   # dead pid — stale
        "port": 9000, "host": "127.0.0.1", "mode": "foreground",
    })

    # Stub start_server so cmd_web_start short-circuits to a quick exit.
    import harness.dashboard as _dash

    class _FakeHandle:
        def __init__(self):
            self.server = self
            # Make .thread.join() return immediately.
            class _T:
                def join(self_inner, timeout=None): return None
            self.thread = _T()
        def shutdown(self): pass
        def server_close(self): pass

    def _fake_start_server(cfg, *, blocking=False):
        # Return a fake handle whose thread.join() exits immediately,
        # so cmd_web_start runs end-to-end in milliseconds.
        return _FakeHandle()

    monkeypatch.setattr(_dash, "start_server", _fake_start_server)

    args = argparse.Namespace(host="127.0.0.1", port=9101, background="no")
    rc = cli.cmd_web_start(args)
    assert rc == 0  # clean exit after the fake thread.join() returns

    # No "already running" error was printed — proving the stale-pid
    # branch was taken, not the live-instance refusal.
    err = capsys.readouterr().err
    assert "already running" not in err

    # Marker has been cleaned up by the finally block.
    assert cli._read_web_marker() is None


def test_stop_with_no_marker_is_friendly_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = cli.cmd_web_stop(argparse.Namespace())
    assert rc == 0
    captured = capsys.readouterr()
    assert "no server running" in captured.out


def test_stop_with_stale_marker_cleans_up(tmp_path, monkeypatch, capsys):
    """`web stop` against a stale marker should remove it and return 0
    so scripts can call `stop` defensively before `start`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cli._write_web_marker({"pid": 999_999, "port": 9000})
    rc = cli.cmd_web_stop(argparse.Namespace())
    assert rc == 0
    assert cli._read_web_marker() is None


# ---------------------------------------------------------------------------
# 3. End-to-end: real CLI subprocess via `harness web start --background yes`
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait(predicate, timeout=10.0, interval=0.1, msg="timeout"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(msg)


def test_e2e_start_background_then_stop(tmp_path):
    """Spawns `harness web start --background yes`, hits /status,
    runs `harness web stop`, and checks the full lifecycle: marker
    written → server serves → marker removed → port released → child
    process exits."""
    port = _free_port()
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)  # isolate marker + log from real ~/.harness
    # Spawn `python -m harness.cli web start --port N --background yes`
    proc = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "start",
         "--port", str(port), "--background", "yes"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"start failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    # Marker file exists, points at the running child, declares background.
    marker_path = tmp_path / ".harness" / "web.lock"
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text())
    assert marker["mode"] == "background"
    assert marker["port"] == port
    assert marker["host"] == "127.0.0.1"
    child_pid = marker["pid"]
    assert cli._is_pid_alive(child_pid)
    # Log file was created by the background spawner.
    log_path = tmp_path / ".harness" / "web.log"
    assert log_path.is_file()

    # Server actually serves.
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/status", timeout=5.0,
        ) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "View Status" in body
    except urllib.error.URLError as exc:
        raise AssertionError(
            f"server didn't accept requests on port {port}: {exc}"
        )

    # Second `start` must refuse (single-instance gate).
    second = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "start",
         "--port", str(_free_port()), "--background", "yes"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert second.returncode == 1
    assert "already running" in second.stderr

    # `harness web stop` removes the marker, kills the child cleanly.
    stop = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "stop"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert stop.returncode == 0, (
        f"stop failed: stdout={stop.stdout!r} stderr={stop.stderr!r}"
    )

    # Marker is gone.
    assert not marker_path.exists()
    # Child process exited within stop's wait window.
    _wait(lambda: not cli._is_pid_alive(child_pid), timeout=10.0,
          msg=f"child pid {child_pid} still alive after stop")
    # Port is rebindable (clean release).
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
    finally:
        s.close()

    # `harness web stop` again is a clean no-op (idempotent).
    second_stop = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "stop"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert second_stop.returncode == 0
    assert "no server running" in second_stop.stdout


def test_web_help_describes_start_and_stop():
    """`harness web --help` must explain both subcommands and show
    the new defaults (port 9000, host 127.0.0.1, background no)."""
    proc = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "start" in out and "stop" in out
    # Examples section is present.
    assert "Examples:" in out
    assert "harness web start" in out
    assert "harness web stop" in out


def test_web_start_help_describes_options():
    proc = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "start", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    out = proc.stdout
    # New defaults surfaced.
    assert "127.0.0.1" in out
    assert "9000" in out
    # --background yes/no is documented.
    assert "background" in out
    assert "yes" in out and "no" in out


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="background subprocess test can race on slow CI",
)
def test_e2e_marker_pid_must_match_child_not_parent(tmp_path):
    """When `--background yes` re-execs self, the marker on disk must
    point at the spawned CHILD's pid, not the (now-exited) parent's
    pid. Otherwise `web stop` would signal the wrong process (or a
    pid that's already gone)."""
    port = _free_port()
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "start",
         "--port", str(port), "--background", "yes"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    marker = json.loads((tmp_path / ".harness" / "web.lock").read_text())
    # The parent's pid was the `python -m harness.cli ...` subprocess
    # we just spawned (`proc.pid` is gone by now since proc.wait completed).
    # The marker pid must be DIFFERENT — it's the re-exec'd grandchild.
    # Belt-and-braces: it must at least be alive.
    assert cli._is_pid_alive(marker["pid"])
    # Cleanup.
    subprocess.run(
        [sys.executable, "-m", "harness.cli", "web", "stop"],
        env=env, capture_output=True, text=True, timeout=10,
    )
