"""P1.7 regression: workspace-level advisory lock prevents two concurrent
`teane run` invocations against the same workspace from clobbering each
other's patches.

The lock is process-scoped via fcntl. We exercise it across two
subprocesses because two flock acquisitions inside the same process from
the same fd succeed (the second one is a no-op on the same lock owner),
which would hide the regression.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(autouse=True)
def _release_module_lock():
    """Some tests above might have left a lock on the module-level slot
    (e.g. when running the suite in the same interpreter). Clear it so
    each test starts fresh. The lock is keyed per-workspace path now
    (see audit §1.9) so we drop every entry rather than overwriting a
    single singleton."""
    import harness.cli as cli_mod
    if hasattr(cli_mod, "_WORKSPACE_LOCK_HANDLES"):
        cli_mod._WORKSPACE_LOCK_HANDLES.clear()
    yield
    if hasattr(cli_mod, "_WORKSPACE_LOCK_HANDLES"):
        cli_mod._WORKSPACE_LOCK_HANDLES.clear()


def _spawn_lock_holder(workspace: str, sleep_seconds: float = 3.0) -> subprocess.Popen:
    """Spawn a tiny Python subprocess that acquires the lock and then sleeps.

    Returns the Popen handle; the caller is responsible for killing it.
    """
    script = (
        "import sys, time, os;"
        "sys.path.insert(0, os.environ['HARNESS_REPO']);"
        "from harness.cli import _acquire_workspace_lock;"
        f"h = _acquire_workspace_lock({workspace!r}, force=False);"
        "assert h not in (False, None), 'first holder should succeed';"
        "print('LOCKED', flush=True);"
        f"time.sleep({sleep_seconds})"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "HARNESS_REPO": repo},
        text=True,
    )
    # Wait for the holder to confirm it acquired the lock.
    for _ in range(30):
        line = (proc.stdout.readline() or "").strip()
        if line == "LOCKED":
            return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("lock holder subprocess never reported LOCKED")


def test_second_session_refused_without_force(tmp_path):
    """A second acquire on a workspace already held by another process
    must return False (operator-facing 'lock held' refusal)."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")

    workspace = str(tmp_path)
    holder = _spawn_lock_holder(workspace, sleep_seconds=3.0)
    try:
        from harness.cli import _acquire_workspace_lock
        result = _acquire_workspace_lock(workspace, force=False)
        assert result is False, "second session must be refused when lock held"
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_force_lock_succeeds_even_when_held(tmp_path):
    """--force-lock takes the lock anyway. Operator owns the resulting
    risk; this exists for the recovery case where a crash stranded the
    lock. We don't test the recovery path itself here — just that the
    bypass actually works."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")

    workspace = str(tmp_path)
    holder = _spawn_lock_holder(workspace, sleep_seconds=2.0)
    try:
        # The holder will release in ~2s; force_lock waits for the lock and
        # should succeed once the holder exits, but we kill the holder so
        # we're not racing the timing.
        time.sleep(0.2)
        holder.kill()
        holder.wait(timeout=5)
        from harness.cli import _acquire_workspace_lock
        result = _acquire_workspace_lock(workspace, force=True)
        assert result is not False
        assert result is not None
    finally:
        if holder.poll() is None:
            holder.kill()


def test_unlocked_workspace_lock_succeeds(tmp_path):
    """Sanity: no holder → first acquire returns the handle."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")

    workspace = str(tmp_path)
    from harness.cli import _acquire_workspace_lock
    result = _acquire_workspace_lock(workspace, force=False)
    assert result not in (False, None)
    # Lock file actually exists.
    assert os.path.isfile(os.path.join(workspace, ".harness_session.lock"))


# ---------------------------------------------------------------------------
# Audit §1.9 — truncate-AFTER-flock + per-workspace handle dict
# ---------------------------------------------------------------------------


def test_pid_written_after_lock_acquired(tmp_path):
    """The lock file should carry the holder's pid, written AFTER flock —
    the earlier 'open mode=w' truncated BEFORE flock, so concurrent
    acquirers could wipe the holder's diagnostic line."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")
    import harness.cli as cli_mod
    workspace = str(tmp_path)
    fh = cli_mod._acquire_workspace_lock(workspace, force=False)
    assert fh not in (False, None)
    contents = open(os.path.join(workspace, ".harness_session.lock"), encoding="utf-8").read()
    assert f"pid={os.getpid()}" in contents


def test_two_distinct_workspaces_share_module_slot_dict(tmp_path, monkeypatch):
    """The handle dict is keyed by workspace realpath so acquiring two
    different workspaces in the same process doesn't overwrite the first
    lock's handle (which would let GC release the fcntl lock)."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")
    import harness.cli as cli_mod
    ws_a = tmp_path / "wa"
    ws_b = tmp_path / "wb"
    ws_a.mkdir()
    ws_b.mkdir()
    fh_a = cli_mod._acquire_workspace_lock(str(ws_a), force=False)
    fh_b = cli_mod._acquire_workspace_lock(str(ws_b), force=False)
    assert fh_a not in (False, None)
    assert fh_b not in (False, None)
    # Per-path dict: both handles must survive in the module slot.
    assert os.path.realpath(str(ws_a)) in cli_mod._WORKSPACE_LOCK_HANDLES
    assert os.path.realpath(str(ws_b)) in cli_mod._WORKSPACE_LOCK_HANDLES


def test_open_does_not_truncate_lock_file_before_flock(tmp_path):
    """Regression: pre-write a sentinel line into the lock file and verify
    a successful acquire still ends with our pid line (the pid is written
    AFTER flock; the seek+truncate explicitly happens under the lock)."""
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")
    import harness.cli as cli_mod
    workspace = str(tmp_path)
    lock_path = os.path.join(workspace, ".harness_session.lock")
    open(lock_path, "w", encoding="utf-8").write("sentinel-before-acquire\n")
    fh = cli_mod._acquire_workspace_lock(workspace, force=False)
    assert fh not in (False, None)
    contents = open(lock_path, encoding="utf-8").read()
    # Sentinel is gone (we truncated AFTER flock); our pid line is present.
    assert "sentinel-before-acquire" not in contents
    assert f"pid={os.getpid()}" in contents


# ---------------------------------------------------------------------------
# Stranded-pid hardening: signal-driven cleanup (#1) + liveness-validated
# read (#2). See harness.cli._install_lock_signal_handlers /
# _read_workspace_lock_holder.
# ---------------------------------------------------------------------------
import signal as _signal  # noqa: E402


@pytest.fixture
def _restore_lock_signals():
    """Save/restore terminating-signal handlers + the install-once flag so a
    test that installs the lock signal handlers doesn't leak into the runner."""
    import harness.cli as cli_mod
    saved = {}
    for _name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        _s = getattr(_signal, _name, None)
        if _s is not None:
            saved[_s] = _signal.getsignal(_s)
    was = cli_mod._LOCK_SIGNAL_HANDLERS_INSTALLED
    try:
        yield
    finally:
        for _s, _h in saved.items():
            try:
                _signal.signal(_s, _h)
            except (ValueError, OSError):
                pass
        cli_mod._LOCK_SIGNAL_HANDLERS_INSTALLED = was


def _dead_pid() -> int:
    """A PID guaranteed dead: spawn a trivial child and reap it."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _write_lock(ws: str, content: str) -> None:
    with open(os.path.join(ws, ".harness_session.lock"), "w", encoding="utf-8") as f:
        f.write(content)


# ---- #2: liveness-validated holder read ----
def test_holder_alive_returns_pid(tmp_path):
    import harness.cli as cli_mod
    _write_lock(str(tmp_path), f"pid={os.getpid()}\n")
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) == os.getpid()


def test_holder_dead_pid_returns_none(tmp_path):
    import harness.cli as cli_mod
    _write_lock(str(tmp_path), f"pid={_dead_pid()}\n")
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) is None


def test_holder_missing_file_returns_none(tmp_path):
    import harness.cli as cli_mod
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) is None


def test_holder_empty_and_malformed_return_none(tmp_path):
    import harness.cli as cli_mod
    _write_lock(str(tmp_path), "")
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) is None
    _write_lock(str(tmp_path), "garbage, no pid here\n")
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) is None


def test_holder_nonpositive_pid_returns_none(tmp_path):
    import harness.cli as cli_mod
    _write_lock(str(tmp_path), "pid=0\n")
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) is None


# ---- #1: signal-driven cleanup ----
def test_install_lock_signal_handlers_idempotent(_restore_lock_signals):
    import harness.cli as cli_mod
    cli_mod._LOCK_SIGNAL_HANDLERS_INSTALLED = False
    cli_mod._install_lock_signal_handlers()
    assert cli_mod._LOCK_SIGNAL_HANDLERS_INSTALLED is True
    assert callable(_signal.getsignal(_signal.SIGTERM))
    cli_mod._install_lock_signal_handlers()  # no-op, must not raise
    assert cli_mod._LOCK_SIGNAL_HANDLERS_INSTALLED is True


def test_cleanup_truncates_held_lock(tmp_path, _restore_lock_signals):
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")
    import harness.cli as cli_mod
    handle = cli_mod._acquire_workspace_lock(str(tmp_path), force=False)
    assert handle not in (False, None)
    lock_path = os.path.join(str(tmp_path), ".harness_session.lock")
    assert f"pid={os.getpid()}" in open(lock_path, encoding="utf-8").read()
    cli_mod._clear_workspace_lock_pids_atexit()   # the signal/atexit routine
    assert open(lock_path, encoding="utf-8").read() == ""


def test_acquire_overwrites_stale_dead_pid(tmp_path, _restore_lock_signals):
    if sys.platform == "win32":
        pytest.skip("fcntl unavailable on Windows native")
    import harness.cli as cli_mod
    _write_lock(str(tmp_path), f"pid={_dead_pid()}\n")   # phantom from a crash
    handle = cli_mod._acquire_workspace_lock(str(tmp_path), force=False)
    assert handle not in (False, None)
    # flock was free → acquire succeeds and rewrites the pid; no phantom left.
    assert cli_mod._read_workspace_lock_holder(str(tmp_path)) == os.getpid()
