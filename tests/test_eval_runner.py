"""Eval-runner watchdog + CLI exit watchdog.

Both exist because of one incident (eval fix_off_by_one): a fatal
dispatch error left the harness parked in interpreter shutdown joining a
wedged non-daemon worker thread, subprocess.run's timeout never fired,
and the kill-on-timeout wouldn't have reaped MCP-server grandchildren
anyway. The runner now owns its deadline and kills the process group;
the CLI arms a daemon watchdog that forces the decided exit code out.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

import evals.run_eval as re_mod
from evals.run_eval import _run_harness

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_dead(pid: int, timeout: float = 6.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


class TestRunnerWatchdog:
    def test_timeout_kills_whole_process_group(self, monkeypatch, tmp_path):
        # Fake harness: spawns a grandchild, prints its pid, then hangs.
        # The watchdog must return 124 AND take the grandchild down with
        # the process group (the old runner leaked it to init).
        code = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "print(f'GRANDCHILD={p.pid}', flush=True)\n"
            "time.sleep(120)\n"
        )
        monkeypatch.setattr(re_mod, "_HARNESS_CMD_PREFIX", [sys.executable, "-c", code])
        t0 = time.monotonic()
        rc, err, tail = _run_harness(
            workspace=tmp_path, session_id="wd-test", prompt="x",
            new_build=True, timeout_s=3,
        )
        elapsed = time.monotonic() - t0
        assert rc == 124
        assert "timeout" in (err or "")
        assert elapsed < 20  # deadline actually enforced
        gc_pid = int(tail.split("GRANDCHILD=")[1].split()[0])
        assert _wait_dead(gc_pid), "grandchild leaked past the group kill"

    def test_clean_exit_passes_through_code_and_no_tail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            re_mod, "_HARNESS_CMD_PREFIX",
            [sys.executable, "-c", "print('ok')"],
        )
        rc, err, tail = _run_harness(
            workspace=tmp_path, session_id="wd-ok", prompt="x",
            new_build=True, timeout_s=10,
        )
        assert rc == 0 and err is None and tail == ""

    def test_failure_exit_captures_stderr_tail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            re_mod, "_HARNESS_CMD_PREFIX",
            [sys.executable, "-c",
             "import sys; print('the actual reason', file=sys.stderr); sys.exit(3)"],
        )
        rc, err, tail = _run_harness(
            workspace=tmp_path, session_id="wd-fail", prompt="x",
            new_build=True, timeout_s=10,
        )
        assert rc == 3 and err is None
        assert "the actual reason" in tail


class TestExitWatchdog:
    def test_forces_exit_past_wedged_nondaemon_thread(self):
        # Without the watchdog this child would hang forever in
        # threading._shutdown joining the non-daemon sleeper. With it,
        # the decided exit code comes out within the grace window.
        script = (
            "import sys, threading, time\n"
            f"sys.path.insert(0, {REPO_ROOT!r})\n"
            "from harness.cli import _arm_exit_watchdog\n"
            "threading.Thread(target=lambda: time.sleep(300)).start()\n"
            "_arm_exit_watchdog(7, grace_seconds=2.0)\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 7
        assert time.monotonic() - t0 < 25
        assert "forcing exit(7)" in proc.stderr

    def test_noop_under_pytest_env(self):
        # Inside a test runner the watchdog must never arm — an armed
        # os._exit would kill the pytest process itself.
        from harness.cli import _arm_exit_watchdog
        import threading
        before = {t.name for t in threading.enumerate()}
        _arm_exit_watchdog(0, grace_seconds=0.1)
        after = {t.name for t in threading.enumerate()}
        assert "exit-watchdog" not in (after - before)
