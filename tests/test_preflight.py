"""Tests for harness/preflight.py.

Each probe is exercised in both a PASS state and a WARN/FAIL state by
monkeypatching the underlying primitive (shutil.which, subprocess.run,
socket.create_connection, os.access, the OS-detection helpers in
harness._platform). The cross-OS dispatch tests verify that run_all()
asks the right set of questions per platform.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from types import SimpleNamespace

from harness import preflight
from harness.preflight import (
    CheckResult,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
)


# ---------------------------------------------------------------------------
# Per-probe tests
# ---------------------------------------------------------------------------

class TestProbePython:
    def test_pass_on_3_11(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
        result = preflight.probe_python()
        assert result.status == STATUS_PASS
        assert "3.11.0" in result.detail

    def test_fail_on_3_10(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 5, "final", 0))
        result = preflight.probe_python()
        assert result.status == STATUS_FAIL
        assert "3.10.5" in result.detail
        assert result.install_cmd  # has install hint


class TestProbeGit:
    def test_fail_when_not_on_path(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
        result = preflight.probe_git()
        assert result.status == STATUS_FAIL
        assert "not on PATH" in result.detail

    def test_pass_when_on_path(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/git")

        def _fake_run(*a, **kw):
            return SimpleNamespace(returncode=0, stdout="git version 2.43.0\n", stderr="")
        monkeypatch.setattr(preflight.subprocess, "run", _fake_run)

        result = preflight.probe_git()
        assert result.status == STATUS_PASS
        assert "2.43" in result.detail


class TestProbeHomeWritable:
    def test_pass_when_home_writable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(preflight.os.path, "expanduser", lambda p: str(tmp_path))
        result = preflight.probe_home_writable()
        assert result.status == STATUS_PASS
        assert str(tmp_path) in result.detail

    def test_fail_when_home_unresolved(self, monkeypatch):
        monkeypatch.setattr(preflight.os.path, "expanduser", lambda p: "~")
        result = preflight.probe_home_writable()
        assert result.status == STATUS_FAIL


class TestProbeTempWritable:
    def test_pass_on_writable_tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            preflight._platform, "harness_temp_dir", lambda subdir="": str(tmp_path),
        )
        result = preflight.probe_temp_writable()
        assert result.status == STATUS_PASS

    def test_fail_on_nonexistent_temp(self, monkeypatch):
        monkeypatch.setattr(
            preflight._platform, "harness_temp_dir",
            lambda subdir="": "/nonexistent/path/that/cannot/exist",
        )
        result = preflight.probe_temp_writable()
        assert result.status == STATUS_FAIL


class TestProbeOutboundHTTPS:
    def test_pass_when_reachable(self, monkeypatch):
        # create_connection returns a real socket; emulate with a context-manager stub.
        class _StubSock:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(preflight.socket, "create_connection", lambda *a, **kw: _StubSock())
        result = preflight.probe_outbound_https("example.com")
        assert result.status == STATUS_PASS
        assert "reachable" in result.detail

    def test_warn_on_dns_failure(self, monkeypatch):
        def _boom(*a, **kw):
            raise socket.gaierror("DNS failure simulation")
        monkeypatch.setattr(preflight.socket, "create_connection", _boom)
        result = preflight.probe_outbound_https("example.com")
        assert result.status == STATUS_WARN


class TestProbeDocker:
    def test_warn_when_docker_missing(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
        result = preflight.probe_docker()
        assert result.status == STATUS_WARN
        assert result.install_cmd  # has install hint

    def test_pass_when_linux_containers(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            preflight.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=0, stdout="Ubuntu 22.04;linux\n", stderr="",
            ),
        )
        result = preflight.probe_docker()
        assert result.status == STATUS_PASS
        assert "linux" in result.detail.lower()

    def test_warn_when_windows_containers(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            preflight.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=0, stdout=";windows\n", stderr="",
            ),
        )
        result = preflight.probe_docker()
        assert result.status == STATUS_WARN
        assert "Linux containers" in result.detail


class TestProbeUnshare:
    def test_skip_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_linux", lambda: False)
        result = preflight.probe_unshare()
        assert result.status == STATUS_SKIP

    def test_pass_on_linux_with_namespaces(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_linux", lambda: True)
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/unshare")
        monkeypatch.setattr(
            preflight.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="ok\n", stderr=""),
        )
        result = preflight.probe_unshare()
        assert result.status == STATUS_PASS

    def test_warn_on_linux_without_userns(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_linux", lambda: True)
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/unshare")
        monkeypatch.setattr(
            preflight.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stdout="", stderr="Operation not permitted",
            ),
        )
        result = preflight.probe_unshare()
        assert result.status == STATUS_WARN


class TestProbeTaskkill:
    def test_skip_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: False)
        result = preflight.probe_taskkill()
        assert result.status == STATUS_SKIP

    def test_pass_on_windows_with_taskkill(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: True)
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "C:\\Windows\\System32\\taskkill.exe")
        result = preflight.probe_taskkill()
        assert result.status == STATUS_PASS

    def test_fail_on_windows_without_taskkill(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: True)
        monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
        result = preflight.probe_taskkill()
        assert result.status == STATUS_FAIL


class TestProbePosixSh:
    def test_skip_on_posix(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: False)
        result = preflight.probe_posix_sh()
        assert result.status == STATUS_SKIP

    def test_pass_on_windows_with_git_bash(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: True)
        monkeypatch.setattr(
            preflight._platform, "posix_shell_path",
            lambda: "C:\\Program Files\\Git\\usr\\bin\\sh.exe",
        )
        result = preflight.probe_posix_sh()
        assert result.status == STATUS_PASS

    def test_warn_on_windows_without_git_bash(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: True)
        monkeypatch.setattr(preflight._platform, "posix_shell_path", lambda: None)
        result = preflight.probe_posix_sh()
        assert result.status == STATUS_WARN
        assert result.install_cmd


class TestProbeLongPaths:
    def test_skip_on_posix(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_windows", lambda: False)
        result = preflight.probe_long_paths()
        assert result.status == STATUS_SKIP


class TestProbeXcodeCLI:
    def test_skip_on_non_macos(self, monkeypatch):
        monkeypatch.setattr(preflight._platform, "is_macos", lambda: False)
        result = preflight.probe_xcode_cli()
        assert result.status == STATUS_SKIP


class TestProbeOptionalBinary:
    def test_warn_when_missing(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
        result = preflight._probe_optional_binary("ruff", "Python formatter", "RECOMMENDED")
        assert result.status == STATUS_WARN
        assert "ruff" in result.detail or "ruff" in result.name

    def test_pass_when_present(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
        result = preflight._probe_optional_binary("ruff", "Python formatter", "RECOMMENDED")
        assert result.status == STATUS_PASS


class TestProbeLLMKeys:
    def test_env_keys_reported(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-xxx")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        results = preflight.probe_llm_api_keys()
        statuses = {r.name: r.status for r in results}
        assert statuses["ANTHROPIC_API_KEY"] == STATUS_PASS
        assert statuses["OPENAI_API_KEY"] == STATUS_SKIP


# ---------------------------------------------------------------------------
# Install-recipe map
# ---------------------------------------------------------------------------

class TestInstallRecipes:
    def test_git_recipe_per_os(self):
        assert "apt" in preflight._install_recipe("git", "linux")
        assert "brew" in preflight._install_recipe("git", "macos")
        assert "winget" in preflight._install_recipe("git", "windows")

    def test_unknown_tool_returns_empty(self):
        assert preflight._install_recipe("not-a-real-tool", "linux") == ""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestRunAllDispatch:
    def test_windows_override_includes_long_paths_probe(self, monkeypatch):
        # Stub the noisy probes so the test stays fast.
        monkeypatch.setattr(preflight, "probe_outbound_https",
                            lambda *a, **kw: CheckResult(name="x", status=STATUS_SKIP, detail=""))
        monkeypatch.setattr(preflight, "probe_docker",
                            lambda: CheckResult(name="Docker daemon", status=STATUS_SKIP, detail=""))
        results = preflight.run_all(platform_override="windows", quick=True)
        names = {r.name for r in results}
        assert "long paths" in names
        assert "taskkill" in names
        # unshare is Linux-only and should NOT appear in the Windows set.
        assert "unshare" not in names

    def test_linux_override_includes_unshare(self, monkeypatch):
        monkeypatch.setattr(preflight, "probe_outbound_https",
                            lambda *a, **kw: CheckResult(name="x", status=STATUS_SKIP, detail=""))
        monkeypatch.setattr(preflight, "probe_docker",
                            lambda: CheckResult(name="Docker daemon", status=STATUS_SKIP, detail=""))
        results = preflight.run_all(platform_override="linux", quick=True)
        names = {r.name for r in results}
        assert "unshare" in names
        assert "long paths" not in names
        assert "taskkill" not in names

    def test_macos_override_includes_xcode(self, monkeypatch):
        monkeypatch.setattr(preflight, "probe_outbound_https",
                            lambda *a, **kw: CheckResult(name="x", status=STATUS_SKIP, detail=""))
        monkeypatch.setattr(preflight, "probe_docker",
                            lambda: CheckResult(name="Docker daemon", status=STATUS_SKIP, detail=""))
        results = preflight.run_all(platform_override="macos", quick=True)
        names = {r.name for r in results}
        assert "Xcode CLI tools" in names
        assert "unshare" not in names
        assert "long paths" not in names

    def test_quick_skips_outbound_probe(self, monkeypatch):
        results = preflight.run_all(quick=True)
        outbound = next((r for r in results if r.name == "outbound HTTPS"), None)
        assert outbound is not None
        assert outbound.status == STATUS_SKIP

    def test_override_restored_after_run(self, monkeypatch):
        # is_linux is True on this Linux dev host. After run_all returns,
        # the predicates must be back to their real value.
        original = preflight._platform.is_linux()
        preflight.run_all(platform_override="windows", quick=True)
        assert preflight._platform.is_linux() == original


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class TestRenderTTY:
    def test_header_uses_override_platform(self):
        results = [CheckResult(name="x", status=STATUS_PASS, detail="ok")]
        out = preflight.render_tty(results, no_color=True, platform_name="windows")
        assert "windows" in out.splitlines()[1].lower()

    def test_pass_warn_fail_skip_markers(self):
        results = [
            CheckResult(name="a", status=STATUS_PASS, detail="ok"),
            CheckResult(name="b", status=STATUS_WARN, detail="meh", install_cmd="pip install b"),
            CheckResult(name="c", status=STATUS_FAIL, detail="bad", install_cmd="apt install c"),
            CheckResult(name="d", status=STATUS_SKIP, detail="n/a"),
        ]
        out = preflight.render_tty(results, no_color=True)
        # Install commands shown for WARN and FAIL but not for PASS/SKIP.
        assert "pip install b" in out
        assert "apt install c" in out
        assert "ok" in out

    def test_summary_footer_counts(self):
        results = [
            CheckResult(name="a", status=STATUS_PASS, detail=""),
            CheckResult(name="b", status=STATUS_PASS, detail=""),
            CheckResult(name="c", status=STATUS_WARN, detail=""),
            CheckResult(name="d", status=STATUS_FAIL, detail=""),
        ]
        out = preflight.render_tty(results, no_color=True)
        assert "2 ✓" in out
        assert "1 ⚠" in out
        assert "1 ✗" in out


class TestRenderJSON:
    def test_shape(self):
        results = [
            CheckResult(name="a", status=STATUS_PASS, detail="ok"),
            CheckResult(name="b", status=STATUS_FAIL, detail="bad", install_cmd="x"),
        ]
        out = preflight.render_json(results, platform_name="linux")
        payload = json.loads(out)
        assert payload["platform"] == "linux"
        assert len(payload["results"]) == 2
        assert payload["summary"]["pass"] == 1
        assert payload["summary"]["fail"] == 1
        # FAIL row sets exit_code=1.
        assert payload["summary"]["exit_code"] == 1

    def test_exit_code_0_when_no_fails(self):
        results = [
            CheckResult(name="a", status=STATUS_PASS, detail=""),
            CheckResult(name="b", status=STATUS_WARN, detail=""),
        ]
        payload = json.loads(preflight.render_json(results))
        assert payload["summary"]["exit_code"] == 0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_pre_flight_cli_json_quick(self):
        """`python -m harness.cli pre-flight --json-dump true --quick` produces parseable JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "harness.cli", "pre-flight", "--json-dump", "true", "--quick"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        # Exit code is 0 or 1 depending on whether any FAIL surfaced; the
        # critical assertion is that the output is well-formed JSON.
        assert result.returncode in (0, 1), (
            f"unexpected exit {result.returncode}: {result.stderr}"
        )
        payload = json.loads(result.stdout)
        assert "platform" in payload
        assert "results" in payload
        assert "summary" in payload
        assert payload["summary"]["exit_code"] == result.returncode

    def test_pre_flight_cli_tty_runs(self):
        """`python -m harness.cli pre-flight --no-color --quick` returns 0 or 1
        and the output contains the header."""
        result = subprocess.run(
            [sys.executable, "-m", "harness.cli", "pre-flight", "--no-color", "--quick"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        assert result.returncode in (0, 1)
        assert "teane pre-flight" in result.stdout
        assert "REQUIRED" in result.stdout
