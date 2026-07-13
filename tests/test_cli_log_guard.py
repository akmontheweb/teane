"""Tests for the --log wipe-target guard.

Finsearch session 156032347 (2026-07-13): the operator invoked `teane
build` with stdout shell-redirected to `logs/build.log` INSIDE the
workspace. Build's first step wipes the workspace root (preserving only
product_spec/, .git/, and optionally docs/), so the log file was
deleted before it could capture the run. The operator had to relocate
the log to `/tmp/` and restart.

`_refuse_log_inside_workspace` catches BOTH mistakes at CLI-parse time:
- `--log <path>` where <path> resolves under workspace_path
- shell-redirected stdout whose /proc/self/fd/1 target lives under
  workspace_path (Linux only; other platforms silently skip)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from harness import cli


class TestRefuseLogInsideWorkspace:
    def test_none_returned_when_no_log_and_stdout_is_terminal(self):
        # No --log, no shell redirection → nothing to refuse.
        with tempfile.TemporaryDirectory() as ws:
            err = cli._refuse_log_inside_workspace(
                workspace_path=ws, log_file=None,
            )
            assert err is None

    def test_none_returned_when_log_outside_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            with tempfile.TemporaryDirectory() as elsewhere:
                out_log = os.path.join(elsewhere, "build.log")
                err = cli._refuse_log_inside_workspace(
                    workspace_path=ws, log_file=out_log,
                )
                assert err is None

    def test_refuses_when_log_directly_inside_workspace(self):
        # The exact finsearch shape: --log workspace/logs/build.log.
        with tempfile.TemporaryDirectory() as ws:
            in_log = os.path.join(ws, "logs", "build.log")
            err = cli._refuse_log_inside_workspace(
                workspace_path=ws, log_file=in_log,
            )
            assert err is not None
            assert "wipes" in err.lower()
            # Message names both the log path and the workspace so the
            # operator can see the collision at a glance.
            assert in_log in err or "build.log" in err
            assert ws in err

    def test_refuses_when_log_uses_relative_path_that_resolves_inside(
        self, monkeypatch,
    ):
        # Relative path that resolves under workspace via the operator's
        # cwd — same collision, different spelling.
        with tempfile.TemporaryDirectory() as ws:
            monkeypatch.chdir(ws)
            err = cli._refuse_log_inside_workspace(
                workspace_path=ws, log_file="logs/build.log",
            )
            assert err is not None
            assert "wipes" in err.lower()

    def test_refuses_via_symlink_alias_of_workspace(self):
        # An operator with `~/proj` symlinked to `/mnt/data/proj`
        # shouldn't be able to bypass the guard by using the alias in
        # --log. realpath comparison catches this.
        with tempfile.TemporaryDirectory() as real_ws:
            with tempfile.TemporaryDirectory() as alias_dir:
                alias = os.path.join(alias_dir, "workspace_alias")
                try:
                    os.symlink(real_ws, alias)
                except (OSError, NotImplementedError):
                    pytest.skip("symlinks not supported on this platform")
                aliased_log = os.path.join(alias, "logs", "build.log")
                err = cli._refuse_log_inside_workspace(
                    workspace_path=real_ws, log_file=aliased_log,
                )
                assert err is not None
                assert "wipes" in err.lower()
