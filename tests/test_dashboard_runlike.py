"""Tests for the subcommand-aware spawner + validation helpers."""

from __future__ import annotations

import pytest


def test_is_valid_subcommand_accepts_each_runlike():
    from harness.dashboard_runlike import is_valid_subcommand
    for sub in ("build", "patch", "deploy", "test", "audit"):
        assert is_valid_subcommand(sub), sub


def test_is_valid_subcommand_rejects_unknown_tokens():
    from harness.dashboard_runlike import is_valid_subcommand
    assert not is_valid_subcommand("run")  # legacy / removed
    assert not is_valid_subcommand("resume")
    assert not is_valid_subcommand("status")
    assert not is_valid_subcommand("")
    assert not is_valid_subcommand("BUILD")  # case-sensitive on purpose


def test_spawn_harness_subcommand_rejects_invalid_subcommand(tmp_path):
    """The spawner is the first line of defense against a typo /
    injection — a malformed subcommand must raise rather than fire
    Popen with a bad token."""
    from harness.dashboard_runlike import spawn_harness_subcommand

    class _FakeCfg:
        log_dir = str(tmp_path)
        web_db_path = str(tmp_path / "w.db")
        host = "127.0.0.1"
        port = 9999
        hitl_webhook_secret = ""
        hitl_webhook_timeout_seconds = 600.0

    with pytest.raises(ValueError, match="unknown subcommand"):
        spawn_harness_subcommand(
            _FakeCfg(),
            subcommand="rm-rf",
            workspace=str(tmp_path),
            prompt="x",
        )


def test_spawn_harness_subcommand_requires_workspace(tmp_path):
    from harness.dashboard_runlike import spawn_harness_subcommand

    class _FakeCfg:
        log_dir = str(tmp_path)
        web_db_path = str(tmp_path / "w.db")
        host = "127.0.0.1"
        port = 9999
        hitl_webhook_secret = ""
        hitl_webhook_timeout_seconds = 600.0

    with pytest.raises(ValueError, match="workspace is required"):
        spawn_harness_subcommand(
            _FakeCfg(),
            subcommand="build",
            workspace="",
            prompt="x",
        )
