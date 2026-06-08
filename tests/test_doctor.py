"""Tests for harness/cli.py — `harness doctor` first-run healthcheck."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.cli import (
    _doctor_check_api_keys,
    _doctor_check_checkpoint_db,
    _doctor_check_config,
    _doctor_check_git,
    _doctor_check_sandbox,
    _format_doctor_line,
)


class TestDoctorGitCheck:
    def test_passes_in_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init", "-q", tmpdir], check=True)
            status, _detail = _doctor_check_git(tmpdir)
            assert status == "pass"

    def test_fails_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, detail = _doctor_check_git(tmpdir)
            assert status == "fail"
            assert "not a git repo" in detail


class TestDoctorApiKeysCheck:
    def test_warns_when_no_routing_models_configured(self):
        config = {"model_routing": {}}
        status, detail = _doctor_check_api_keys(config)
        assert status == "warn"
        assert "no non-ollama models" in detail

    def test_warns_when_only_ollama_configured(self):
        config = {"model_routing": {"planning_primary": "ollama:llama3"}}
        status, detail = _doctor_check_api_keys(config)
        assert status == "warn"

    def test_fails_when_provider_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o"}}
        status, detail = _doctor_check_api_keys(config)
        assert status == "fail"
        assert "OPENAI_API_KEY" in detail

    def test_passes_when_all_keys_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config = {
            "model_routing": {
                "planning_primary": "anthropic:claude-opus-4-5",
                "patching_primary": "deepseek:deepseek-coder",
            },
        }
        status, detail = _doctor_check_api_keys(config)
        assert status == "pass"
        assert "ANTHROPIC_API_KEY" in detail
        assert "DEEPSEEK_API_KEY" in detail


class TestDoctorSandboxCheck:
    def test_bare_backend_warns(self):
        status, detail = _doctor_check_sandbox({"sandbox": {"backend": "bare"}})
        assert status == "warn"
        assert "bare" in detail

    def test_docker_missing_fails(self, monkeypatch):
        # Force shutil.which("docker") to return None inside the check.
        import shutil
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda name: None if name == "docker" else real_which(name))
        status, detail = _doctor_check_sandbox({"sandbox": {"backend": "docker"}})
        assert status == "fail"
        assert "docker" in detail.lower()


class TestDoctorCheckpointDbCheck:
    def test_writable_db_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "subdir", "ckpt.db")
            status, detail = _doctor_check_checkpoint_db(
                {"persistence": {"db_path": db}}
            )
            assert status == "pass"
            assert db in detail


class TestDoctorConfigCheck:
    def test_clean_config_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, detail = _doctor_check_config(tmpdir)
            assert status == "pass"

    def test_typo_in_workspace_config_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / ".harness_config.json"
            cfg_path.write_text(
                '{"token_budget": {"hrad_cap_usd": 1.0}}'
            )
            status, detail = _doctor_check_config(tmpdir)
            assert status == "warn"
            assert "hrad_cap_usd" in detail


class TestDoctorLineFormatting:
    def test_line_includes_label_and_detail(self):
        line = _format_doctor_line("pass", "api keys", "all present")
        assert "api keys" in line
        assert "all present" in line

    def test_status_marker_present(self):
        assert "OK" in _format_doctor_line("pass", "x", "y")
        assert "WARN" in _format_doctor_line("warn", "x", "y")
        assert "FAIL" in _format_doctor_line("fail", "x", "y")
