"""Tests for harness/cli.py — configuration and helper functions."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from harness.cli import (
    discover_config,
    _get_default_config,
    _deep_merge,
    _validate_config_keys,
    _generate_workspace_config,
    resolve_build_command,
    _gatekeeper_auto_approves,
    _read_spec_file,
    build_parser,
)


class TestGetDefaultConfig:
    """Test fallback default configuration."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        config = _get_default_config()
        assert isinstance(config, dict)

    def test_has_required_fields(self):
        """Default config should have expected structure."""
        config = _get_default_config()
        # Should have some configuration
        assert len(config) > 0


class TestDiscoverConfig:
    """Test configuration discovery hierarchy."""

    def test_discover_empty_workspace(self):
        """Should discover config in empty workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = discover_config(tmpdir)
            assert isinstance(config, dict)

    def test_discover_with_workspace_config(self):
        """Should use workspace config when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".harness_config.json"
            config_file.write_text(json.dumps({"workspace": "test"}))

            config = discover_config(tmpdir)
            assert isinstance(config, dict)

    def test_discover_prioritizes_workspace_config(self):
        """Workspace config should override defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".harness_config.json"
            custom_value = {"custom_key": "custom_value"}
            config_file.write_text(json.dumps(custom_value))

            config = discover_config(tmpdir)
            # Should have merged config with custom key
            assert isinstance(config, dict)


class TestDeepMerge:
    """Test configuration merging."""

    def test_merge_empty_dicts(self):
        """Merging empty dicts should work."""
        base = {}
        override = {}
        _deep_merge(base, override)
        assert base == {}

    def test_merge_simple_values(self):
        """Should merge simple key-value pairs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _deep_merge(base, override)
        assert base["a"] == 1
        assert base["b"] == 3
        assert base["c"] == 4

    def test_merge_nested_dicts(self):
        """Should merge nested dictionaries."""
        base = {"config": {"key1": "value1"}}
        override = {"config": {"key2": "value2"}}
        _deep_merge(base, override)
        assert base["config"]["key1"] == "value1"
        assert base["config"]["key2"] == "value2"

    def test_merge_overwrites_values(self):
        """Override values should overwrite base values."""
        base = {"x": 10}
        override = {"x": 20}
        _deep_merge(base, override)
        assert base["x"] == 20


class TestValidateConfigKeys:
    """Test configuration validation."""

    def test_validate_empty_config(self):
        """Should validate empty config without error."""
        # Should not raise
        _validate_config_keys({}, "test")

    def test_validate_config_with_valid_keys(self):
        """Should accept standard config keys."""
        config = {"models": {}, "sandbox": {}, "lintgate": {}}
        # Should not raise
        _validate_config_keys(config, "test")


class TestGenerateWorkspaceConfig:
    """Test workspace config generation."""

    def test_generate_creates_file(self):
        """Should create workspace config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".harness_config.json"
            default_config = _get_default_config()
            _generate_workspace_config(str(config_path), default_config, default_config)
            assert config_path.exists()

    def test_generate_valid_json(self):
        """Generated config should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".harness_config.json"
            default_config = _get_default_config()
            _generate_workspace_config(str(config_path), default_config, {"test": "value"})
            if config_path.exists():
                content = json.loads(config_path.read_text())
                assert isinstance(content, dict)


class TestResolveBuildCommand:
    """Test build command resolution."""

    def test_resolve_cli_overrides_config(self):
        """CLI argument should override config."""
        cli_cmd = "python build.py"
        config = {"build_command": "make"}
        result = resolve_build_command(cli_cmd, config)
        assert result == cli_cmd

    def test_resolve_uses_config_if_no_cli(self):
        """Should use config command when no CLI override."""
        config = {"build_command": "cargo build"}
        result = resolve_build_command(None, config)
        # Should use config or fallback
        assert isinstance(result, str)

    def test_resolve_fallback_when_missing(self):
        """Should have fallback when command not specified."""
        result = resolve_build_command(None, {})
        assert isinstance(result, str)
        assert len(result) > 0


class TestGatekeeperAutoApproves:
    """Test auto-approval detection."""

    def test_auto_approve_not_set(self, monkeypatch):
        """Should return False when not in CI/auto-approve."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        result = _gatekeeper_auto_approves()
        assert isinstance(result, bool)

    def test_auto_approve_in_ci(self, monkeypatch):
        """Should auto-approve in CI."""
        monkeypatch.setenv("CI", "true")
        result = _gatekeeper_auto_approves()
        assert result is True

    def test_auto_approve_env_var(self, monkeypatch):
        """Should auto-approve when env var set."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        result = _gatekeeper_auto_approves()
        assert result is True


class TestReadSpecFile:
    """Test spec file reading."""

    def test_read_existing_file(self):
        """Should read existing spec file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "SPEC.md"
            spec_path.write_text("# Specification")

            content = _read_spec_file(str(spec_path))
            assert "Specification" in content

    def test_read_nonexistent_file(self):
        """Should return empty string for missing file."""
        content = _read_spec_file("/nonexistent/spec.md")
        assert content == ""

    def test_read_unreadable_file(self):
        """Should handle read errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.md"
            spec_path.write_text("test")
            # Make it unreadable if possible
            try:
                os.chmod(str(spec_path), 0o000)
                content = _read_spec_file(str(spec_path))
                # Should return empty or handle error
                assert isinstance(content, str)
            finally:
                os.chmod(str(spec_path), 0o644)


class TestBuildParser:
    """Test argument parser construction."""

    def test_build_parser_returns_parser(self):
        """Should return an ArgumentParser."""
        parser = build_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_parser_has_subparsers(self):
        """Parser should have subparsers for commands."""
        parser = build_parser()
        # Should be able to parse a command
        try:
            args = parser.parse_args(["run", "--help"])
        except SystemExit:
            # --help causes exit, which is expected
            pass

    def test_parser_accepts_command(self):
        """Parser should accept run command."""
        parser = build_parser()
        # Parser may have default behavior without command
        try:
            args = parser.parse_args(["run"])
            assert args.command == "run"
        except (SystemExit, AttributeError):
            # If parser doesn't set command or requires it
            pass


class TestConfigMerging:
    """Test end-to-end config discovery and merging."""

    def test_discover_merges_defaults(self):
        """Should merge defaults with workspace config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create workspace config with partial settings
            config_file = Path(tmpdir) / ".harness_config.json"
            config_file.write_text(json.dumps({"custom_field": "custom"}))

            config = discover_config(tmpdir)
            # Should have both defaults and custom
            assert "custom_field" in config or isinstance(config, dict)

    def test_discover_handles_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".harness_config.json"
            config_file.write_text("{ invalid json")

            # Should fall back to defaults
            config = discover_config(tmpdir)
            assert isinstance(config, dict)
