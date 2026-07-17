"""Env-placeholder expansion in the canonical config.

Machine-local paths (MCP filesystem root, deployment storage) resolve
from the environment the same way API keys do: the repo commits
``${TEANE_X:-default}``, each machine exports overrides, and the
default covers the common case. Only the TEANE_/HARNESS_ namespaces
expand — ``${POSTGRES_PASSWORD}``-style compose/shell template strings
pass through untouched.
"""

from __future__ import annotations

import os

import pytest

from harness.cli import (
    ConfigError,
    _doctor_check_env_placeholders,
    _expand_env_placeholders,
    load_raw_config,
)


class TestExpansion:
    def test_env_value_wins(self, monkeypatch):
        monkeypatch.setenv("TEANE_MCP_FS_ROOT", "/srv/work")
        out = _expand_env_placeholders(
            {"a": "${TEANE_MCP_FS_ROOT:-~/fallback}"},
        )
        assert out["a"] == "/srv/work"

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEANE_MCP_FS_ROOT", raising=False)
        out = _expand_env_placeholders({"a": "${TEANE_MCP_FS_ROOT:-/d/efault}"})
        assert out["a"] == "/d/efault"

    def test_tilde_default_expands_to_home(self, monkeypatch):
        monkeypatch.delenv("TEANE_VOLUME_ROOT", raising=False)
        out = _expand_env_placeholders({"a": "${TEANE_VOLUME_ROOT:-~/.harness/v}"})
        assert out["a"] == os.path.expanduser("~/.harness/v")
        assert "~" not in out["a"]

    def test_bare_tilde_value_expands(self):
        out = _expand_env_placeholders({"a": "~"})
        assert out["a"] == os.path.expanduser("~")

    def test_unset_without_default_raises_naming_var_and_location(
        self, monkeypatch,
    ):
        monkeypatch.delenv("TEANE_NOPE", raising=False)
        with pytest.raises(ConfigError) as exc:
            _expand_env_placeholders({"mcp": {"servers": ["${TEANE_NOPE}"]}})
        msg = str(exc.value)
        assert "TEANE_NOPE" in msg
        assert "mcp.servers[0]" in msg

    def test_non_namespace_placeholders_untouched(self):
        # Compose/shell templates must survive verbatim.
        out = _expand_env_placeholders(
            {"cmd": "docker run -e PW=${POSTGRES_PASSWORD} img"},
        )
        assert out["cmd"] == "docker run -e PW=${POSTGRES_PASSWORD} img"

    def test_malformed_namespace_placeholder_raises(self):
        with pytest.raises(ConfigError) as exc:
            _expand_env_placeholders({"a": "${TEANE_OOPS"})
        assert "malformed" in str(exc.value)

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        # Shell ':-' semantics: falls back on unset OR EMPTY. An empty
        # `export TEANE_X=` profile line previously resolved to "" and the
        # fs MCP server got an empty root while the preflight warning
        # (guarded on truthiness) stayed silent.
        monkeypatch.setenv("TEANE_MCP_FS_ROOT", "")
        out = _expand_env_placeholders({"a": "${TEANE_MCP_FS_ROOT:-/d/efault}"})
        assert out["a"] == "/d/efault"

    def test_empty_env_var_without_default_raises(self, monkeypatch):
        monkeypatch.setenv("TEANE_EMPTYX", "")
        with pytest.raises(ConfigError) as exc:
            _expand_env_placeholders({"a": "${TEANE_EMPTYX}"})
        msg = str(exc.value)
        assert "TEANE_EMPTYX" in msg
        assert "empty" in msg

    def test_nested_placeholder_rejected_loudly(self, monkeypatch):
        # The default group stops at the first '}' — supporting nesting
        # would silently truncate. Must raise whether or not the outer
        # var is set (the set case previously produced `/x}`).
        monkeypatch.setenv("TEANE_A", "/x")
        with pytest.raises(ConfigError) as exc:
            _expand_env_placeholders({"a": "${TEANE_A:-${TEANE_B}}"})
        assert "nest" in str(exc.value).lower()
        monkeypatch.delenv("TEANE_A", raising=False)
        with pytest.raises(ConfigError):
            _expand_env_placeholders({"a": "${TEANE_A:-${TEANE_B}}"})

    def test_nested_structures_and_non_strings(self, monkeypatch):
        monkeypatch.setenv("HARNESS_X", "resolved")
        out = _expand_env_placeholders({
            "list": [{"deep": "${HARNESS_X}"}, 7, None],
            "num": 3,
            "flag": True,
        })
        assert out["list"][0]["deep"] == "resolved"
        assert out["list"][1:] == [7, None]
        assert out["num"] == 3 and out["flag"] is True

    def test_multiple_placeholders_in_one_string(self, monkeypatch):
        monkeypatch.setenv("TEANE_A", "x")
        monkeypatch.setenv("TEANE_B", "y")
        out = _expand_env_placeholders({"a": "${TEANE_A}/${TEANE_B}"})
        assert out["a"] == "x/y"


class TestShippedConfig:
    def test_repo_config_loads_with_no_env_and_no_leftovers(self, monkeypatch):
        # With NO TEANE_/HARNESS_ overrides exported, the shipped
        # config's :-defaults must fully resolve.
        for var in list(os.environ):
            if var.startswith(("TEANE_", "HARNESS_")):
                monkeypatch.delenv(var)
        cfg = load_raw_config()

        def _assert_no_placeholders(node):
            if isinstance(node, str):
                assert "${TEANE_" not in node and "${HARNESS_" not in node
            elif isinstance(node, dict):
                for k, v in node.items():
                    # _-prefixed keys are documentation; the expander
                    # leaves them verbatim (they may mention placeholders).
                    if not (isinstance(k, str) and k.startswith("_")):
                        _assert_no_placeholders(v)
            elif isinstance(node, list):
                for v in node:
                    _assert_no_placeholders(v)

        _assert_no_placeholders(cfg)
        storage = cfg["deployment_defaults"]["storage"]
        assert storage["volume_root"] == os.path.expanduser(
            "~/.harness/deploy/volumes",
        )

    def test_env_override_reaches_loaded_config(self, monkeypatch):
        monkeypatch.setenv("TEANE_VOLUME_ROOT", "/data/teane-volumes")
        cfg = load_raw_config()
        storage = cfg["deployment_defaults"]["storage"]
        assert storage["volume_root"] == "/data/teane-volumes"


class TestDoctorEnvPlaceholders:
    def test_reports_default_and_override(self, monkeypatch):
        monkeypatch.setenv("TEANE_MCP_FS_ROOT", "/srv/work")
        monkeypatch.delenv("TEANE_VOLUME_ROOT", raising=False)
        rows = dict(_doctor_check_env_placeholders())
        assert rows["env override: TEANE_MCP_FS_ROOT"][0] == "pass"
        assert rows["env override: TEANE_VOLUME_ROOT"][0] == "pass"
        assert "default used" in rows["env override: TEANE_VOLUME_ROOT"][1]

    def test_resolved_env_value_never_echoed(self, monkeypatch):
        # The placeholder mechanism accepts any ${TEANE_*} anywhere in
        # config, so the resolved value can be a credentialed URL — and the
        # detail string lands in doctor output and every session log at
        # INFO. Names only, like _doctor_check_env_vars_from_config.
        secret = "https://user:hunter2@backups.internal/container/"
        monkeypatch.setenv("TEANE_MCP_FS_ROOT", secret)
        rows = dict(_doctor_check_env_placeholders())
        detail = rows["env override: TEANE_MCP_FS_ROOT"][1]
        assert secret not in detail
        assert "hunter2" not in detail

    def test_empty_env_var_reported_as_unset(self, monkeypatch):
        # Mirrors the expander's shell ':-' semantics: empty counts as
        # unset, so the default-used row must fire, not "set in
        # environment".
        monkeypatch.setenv("TEANE_MCP_FS_ROOT", "")
        rows = dict(_doctor_check_env_placeholders())
        detail = rows["env override: TEANE_MCP_FS_ROOT"][1]
        assert "default used" in detail
        assert "empty" in detail


class TestPreflightEnvReport:
    def test_logs_resolutions_and_warns_on_missing_fs_root(self, caplog):
        from harness.cli import _preflight_config_env_report
        config = {
            "mcp": {
                "enabled": True,
                "servers": [{
                    "name": "fs",
                    "command": [
                        "npx", "-y",
                        "@modelcontextprotocol/server-filesystem",
                        "/nonexistent/root/for/test",
                    ],
                }],
            },
        }
        with caplog.at_level("INFO", logger="harness.cli"):
            _preflight_config_env_report(config)
        text = caplog.text
        assert "env override: TEANE_MCP_FS_ROOT" in text  # resolution lines
        assert "does not exist on this host" in text      # missing-root warning
        assert "TEANE_MCP_FS_ROOT" in text

    def test_existing_root_produces_no_warning(self, caplog, tmp_path):
        from harness.cli import _preflight_config_env_report
        config = {
            "mcp": {
                "enabled": True,
                "servers": [{
                    "name": "fs",
                    "command": [
                        "npx", "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(tmp_path),
                    ],
                }],
            },
        }
        with caplog.at_level("WARNING", logger="harness.cli"):
            _preflight_config_env_report(config)
        assert "does not exist" not in caplog.text

    def test_never_raises(self):
        from harness.cli import _preflight_config_env_report
        _preflight_config_env_report({"mcp": {"enabled": True, "servers": [None, 42]}})
