"""Tests for the pure helpers in scripts/setup.py.

The script's interactive prompts, subprocess calls, and venv-creation
side effects can't be unit-tested without a TTY and a docker daemon.
This file covers the stateless helpers that drive the wizard:
platform detection, Python lookup, config-shape construction, the
idempotent shell-rc append, and the per-platform install command
table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# Load scripts/setup.py as a module so the tests can import its symbols.
# Done this way (rather than adding scripts/ to sys.path) so the test file
# stays self-contained and `pytest` doesn't try to collect scripts/.
_SETUP_PATH = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"


@pytest.fixture(scope="module")
def setup_module_obj():
    spec = importlib.util.spec_from_file_location("harness_setup_script", _SETUP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _detect_platform
# ---------------------------------------------------------------------------

class TestDetectPlatform:

    def test_darwin(self, setup_module_obj, monkeypatch):
        import platform as p
        monkeypatch.setattr(p, "system", lambda: "Darwin")
        assert setup_module_obj._detect_platform() == "darwin"

    def test_windows(self, setup_module_obj, monkeypatch):
        import platform as p
        monkeypatch.setattr(p, "system", lambda: "Windows")
        assert setup_module_obj._detect_platform() == "windows"

    def test_linux_native(self, setup_module_obj, monkeypatch, tmp_path):
        import platform as p
        monkeypatch.setattr(p, "system", lambda: "Linux")
        # Stub /proc/version to contain a non-WSL kernel string.
        fake_proc = tmp_path / "proc_version"
        fake_proc.write_text("Linux version 6.5.0-generic ...")
        # Patch Path("/proc/version").read_text via the script's own Path import
        original_read = setup_module_obj.Path.read_text

        def fake_read_text(self, *a, **kw):
            if str(self) == "/proc/version":
                return fake_proc.read_text()
            return original_read(self, *a, **kw)

        monkeypatch.setattr(setup_module_obj.Path, "read_text", fake_read_text)
        assert setup_module_obj._detect_platform() == "linux"

    def test_wsl2(self, setup_module_obj, monkeypatch, tmp_path):
        import platform as p
        monkeypatch.setattr(p, "system", lambda: "Linux")
        fake_proc = tmp_path / "proc_version"
        fake_proc.write_text(
            "Linux version 5.15.0-microsoft-standard-WSL2 ..."
        )
        original_read = setup_module_obj.Path.read_text

        def fake_read_text(self, *a, **kw):
            if str(self) == "/proc/version":
                return fake_proc.read_text()
            return original_read(self, *a, **kw)

        monkeypatch.setattr(setup_module_obj.Path, "read_text", fake_read_text)
        assert setup_module_obj._detect_platform() == "wsl2"


# ---------------------------------------------------------------------------
# _build_default_config
# ---------------------------------------------------------------------------

class TestBuildDefaultConfig:

    def test_shape_matches_cli_schema(self, setup_module_obj):
        # The config the wizard writes MUST validate against the harness's
        # own known-keys table — otherwise the config-parse doctor check
        # would warn on first run.
        from harness.cli import _KNOWN_TOP_LEVEL_KEYS, _KNOWN_NESTED_KEYS
        cfg = setup_module_obj._build_default_config(
            "anthropic", "anthropic:claude-sonnet-4-6",
        )
        # The internal `_comment` key starts with underscore — the validator
        # ignores it by convention. All other top-level keys must be known.
        for key in cfg:
            if key.startswith("_"):
                continue
            assert key in _KNOWN_TOP_LEVEL_KEYS, (
                f"Setup script wrote unknown top-level key {key!r}; "
                f"would trip the doctor's config-parse warning"
            )
        # The model_routing section must use only known nested keys.
        # `_`-prefixed keys are comments and are skipped by the validator at
        # runtime (see cli.py _validate_config_keys), so mirror that here.
        routing_known = _KNOWN_NESTED_KEYS["model_routing"]
        for nested in cfg["model_routing"]:
            if nested.startswith("_"):
                continue
            assert nested in routing_known, (
                f"Setup script wrote unknown model_routing key {nested!r}"
            )
        # Same skip for any other nested sections (e.g. node_throttle) that
        # use inline `_comment` documentation keys.
        for section, section_value in cfg.items():
            if not isinstance(section_value, dict):
                continue
            known_nested = _KNOWN_NESTED_KEYS.get(section)
            if known_nested is None:
                continue
            for nested in section_value:
                if nested.startswith("_"):
                    continue
                assert nested in known_nested, (
                    f"Setup script wrote unknown {section} key {nested!r}"
                )

    def test_routes_all_three_roles_to_chosen_model(self, setup_module_obj):
        cfg = setup_module_obj._build_default_config(
            "openai", "openai:gpt-4o-mini",
        )
        routing = cfg["model_routing"]
        assert routing["planning_primary"] == "openai:gpt-4o-mini"
        assert routing["patching_primary"] == "openai:gpt-4o-mini"
        assert routing["repair_primary"] == "openai:gpt-4o-mini"

    def test_default_models_table_references_catalogue_keys(self, setup_module_obj):
        # Every model the wizard offers MUST exist in the shipped
        # model_prices.json — otherwise the wizard would write a config
        # the gateway can't resolve.
        catalogue = setup_module_obj._load_model_catalogue()
        if not catalogue:
            pytest.skip("model_prices.json unreadable in this env")
        for provider, model_key in setup_module_obj.DEFAULT_MODELS_BY_PROVIDER.items():
            assert model_key in catalogue, (
                f"DEFAULT_MODELS_BY_PROVIDER[{provider!r}] = {model_key!r} is "
                f"not in harness/model_prices.json"
            )


# ---------------------------------------------------------------------------
# _idempotent_append
# ---------------------------------------------------------------------------

class TestIdempotentAppend:

    def test_appends_when_missing(self, setup_module_obj, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("# user rc\n")
        line = 'export ANTHROPIC_API_KEY="sk-test"'
        assert setup_module_obj._idempotent_append(rc, line) is True
        assert line in rc.read_text()

    def test_skips_when_already_present(self, setup_module_obj, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text('export ANTHROPIC_API_KEY="sk-test"\n')
        assert setup_module_obj._idempotent_append(
            rc, 'export ANTHROPIC_API_KEY="sk-test"',
        ) is False
        # Still exactly one occurrence.
        assert rc.read_text().count("ANTHROPIC_API_KEY") == 1

    def test_creates_file_when_absent(self, setup_module_obj, tmp_path):
        rc = tmp_path / ".bashrc"
        # File doesn't exist yet
        line = "export FOO=1"
        assert setup_module_obj._idempotent_append(rc, line) is True
        assert rc.read_text().strip().endswith(line)


# ---------------------------------------------------------------------------
# _install_command_for
# ---------------------------------------------------------------------------

class TestInstallCommandFor:

    def test_python_linux(self, setup_module_obj):
        cmd = setup_module_obj._install_command_for("python3.14", "linux")
        assert "apt install" in cmd
        assert "python3.14" in cmd

    def test_python_macos(self, setup_module_obj):
        cmd = setup_module_obj._install_command_for("python3.14", "darwin")
        assert "brew install python@3.14" in cmd

    def test_python_windows_points_at_python_org(self, setup_module_obj):
        cmd = setup_module_obj._install_command_for("python3.14", "windows")
        assert "python.org" in cmd

    def test_docker_linux_includes_group_step(self, setup_module_obj):
        cmd = setup_module_obj._install_command_for("docker", "linux")
        # Without the docker-group step, the doctor reports
        # "docker: permission denied" — surface it in the install command.
        assert "docker.io" in cmd
        assert "docker" in cmd.lower()


# ---------------------------------------------------------------------------
# _detect_shell_rc
# ---------------------------------------------------------------------------

class TestDetectShellRc:

    def test_zsh(self, setup_module_obj, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = setup_module_obj._detect_shell_rc()
        assert rc is not None
        assert rc.name == ".zshrc"

    def test_bash(self, setup_module_obj, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/bash")
        rc = setup_module_obj._detect_shell_rc()
        assert rc is not None
        assert rc.name == ".bashrc"

    def test_unknown_falls_back_to_profile(self, setup_module_obj, monkeypatch):
        monkeypatch.setenv("SHELL", "/usr/bin/dash")
        rc = setup_module_obj._detect_shell_rc()
        assert rc is not None
        assert rc.name == ".profile"
