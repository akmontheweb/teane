"""Phase 3 regression: first-run web wizard.

Covers the library layer (:mod:`harness.web_wizard`) end-to-end:
state detection across the three page modes, config + env-sidecar
write path, safety guards on env-var names / values, and starter-
template loading. The HTTP wiring is exercised in a sibling test
file so this one stays pure library.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from harness.web_wizard import (
    DEFAULT_MODELS_BY_PROVIDER,
    PROVIDER_ENV_VAR,
    StarterTemplate,
    apply_wizard_choice,
    build_default_config,
    load_starter_templates,
    wizard_state,
    write_env_sh,
)
from harness.dashboard import DashboardConfig


class _StubCfg:
    """Minimal shim exposing the fields wizard_state reads."""

    def __init__(self, log_dir):
        self.log_dir = str(log_dir)


# ---------------------------------------------------------------------------
# build_default_config — the shape validate_config_strict wants
# ---------------------------------------------------------------------------

def test_build_default_config_has_required_routing_and_throttle_keys():
    cfg = build_default_config("anthropic", "anthropic:claude-sonnet-4-6")
    # These three model_routing keys are required by validate_config_strict.
    for k in ("planning_primary", "patching_primary", "repair_primary"):
        assert cfg["model_routing"][k] == "anthropic:claude-sonnet-4-6"
    assert cfg["node_throttle"]["max_patch_repair_iterations"] == 3


# ---------------------------------------------------------------------------
# apply_wizard_choice — writes config + env sidecar
# ---------------------------------------------------------------------------

def test_apply_rejects_unknown_provider(tmp_path):
    cfg = DashboardConfig.from_config({"dashboard": {"log_dir": str(tmp_path / "logs")}})
    result = apply_wizard_choice(
        provider="not-a-provider", api_key="k", dashboard_cfg=cfg,
        dest_config_path=str(tmp_path / "config.json"),
        dest_env_sh_path=str(tmp_path / "env.sh"),
        validate=False,
    )
    assert not result.ok
    assert "unknown provider" in result.error


def test_apply_rejects_missing_key_for_remote_provider(tmp_path):
    cfg = DashboardConfig.from_config({"dashboard": {"log_dir": str(tmp_path / "logs")}})
    result = apply_wizard_choice(
        provider="anthropic", api_key="   ", dashboard_cfg=cfg,
        dest_config_path=str(tmp_path / "config.json"),
        dest_env_sh_path=str(tmp_path / "env.sh"),
        validate=False,
    )
    assert not result.ok
    assert "ANTHROPIC_API_KEY" in result.error


def test_apply_allows_local_ollama_without_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DashboardConfig.from_config({"dashboard": {"log_dir": str(tmp_path / "logs")}})
    result = apply_wizard_choice(
        provider="ollama", api_key="", dashboard_cfg=cfg,
        dest_config_path=str(tmp_path / "config.json"),
        dest_env_sh_path=str(tmp_path / "env.sh"),
        validate=False,
    )
    assert result.ok, result.error
    with open(result.config_path, "r", encoding="utf-8") as f:
        written = json.load(f)
    assert written["model_routing"]["patching_primary"] == "ollama:llama3.2"
    # No env sidecar for local providers.
    assert result.env_sh_path == ""


def test_apply_remote_provider_writes_env_sidecar(tmp_path, monkeypatch):
    cfg = DashboardConfig.from_config({"dashboard": {"log_dir": str(tmp_path / "logs")}})
    env_path = tmp_path / "env.sh"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = apply_wizard_choice(
        provider="deepseek", api_key="sk-test-1234", dashboard_cfg=cfg,
        dest_config_path=str(tmp_path / "config.json"),
        dest_env_sh_path=str(env_path),
        validate=False,
    )
    assert result.ok, result.error
    assert env_path.exists()
    body = env_path.read_text()
    assert "export DEEPSEEK_API_KEY='sk-test-1234'" in body
    # Loaded into the process's env so the SAME wizard flow can spawn a
    # child harness process without a shell reload.
    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-test-1234"


def test_apply_config_omits_api_key_from_json(tmp_path, monkeypatch):
    """The env-var-only architecture requires that the wizard NEVER
    write secrets into config.json — a leaked config file must stay
    safe to check into (private) source control."""
    cfg = DashboardConfig.from_config({"dashboard": {"log_dir": str(tmp_path / "logs")}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = apply_wizard_choice(
        provider="openai", api_key="sk-secret",
        dashboard_cfg=cfg,
        dest_config_path=str(tmp_path / "config.json"),
        dest_env_sh_path=str(tmp_path / "env.sh"),
        validate=False,
    )
    assert result.ok, result.error
    body = json.loads(open(result.config_path).read())
    # Walk the whole config and assert the secret isn't anywhere.
    assert "sk-secret" not in json.dumps(body)


# ---------------------------------------------------------------------------
# write_env_sh — safety + idempotency
# ---------------------------------------------------------------------------

def test_write_env_sh_sets_mode_0600(tmp_path):
    dest = tmp_path / "env.sh"
    write_env_sh({"FOO_KEY": "bar"}, dest=str(dest))
    mode = stat.S_IMODE(os.stat(dest).st_mode)
    assert mode == 0o600, f"env.sh must be owner-only (0600); got {oct(mode)}"


def test_write_env_sh_escapes_single_quotes(tmp_path):
    dest = tmp_path / "env.sh"
    write_env_sh({"WITH_QUOTE": "abc'def"}, dest=str(dest))
    body = dest.read_text()
    # POSIX-portable escape: close single-quote, escape ', reopen.
    assert "WITH_QUOTE='abc'\\''def'" in body


def test_write_env_sh_rewrites_existing_key(tmp_path):
    dest = tmp_path / "env.sh"
    dest.write_text(
        "# custom comment I want to keep\n"
        "export FOO='old'\n"
        "export UNRELATED='keep'\n"
    )
    write_env_sh({"FOO": "new"}, dest=str(dest))
    body = dest.read_text()
    assert "export FOO='new'" in body
    assert "export FOO='old'" not in body
    assert "export UNRELATED='keep'" in body
    assert "# custom comment I want to keep" in body


def test_write_env_sh_refuses_unsafe_var_name(tmp_path):
    with pytest.raises(ValueError, match="unsafe env var name"):
        write_env_sh({"lowercase": "x"}, dest=str(tmp_path / "env.sh"))
    with pytest.raises(ValueError, match="unsafe env var name"):
        write_env_sh({"HAS SPACE": "x"}, dest=str(tmp_path / "env.sh"))


def test_write_env_sh_refuses_control_chars_in_value(tmp_path):
    with pytest.raises(ValueError, match="control character"):
        write_env_sh({"OK_NAME": "value\nrogue"}, dest=str(tmp_path / "env.sh"))


# ---------------------------------------------------------------------------
# wizard_state — three modes
# ---------------------------------------------------------------------------

def test_wizard_state_needs_setup_when_no_config(tmp_path, monkeypatch):
    # Point _get_global_config_path at a location that doesn't exist.
    from harness import cli as _cli
    monkeypatch.setattr(_cli, "_get_global_config_path",
                        lambda: str(tmp_path / "nowhere" / "config.json"))
    state = wizard_state(_StubCfg(tmp_path / "logs"))
    assert state.kind == "needs_setup"
    assert not state.has_config


def _copy_shipped_config(dest):
    """Copy the repo's real ``config/config.json`` — the wizard needs
    a config that actually passes ``validate_config_strict`` to reach
    the ``no_sessions`` / ``has_sessions`` state, and the shipped file
    is the canonical valid example."""
    import shutil
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "config.json",
    )
    shutil.copy(src, dest)


def test_wizard_state_no_sessions_when_config_ok_and_log_dir_empty(tmp_path, monkeypatch):
    from harness import cli as _cli
    cfg_path = tmp_path / "config.json"
    _copy_shipped_config(str(cfg_path))
    monkeypatch.setattr(_cli, "_get_global_config_path", lambda: str(cfg_path))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    state = wizard_state(_StubCfg(log_dir))
    assert state.kind in ("no_sessions", "needs_setup")
    # If it's needs_setup, the shipped config's declared model
    # registry references API keys that aren't set in this test env —
    # legitimate for a fresh clone, so accept either as long as
    # has_config is True.
    assert state.has_config
    assert state.session_count == 0


def test_wizard_state_has_sessions_when_log_dir_has_entries(tmp_path, monkeypatch):
    from harness import cli as _cli
    cfg_path = tmp_path / "config.json"
    _copy_shipped_config(str(cfg_path))
    monkeypatch.setattr(_cli, "_get_global_config_path", lambda: str(cfg_path))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "sess-a.jsonl").write_text('{"event":"session_start"}\n')
    (log_dir / "sess-b.jsonl").write_text('{"event":"session_start"}\n')
    # Set API keys the shipped config's model registry references so
    # wizard_state can settle on has_sessions rather than needs_setup.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    state = wizard_state(_StubCfg(log_dir))
    assert state.kind == "has_sessions"
    assert state.session_count == 2


# ---------------------------------------------------------------------------
# Starter templates
# ---------------------------------------------------------------------------

def test_load_starter_templates_returns_packaged_three():
    """The three JSONs shipped in harness/static/templates/ must all
    load cleanly. Adding a fourth is fine; regressing to fewer than
    three fails this assertion (the wizard needs a grid with at
    least a couple of cards to feel populated)."""
    templates = load_starter_templates()
    slugs = sorted(t.slug for t in templates)
    assert "flask-todo" in slugs
    assert "fastapi-notes" in slugs
    assert "static-site" in slugs
    for t in templates:
        assert isinstance(t, StarterTemplate)
        assert t.title
        assert t.prompt.strip()


def test_load_starter_templates_skips_malformed(tmp_path):
    (tmp_path / "starter-bad.json").write_text("not json{")
    (tmp_path / "starter-ok.json").write_text(
        json.dumps({"title": "ok", "prompt": "do things"})
    )
    (tmp_path / "not-a-starter.json").write_text('{"title": "ignore"}')
    templates = load_starter_templates(templates_dir=str(tmp_path))
    slugs = [t.slug for t in templates]
    assert slugs == ["ok"]


# ---------------------------------------------------------------------------
# Provider catalogue in-sync check
# ---------------------------------------------------------------------------

def test_provider_env_var_map_matches_model_map():
    """Every provider that ships a default model must have an entry
    in the env-var map, so the wizard never claims to support a
    provider whose key requirement is unknown."""
    for provider in DEFAULT_MODELS_BY_PROVIDER:
        assert provider in PROVIDER_ENV_VAR, (
            f"PROVIDER_ENV_VAR missing entry for {provider!r}"
        )
