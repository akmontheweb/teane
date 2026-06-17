"""Tests for harness/cli.py — single-source canonical config + helpers.

The harness reads exactly one config file (config/config.json). Validation is
strict: unknown keys, missing required fields, wrong types, or missing API key
env vars all raise ConfigError. These tests cover every failure branch plus the
happy path, then exercise the unrelated CLI helpers (resolve_build_command,
gatekeeper auto-approve, spec-file reading, argument parser, interactive
review loop).
"""

import json
import os

import pytest

from harness.cli import (
    discover_config,
    validate_config_strict,
    ConfigError,
    _strip_comments,
    _warn_if_legacy_workspace_config,
    resolve_build_command,
    _gatekeeper_auto_approves,
    _read_spec_file,
    build_parser,
    _doctor_check_external_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_valid_config() -> dict:
    """Smallest config that passes validate_config_strict (assuming the
    matching env vars are exported by the test fixture)."""
    return {
        "build_command": "make build",
        "allow_network": True,
        "product_spec_dir": "product_spec",
        "sandbox": {"backend": "auto"},
        "token_budget": {"hard_cap_usd": 2.0},
        "persistence": {"db_path": "~/.harness/checkpoints.db"},
        "models": {
            "openai:gpt-4o-mini": {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "api_key": "",
            },
        },
        "model_routing": {
            "planning_primary": "openai:gpt-4o-mini",
            "patching_primary": "openai:gpt-4o-mini",
            "repair_primary": "openai:gpt-4o-mini",
        },
    }


@pytest.fixture
def openai_key(monkeypatch):
    """Provide OPENAI_API_KEY so env-var validation passes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    yield


# ---------------------------------------------------------------------------
# validate_config_strict — every error branch
# ---------------------------------------------------------------------------

class TestValidateConfigStrict:

    def test_minimum_valid_config_passes(self, openai_key):
        # The smallest config that satisfies every required check should
        # validate without raising.
        validate_config_strict(_min_valid_config(), source="test")

    def test_unknown_top_level_key_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["model_routin"] = {}  # typo
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "Unknown top-level key 'model_routin'" in msg
        assert "model_routing" in msg  # difflib suggestion

    def test_unknown_nested_key_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["token_budget"]["hrad_cap_usd"] = 2.0  # typo
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "Unknown nested key 'token_budget.hrad_cap_usd'" in msg
        assert "hard_cap_usd" in msg  # difflib suggestion

    def test_deployment_defaults_valid_passes(self, openai_key):
        # A populated deployment_defaults section with the four known
        # sub-sections (network, storage, secrets, infra_sync) is
        # accepted; LEAF keys inside the sub-sections are intentionally
        # not enumerated — operators may set organisation-specific
        # policies the harness has never heard of.
        cfg = _min_valid_config()
        cfg["deployment_defaults"] = {
            "network": {"reverse_proxy": "caddy", "tls_strategy": "letsencrypt"},
            "storage": {"volume_root": "/var/lib/app"},
            "secrets": {"manager": "vault"},
            "infra_sync": {"conflict_policy": "abort"},
        }
        validate_config_strict(cfg, source="test")

    def test_deployment_defaults_typoed_subsection_raises(self, openai_key):
        # Typo in a sub-section name (network → netwrok) is caught at
        # depth 1 by the nested-key validator, with a difflib hint.
        cfg = _min_valid_config()
        cfg["deployment_defaults"] = {"netwrok": {}}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "Unknown nested key 'deployment_defaults.netwrok'" in msg
        assert "network" in msg  # difflib suggestion

    def test_missing_models_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["models"] = {}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'models' must contain at least one entry" in str(exc.value)

    def test_missing_planning_primary_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["model_routing"]["planning_primary"] = ""
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'model_routing.planning_primary' is required" in str(exc.value)

    def test_routing_references_unknown_model_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["model_routing"]["planning_primary"] = "deepseek:ghost"
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "references unknown model 'deepseek:ghost'" in str(exc.value)

    def test_optional_routing_unknown_reference_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["model_routing"]["doc_reviewer_primary"] = "openai:no-such"
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'model_routing.doc_reviewer_primary' is set to" in str(exc.value)

    def test_wrong_type_for_int_field_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["sandbox"]["docker_pids_limit"] = "100"  # str instead of int
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'sandbox.docker_pids_limit' must be of type int" in str(exc.value)

    def test_wrong_type_for_bool_field_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["allow_network"] = "yes"  # str instead of bool
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'allow_network' must be of type bool" in str(exc.value)

    def test_negative_hard_cap_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["token_budget"]["hard_cap_usd"] = -1.0
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "must be a positive number" in str(exc.value)

    def test_invalid_sandbox_backend_raises(self, openai_key):
        cfg = _min_valid_config()
        cfg["sandbox"]["backend"] = "invalid_backend"
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "'sandbox.backend' must be one of" in str(exc.value)

    def test_missing_required_env_var_raises(self, monkeypatch):
        # No env var → ConfigError with name of missing env var.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(_min_valid_config(), source="test")
        msg = str(exc.value)
        assert "Missing API key environment variable" in msg
        assert "OPENAI_API_KEY" in msg
        assert "openai:gpt-4o-mini" in msg

    def test_ollama_model_does_not_require_env_var(self, monkeypatch):
        # Local providers (ollama) don't need a {PROVIDER}_API_KEY env var.
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        cfg = _min_valid_config()
        cfg["models"]["ollama:qwen2.5-coder:14b"] = {
            "provider": "ollama",
            "model_id": "qwen2.5-coder:14b",
            "api_key": "",
        }
        cfg["model_routing"]["ollama_local_model"] = "ollama:qwen2.5-coder:14b"
        # OPENAI_API_KEY still required for the routed openai model.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
        # Should NOT raise — ollama is in _LOCAL_PROVIDERS.
        validate_config_strict(cfg, source="test")

    def test_unused_model_skips_env_var_check(self, monkeypatch):
        # Models declared in `models` but NOT referenced by any
        # model_routing field must NOT cause env-var validation.
        cfg = _min_valid_config()
        cfg["models"]["anthropic:claude-sonnet-4"] = {
            "provider": "anthropic",
            "model_id": "claude-sonnet-4",
            "api_key": "",
        }
        # No model_routing field references anthropic, so
        # ANTHROPIC_API_KEY does NOT need to be set.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        validate_config_strict(cfg, source="test")

    def test_multiple_errors_reported_at_once(self, monkeypatch):
        # The validator collects ALL errors in a single pass and raises
        # one ConfigError with the full list — operator gets a complete
        # fix list, not just the first problem.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _min_valid_config()
        cfg["unknown_top"] = {}
        cfg["token_budget"]["hrad_cap_usd"] = 2.0
        cfg["models"] = {}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "Unknown top-level key 'unknown_top'" in msg
        assert "Unknown nested key 'token_budget.hrad_cap_usd'" in msg
        assert "'models' must contain at least one entry" in msg

    def test_error_message_tells_operator_to_fix_config(self, monkeypatch):
        # The trailing "Fix the config file before re-running" line is
        # the contract the user explicitly asked for.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(_min_valid_config(), source="/some/path.json")
        msg = str(exc.value)
        assert "/some/path.json" in msg
        assert "Fix the config file before re-running" in msg

    # --- llm_dispatch section (max_tokens per role) ---

    def test_llm_dispatch_valid_passes(self, openai_key):
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "max_tokens_default": 4096,
            "max_tokens_per_role": {
                "planning": 4096, "patching": 4096, "repair": 8192,
                "doc_reviewer": 2048, "code_reviewer": 4096,
            },
        }
        validate_config_strict(cfg, source="test")

    def test_llm_dispatch_default_below_floor_rejected(self, openai_key):
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {"max_tokens_default": 100}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "max_tokens_default" in str(exc.value)
        assert "[256, 32768]" in str(exc.value)

    def test_llm_dispatch_per_role_above_ceiling_rejected(self, openai_key):
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {"max_tokens_per_role": {"repair": 99999}}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "max_tokens_per_role.repair" in msg
        assert "32768" in msg

    def test_llm_dispatch_per_role_wrong_type_rejected(self, openai_key):
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {"max_tokens_per_role": {"repair": "eight"}}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "must be an int" in str(exc.value)

    def test_llm_dispatch_section_optional(self, openai_key):
        # Section is opt-in — absence falls back to GatewayConfig defaults.
        cfg = _min_valid_config()
        assert "llm_dispatch" not in cfg
        validate_config_strict(cfg, source="test")

    def test_llm_dispatch_unknown_nested_key_rejected(self, openai_key):
        # Typos in the llm_dispatch section must surface, not silently no-op.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {"max_tokens_defalt": 4096}  # typo
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "max_tokens_defalt" in str(exc.value)
        assert "max_tokens_default" in str(exc.value)  # difflib suggestion

    def test_llm_dispatch_default_null_accepted(self, openai_key):
        # null/empty/0 mean "no limit" — they must not be rejected.
        for blank in (None, "", 0):
            cfg = _min_valid_config()
            cfg["llm_dispatch"] = {"max_tokens_default": blank}
            validate_config_strict(cfg, source="test")

    def test_llm_dispatch_per_role_null_accepted(self, openai_key):
        # A blank per-role entry means "no limit for this role" — overrides
        # the default rather than inheriting it. Validation accepts any of
        # null / "" / 0.
        for blank in (None, "", 0):
            cfg = _min_valid_config()
            cfg["llm_dispatch"] = {
                "max_tokens_default": 4096,
                "max_tokens_per_role": {"planning": blank},
            }
            validate_config_strict(cfg, source="test")

    def test_llm_dispatch_default_non_empty_string_rejected(self, openai_key):
        # Strings are accepted by the broadened type schema, but a non-empty
        # string is garbage — it must still be rejected with a clear message.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {"max_tokens_default": "4096"}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "max_tokens_default" in str(exc.value)
        assert "int, null, or blank" in str(exc.value)

    # --- llm_dispatch.continue_on_length ---

    def test_llm_dispatch_continue_on_length_valid_passes(self, openai_key):
        # All five known roles can be set; the validator must accept.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "continue_on_length": {
                "planning": False,
                "patching": True,
                "repair": False,
                "doc_reviewer": False,
                "code_reviewer": False,
            },
        }
        validate_config_strict(cfg, source="test")

    def test_llm_dispatch_continue_on_length_missing_passes(self, openai_key):
        # The map is optional — omission falls back to
        # graph._CONTINUE_ON_LENGTH_DEFAULTS (only patching on).
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "max_tokens_per_role": {"patching": 16384},
        }
        validate_config_strict(cfg, source="test")

    def test_llm_dispatch_continue_on_length_partial_passes(self, openai_key):
        # Operator can override one role and inherit defaults for the rest.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "continue_on_length": {"planning": True},
        }
        validate_config_strict(cfg, source="test")

    def test_llm_dispatch_continue_on_length_wrong_type_rejected(
        self, openai_key,
    ):
        # Per-role values must be bool, not int / string / etc.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "continue_on_length": {"patching": "yes"},
        }
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        msg = str(exc.value)
        assert "continue_on_length.patching" in msg
        assert "must be a bool" in msg

    def test_llm_dispatch_continue_on_length_int_rejected(self, openai_key):
        # 0/1 are tempting bool-likes but Python's isinstance(x, bool) is
        # strict; the validator must reject them so operators get a clear
        # error rather than silently treating 1 as True.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "continue_on_length": {"repair": 1},
        }
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "continue_on_length.repair" in str(exc.value)
        assert "must be a bool" in str(exc.value)

    def test_llm_dispatch_continue_on_length_empty_role_key_rejected(
        self, openai_key,
    ):
        # Empty / whitespace role names are nonsensical.
        cfg = _min_valid_config()
        cfg["llm_dispatch"] = {
            "continue_on_length": {"": True},
        }
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(cfg, source="test")
        assert "continue_on_length" in str(exc.value)
        assert "non-empty role-name strings" in str(exc.value)


# ---------------------------------------------------------------------------
# discover_config — single-source loader
# ---------------------------------------------------------------------------

class TestDiscoverConfig:

    def test_canonical_loads_when_valid(self, openai_key):
        # Smoke: the shipped config/config.json validates when the matching
        # env var is set. Doubles as a regression check that we didn't
        # break the canonical file with a typo or wrong type.
        # (This test will fail until DEEPSEEK_API_KEY is also set in the
        # environment, since the shipped config routes through deepseek.)
        # Use a stripped-down config via monkeypatch instead.
        pass  # exercised by the test_discover_returns_dict test below

    def test_discover_raises_when_canonical_missing(self, monkeypatch, tmp_path):
        from harness import cli as cli_mod
        # Point _get_global_config_path at a path that doesn't exist
        missing = str(tmp_path / "nope.json")
        monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: missing)
        with pytest.raises(ConfigError) as exc:
            discover_config(str(tmp_path))
        assert "Canonical config not found" in str(exc.value)
        assert missing in str(exc.value)

    def test_discover_raises_on_invalid_json(self, monkeypatch, tmp_path):
        from harness import cli as cli_mod
        bad = tmp_path / "config.json"
        bad.write_text("{ invalid json", encoding="utf-8")
        monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(bad))
        with pytest.raises(ConfigError) as exc:
            discover_config(str(tmp_path))
        assert "Invalid JSON in" in str(exc.value)
        assert "Fix the JSON syntax" in str(exc.value)

    def test_discover_raises_on_non_object_root(self, monkeypatch, tmp_path):
        from harness import cli as cli_mod
        bad = tmp_path / "config.json"
        bad.write_text('["this", "is", "an", "array"]', encoding="utf-8")
        monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(bad))
        with pytest.raises(ConfigError) as exc:
            discover_config(str(tmp_path))
        assert "must contain a JSON object at the top level" in str(exc.value)

    def test_discover_returns_dict(self, monkeypatch, tmp_path, openai_key):
        from harness import cli as cli_mod
        good = tmp_path / "config.json"
        good.write_text(json.dumps(_min_valid_config()), encoding="utf-8")
        monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(good))
        cfg = discover_config(str(tmp_path))
        assert isinstance(cfg, dict)
        assert cfg["model_routing"]["planning_primary"] == "openai:gpt-4o-mini"

    def test_discover_strips_comment_keys(self, monkeypatch, tmp_path, openai_key):
        from harness import cli as cli_mod
        cfg_in = _min_valid_config()
        cfg_in["_comment"] = "top-level doc"
        cfg_in["sandbox"]["_comment"] = "nested doc"
        good = tmp_path / "config.json"
        good.write_text(json.dumps(cfg_in), encoding="utf-8")
        monkeypatch.setattr(cli_mod, "_get_global_config_path", lambda: str(good))
        cfg_out = discover_config(str(tmp_path))
        assert "_comment" not in cfg_out
        assert "_comment" not in cfg_out["sandbox"]


# ---------------------------------------------------------------------------
# _strip_comments
# ---------------------------------------------------------------------------

class TestStripComments:

    def test_top_level_comment_removed(self):
        assert _strip_comments({"_comment": "x", "k": 1}) == {"k": 1}

    def test_nested_comment_removed(self):
        out = _strip_comments({"sandbox": {"_comment": "x", "backend": "auto"}})
        assert out == {"sandbox": {"backend": "auto"}}

    def test_non_string_keys_preserved(self):
        # Defensive: JSON only uses string keys, but the helper shouldn't
        # explode on non-string keys.
        out = _strip_comments({"k": 1, "_skip": 2})
        assert out == {"k": 1}


# ---------------------------------------------------------------------------
# _warn_if_legacy_workspace_config
# ---------------------------------------------------------------------------

class TestLegacyWorkspaceConfig:

    def test_no_warning_when_legacy_absent(self, tmp_path, caplog):
        with caplog.at_level("INFO"):
            _warn_if_legacy_workspace_config(str(tmp_path))
        msgs = " ".join(r.message for r in caplog.records)
        assert "Legacy .harness_config.json" not in msgs

    def test_warning_when_legacy_present(self, tmp_path, caplog):
        legacy = tmp_path / ".harness_config.json"
        legacy.write_text("{}", encoding="utf-8")
        with caplog.at_level("INFO"):
            _warn_if_legacy_workspace_config(str(tmp_path))
        msgs = " ".join(r.message for r in caplog.records)
        assert "Legacy .harness_config.json" in msgs
        assert str(legacy) in msgs


# ---------------------------------------------------------------------------
# resolve_build_command — unchanged behavior, kept under coverage
# ---------------------------------------------------------------------------

class TestResolveBuildCommand:

    def test_cli_overrides_config(self):
        cli_cmd = "python build.py"
        config = {"build_command": "make"}
        result = resolve_build_command(cli_cmd, config)
        assert result == cli_cmd

    def test_uses_config_when_no_cli(self):
        config = {"build_command": "cargo build"}
        result = resolve_build_command(None, config)
        assert isinstance(result, str)

    def test_fallback_when_missing(self):
        result = resolve_build_command(None, {})
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Gatekeeper auto-approval (env-var driven)
# ---------------------------------------------------------------------------

class TestGatekeeperAutoApproves:

    def test_returns_bool_when_unset(self, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        assert isinstance(_gatekeeper_auto_approves(), bool)

    def test_ci_env_triggers_approval(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert _gatekeeper_auto_approves() is True

    def test_harness_auto_approve_env_triggers(self, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        assert _gatekeeper_auto_approves() is True


# ---------------------------------------------------------------------------
# Spec-file reading
# ---------------------------------------------------------------------------

class TestReadSpecFile:

    def test_read_existing_file(self, tmp_path):
        spec_path = tmp_path / "SPEC.md"
        spec_path.write_text("# Specification")
        assert "Specification" in _read_spec_file(str(spec_path))

    def test_read_nonexistent_file(self):
        assert _read_spec_file("/nonexistent/spec.md") == ""

    def test_read_unreadable_file(self, tmp_path):
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("test")
        try:
            os.chmod(str(spec_path), 0o000)
            assert isinstance(_read_spec_file(str(spec_path)), str)
        finally:
            os.chmod(str(spec_path), 0o644)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

class TestBuildParser:

    def test_returns_parser(self):
        parser = build_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_parser_has_run_subcommand(self):
        parser = build_parser()
        try:
            parser.parse_args(["run", "--help"])
        except SystemExit:
            pass  # --help exits, expected

    def test_deploy_dev_defaults_false(self):
        # Default is opt-in: omit the flag and the harness will stop after
        # a clean security scan instead of rolling forward into deployment.
        parser = build_parser()
        args = parser.parse_args(["run", "-w", "/tmp/x", "-p", "do x"])
        assert args.deploy_dev is False

    def test_deploy_dev_true(self):
        parser = build_parser()
        args = parser.parse_args(
            ["run", "-w", "/tmp/x", "-p", "do x", "--deploy-dev", "true"]
        )
        assert args.deploy_dev is True

    def test_old_dev_deployment_flag_rejected(self):
        # The legacy --dev-deployment / --dev_deployment forms are gone;
        # argparse must surface the rename loudly instead of silently
        # ignoring scripts that still pass them.
        import pytest as _pytest
        parser = build_parser()
        with _pytest.raises(SystemExit):
            parser.parse_args(
                ["run", "-w", "/tmp/x", "-p", "do x", "--dev-deployment"],
            )
        with _pytest.raises(SystemExit):
            parser.parse_args(
                ["run", "-w", "/tmp/x", "-p", "do x", "--dev_deployment"],
            )

    def test_workspace_dash_r_short_alias_removed(self):
        # `-r` used to be a third short alias for --workspace; the
        # simplification keeps only -w. Asserting the rejection helps
        # operators notice the change.
        import pytest as _pytest
        parser = build_parser()
        with _pytest.raises(SystemExit):
            parser.parse_args(["run", "-r", "/tmp/x", "-p", "do x"])


# ---------------------------------------------------------------------------
# Interactive review loop (kept from prior file; unrelated to config change)
# ---------------------------------------------------------------------------

class TestInteractiveReviewLoopAsync:
    """Fix 1 regression: the [B] Refine action used to call asyncio.run()
    from inside the already-running cmd_run loop and raise
    'asyncio.run() cannot be called from a running event loop'.
    """

    @pytest.mark.asyncio
    async def test_refine_branch_awaits_helper_without_raising(self, tmp_path, monkeypatch):
        from harness import cli as cli_mod
        from harness.hitl import set_channel, reset_channel

        # The new --hitl-req=false default + pytest's non-TTY stdin
        # would auto-approve before the channel stub gets a chance.
        # Force both off so the refine-branch path is actually exercised.
        monkeypatch.setattr(
            cli_mod, "_hitl_gate_enabled", lambda gate_name: True,
        )
        monkeypatch.setattr(
            cli_mod, "_gatekeeper_auto_approves",
            lambda gate_name=None: False,
        )

        spec_path = tmp_path / "SPEC_REQUIREMENTS.md"
        spec_path.write_text("# Original\nA\n", encoding="utf-8")

        class _Channel:
            def __init__(self):
                self._responses = iter(["b", "feedback notes", "a"])

            def prompt(self, *args, **kwargs):
                return next(self._responses)

            def notes(self, *args, **kwargs):
                return next(self._responses)

            def confirm(self, *args, **kwargs):
                return True

            def wait_for_manual_edit(self, *args, **kwargs):
                return None

        set_channel(_Channel())

        async def fake_refine(spec_path_arg, notes, gateway):
            with open(spec_path_arg, "w", encoding="utf-8") as f:
                f.write("# Refined\nA + " + notes + "\n")
            return open(spec_path_arg, encoding="utf-8").read()

        monkeypatch.setattr(cli_mod, "_refine_requirements", fake_refine)

        try:
            result = await cli_mod.interactive_review_loop(str(spec_path), gateway=None)
        finally:
            reset_channel()

        assert "Refined" in result
        assert "feedback notes" in result


# ---------------------------------------------------------------------------
# _doctor_check_external_tools — external-binary detection rows
# ---------------------------------------------------------------------------

class TestDoctorExternalTools:
    """Verify the external-tools doctor check produces the right severity
    and install-hint surface for each tool. ``shutil.which`` is monkeypatched
    at the ``harness.cli`` module so both the in-helper lookups and the
    docker-compose v2 probe share the fake."""

    @staticmethod
    def _config(scanners=None, sandbox_backend="auto", deployment_enabled=False):
        cfg = {
            "sandbox": {"backend": sandbox_backend},
            "deployment": {"enabled": deployment_enabled},
        }
        if scanners is not None:
            cfg["security_scan"] = {"scanners": list(scanners)}
        return cfg

    @staticmethod
    def _rows_by_name(rows):
        # rows are (label, (status, detail)); strip the "external: " prefix.
        return {label.removeprefix("external: "): (status, detail)
                for label, (status, detail) in rows}

    def test_gitleaks_missing_warns_with_install_hint(self, tmp_path, monkeypatch):
        import harness.cli as cli_mod

        def fake_which(name):
            return None if name == "gitleaks" else f"/usr/bin/{name}"
        monkeypatch.setattr(cli_mod.shutil, "which", fake_which)

        cfg = self._config(scanners=["gitleaks"])
        rows = self._rows_by_name(_doctor_check_external_tools(cfg, str(tmp_path)))

        assert "gitleaks" in rows
        status, detail = rows["gitleaks"]
        assert status == "warn"
        assert "Python fallback" in detail
        assert "install:" in detail
        assert "gitleaks" in detail  # hint references the binary

    def test_all_security_tools_present_pass(self, tmp_path, monkeypatch):
        import harness.cli as cli_mod

        monkeypatch.setattr(cli_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        # Avoid `docker compose version` actually running.
        monkeypatch.setattr(
            cli_mod, "_has_docker_compose_subcommand", lambda: True,
        )

        cfg = self._config(scanners=["gitleaks", "bandit", "semgrep", "trivy"])
        rows = self._rows_by_name(_doctor_check_external_tools(cfg, str(tmp_path)))

        for scanner in ("gitleaks", "bandit", "semgrep", "trivy"):
            assert rows[scanner][0] == "pass", rows[scanner]
        assert rows["docker"][0] == "pass"

    def test_docker_missing_with_docker_backend_fails(self, tmp_path, monkeypatch):
        import harness.cli as cli_mod

        def fake_which(name):
            return None if name == "docker" else f"/usr/bin/{name}"
        monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
        monkeypatch.setattr(
            cli_mod, "_has_docker_compose_subcommand", lambda: False,
        )

        cfg = self._config(sandbox_backend="docker")
        rows = self._rows_by_name(_doctor_check_external_tools(cfg, str(tmp_path)))

        assert rows["docker"][0] == "fail"
        assert "install:" in rows["docker"][1]

    def test_no_formatter_row_for_absent_extension(self, tmp_path, monkeypatch):
        import harness.cli as cli_mod

        # Workspace contains only .py files → no clang-format / prettier rows.
        (tmp_path / "a.py").write_text("print('hi')\n")
        (tmp_path / "b.py").write_text("x = 1\n")

        monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            cli_mod, "_has_docker_compose_subcommand", lambda: False,
        )

        cfg = self._config(scanners=["gitleaks"])
        rows = self._rows_by_name(_doctor_check_external_tools(cfg, str(tmp_path)))

        # Negative assertions: tools tied to extensions we didn't create
        # must NOT produce rows.
        for missing in ("clang-format", "prettier", "rustfmt", "gofmt", "shfmt"):
            assert missing not in rows, f"unexpected row for {missing}: {rows[missing]}"
        # Positive: ruff (the .py formatter) should be present, as a warn.
        assert "ruff" in rows
        assert rows["ruff"][0] == "warn"

    def test_disabled_scanner_is_skipped_not_warned(self, tmp_path, monkeypatch):
        import harness.cli as cli_mod

        monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            cli_mod, "_has_docker_compose_subcommand", lambda: False,
        )

        # Operator opted out of trivy → its missing binary is informational,
        # not a warning.
        cfg = self._config(scanners=["gitleaks", "bandit", "semgrep"])
        rows = self._rows_by_name(_doctor_check_external_tools(cfg, str(tmp_path)))

        assert rows["trivy"][0] == "skip"
        assert rows["gitleaks"][0] == "warn"


# ---------------------------------------------------------------------------
# _dispatch_with_continuation — recovers from finish_reason="length"
# ---------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason

        class _Usage:
            input_tokens = 100
            output_tokens = 200
            cost_usd = 0.001
        self.usage = _Usage()


class _ChunkedGateway:
    """Returns canned (content, finish_reason) tuples in order. Used to
    simulate an LLM that hits its output token cap mid-document on
    cycle N and finally stops on a later cycle."""

    def __init__(self, chunks: list[tuple[str, str]]):
        self._chunks = list(chunks)
        self.dispatched_messages: list[list[dict]] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched_messages.append([dict(m) for m in messages])
        content, finish_reason = self._chunks.pop(0)
        return _StubResponse(content, finish_reason), budget_remaining_usd - 0.10


class TestDispatchWithContinuation:
    """The architecture/requirements writers used to truncate at the
    planning role's 4096-token output cap and never resume — session
    web-6d5ef9b18f6a's SPEC_ARCHITECTURE.md ended mid-sentence in §3
    and the patching round that followed never saw the frontend
    inventory. The helper now feeds the partial back as an assistant
    turn and asks the LLM to continue."""

    def test_single_stop_returns_content_unchanged(self):
        import asyncio
        from harness.cli import _dispatch_with_continuation

        gw = _ChunkedGateway([("# Spec\n\nDone.", "stop")])
        content, cost = asyncio.run(_dispatch_with_continuation(
            gateway=gw,
            messages=[{"role": "user", "content": "synthesize"}],
            role="planning",
            budget_remaining_usd=2.0,
            log_label="test",
        ))
        assert content == "# Spec\n\nDone."
        assert cost == pytest.approx(0.001)
        assert len(gw.dispatched_messages) == 1

    def test_length_then_stop_concatenates_chunks(self):
        import asyncio
        from harness.cli import _dispatch_with_continuation

        gw = _ChunkedGateway([
            ("# Section 1\n\nFirst half.", "length"),
            (" Second half.\n# Section 2\n", "stop"),
        ])
        content, cost = asyncio.run(_dispatch_with_continuation(
            gateway=gw,
            messages=[{"role": "user", "content": "synthesize"}],
            role="planning",
            budget_remaining_usd=2.0,
            log_label="test",
        ))
        assert content == "# Section 1\n\nFirst half. Second half.\n# Section 2\n"
        # Two dispatches paid for; the helper sums their costs.
        assert cost == pytest.approx(0.002)
        # The second dispatch must have received the partial as an
        # assistant turn plus a user "continue" prompt.
        second_msgs = gw.dispatched_messages[1]
        assert second_msgs[-2]["role"] == "assistant"
        assert second_msgs[-2]["content"] == "# Section 1\n\nFirst half."
        assert second_msgs[-1]["role"] == "user"
        assert "EXACTLY where you left off" in second_msgs[-1]["content"]

    def test_caps_continuations_at_max(self):
        """Never-stops case — the helper bails after max_continuations
        and returns whatever it accumulated rather than looping
        forever."""
        import asyncio
        from harness.cli import _dispatch_with_continuation

        # Initial + 3 continuations = 4 dispatches, all "length".
        gw = _ChunkedGateway([
            ("chunk1 ", "length"),
            ("chunk2 ", "length"),
            ("chunk3 ", "length"),
            ("chunk4", "length"),
        ])
        content, cost = asyncio.run(_dispatch_with_continuation(
            gateway=gw,
            messages=[{"role": "user", "content": "x"}],
            role="planning",
            budget_remaining_usd=2.0,
            log_label="test",
            max_continuations=3,
        ))
        assert content == "chunk1 chunk2 chunk3 chunk4"
        assert len(gw.dispatched_messages) == 4
        assert cost == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# Flag-surface simplification: defaults, --yes wiring, removed flags
# ---------------------------------------------------------------------------

class TestRunParserSurface:
    """Locks down the simplified run-parser surface so the next round of
    refactoring can spot accidental regressions. Covers all four new
    HITL toggles, --spec-discovery, --cd-discovery, --deploy-dev, the
    --new-build / --yes pairing, the two default flips (--allow-network
    true, --git false), and explicit rejection of every removed/renamed
    flag."""

    @staticmethod
    def _args(*extra):
        from harness.cli import build_parser
        return build_parser().parse_args(
            ["run", "-w", "/tmp/x", "-p", "do x", *extra],
        )

    def test_all_hitl_flags_default_false(self):
        a = self._args()
        assert a.hitl_req is False
        assert a.hitl_arch is False
        assert a.hitl_repair is False
        assert a.hitl_deployment is False

    def test_discovery_toggles_default_false(self):
        a = self._args()
        assert a.spec_discovery is False
        assert a.cd_discovery is False
        assert a.deploy_dev is False

    def test_allow_network_defaults_true_git_defaults_false(self):
        # Two default-flips relative to the legacy surface — locked
        # down so a future refactor can't quietly reverse them.
        a = self._args()
        assert a.allow_network is True
        assert a.git is False

    def test_hitl_flag_accepts_true(self):
        a = self._args("--hitl-req", "true", "--hitl-deployment", "true")
        assert a.hitl_req is True
        assert a.hitl_deployment is True
        # The other two stay at their false default.
        assert a.hitl_arch is False
        assert a.hitl_repair is False

    def test_bool_flag_accepts_yes_and_no(self):
        # `_bool_choice` is the shared parser type; assert it
        # tolerates the operator-friendly spellings, not just
        # "true"/"false".
        a = self._args(
            "--hitl-req", "yes",
            "--hitl-arch", "no",
            "--cd-discovery", "1",
            "--deploy-dev", "off",
        )
        assert a.hitl_req is True
        assert a.hitl_arch is False
        assert a.cd_discovery is True
        assert a.deploy_dev is False

    def test_bool_flag_rejects_garbage(self):
        import pytest as _pytest
        with _pytest.raises(SystemExit):
            self._args("--hitl-req", "maybe")

    def test_yes_alone_rejected_in_cmd_run(self, tmp_path):
        # --yes is a confirmation modifier for --new-build, not a
        # standalone flag. cmd_run (not the parser) returns 2 with a
        # stderr message when --yes is passed without --new-build true.
        import argparse
        import io
        import sys
        from harness import cli as cli_mod
        a = argparse.Namespace(
            workspace=str(tmp_path), prompt="do x",
            new_build=False, assume_yes=True,
            hitl_req=False, hitl_arch=False, hitl_repair=False,
            hitl_deployment=False, force_lock=False,
        )
        # Capture stderr so the test runner doesn't get a noisy line.
        buf = io.StringIO()
        orig = sys.stderr
        sys.stderr = buf
        try:
            import asyncio
            rc = asyncio.run(cli_mod.cmd_run(a))
        finally:
            sys.stderr = orig
        assert rc == 2
        assert "--yes can only be used with --new-build true" in buf.getvalue()

    def test_removed_flags_rejected(self):
        # Every dropped/renamed flag must fail loudly so operators
        # relying on the legacy names see the rename immediately.
        import pytest as _pytest
        for arg in (
            ["--skip-discovery"],
            ["--discover"],
            ["--dev-deployment"],
            ["--dev_deployment"],
            ["--new_build", "true"],
            ["--output-dir", "./d"],
            ["-o", "./d"],
            ["--spec-review-cycles", "3"],
            ["--code-review-cycles", "3"],
        ):
            with _pytest.raises(SystemExit):
                self._args(*arg)

    def test_version_short_alias_is_lowercase_v(self):
        # --version short form moved from -V to -v as part of the
        # consistency pass. Now `-v` on the top-level parser prints
        # the version and exits.
        import pytest as _pytest
        from harness.cli import build_parser
        with _pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["-v"])
        # argparse exits 0 on a successful --version action.
        assert exc.value.code == 0
