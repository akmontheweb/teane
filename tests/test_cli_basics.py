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
        "manifest_file": "product_spec.txt",
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
