"""Regression tests for the speculative rebuild (#12).

Covers:
    - ``SpeculativeConfig.normalize`` accepts known enum values, clamps
      numerics, and substitutes defaults for unknowns.
    - ``_upgrade_legacy_config`` infers the backwards-compatible defaults
      when the operator hasn't migrated to the new schema.
    - ``_trigger_met`` correctly gates engagement on the four triggers.
    - ``_build_variant_specs`` produces the right per-variant routing
      under each (diversity_mode, cost_strategy) pairing.
    - ``_seed_messages_with_style`` appends the style hint to the first
      system message and leaves others alone.
    - ``_select_winner`` honours the legacy ``first_success`` alias.
"""

from __future__ import annotations

import pytest

from harness.speculative import (
    COST_ALL_CHEAP,
    COST_CHEAP_FIRST_SEQUENTIAL,
    COST_CHEAP_PARALLEL_THEN_EXPENSIVE,
    COST_EQUAL,
    DIVERSITY_MODEL,
    DIVERSITY_PROMPT,
    DIVERSITY_TEMPERATURE,
    SALVAGE_MERGE,
    SALVAGE_NONE,
    SELECT_ALL_PASS,
    SELECT_FEWEST_CHANGES,
    SELECT_FIRST_PASS,
    SELECT_FIRST_SUCCESS,
    TRIGGER_AFTER_N_REPAIR_FAILURES,
    TRIGGER_ALWAYS,
    TRIGGER_FIRST_ATTEMPT_ONLY,
    TRIGGER_MANUAL,
    SpeculativeConfig,
    VariantResult,
    _build_variant_specs,
    _seed_messages_with_style,
    _select_winner,
    _trigger_met,
    _upgrade_legacy_config,
)


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

def test_normalize_accepts_full_schema_defaults():
    cfg = SpeculativeConfig.normalize({})
    assert cfg.enabled is False
    assert cfg.trigger == TRIGGER_AFTER_N_REPAIR_FAILURES
    assert cfg.diversity_mode == DIVERSITY_MODEL
    assert cfg.cost_strategy == COST_CHEAP_FIRST_SEQUENTIAL
    assert cfg.selection_strategy == SELECT_FIRST_PASS
    assert cfg.salvage_strategy == SALVAGE_NONE
    assert cfg.num_variants == 3
    assert cfg.max_concurrency == 3
    assert cfg.voting.n_judges == 3


def test_normalize_clamps_num_variants_into_range():
    a = SpeculativeConfig.normalize({"enabled": True, "num_variants": 99})
    b = SpeculativeConfig.normalize({"enabled": True, "num_variants": 0})
    assert a.num_variants == 10
    assert b.num_variants == 1


def test_normalize_clamps_temperature_into_range():
    cfg = SpeculativeConfig.normalize({"temperature": 5.0})
    assert cfg.temperature == 1.5
    cfg2 = SpeculativeConfig.normalize({"temperature": -1.0})
    assert cfg2.temperature == 0.0


def test_normalize_substitutes_default_on_unknown_enum(caplog):
    with caplog.at_level("WARNING"):
        cfg = SpeculativeConfig.normalize({
            "diversity_mode": "fancy-new-mode",
            "trigger": "nonsense",
            "salvage_strategy": "asdf",
        })
    assert cfg.diversity_mode == DIVERSITY_MODEL
    assert cfg.trigger == TRIGGER_AFTER_N_REPAIR_FAILURES
    assert cfg.salvage_strategy == SALVAGE_NONE
    assert any("unknown value" in r.message for r in caplog.records)


def test_normalize_voting_clamped():
    cfg = SpeculativeConfig.normalize({"voting": {"n_judges": 99}})
    assert cfg.voting.n_judges == 7
    cfg2 = SpeculativeConfig.normalize({"voting": {"n_judges": 0}})
    assert cfg2.voting.n_judges == 1


def test_normalize_propagates_models_and_styles():
    cfg = SpeculativeConfig.normalize({
        "variant_models": ["openai:gpt-4o-mini", "deepseek:chat", ""],
        "variant_prompt_styles": ["minimal-diff", "thorough"],
        "expensive_model": "anthropic:claude-sonnet-4-6",
        "cheap_model": "deepseek:chat",
    })
    assert cfg.variant_models == ["openai:gpt-4o-mini", "deepseek:chat"]
    assert cfg.variant_prompt_styles == ["minimal-diff", "thorough"]
    assert cfg.expensive_model == "anthropic:claude-sonnet-4-6"
    assert cfg.cheap_model == "deepseek:chat"


# ---------------------------------------------------------------------------
# 2. Legacy upgrader
# ---------------------------------------------------------------------------

def test_upgrade_legacy_config_disabled_no_warning(caplog):
    raw = {"enabled": False, "num_variants": 3, "temperature": 0.3}
    with caplog.at_level("WARNING"):
        out = _upgrade_legacy_config(raw)
    # Disabled configs are left alone (no implicit fields).
    assert "diversity_mode" not in out
    assert "trigger" not in out
    assert not any("legacy config" in r.message for r in caplog.records)


def test_upgrade_legacy_config_enabled_warns_and_seeds_defaults(caplog):
    raw = {
        "enabled": True,
        "num_variants": 3,
        "temperature": 0.3,
        "selection_strategy": "first_success",
    }
    with caplog.at_level("WARNING"):
        out = _upgrade_legacy_config(raw)
    assert out["diversity_mode"] == DIVERSITY_TEMPERATURE
    assert out["cost_strategy"] == COST_EQUAL
    assert out["salvage_strategy"] == SALVAGE_MERGE
    assert out["trigger"] == TRIGGER_FIRST_ATTEMPT_ONLY
    # selection_strategy was aliased.
    assert out["selection_strategy"] == SELECT_FIRST_PASS
    assert any("legacy config" in r.message for r in caplog.records)


def test_upgrade_no_op_when_new_keys_present():
    raw = {
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "selection_strategy": "fewest_changes",
    }
    out = _upgrade_legacy_config(raw)
    # selection_strategy mapping still applies (no-op for non-alias values).
    assert out["selection_strategy"] == "fewest_changes"
    # No defaults injected — operator chose to omit them.
    assert "trigger" not in out
    assert "cost_strategy" not in out


def test_upgrade_legacy_alias_first_success_normalizes_to_first_pass():
    # `first_success` is itself in SELECTION_STRATEGIES so it round-trips,
    # but _upgrade_legacy_config aliases it. Both paths must end at
    # first_pass when normalize runs after upgrade.
    raw = _upgrade_legacy_config({"selection_strategy": SELECT_FIRST_SUCCESS})
    cfg2 = SpeculativeConfig.normalize(raw)
    assert cfg2.selection_strategy == SELECT_FIRST_PASS


# ---------------------------------------------------------------------------
# 3. Trigger evaluation
# ---------------------------------------------------------------------------

def _cfg_with(trigger: str, **kw) -> SpeculativeConfig:
    return SpeculativeConfig.normalize({"enabled": True, "trigger": trigger, **kw})


def test_trigger_always_engages():
    cfg = _cfg_with(TRIGGER_ALWAYS)
    engage, _ = _trigger_met(cfg, {"loop_counter": {"patching": 5, "repair": 3}})
    assert engage is True


def test_trigger_first_attempt_only():
    cfg = _cfg_with(TRIGGER_FIRST_ATTEMPT_ONLY)
    engage, _ = _trigger_met(cfg, {"loop_counter": {"patching": 1}})
    assert engage is True
    engage2, reason2 = _trigger_met(cfg, {"loop_counter": {"patching": 2}})
    assert engage2 is False
    assert "patching_count=2" in reason2


def test_trigger_after_n_repair_failures():
    cfg = _cfg_with(TRIGGER_AFTER_N_REPAIR_FAILURES, n_repair_failures_threshold=2)
    engage, _ = _trigger_met(cfg, {"loop_counter": {"repair": 1}})
    assert engage is False
    engage2, _ = _trigger_met(cfg, {"loop_counter": {"repair": 2}})
    assert engage2 is True
    engage3, _ = _trigger_met(cfg, {"loop_counter": {"repair": 4}})
    assert engage3 is True


def test_trigger_manual_requires_state_flag():
    cfg = _cfg_with(TRIGGER_MANUAL)
    engage1, _ = _trigger_met(cfg, {})
    assert engage1 is False
    engage2, _ = _trigger_met(cfg, {"force_speculative": True})
    assert engage2 is True


# ---------------------------------------------------------------------------
# 4. Variant spec builder
# ---------------------------------------------------------------------------

def test_spec_builder_diversity_temperature_spreads_temps():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_TEMPERATURE,
        "cost_strategy": COST_EQUAL,
        "num_variants": 3,
        "temperature": 0.2,
    })
    specs = _build_variant_specs(cfg)
    assert len(specs) == 3
    # Each subsequent variant gets a higher temp.
    temps = [s.temperature for s in specs]
    assert temps[0] == pytest.approx(0.2)
    assert temps[1] == pytest.approx(0.35, abs=1e-3)
    assert temps[2] == pytest.approx(0.5, abs=1e-3)
    # No model override for temperature-only diversity.
    assert all(s.model_override is None for s in specs)


def test_spec_builder_diversity_model_routes_each_variant():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "cost_strategy": COST_EQUAL,
        "num_variants": 3,
        "variant_models": ["openai:gpt-4o-mini", "deepseek:chat", "anthropic:claude-haiku"],
    })
    specs = _build_variant_specs(cfg)
    assert [s.model_override for s in specs] == [
        "openai:gpt-4o-mini", "deepseek:chat", "anthropic:claude-haiku",
    ]


def test_spec_builder_diversity_prompt_uses_styles():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_PROMPT,
        "cost_strategy": COST_EQUAL,
        "num_variants": 3,
        "variant_prompt_styles": ["minimal-diff", "balanced", "thorough"],
    })
    specs = _build_variant_specs(cfg)
    # Style suffixes come from the built-in library.
    assert "smallest possible diff" in specs[0].system_prompt_suffix
    assert "balance correctness" in specs[1].system_prompt_suffix
    assert "thorough" in specs[2].system_prompt_suffix.lower()


def test_spec_builder_cost_cheap_first_sequential_last_is_expensive():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "cost_strategy": COST_CHEAP_FIRST_SEQUENTIAL,
        "num_variants": 3,
        "cheap_model": "deepseek:chat",
        "expensive_model": "anthropic:claude-sonnet-4-6",
        "variant_models": ["x", "y", "z"],  # ignored when cost_strategy overrides
    })
    specs = _build_variant_specs(cfg)
    # First N-1 cheap; last expensive (flagged).
    assert specs[0].model_override == "deepseek:chat"
    assert specs[1].model_override == "deepseek:chat"
    assert specs[2].model_override == "anthropic:claude-sonnet-4-6"
    assert specs[2].is_expensive is True
    assert specs[0].is_expensive is False


def test_spec_builder_cost_cheap_parallel_then_expensive_first_is_expensive():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "cost_strategy": COST_CHEAP_PARALLEL_THEN_EXPENSIVE,
        "num_variants": 3,
        "cheap_model": "deepseek:chat",
        "expensive_model": "anthropic:claude-sonnet-4-6",
    })
    specs = _build_variant_specs(cfg)
    assert specs[0].model_override == "anthropic:claude-sonnet-4-6"
    assert specs[0].is_expensive is True
    assert specs[1].model_override == "deepseek:chat"
    assert specs[2].model_override == "deepseek:chat"


def test_spec_builder_cost_all_cheap_overrides_diversity_models():
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "cost_strategy": COST_ALL_CHEAP,
        "num_variants": 3,
        "cheap_model": "deepseek:chat",
        "variant_models": ["openai:gpt-4o", "anthropic:claude-opus"],
    })
    specs = _build_variant_specs(cfg)
    assert all(s.model_override == "deepseek:chat" for s in specs)


def test_spec_builder_cost_gradient_low_to_high_consumes_variant_models_in_order():
    """The new gradient strategy walks ``variant_models`` monotonically
    (no modulo cycling) so a 3-entry list assigns model[i] to variant i
    and flags the last one as expensive."""
    from harness.speculative import COST_GRADIENT_LOW_TO_HIGH
    cfg = SpeculativeConfig.normalize({
        "enabled": True,
        "diversity_mode": DIVERSITY_MODEL,
        "cost_strategy": COST_GRADIENT_LOW_TO_HIGH,
        "num_variants": 3,
        "variant_models": ["ollama:tiny", "openai:gpt-4o-mini", "anthropic:claude-opus"],
    })
    specs = _build_variant_specs(cfg)
    assert specs[0].model_override == "ollama:tiny"
    assert specs[1].model_override == "openai:gpt-4o-mini"
    assert specs[2].model_override == "anthropic:claude-opus"
    assert specs[0].is_expensive is False
    assert specs[1].is_expensive is False
    assert specs[2].is_expensive is True


# ---------------------------------------------------------------------------
# 5. Message style seeding
# ---------------------------------------------------------------------------

def test_seed_messages_appends_style_to_first_system_block():
    from harness.speculative import _VariantSpec
    spec = _VariantSpec(
        index=0, model_override=None, temperature=0.3,
        system_prompt_suffix="Style: be terse.",
    )
    messages = [
        {"role": "system", "content": "Base system prompt."},
        {"role": "user", "content": "Fix the bug."},
    ]
    out = _seed_messages_with_style(messages, spec)
    assert out[0]["content"].endswith("Style: be terse.")
    assert "Base system prompt." in out[0]["content"]
    # User message untouched.
    assert out[1] == {"role": "user", "content": "Fix the bug."}


def test_seed_messages_empty_suffix_returns_copy():
    from harness.speculative import _VariantSpec
    spec = _VariantSpec(0, None, 0.3, "")
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = _seed_messages_with_style(messages, spec)
    assert out == messages
    # Returns a new list (not the original — protects against mutation).
    assert out is not messages


def test_seed_messages_inserts_system_when_missing():
    from harness.speculative import _VariantSpec
    spec = _VariantSpec(0, None, 0.3, "Style: thorough.")
    messages = [{"role": "user", "content": "Fix the bug."}]
    out = _seed_messages_with_style(messages, spec)
    assert out[0]["role"] == "system"
    assert "thorough" in out[0]["content"]
    assert out[1]["role"] == "user"


# ---------------------------------------------------------------------------
# 6. Selection — alias handling
# ---------------------------------------------------------------------------

def test_select_winner_legacy_alias_returns_first_pass():
    results = [
        VariantResult(index=0, variant_id="a", worktree_path="", exit_code=0),
        VariantResult(index=1, variant_id="b", worktree_path="", exit_code=0),
    ]
    winner = _select_winner(results, "first_success")
    assert winner is not None
    assert winner.index == 0


def test_select_winner_returns_none_when_no_pass():
    results = [
        VariantResult(index=0, variant_id="a", worktree_path="", exit_code=1),
        VariantResult(index=1, variant_id="b", worktree_path="", exit_code=2),
    ]
    assert _select_winner(results, SELECT_FIRST_PASS) is None


def test_select_winner_all_pass_requires_full_pass():
    one = VariantResult(index=0, variant_id="a", worktree_path="", exit_code=0)
    two = VariantResult(index=1, variant_id="b", worktree_path="", exit_code=1)
    assert _select_winner([one, two], SELECT_ALL_PASS) is None
    one2 = VariantResult(index=0, variant_id="a", worktree_path="", exit_code=0)
    two2 = VariantResult(index=1, variant_id="b", worktree_path="", exit_code=0)
    assert _select_winner([one2, two2], SELECT_ALL_PASS) is one2


def test_select_winner_fewest_changes_takes_smaller_diff():
    from harness.patcher import PatchResult
    a = VariantResult(index=0, variant_id="a", worktree_path="", exit_code=0)
    from harness.patcher import OperationType as _Op
    a.patch_results = [PatchResult(success=True, file="x", operation=_Op.CREATE_FILE, lines_changed=50)]
    b = VariantResult(index=1, variant_id="b", worktree_path="", exit_code=0)
    b.patch_results = [PatchResult(success=True, file="x", operation=_Op.CREATE_FILE, lines_changed=10)]
    winner = _select_winner([a, b], SELECT_FEWEST_CHANGES)
    assert winner is b
