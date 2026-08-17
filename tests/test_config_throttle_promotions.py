"""Regression tests for the 2026-07-06 config throttle promotions.

Six per-node throttles moved from hard-coded module constants to
``config.json`` — three router tripwires (``stuck_target_limit``,
``generic_no_progress_limit``, ``same_missing_dep_limit``), the
security-scan hard-ceiling multiplier (``security.hard_ceiling_multiplier``),
and the fanout worker pair (``fanout.max_concurrency`` /
``fanout.timeout_seconds``). Each promotion follows the same shape as
the existing ``node_throttle.total_hard_cap_multiplier`` path.

These tests wire config dicts through ``create_gateway_from_config`` and
assert the resolved ``GatewayConfig`` fields land at the expected
values (including default fallback + range clamping).
"""

from __future__ import annotations

from harness.gateway import (
    GatewayConfig,
    ModelSpec,
    create_gateway_from_config,
    register_model,
)


def _stub_routing() -> dict:
    """Minimal model_routing block referencing a single stub model —
    required so ``create_gateway_from_config`` doesn't fail on
    unregistered routing keys."""
    register_model("stub:throttle-check", ModelSpec(
        provider="stub", model_id="throttle", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x",
    ))
    return {
        "model_routing": {
            "planning_primary": "stub:throttle-check",
            "patching_primary": "stub:throttle-check",
            "repair_primary": "stub:throttle-check",
        },
    }


class TestRouterTripwireDefaults:
    """The GatewayConfig dataclass defaults must match the shipped
    config.json values so a config missing the new keys still runs
    with the intended behaviour (not a silent zero / disabled gate)."""

    def test_dataclass_defaults(self):
        cfg = GatewayConfig()
        assert cfg.stuck_target_limit == 3
        assert cfg.generic_no_progress_limit == 5
        assert cfg.same_missing_dep_limit == 3
        assert cfg.hard_security_ceiling_multiplier == 3
        assert cfg.fanout_max_concurrency == 8
        assert cfg.fanout_timeout_seconds == 180.0


class TestRouterTripwireConfigLoad:
    """Values under ``node_throttle`` land on ``gateway.config`` at the
    expected attribute names."""

    def test_router_tripwires_from_config(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "node_throttle": {
                "stuck_target_limit": 7,
                "generic_no_progress_limit": 10,
                "same_missing_dep_limit": 4,
            },
        })
        assert gw.config.stuck_target_limit == 7
        assert gw.config.generic_no_progress_limit == 10
        assert gw.config.same_missing_dep_limit == 4

    def test_missing_router_keys_fall_back_to_defaults(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "node_throttle": {},
        })
        assert gw.config.stuck_target_limit == 3
        assert gw.config.generic_no_progress_limit == 5
        assert gw.config.same_missing_dep_limit == 3

    def test_router_tripwires_clamp_low(self):
        # Zero / negative would silently disable the gate — clamp up.
        gw = create_gateway_from_config({
            **_stub_routing(),
            "node_throttle": {
                "stuck_target_limit": 0,
                "generic_no_progress_limit": -3,
                "same_missing_dep_limit": 0,
            },
        })
        assert gw.config.stuck_target_limit == 1
        assert gw.config.generic_no_progress_limit == 1
        assert gw.config.same_missing_dep_limit == 1

    def test_router_tripwires_clamp_high(self):
        # Absurdly large values would defeat the runaway-loop guard —
        # clamp down.
        gw = create_gateway_from_config({
            **_stub_routing(),
            "node_throttle": {
                "stuck_target_limit": 9999,
                "generic_no_progress_limit": 9999,
                "same_missing_dep_limit": 9999,
            },
        })
        assert gw.config.stuck_target_limit == 50
        assert gw.config.generic_no_progress_limit == 50
        assert gw.config.same_missing_dep_limit == 50

    def test_router_tripwires_reject_garbage_types(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "node_throttle": {
                "stuck_target_limit": "not an int",
            },
        })
        # Garbage falls back to the code default rather than crashing.
        assert gw.config.stuck_target_limit == 3


class TestSecurityCeilingConfigLoad:
    """The security-scan hard-ceiling multiplier lives under the
    ``security`` block (not ``node_throttle``) because it's scoped to
    the security-fix loop and reads alongside the other security
    policy knobs."""

    def test_security_ceiling_from_config(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "security": {"hard_ceiling_multiplier": 5},
        })
        assert gw.config.hard_security_ceiling_multiplier == 5

    def test_missing_security_ceiling_falls_back(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "security": {},
        })
        assert gw.config.hard_security_ceiling_multiplier == 3

    def test_security_ceiling_clamp_range(self):
        # Floor 1 (immediate escalation) / ceiling 20 (runaway risk).
        low = create_gateway_from_config({
            **_stub_routing(),
            "security": {"hard_ceiling_multiplier": 0},
        })
        assert low.config.hard_security_ceiling_multiplier == 1

        high = create_gateway_from_config({
            **_stub_routing(),
            "security": {"hard_ceiling_multiplier": 100},
        })
        assert high.config.hard_security_ceiling_multiplier == 20


class TestFanoutConfigLoad:
    """The fanout section is new (not aliased from an existing block)
    so both keys need explicit coverage — a missing ``fanout`` block
    must still yield the shipped defaults, not zero (disabled)."""

    def test_fanout_from_config(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "fanout": {"max_concurrency": 16, "timeout_seconds": 300},
        })
        assert gw.config.fanout_max_concurrency == 16
        assert gw.config.fanout_timeout_seconds == 300.0

    def test_missing_fanout_block_falls_back(self):
        gw = create_gateway_from_config(_stub_routing())
        assert gw.config.fanout_max_concurrency == 8
        assert gw.config.fanout_timeout_seconds == 180.0

    def test_fanout_concurrency_clamp(self):
        low = create_gateway_from_config({
            **_stub_routing(),
            "fanout": {"max_concurrency": 0},
        })
        assert low.config.fanout_max_concurrency == 1

        high = create_gateway_from_config({
            **_stub_routing(),
            "fanout": {"max_concurrency": 9999},
        })
        assert high.config.fanout_max_concurrency == 64

    def test_fanout_timeout_clamp(self):
        low = create_gateway_from_config({
            **_stub_routing(),
            "fanout": {"timeout_seconds": 0.1},
        })
        assert low.config.fanout_timeout_seconds == 1.0

        high = create_gateway_from_config({
            **_stub_routing(),
            "fanout": {"timeout_seconds": 100_000},
        })
        assert high.config.fanout_timeout_seconds == 3600.0


class TestFanoutRuntimeResolution:
    """The fanout module resolves ``fanout_max_concurrency`` /
    ``fanout_timeout_seconds`` LAZILY from the process-wide gateway
    config, so an operator can edit ``config.json`` mid-run and the
    next call picks up the new value. Verify the resolver falls back
    cleanly when no gateway is registered."""

    def test_resolvers_fall_back_when_gateway_absent(self):
        from harness.fanout import (
            _default_max_concurrency,
            _default_timeout_seconds,
            _DEFAULT_MAX_CONCURRENCY,
            _DEFAULT_TIMEOUT_SECONDS,
        )
        from harness.graph import set_gateway_config

        set_gateway_config(None)
        try:
            assert _default_max_concurrency() == _DEFAULT_MAX_CONCURRENCY
            assert _default_timeout_seconds() == _DEFAULT_TIMEOUT_SECONDS
        finally:
            set_gateway_config(None)


class TestBudgetCapPromotions:
    """2026-08-17 promotion: every hardcoded budget cap / threshold moved to
    ``config.json`` ``token_budget`` (and ``token_budget.gates``) so operators
    control them. Same shape as the throttle promotions above."""

    def test_dataclass_defaults(self):
        from harness.gateway import (
            DEFAULT_HARD_CAP_USD, DEFAULT_SYNTHESIS_ENVELOPE_USD,
            DEFAULT_INSTALLATION_DOC_FLOOR_USD, DEFAULT_FANOUT_BUDGET_USD,
            DEFAULT_FORCE_LOCAL_BELOW_USD, DEFAULT_REFUSE_EST_COST_FRACTION,
            DEFAULT_LOW_BUDGET_SKIP_USD, DEFAULT_MIN_CALL_BUDGET_USD,
        )
        cfg = GatewayConfig()
        assert cfg.default_hard_cap_usd == DEFAULT_HARD_CAP_USD
        assert cfg.synthesis_envelope_usd == DEFAULT_SYNTHESIS_ENVELOPE_USD
        assert cfg.installation_doc_floor_usd == DEFAULT_INSTALLATION_DOC_FLOOR_USD
        assert cfg.fanout_default_budget_usd == DEFAULT_FANOUT_BUDGET_USD
        assert cfg.force_local_below_usd == DEFAULT_FORCE_LOCAL_BELOW_USD
        assert cfg.refuse_if_est_cost_fraction_over == DEFAULT_REFUSE_EST_COST_FRACTION
        assert cfg.low_budget_skip_usd == DEFAULT_LOW_BUDGET_SKIP_USD
        assert cfg.min_call_budget_usd == DEFAULT_MIN_CALL_BUDGET_USD

    def test_budget_caps_from_config(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "token_budget": {
                "hard_cap_usd": 25.0,
                "default_hard_cap_usd": 3.0,
                "synthesis_envelope_usd": 4.0,
                "installation_doc_floor_usd": 1.5,
                "fanout_default_budget_usd": 2.5,
                "gates": {
                    "force_local_below_usd": 0.02,
                    "refuse_if_est_cost_fraction_over": 0.9,
                    "low_budget_skip_usd": 0.20,
                    "min_call_budget_usd": 0.03,
                },
            },
        })
        assert gw.config.hard_cap_usd == 25.0
        assert gw.config.default_hard_cap_usd == 3.0
        assert gw.config.synthesis_envelope_usd == 4.0
        assert gw.config.installation_doc_floor_usd == 1.5
        assert gw.config.fanout_default_budget_usd == 2.5
        assert gw.config.force_local_below_usd == 0.02
        assert gw.config.refuse_if_est_cost_fraction_over == 0.9
        assert gw.config.low_budget_skip_usd == 0.20
        assert gw.config.min_call_budget_usd == 0.03

    def test_missing_budget_keys_fall_back_to_defaults(self):
        from harness.gateway import (
            DEFAULT_SYNTHESIS_ENVELOPE_USD, DEFAULT_LOW_BUDGET_SKIP_USD,
        )
        gw = create_gateway_from_config({
            **_stub_routing(),
            "token_budget": {"hard_cap_usd": 10.0},
        })
        assert gw.config.hard_cap_usd == 10.0
        assert gw.config.synthesis_envelope_usd == DEFAULT_SYNTHESIS_ENVELOPE_USD
        assert gw.config.low_budget_skip_usd == DEFAULT_LOW_BUDGET_SKIP_USD

    def test_refuse_fraction_clamps_to_unit_interval(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "token_budget": {"gates": {"refuse_if_est_cost_fraction_over": 5.0}},
        })
        assert gw.config.refuse_if_est_cost_fraction_over == 1.0

    def test_resolve_hard_cap_usd_precedence(self):
        from harness.gateway import resolve_hard_cap_usd, DEFAULT_HARD_CAP_USD
        # hard_cap_usd wins
        assert resolve_hard_cap_usd({"hard_cap_usd": 12.0}) == 12.0
        # falls back to default_hard_cap_usd when hard_cap_usd absent
        assert resolve_hard_cap_usd({"default_hard_cap_usd": 3.0}) == 3.0
        # then to the named constant when both absent
        assert resolve_hard_cap_usd({}) == DEFAULT_HARD_CAP_USD
        # garbage → constant, never crashes
        assert resolve_hard_cap_usd({"hard_cap_usd": "oops"}) == DEFAULT_HARD_CAP_USD


class TestResilienceAndTimeoutPromotions:
    """2026-08-17: provider retry/backoff (llm_dispatch) and the pytest
    per-test timeout (sandbox.test_timeout_seconds) moved from hardcoded
    literals to config."""

    def test_retry_dataclass_defaults(self):
        from harness.gateway import (
            DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY_SECONDS,
            DEFAULT_RETRY_MAX_DELAY_SECONDS, DEFAULT_RETRY_MAX_TOTAL_SECONDS,
            DEFAULT_EMPTY_CONTENT_RETRIES,
        )
        cfg = GatewayConfig()
        assert cfg.max_retries == DEFAULT_MAX_RETRIES
        assert cfg.base_delay == DEFAULT_RETRY_BASE_DELAY_SECONDS
        assert cfg.retry_max_delay_seconds == DEFAULT_RETRY_MAX_DELAY_SECONDS
        assert cfg.retry_max_total_seconds == DEFAULT_RETRY_MAX_TOTAL_SECONDS
        assert cfg.empty_content_retries == DEFAULT_EMPTY_CONTENT_RETRIES

    def test_retry_from_config(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "llm_dispatch": {
                "max_retries": 8,
                "retry_base_delay_seconds": 2.0,
                "retry_max_delay_seconds": 90.0,
                "retry_max_total_seconds": 600.0,
                "empty_content_retries": 4,
            },
        })
        assert gw.config.max_retries == 8
        assert gw.config.base_delay == 2.0
        assert gw.config.retry_max_delay_seconds == 90.0
        assert gw.config.retry_max_total_seconds == 600.0
        assert gw.config.empty_content_retries == 4

    def test_retry_clamps(self):
        gw = create_gateway_from_config({
            **_stub_routing(),
            "llm_dispatch": {"max_retries": -1, "empty_content_retries": 999},
        })
        assert gw.config.max_retries == 0        # floor
        assert gw.config.empty_content_retries == 20  # ceiling

    def test_pytest_run_uses_given_timeout(self):
        from harness.cli import _pytest_run, DEFAULT_PYTEST_TIMEOUT_SECONDS
        assert "--timeout=45" in _pytest_run(45)
        assert f"--timeout={DEFAULT_PYTEST_TIMEOUT_SECONDS}" in _pytest_run()

    def test_build_command_threads_test_timeout(self, tmp_path):
        from harness.cli import _detect_default_build_command, resolve_build_command
        (tmp_path / "main.py").write_text("print('hi')\n")
        # Helper honours an explicit timeout...
        cmd = _detect_default_build_command(str(tmp_path), test_timeout_seconds=45)
        assert cmd is not None and "--timeout=45" in cmd
        # ...and resolve_build_command reads it from config.sandbox.
        cmd2 = resolve_build_command(
            {"sandbox": {"test_timeout_seconds": 77}}, str(tmp_path),
        )
        assert "--timeout=77" in cmd2
