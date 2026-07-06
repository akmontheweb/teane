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
