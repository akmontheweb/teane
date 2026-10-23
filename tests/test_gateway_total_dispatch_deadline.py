"""The end-to-end ceiling on one dispatch attempt.

Regression cover for lumina 01a16872, which sat in a single HTTP read for
68 minutes (20:24:18 -> 21:32:50) with ``request_timeout_seconds=600`` in
force, then succeeded on retry in two seconds. ``httpx.Timeout(read=...)``
bounds the gap between consecutive bytes, not the call, so a provider that
trickles keep-alive bytes resets that clock forever.

Every test here EXECUTES the deadline path against a provider that hangs.
A test that only asserted on source text would have passed against the
broken code, which is exactly how the in-turn retry shipped with a
TypeError that fired eleven times before anyone noticed.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from harness.gateway import (
    DEFAULT_TOTAL_REQUEST_TIMEOUT_SECONDS,
    GatewayConfig,
    NodeRole,
)


class _StubGateway:
    """Just enough Gateway to exercise the resolver and the bounded call."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config

    _total_timeout_for = None  # bound below


# Bind the real methods so we test the shipped implementations, not copies.
from harness.gateway import Gateway  # noqa: E402

_StubGateway._total_timeout_for = Gateway._total_timeout_for
_StubGateway._request_timeout_for = Gateway._request_timeout_for


def _gw(**kw) -> _StubGateway:
    return _StubGateway(GatewayConfig(**kw))


def _loaded(llm_dispatch: dict) -> GatewayConfig:
    """Config as the REAL loader builds it, clamps and all."""
    from harness.gateway import create_gateway_from_config

    return create_gateway_from_config(
        {"llm_dispatch": llm_dispatch}
    ).config


class TestTheResolver:
    def test_it_defaults_to_the_module_constant(self):
        gw = _gw()
        assert (
            gw._total_timeout_for(NodeRole.REPAIR)
            == DEFAULT_TOTAL_REQUEST_TIMEOUT_SECONDS
        )

    def test_a_per_role_entry_wins(self):
        gw = _gw(
            total_request_timeout_seconds=1200.0,
            total_request_timeout_per_role={"repair": 60.0},
        )
        assert gw._total_timeout_for(NodeRole.REPAIR) == 60.0

    def test_a_role_absent_from_the_map_inherits_the_default(self):
        gw = _gw(
            total_request_timeout_seconds=1200.0,
            total_request_timeout_per_role={"repair": 60.0},
        )
        assert gw._total_timeout_for(NodeRole.PATCHING) == 1200.0

    def test_it_is_a_different_knob_from_the_httpx_read_timeout(self):
        """The whole point: the two must be independently settable, because
        one is a between-bytes clock and the other is a deadline."""
        gw = _gw(
            request_timeout_seconds=600.0,
            total_request_timeout_seconds=90.0,
        )
        assert gw._total_timeout_for(NodeRole.REPAIR) == 90.0
        assert gw._request_timeout_for(NodeRole.REPAIR).read == 600.0


class TestTheDeadlineActuallyFires:
    """The behaviour the 68-minute stall needed and did not have."""

    @staticmethod
    async def _bounded(deadline: float, hang_for: float):
        """Mirror of ``_call._bounded``: run under a deadline, convert the
        expiry to ReadTimeout so existing timeout paths handle it."""
        async def _chat():
            await asyncio.sleep(hang_for)
            return "response"

        try:
            async with asyncio.timeout(deadline):
                return await _chat()
        except TimeoutError as exc:
            raise httpx.ReadTimeout(
                f"total dispatch deadline of {deadline:.0f}s exceeded"
            ) from exc

    def test_a_hanging_read_raises_instead_of_hanging_forever(self):
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._bounded(deadline=0.05, hang_for=30.0))

    def test_it_gives_up_at_the_deadline_not_at_the_hang_length(self):
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._bounded(deadline=0.05, hang_for=30.0))
        assert time.monotonic() - started < 5.0, (
            "the deadline did not bound the call — this is the 68-minute "
            "stall reproduced in miniature"
        )

    def test_the_error_is_a_timeout_the_retry_path_recognises(self):
        """retry_with_backoff and the dispatch except clause both key off
        httpx.TimeoutException. A bare TimeoutError would escape them and
        crash the run instead of retrying -- and the real retry succeeded in
        two seconds, so losing it would be the expensive bug."""
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(self._bounded(deadline=0.05, hang_for=30.0))

    def test_the_message_names_the_deadline(self):
        with pytest.raises(httpx.ReadTimeout, match=r"deadline of 1s"):
            asyncio.run(self._bounded(deadline=1.0, hang_for=30.0))

    def test_a_call_that_finishes_in_time_is_untouched(self):
        got = asyncio.run(self._bounded(deadline=30.0, hang_for=0.0))
        assert got == "response"


class TestTheConfigLoader:
    def test_the_key_is_read_from_llm_dispatch(self):
        cfg = _loaded({"total_request_timeout_seconds": 300.0})
        assert cfg.total_request_timeout_seconds == 300.0

    def test_a_missing_key_falls_back_to_the_default(self):
        cfg = _loaded({})
        assert (
            cfg.total_request_timeout_seconds
            == DEFAULT_TOTAL_REQUEST_TIMEOUT_SECONDS
        )

    def test_a_nonsense_value_is_clamped_not_obeyed(self):
        """A zero or negative deadline would abort every dispatch
        instantly; the clamp floor is what keeps a typo from bricking a
        run."""
        cfg = _loaded({"total_request_timeout_seconds": 0})
        assert cfg.total_request_timeout_seconds >= 1.0

    def test_per_role_entries_load_and_drop_non_positive_ones(self):
        cfg = _loaded({
            "total_request_timeout_per_role": {
                "repair": 900, "patching": 0, "planning": -5,
            },
        })
        assert cfg.total_request_timeout_per_role == {"repair": 900.0}


class TestTheShippedConfig:
    def test_config_json_sets_the_key(self):
        import json
        import pathlib

        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent
             / "config" / "config.json").read_text()
        )
        val = cfg["llm_dispatch"]["total_request_timeout_seconds"]
        assert val > 0

    def test_it_leaves_headroom_over_the_longest_real_call(self):
        """252s was the longest legitimate call on lumina 01a16872. A
        deadline near that turns working calls into retries, which costs
        double rather than saving anything."""
        import json
        import pathlib

        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent
             / "config" / "config.json").read_text()
        )
        assert cfg["llm_dispatch"]["total_request_timeout_seconds"] >= 600

    def test_config_help_documents_both_new_keys(self):
        import pathlib

        help_md = (
            pathlib.Path(__file__).parent.parent
            / "config" / "config_help.md"
        ).read_text()
        assert "total_request_timeout_seconds" in help_md
        assert "total_request_timeout_per_role" in help_md

    def test_config_help_no_longer_calls_the_read_timeout_an_overall_one(self):
        """The old wording ("read/overall") is the misconception that let a
        68-minute stall look configured-for."""
        import pathlib

        help_md = (
            pathlib.Path(__file__).parent.parent
            / "config" / "config_help.md"
        ).read_text()
        assert "read/overall HTTP timeout" not in help_md


class TestTheRealDispatchPathIsBounded:
    """Drives ``Gateway.dispatch`` against a provider that HANGS.

    The mirror tests above prove the pattern works; this one proves the
    shipped ``_call`` actually uses it. Both are needed: 33 tests passed
    against the in-turn retry while it raised a TypeError on every firing,
    because not one of them executed the real code.
    """

    @staticmethod
    def _hanging_gateway(deadline: float):
        from harness.gateway import (
            Gateway,
            GatewayConfig,
            ModelSpec,
            register_model,
        )

        key = "stub:hangs"
        register_model(key, ModelSpec(
            provider="stub", model_id="hangs", context_window=128_000,
            input_cost_per_1m=0.5, output_cost_per_1m=1.0,
            api_base_url="", api_key="x",
        ))

        class _Hangs:
            def __init__(self) -> None:
                self.calls = 0
                self.spec = ModelSpec(
                    provider="stub", model_id="hangs",
                    context_window=128_000, input_cost_per_1m=0.5,
                    output_cost_per_1m=1.0, api_base_url="", api_key="x",
                )
                self.api_key = "x"

            async def chat_completion(self, **_kw):
                # The lumina shape: headers arrived, the body never did.
                self.calls += 1
                await asyncio.sleep(3600)

            async def close(self):
                return None

        provider = _Hangs()
        gw = Gateway(GatewayConfig(
            planning_primary=key, patching_primary=key, repair_primary=key,
            total_request_timeout_seconds=deadline,
            max_retries=0,
            retry_network_cooloff_seconds=0.0,
            network_cooloff_attempts=0,
            fallback_on_timeout=False,
        ))
        gw._providers[key] = provider  # type: ignore[index]
        return gw, provider

    @pytest.mark.asyncio
    async def test_a_hanging_provider_does_not_hang_the_dispatch(self):
        gw, provider = self._hanging_gateway(deadline=0.1)
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 — any raise beats hanging
            await asyncio.wait_for(
                gw.dispatch(
                    messages=[
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                    ],
                    role=NodeRole.REPAIR,
                    budget_remaining_usd=10.0,
                ),
                timeout=30.0,
            )
        elapsed = time.monotonic() - started
        assert provider.calls >= 1, "the provider was never called"
        assert elapsed < 25.0, (
            f"dispatch took {elapsed:.1f}s against a 0.1s deadline — the "
            "shipped path is NOT bounded, which is the 68-minute stall"
        )
