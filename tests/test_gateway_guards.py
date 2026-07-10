"""Regression tests for P1 gateway guards: EmptyLLMResponseError (P1.5)
and BudgetTooLowError (P1.4).

These tests stub the provider layer so the gateway's behaviour around an
empty content body and a pre-flight cost estimate can be exercised without
a real network call.
"""

from __future__ import annotations

import pytest

from harness.gateway import (
    BudgetTooLowError,
    EmptyLLMResponseError,
    Gateway,
    GatewayConfig,
    LLMResponse,
    ModelSpec,
    NodeRole,
    TokenUsage,
    register_model,
)


def _stub_model(model_key: str = "stub:fast") -> str:
    """Register a cheap stub model in the gateway registry. Idempotent."""
    register_model(model_key, ModelSpec(
        provider="stub",
        model_id="fast",
        context_window=128_000,
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.0,
        api_base_url="",
        api_key="x",  # non-ollama needs *some* key for select-model paths
    ))
    return model_key


class _StubProvider:
    """Minimal provider double whose chat_completion returns a scripted
    sequence of LLMResponse objects (advances on each call)."""

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self._idx = 0
        self.calls = 0
        # Mirror the BaseLLM interface bits the gateway actually pokes at.
        self.spec = ModelSpec(
            provider="stub", model_id="fast", context_window=128_000,
            input_cost_per_1m=0.5, output_cost_per_1m=1.0,
            api_base_url="", api_key="x",
        )
        # The gateway treats `provider.api_key` as truthy when checking the
        # smart-key-resolution path; mirror BaseLLM here so dispatch flows.
        self.api_key = "x"

    async def chat_completion(self, **_kwargs):
        self.calls += 1
        if self._idx >= len(self._responses):
            return self._responses[-1]
        out = self._responses[self._idx]
        self._idx += 1
        return out

    async def close(self):
        return None


def _make_gateway_with_stub_provider(stub: _StubProvider, model_key: str) -> Gateway:
    """Pin the gateway to a stub provider for `model_key`."""
    _stub_model(model_key)
    cfg = GatewayConfig(
        planning_primary=model_key,
        patching_primary=model_key,
        repair_primary=model_key,
    )
    gateway = Gateway(cfg)
    gateway._providers[model_key] = stub  # type: ignore[index]
    return gateway


@pytest.mark.asyncio
async def test_empty_llm_response_raises_after_retries():
    """P1.5: persistent empty content must raise EmptyLLMResponseError
    instead of silently returning, so the repair/HITL router can route to
    a clear operator message rather than waste retry budget."""
    empty_usage = TokenUsage(input_tokens=10, output_tokens=0, model_name="stub:fast", cost_usd=0.0)
    stub = _StubProvider([
        LLMResponse(content="", usage=empty_usage, model="stub:fast"),
        LLMResponse(content="", usage=empty_usage, model="stub:fast"),
        LLMResponse(content="", usage=empty_usage, model="stub:fast"),
        LLMResponse(content="", usage=empty_usage, model="stub:fast"),
    ])
    gateway = _make_gateway_with_stub_provider(stub, "stub:fast")
    with pytest.raises(EmptyLLMResponseError):
        await gateway.dispatch(
            messages=[
                {"role": "system", "content": "you are a test"},
                {"role": "user", "content": "do the thing"},
            ],
            role=NodeRole.PATCHING,
            budget_remaining_usd=1.0,
        )
    # 1 initial + 2 empty-retry attempts = 3
    assert stub.calls >= 3, "empty-retry loop must fire before raising"


@pytest.mark.asyncio
async def test_empty_then_recovers_succeeds():
    """One empty response followed by real content should NOT raise — the
    empty-retry loop is expected to recover gracefully when the provider
    blinks for a single call."""
    usage = TokenUsage(input_tokens=10, output_tokens=20, model_name="stub:fast", cost_usd=0.0001)
    stub = _StubProvider([
        LLMResponse(content="", usage=TokenUsage(model_name="stub:fast"), model="stub:fast"),
        LLMResponse(content="hello world", usage=usage, model="stub:fast"),
    ])
    gateway = _make_gateway_with_stub_provider(stub, "stub:fast")
    response, new_budget = await gateway.dispatch(
        messages=[
            {"role": "system", "content": "stub"},
            {"role": "user", "content": "hi"},
        ],
        role=NodeRole.PATCHING,
        budget_remaining_usd=1.0,
    )
    assert response.content == "hello world"
    assert new_budget == pytest.approx(1.0 - 0.0001, rel=1e-6)
    assert stub.calls == 2


def test_rate_limit_circuit_breaker_opens_after_threshold():
    """P1.9: after the configured number of 429/503 failures inside the
    rolling window, _circuit_is_open() returns True so dispatch can divert
    to a local fallback instead of burning retries against a broken provider."""
    cfg = GatewayConfig(
        planning_primary="stub:fast",
        patching_primary="stub:fast",
        repair_primary="stub:fast",
    )
    gateway = Gateway(cfg)
    assert gateway._circuit_is_open() is False
    # Default threshold is 3; record 2 then check (still closed), then a
    # third should open the circuit on the next check.
    gateway._record_rate_limit_failure()
    gateway._record_rate_limit_failure()
    assert gateway._circuit_is_open() is False
    gateway._record_rate_limit_failure()
    assert gateway._circuit_is_open() is True


@pytest.mark.asyncio
async def test_preflight_budget_refuses_oversized_call():
    """P1.4: the pre-flight estimator must refuse a call whose projected
    cost exceeds the remaining budget, instead of letting it dispatch and
    overspending the hard cap by its own cost."""
    # Register an EXPENSIVE model so a payload that fits the context window
    # still has a projected cost greater than our tight budget. The default
    # 128k context lets a ~80k-token payload pass the context guard while
    # the high price rates push the estimate above $0.10.
    register_model("stub:premium", ModelSpec(
        provider="stub", model_id="premium", context_window=128_000,
        input_cost_per_1m=10.0, output_cost_per_1m=30.0,
        api_base_url="", api_key="x",
    ))
    big_message = "x" * 320_000  # ~80_000 tokens at chars/4; safely under 128k
    stub = _StubProvider([
        LLMResponse(
            content="should never be returned",
            usage=TokenUsage(model_name="stub:premium"),
            model="stub:premium",
        ),
    ])
    cfg = GatewayConfig(
        planning_primary="stub:premium",
        patching_primary="stub:premium",
        repair_primary="stub:premium",
    )
    gateway = Gateway(cfg)
    gateway._providers["stub:premium"] = stub  # type: ignore[index]
    # Override the stub provider's spec so the gateway picks up the
    # expensive rates when estimating cost.
    stub.spec = ModelSpec(
        provider="stub", model_id="premium", context_window=128_000,
        input_cost_per_1m=10.0, output_cost_per_1m=30.0,
        api_base_url="", api_key="x",
    )
    with pytest.raises(BudgetTooLowError):
        await gateway.dispatch(
            messages=[
                {"role": "system", "content": "stub"},
                {"role": "user", "content": big_message},
            ],
            role=NodeRole.PATCHING,
            # Tight budget — large enough to avoid the low-budget Ollama
            # fallback ($0.05 trigger), small enough that the projected
            # ~$0.80 input + ~$0.12 output cost blows past it.
            budget_remaining_usd=0.10,
        )
    # Critical: the provider must NEVER have been called.
    assert stub.calls == 0


@pytest.mark.asyncio
async def test_session_tracker_accumulates_across_dispatches():
    """Every successful dispatch must land in ``gateway.session_tracker``
    — this is what the end-of-run "Token Cost" summary now reads, so a
    caller that forgets to call ``aggregate_tokens(state[...], ...)``
    can no longer drop costs from the displayed total."""
    usage = TokenUsage(
        input_tokens=10, output_tokens=20,
        model_name="stub:fast", cost_usd=0.0005,
    )
    stub = _StubProvider([
        LLMResponse(content="one", usage=usage, model="stub:fast"),
        LLMResponse(content="two", usage=usage, model="stub:fast"),
        LLMResponse(content="three", usage=usage, model="stub:fast"),
    ])
    gateway = _make_gateway_with_stub_provider(stub, "stub:fast")
    assert gateway.session_tracker.get("total_cost_usd", 0.0) == 0.0
    for _ in range(3):
        await gateway.dispatch(
            messages=[
                {"role": "system", "content": "stub"},
                {"role": "user", "content": "go"},
            ],
            role=NodeRole.PATCHING,
            budget_remaining_usd=1.0,
        )
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(
        0.0015, rel=1e-6
    )
    assert gateway.session_tracker["total_input_tokens"] == 30
    assert gateway.session_tracker["total_output_tokens"] == 60
    # session_cost_summary returns a defensive copy.
    snap = gateway.session_cost_summary()
    snap["total_cost_usd"] = 999.0
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(
        0.0015, rel=1e-6
    )


@pytest.mark.asyncio
async def test_session_tracker_reflects_empty_retry_tail():
    """When the provider blinks with empty content on the first attempt
    and the empty-retry loop recovers, the session tracker must reflect
    the BILLED total across both attempts — not just the last call's
    cost. The empty attempt was charged server-side; dropping it would
    undercount the session total exactly the way audit §4.4 warns
    against for the budget."""
    empty_usage = TokenUsage(
        input_tokens=5, output_tokens=0,
        model_name="stub:fast", cost_usd=0.0002,
    )
    real_usage = TokenUsage(
        input_tokens=10, output_tokens=20,
        model_name="stub:fast", cost_usd=0.0007,
    )
    stub = _StubProvider([
        LLMResponse(content="", usage=empty_usage, model="stub:fast"),
        LLMResponse(content="recovered", usage=real_usage, model="stub:fast"),
    ])
    gateway = _make_gateway_with_stub_provider(stub, "stub:fast")
    response, new_budget = await gateway.dispatch(
        messages=[
            {"role": "system", "content": "stub"},
            {"role": "user", "content": "go"},
        ],
        role=NodeRole.PATCHING,
        budget_remaining_usd=1.0,
    )
    assert response.content == "recovered"
    # Budget deduction was already accumulated across the empty tail.
    assert new_budget == pytest.approx(1.0 - 0.0009, rel=1e-6)
    # Session tracker must agree — this is the dual-display fix.
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(
        0.0009, rel=1e-6
    )
    # And the response.usage.cost_usd carries the BILLED amount so
    # downstream consumers (state-mirror aggregators, log lines,
    # debug dumps) see the same truthful number.
    assert response.usage.cost_usd == pytest.approx(0.0009, rel=1e-6)


class TestOllamaLocalKeyNormalization:
    """Bug (2026-07-10): session 44c5e194's post-resume run crashed
    inside ``_get_provider`` with ``HarnessConfigError: Model
    'ollama:ollama:qwen2.5-coder:14b' is not registered.`` The config
    already stored ``ollama_local_model = "ollama:qwen2.5-coder:14b"``
    (matching the shape of every other ``model_routing.*_primary``
    value) and the code naively prepended ``ollama:`` on top,
    producing the double-prefixed key. Both call sites now go
    through ``_normalized_ollama_local_key`` which strips a leading
    ``ollama:`` before adding the canonical one.
    """

    def _make_gateway(self, ollama_local_model: str) -> Gateway:
        cfg = GatewayConfig(
            planning_primary="stub:fast",
            patching_primary="stub:fast",
            repair_primary="stub:fast",
            ollama_local_model=ollama_local_model,
        )
        return Gateway(cfg)

    def test_bare_config_value_gets_one_prefix(self):
        # Original / documented shape: bare model id, no provider prefix.
        gw = self._make_gateway("qwen2.5-coder:14b")
        assert gw._normalized_ollama_local_key() == "ollama:qwen2.5-coder:14b"

    def test_prefixed_config_value_does_not_double_up(self):
        # Widely-seen operator shape in the wild (mirrors *_primary keys).
        # Must NOT become "ollama:ollama:qwen2.5-coder:14b".
        gw = self._make_gateway("ollama:qwen2.5-coder:14b")
        assert gw._normalized_ollama_local_key() == "ollama:qwen2.5-coder:14b"

    def test_select_model_force_local_uses_normalized_key(self):
        # select_model(force_local=True) is the first of two crash sites.
        gw = self._make_gateway("ollama:qwen2.5-coder:14b")
        key = gw.select_model(NodeRole.PATCHING, force_local=True)
        assert key == "ollama:qwen2.5-coder:14b"
        assert not key.startswith("ollama:ollama:")

    def test_select_model_bare_value_still_produces_valid_key(self):
        gw = self._make_gateway("qwen2.5-coder:14b")
        key = gw.select_model(NodeRole.PATCHING, force_local=True)
        assert key == "ollama:qwen2.5-coder:14b"

    def test_whitespace_around_prefix_tolerated(self):
        # Prefixed then space accidentally landed. Also normalize.
        gw = self._make_gateway("ollama:  qwen2.5-coder:14b  ")
        assert gw._normalized_ollama_local_key() == "ollama:qwen2.5-coder:14b"

    def test_empty_config_value_produces_bare_prefix(self):
        # Preserves the pre-fix semantic ("ollama:") so the outer guard
        # in dispatch (which checks for empty ollama_local_model) still
        # short-circuits before we ever try to look this key up.
        gw = self._make_gateway("")
        assert gw._normalized_ollama_local_key() == "ollama:"
