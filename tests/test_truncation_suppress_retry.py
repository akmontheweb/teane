"""A length-truncation with no content must be retried with reasoning OFF.

Both lumina 01a1de12 and 01a21907 died on the run's FIRST call this way:
``finish_reason=length`` with ~120k reasoning chars and zero content, i.e. the
model spent its whole ``max_tokens`` budget deliberating. The gateway correctly
refused to repeat the identical call — and then "routed to fallback", which is
the role's OWN primary for every role, so there was nowhere to go and the run
ended at HITL with no TTY.

``max_tokens`` is clamped to 32768 in the gateway, so more budget is not
purchasable. Suppressing reasoning is the only headroom there is.

These tests drive the real ``Gateway.dispatch`` against a recording provider
double and assert on the kwargs the provider actually received.
"""

from __future__ import annotations

import pytest

from harness.gateway import (
    EmptyLLMResponseError,
    Gateway,
    GatewayConfig,
    LLMResponse,
    ModelSpec,
    NodeRole,
    TokenUsage,
    register_model,
)

MODEL = "stub:reasoner"


def _spec(*, suppress: bool) -> ModelSpec:
    return ModelSpec(
        provider="stub", model_id="reasoner", context_window=128_000,
        input_cost_per_1m=0.5, output_cost_per_1m=1.0,
        api_base_url="", api_key="x",
        suppress_reasoning_when_off=suppress,
    )


def _truncated(reasoning_chars: int = 120_000) -> LLMResponse:
    return LLMResponse(
        content="",
        usage=TokenUsage(input_tokens=50_000, output_tokens=32_768, cost_usd=0.066),
        model=MODEL,
        finish_reason="length",
        reasoning_content="x" * reasoning_chars,
    )


def _recovered() -> LLMResponse:
    return LLMResponse(
        content="",
        usage=TokenUsage(input_tokens=50_000, output_tokens=13_166, cost_usd=0.028),
        model=MODEL,
        finish_reason="tool_calls",
        tool_calls=[{"name": "submit_decomposition", "input": {"ok": True},
                     "id": "call_1"}],
    )


class _Recorder:
    """Provider double that records the kwargs of every call."""

    def __init__(self, scripted, *, suppress: bool):
        self._scripted = list(scripted)
        self._i = 0
        self.kwargs: list[dict] = []
        self.spec = _spec(suppress=suppress)
        self.api_key = "x"

    async def chat_completion(self, **kw):
        self.kwargs.append(kw)
        out = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return out

    async def close(self):
        return None

    @property
    def calls(self) -> int:
        return len(self.kwargs)

    @property
    def thinking_flags(self) -> list:
        return [k.get("thinking") for k in self.kwargs]


def _gateway(rec: _Recorder, *, suppress_retries: int = 1) -> Gateway:
    register_model(MODEL, rec.spec)
    cfg = GatewayConfig(
        planning_primary=MODEL,
        planning_fallback=MODEL,
        planning_mode="thinking",
        empty_truncation_suppress_retries=suppress_retries,
        max_retries=0,
    )
    gw = Gateway(cfg)
    gw._providers[MODEL] = rec  # type: ignore[index]
    return gw


@pytest.mark.asyncio
async def test_truncation_retries_with_reasoning_suppressed_and_recovers():
    rec = _Recorder([_truncated(), _recovered()], suppress=True)
    gw = _gateway(rec)
    resp, _ = await gw.dispatch(
        messages=[{"role": "user", "content": "decompose this"}],
        role=NodeRole.PLANNING, budget_remaining_usd=5.0,
    )
    assert rec.calls == 2, f"expected one suppressed retry, got {rec.calls} call(s)"
    assert rec.thinking_flags[0] is True, "first call should have been thinking-on"
    assert rec.thinking_flags[1] is False, (
        "the retry must suppress reasoning — otherwise it is the identical "
        "call the gateway already refuses to make"
    )
    assert resp.tool_calls, "the recovered response should carry the tool call"
    assert resp.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_cost_of_the_truncated_attempt_is_not_lost():
    """The provider billed the burned budget; the local accounting must see it."""
    rec = _Recorder([_truncated(), _recovered()], suppress=True)
    gw = _gateway(rec)
    _, remaining = await gw.dispatch(
        messages=[{"role": "user", "content": "x"}],
        role=NodeRole.PLANNING, budget_remaining_usd=5.0,
    )
    spent = 5.0 - remaining
    assert spent >= 0.066 + 0.028 - 1e-6, (
        f"only ${spent:.4f} deducted — the truncated attempt's cost was dropped"
    )


@pytest.mark.asyncio
async def test_no_retry_when_the_model_cannot_suppress_reasoning():
    """Without suppress_reasoning_when_off, thinking=False merely omits the
    parameter and a reasoning-by-default model keeps reasoning — so the retry
    would truncate identically. Don't pay for it."""
    rec = _Recorder([_truncated(), _truncated()], suppress=False)
    gw = _gateway(rec)
    with pytest.raises(EmptyLLMResponseError):
        await gw.dispatch(
            messages=[{"role": "user", "content": "x"}],
            role=NodeRole.PLANNING, budget_remaining_usd=5.0,
        )
    assert rec.calls == 1, f"should not have retried, made {rec.calls} calls"


@pytest.mark.asyncio
async def test_disabled_by_config():
    rec = _Recorder([_truncated(), _recovered()], suppress=True)
    gw = _gateway(rec, suppress_retries=0)
    with pytest.raises(EmptyLLMResponseError):
        await gw.dispatch(
            messages=[{"role": "user", "content": "x"}],
            role=NodeRole.PLANNING, budget_remaining_usd=5.0,
        )
    assert rec.calls == 1


@pytest.mark.asyncio
async def test_retry_is_bounded_by_the_configured_count():
    """A model that truncates again must not loop."""
    rec = _Recorder([_truncated(), _truncated(), _truncated()], suppress=True)
    gw = _gateway(rec, suppress_retries=1)
    with pytest.raises(EmptyLLMResponseError):
        await gw.dispatch(
            messages=[{"role": "user", "content": "x"}],
            role=NodeRole.PLANNING, budget_remaining_usd=5.0,
        )
    assert rec.calls == 2, f"expected exactly 1 retry, got {rec.calls - 1}"


@pytest.mark.asyncio
async def test_an_ordinary_empty_response_is_untouched_by_this_path():
    """finish_reason != 'length' is a provider null, not a budget burn — it
    must still use the ordinary empty-content retries, not suppression."""
    plain_empty = LLMResponse(
        content="", usage=TokenUsage(input_tokens=10, output_tokens=0, cost_usd=0.0),
        model=MODEL, finish_reason="stop",
    )
    rec = _Recorder([plain_empty, _recovered()], suppress=True)
    gw = _gateway(rec)
    resp, _ = await gw.dispatch(
        messages=[{"role": "user", "content": "x"}],
        role=NodeRole.PLANNING, budget_remaining_usd=5.0,
    )
    assert resp.tool_calls
    # Recovered via the ordinary retry, which does NOT change thinking.
    assert rec.thinking_flags[1] is True, (
        "a plain empty response must not silently disable reasoning"
    )
