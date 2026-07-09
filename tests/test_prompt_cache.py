"""Regression tests for the prompt-caching slice.

Covers:
    - ``GatewayConfig.prompt_cache_enabled`` default + config wiring.
    - ``AnthropicProvider`` emits ``cache_control`` markers on the system
      block (and on a substantial first user message) when caching is
      enabled.
    - The Anthropic provider falls back to the legacy string-form system
      payload when caching is disabled or the model declares
      ``supports_cache=False``.
    - ``hash_stable_prefix`` is deterministic and changes when any of
      the first N messages mutate.
    - The gateway detects prefix drift across consecutive dispatches and
      emits the ``cache_prefix_drift`` observability event.
"""

from __future__ import annotations

from typing import Any

import pytest

from harness.gateway import (
    AnthropicProvider,
    Gateway,
    GatewayConfig,
    LLMResponse,
    ModelSpec,
    NodeRole,
    TokenUsage,
    create_gateway_from_config,
    hash_stable_prefix,
    register_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubHttpResponse:
    """Mimics the shape ``httpx.Response`` exposes to the providers."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        # process_llm_patch_output / retry_with_backoff inspect these
        self.status_code = 200
        self.request = None
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _RecordingAnthropicClient:
    """Captures the request payload the AnthropicProvider POSTs."""

    def __init__(self, response_payload: dict[str, Any]):
        self.last_payload: dict[str, Any] | None = None
        self._response_payload = response_payload

    async def post(self, _path: str, json: dict[str, Any]) -> _StubHttpResponse:
        self.last_payload = json
        return _StubHttpResponse(self._response_payload)


def _make_anthropic_provider(
    *,
    supports_cache: bool = True,
    prompt_cache_enabled: bool = True,
) -> tuple[AnthropicProvider, _RecordingAnthropicClient]:
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-sonnet-test",
        context_window=200_000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        cached_input_cost_per_1m=0.30,
        cache_creation_cost_per_1m=3.75,
        api_base_url="https://api.anthropic.com/v1",
        api_key="x",  # provider checks env first; this is the spec fallback
        supports_cache=supports_cache,
    )
    provider = AnthropicProvider(spec, api_key="x")
    provider.prompt_cache_enabled = prompt_cache_enabled  # type: ignore[attr-defined]

    fake_response = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "stop_reason": "end_turn",
        "model": spec.model_id,
    }
    fake_client = _RecordingAnthropicClient(fake_response)
    provider._client = fake_client  # type: ignore[assignment]
    return provider, fake_client


# ---------------------------------------------------------------------------
# Anthropic marker emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_emits_cache_control_when_enabled():
    provider, client = _make_anthropic_provider()
    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "you are a test"},
            {"role": "user", "content": "do the thing"},
        ],
    )
    assert client.last_payload is not None
    system = client.last_payload["system"]
    # List-of-blocks form with cache_control on the system text block.
    assert isinstance(system, list), f"expected list-of-blocks, got {type(system)}"
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # Short user message → stays as a plain string (below the 4 KB
    # threshold for the second breakpoint).
    user_msg = client.last_payload["messages"][0]
    assert isinstance(user_msg["content"], str)


@pytest.mark.asyncio
async def test_anthropic_marks_substantial_first_user_message_as_second_breakpoint():
    provider, client = _make_anthropic_provider()
    big_user = "x" * 5000  # > 4096 char threshold
    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big_user},
        ],
    )
    assert client.last_payload is not None
    user_msg = client.last_payload["messages"][0]
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert user_msg["content"][0]["text"] == big_user


@pytest.mark.asyncio
async def test_anthropic_legacy_string_form_when_cache_disabled():
    provider, client = _make_anthropic_provider(prompt_cache_enabled=False)
    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
    )
    assert client.last_payload is not None
    # Falls back to the legacy single-string system payload.
    assert client.last_payload["system"] == "sys"
    # User message stays a plain string regardless of size.
    assert client.last_payload["messages"][0]["content"] == "u"


@pytest.mark.asyncio
async def test_anthropic_no_cache_control_when_model_does_not_support_cache():
    provider, client = _make_anthropic_provider(supports_cache=False)
    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 8000},
        ],
    )
    assert client.last_payload is not None
    assert client.last_payload["system"] == "sys"
    # Even a long user message stays unmarked when the model can't cache.
    assert isinstance(client.last_payload["messages"][0]["content"], str)


# ---------------------------------------------------------------------------
# Hash + drift detection
# ---------------------------------------------------------------------------

def test_hash_stable_prefix_is_deterministic_and_sensitive():
    a = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "preamble"},
        {"role": "assistant", "content": "old reply"},
    ]
    b = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "preamble"},
        {"role": "assistant", "content": "DIFFERENT reply"},
    ]
    # Different message[2], but n_stable=2 → hash unchanged.
    assert hash_stable_prefix(a, n_stable=2) == hash_stable_prefix(b, n_stable=2)
    # Mutate message[1] → hash changes.
    c = list(a)
    c[1] = {"role": "user", "content": "preamble "}  # trailing whitespace
    assert hash_stable_prefix(a, n_stable=2) != hash_stable_prefix(c, n_stable=2)


@pytest.mark.asyncio
async def test_gateway_emits_cache_prefix_drift_on_second_call(monkeypatch):
    """Two consecutive dispatches for the same role with a mutated preamble
    must surface a ``cache_prefix_drift`` observability event."""
    events: list[tuple[str, dict[str, Any]]] = []

    def _fake_emit(name: str, **fields: Any) -> None:
        events.append((name, fields))

    monkeypatch.setattr("harness.observability.emit_event", _fake_emit)
    monkeypatch.setattr(
        "harness.observability.get_active_session_id", lambda: "sid-test-1234"
    )

    register_model("stub:cache-test", ModelSpec(
        provider="stub",
        model_id="cache-test",
        context_window=128_000,
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.0,
        api_base_url="",
        api_key="x",
    ))

    cfg = GatewayConfig(
        planning_primary="stub:cache-test",
        patching_primary="stub:cache-test",
        repair_primary="stub:cache-test",
    )
    gw = Gateway(cfg)

    class _Stub:
        spec = ModelSpec(
            provider="stub", model_id="cache-test", context_window=128_000,
            input_cost_per_1m=0.5, output_cost_per_1m=1.0,
            api_base_url="", api_key="x",
        )
        api_key = "x"

        async def chat_completion(self, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1, model_name="stub:cache-test", cost_usd=0.0),
                model="stub:cache-test",
            )

        async def close(self) -> None:
            return None

    gw._providers["stub:cache-test"] = _Stub()  # type: ignore[assignment]

    # Call 1 — establishes the baseline hash for (sid, PATCHING).
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "preamble v1"},
        ],
        role=NodeRole.PATCHING,
        budget_remaining_usd=2.00,
    )
    # Call 2 — same role, mutated preamble → drift event must fire.
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "preamble v2"},
        ],
        role=NodeRole.PATCHING,
        budget_remaining_usd=2.00,
    )
    drift_events = [name for name, _ in events if name == "cache_prefix_drift"]
    assert drift_events, f"expected cache_prefix_drift event; got {events}"


@pytest.mark.asyncio
async def test_gateway_cache_family_isolates_drift_buckets(monkeypatch):
    """Two dispatches with the same role but distinct ``cache_family``
    strings must NOT fire drift — they're separate buckets. Two
    dispatches with the same ``cache_family`` and a mutated prefix
    must still fire drift within that family."""
    events: list[tuple[str, dict[str, Any]]] = []

    def _fake_emit(name: str, **fields: Any) -> None:
        events.append((name, fields))

    monkeypatch.setattr("harness.observability.emit_event", _fake_emit)
    monkeypatch.setattr(
        "harness.observability.get_active_session_id", lambda: "sid-cf-1234"
    )

    register_model("stub:cf-test", ModelSpec(
        provider="stub", model_id="cf-test", context_window=128_000,
        input_cost_per_1m=0.5, output_cost_per_1m=1.0,
        api_base_url="", api_key="x",
    ))
    cfg = GatewayConfig(
        planning_primary="stub:cf-test",
        patching_primary="stub:cf-test",
        repair_primary="stub:cf-test",
    )
    gw = Gateway(cfg)

    class _Stub:
        spec = ModelSpec(
            provider="stub", model_id="cf-test", context_window=128_000,
            input_cost_per_1m=0.5, output_cost_per_1m=1.0,
            api_base_url="", api_key="x",
        )
        api_key = "x"

        async def chat_completion(self, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1, model_name="stub:cf-test", cost_usd=0.0),
                model="stub:cf-test",
            )

        async def close(self) -> None:
            return None

    gw._providers["stub:cf-test"] = _Stub()  # type: ignore[assignment]

    # Two PLANNING dispatches with different families + different
    # prefixes — no drift because the families isolate them.
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "requirements prompt"},
        ],
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
        cache_family="planning:requirements_synthesis",
    )
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "architecture prompt"},
        ],
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
        cache_family="planning:architecture_synthesis",
    )
    assert not [name for name, _ in events if name == "cache_prefix_drift"], (
        "cross-family calls must not trigger drift"
    )

    # Two dispatches sharing the same family with a mutated prefix →
    # drift fires (real cache leak within a family).
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "requirements prompt v1"},
        ],
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
        cache_family="planning:requirements_refine",
    )
    await gw.dispatch(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "requirements prompt v2"},
        ],
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
        cache_family="planning:requirements_refine",
    )
    within_family = [
        fields for name, fields in events
        if name == "cache_prefix_drift"
        and fields.get("cache_family") == "planning:requirements_refine"
    ]
    assert within_family, (
        f"within-family drift must still fire; got events={events}"
    )


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_gateway_config_default_is_cache_enabled():
    cfg = GatewayConfig()
    assert cfg.prompt_cache_enabled is True


def test_create_gateway_from_config_threads_prompt_cache_flag():
    register_model("stub:wire-check", ModelSpec(
        provider="stub", model_id="wire", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x",
    ))
    enabled = create_gateway_from_config({
        "model_routing": {
            "planning_primary": "stub:wire-check",
            "patching_primary": "stub:wire-check",
            "repair_primary": "stub:wire-check",
        },
        "llm_dispatch": {"prompt_cache_enabled": True},
    })
    assert enabled.config.prompt_cache_enabled is True

    disabled = create_gateway_from_config({
        "model_routing": {
            "planning_primary": "stub:wire-check",
            "patching_primary": "stub:wire-check",
            "repair_primary": "stub:wire-check",
        },
        "llm_dispatch": {"prompt_cache_enabled": False},
    })
    assert disabled.config.prompt_cache_enabled is False


def test_create_gateway_from_config_defaults_to_cache_on_when_key_missing():
    register_model("stub:default-check", ModelSpec(
        provider="stub", model_id="d", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x",
    ))
    gw = create_gateway_from_config({
        "model_routing": {
            "planning_primary": "stub:default-check",
            "patching_primary": "stub:default-check",
            "repair_primary": "stub:default-check",
        },
    })
    assert gw.config.prompt_cache_enabled is True
