"""Moonshot AI (Kimi) provider wiring.

Moonshot is OpenAI-compatible, so the provider piggybacks on
:class:`OpenAIProvider`. These tests pin the pieces that make it a
first-class provider rather than a config accident: factory
registration, MOONSHOT_API_KEY resolution, the international-endpoint
default, OpenAI-shape request/response handling (tool calls + usage +
reasoning), and the config-validation / API-key-gating integration that
every remote provider must satisfy.
"""

import os

import pytest

from harness.gateway import (
    ModelSpec,
    MoonshotProvider,
    OpenAIProvider,
    create_provider,
    get_model_spec,
    register_model,
    _provider_classes,
)


class _StubHttpResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.request = None
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self, response_payload):
        self.last_payload = None
        self._response_payload = response_payload

    async def post(self, _path, json):
        self.last_payload = json
        return _StubHttpResponse(self._response_payload)


def _spec(*, api_base_url="https://api.moonshot.ai/v1", supports_tools=True):
    return ModelSpec(
        provider="moonshot",
        model_id="kimi-latest",
        context_window=256_000,
        input_cost_per_1m=0.60,
        output_cost_per_1m=2.50,
        cached_input_cost_per_1m=0.15,
        api_base_url=api_base_url,
        supports_tools=supports_tools,
    )


class TestRegistration:
    def test_provider_registered_in_factory(self):
        assert _provider_classes.get("moonshot") is MoonshotProvider
        assert issubclass(MoonshotProvider, OpenAIProvider)

    def test_create_provider_builds_moonshot(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-live")
        register_model("moonshot:kimi-test", _spec())
        prov = create_provider("moonshot:kimi-test")
        assert isinstance(prov, MoonshotProvider)
        assert prov.provider_name == "moonshot"
        assert get_model_spec("moonshot:kimi-test").model_id == "kimi-latest"


class TestKeyAndEndpoint:
    def test_key_resolves_from_moonshot_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moonshot")
        prov = MoonshotProvider(_spec())
        assert prov.api_key == "sk-moonshot"
        assert prov._build_headers()["Authorization"] == "Bearer sk-moonshot"

    def test_does_not_borrow_openai_key(self, monkeypatch):
        # Distinct provider name => distinct env var; must NOT pick up
        # OPENAI_API_KEY the way a bare OpenAIProvider mapping would.
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        prov = MoonshotProvider(_spec())
        assert prov.api_key == ""

    def test_empty_base_url_falls_back_to_international_host(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        prov = MoonshotProvider(_spec(api_base_url=""))
        assert prov.spec.api_base_url == "https://api.moonshot.ai/v1"

    def test_explicit_cn_base_url_is_respected(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        prov = MoonshotProvider(_spec(api_base_url="https://api.moonshot.cn/v1"))
        assert prov.spec.api_base_url == "https://api.moonshot.cn/v1"

    def test_base_url_default_does_not_mutate_shared_spec(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        spec = _spec(api_base_url="")
        MoonshotProvider(spec)
        # dataclasses.replace copies — the caller's spec stays untouched.
        assert spec.api_base_url == ""


class TestWireShape:
    @pytest.mark.asyncio
    async def test_chat_completion_parses_content_usage_toolcalls(
        self, monkeypatch,
    ):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        payload = {
            "choices": [{
                "message": {
                    "content": "hello from kimi",
                    "reasoning_content": "let me think",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "create_file",
                            "arguments": '{"file_path": "a.py", "content": "x"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
        prov = MoonshotProvider(_spec())
        prov._client = _RecordingClient(payload)

        resp = await prov.chat_completion(
            [{"role": "user", "content": "hi"}], max_tokens=256,
        )
        assert resp.content == "hello from kimi"
        assert resp.finish_reason == "tool_calls"
        assert resp.reasoning_content == "let me think"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "create_file"
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 20
        assert resp.usage.cached_tokens == 40
        # cost: (60 uncached/1M * .60) + (40/1M * .15) + (20/1M * 2.50)
        expected = (60/1e6)*0.60 + (40/1e6)*0.15 + (20/1e6)*2.50
        assert resp.usage.cost_usd == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_request_targets_model_id_and_openai_endpoint(
        self, monkeypatch,
    ):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        prov = MoonshotProvider(_spec())
        client = _RecordingClient({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        prov._client = client
        await prov.chat_completion([{"role": "user", "content": "hi"}])
        assert client.last_payload["model"] == "kimi-latest"
        assert client.last_payload["stream"] is False


class TestFixedTemperature:
    """Kimi K3 / kimi-latest 400 on any temperature but 1; the spec's
    ``fixed_temperature`` override must win over whatever the caller (e.g.
    the repair-node escalation, which dispatches at temperature=0.0) asks
    for. Without this the model returns a non-retryable HTTP 400 and aborts
    the build.
    """

    @staticmethod
    def _client():
        return _RecordingClient({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    @pytest.mark.asyncio
    async def test_override_replaces_caller_temperature(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        spec = _spec()
        object.__setattr__(spec, "fixed_temperature", 1.0)
        prov = MoonshotProvider(spec)
        prov._client = self._client()
        # Repair escalation dispatches with the default temperature=0.0.
        await prov.chat_completion(
            [{"role": "user", "content": "hi"}], temperature=0.0,
        )
        assert prov._client.last_payload["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_none_leaves_caller_temperature_untouched(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        prov = MoonshotProvider(_spec())  # fixed_temperature defaults to None
        prov._client = self._client()
        await prov.chat_completion(
            [{"role": "user", "content": "hi"}], temperature=0.3,
        )
        assert prov._client.last_payload["temperature"] == 0.3

    def test_shipped_catalogue_pins_kimi_to_one(self):
        # The price catalogue is the source of truth for the constraint.
        assert get_model_spec("moonshot:kimi-k3").fixed_temperature == 1.0
        assert get_model_spec("moonshot:kimi-latest").fixed_temperature == 1.0


class TestConfigIntegration:
    def test_find_missing_api_keys_gates_on_moonshot_key(self, monkeypatch):
        from harness.cli import find_missing_api_keys

        for k in ("MOONSHOT_API_KEY",):
            monkeypatch.delenv(k, raising=False)
        cfg = {
            "models": {
                "moonshot:kimi-latest": {
                    "provider": "moonshot",
                    "model_id": "kimi-latest",
                    "context_window": 256000,
                    "input_cost_per_1m": 0.6,
                    "output_cost_per_1m": 2.5,
                    "api_base_url": "https://api.moonshot.ai/v1",
                },
            },
            "model_routing": {"planning_primary": "moonshot:kimi-latest"},
        }
        missing = find_missing_api_keys(cfg)
        assert "MOONSHOT_API_KEY" in missing
        assert missing["MOONSHOT_API_KEY"] == ["moonshot:kimi-latest"]

        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-x")
        assert "MOONSHOT_API_KEY" not in find_missing_api_keys(cfg)

    def test_shipped_config_registers_kimi(self, monkeypatch):
        import json
        from harness.gateway import register_models_from_config

        cfg = json.loads(open("config/config.json").read())

        def strip(d):
            if isinstance(d, dict):
                return {k: strip(v) for k, v in d.items()
                        if not k.startswith("_")}
            if isinstance(d, list):
                return [strip(x) for x in d]
            return d

        register_models_from_config(strip(cfg))
        spec = get_model_spec("moonshot:kimi-latest")
        assert spec is not None
        assert spec.provider == "moonshot"
        assert spec.supports_tools is True


class TestWizardMaps:
    def test_wizard_and_env_maps_include_moonshot(self):
        from harness.web_wizard import (
            DEFAULT_MODELS_BY_PROVIDER,
            PROVIDER_ENV_VAR,
        )
        assert DEFAULT_MODELS_BY_PROVIDER["moonshot"] == "moonshot:kimi-latest"
        assert PROVIDER_ENV_VAR["moonshot"] == "MOONSHOT_API_KEY"
