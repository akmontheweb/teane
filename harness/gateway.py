"""
Model-agnostic LLM Gateway with prefix caching, token tracking, budget enforcement,
and exponential backoff for all provider API calls.

This module implements:
    - BaseLLM abstract interface for any provider (OpenAI, Anthropic, DeepSeek, Ollama)
    - Provider-specific HTTP clients with async httpx transport
    - Token usage extraction parsers for each provider's response payload shape
    - Prefix caching anchor utility — ensures system prompts are locked at messages[0]
    - Pre-flight context window guardrail (85% threshold with aggressive truncation)
    - Token-budget-aware dispatch: refuses calls when budget_remaining_usd <= 0
    - Exponential backoff with random jitter for HTTP 429 rate limit handling
    - Model auto-selection based on node role and .harness_config.json routing rules
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Types
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """Identifies which graph node is making the LLM call."""
    PLANNING = "planning"
    PATCHING = "patching"
    REPAIR = "repair"
    HUMAN_INTERVENTION = "human_intervention"
    DOC_REVIEWER = "doc_reviewer"
    CODE_REVIEWER = "code_reviewer"


class EmptyLLMResponseError(RuntimeError):
    """Raised by the gateway when the provider returns an empty content body
    after every retry. Distinct from generic RuntimeError so callers
    (repair / HITL routers) can short-circuit to a clear operator message
    instead of looping for several rounds while the provider stays silent.
    """


class BudgetTooLowError(RuntimeError):
    """Raised by the gateway's pre-flight budget estimate when the projected
    cost of a single call already exceeds the remaining budget. Stops the
    advisory hard-cap from being silently overspent by a single big call.
    """


@dataclass
class TokenUsage:
    """Extracted token usage metadata from a single LLM response.

    ``cached_tokens`` counts cache *reads* (priced at the discounted
    cached input rate). ``cache_creation_tokens`` counts tokens written
    into the cache for the first time — Anthropic charges these at a
    surcharge (~1.25× input), other providers leave it 0.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    model_name: str = ""
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "model_name": self.model_name,
            "cost_usd": self.cost_usd,
        }


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    usage: TokenUsage
    model: str
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Specification for a model including cost rates and context window limits."""
    provider: str  # "deepseek", "anthropic", "openai", "ollama"
    model_id: str
    context_window: int  # maximum tokens the model accepts
    input_cost_per_1m: float  # cost per 1M input tokens in USD
    output_cost_per_1m: float  # cost per 1M output tokens in USD
    cached_input_cost_per_1m: float = 0.0  # cache READ discount (Anthropic, OpenAI, DeepSeek)
    cache_creation_cost_per_1m: float = 0.0  # cache WRITE surcharge (Anthropic only; ~1.25× input)
    api_base_url: str = ""
    api_key: str = ""  # Optional: API key stored in config (env var takes precedence)
    supports_thinking: bool = False
    supports_cache: bool = False
    # Anthropic API version header. Default is the documented stable; users
    # can override per-model via .harness_config.json when newer features
    # need a different version.
    anthropic_version: str = "2023-06-01"
    # Default thinking budget in tokens when the role asks for thinking and
    # the model supports it. Anthropic requires this to be < max_tokens.
    thinking_budget_tokens: int = 8000


# Model registry — populated from model_prices.json at import time, then
# overridden by user .harness_config.json 'models' entries, then by
# explicit register_model() calls. This gives a sensible price catalogue
# out-of-the-box while letting users override without editing source.
_MODEL_REGISTRY: dict[str, ModelSpec] = {}

# Path of the shipped price catalogue — installed alongside this module.
_PRICES_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_prices.json")


def load_model_prices(prices_path: Optional[str] = None, override: bool = False) -> int:
    """
    Load model specifications from a price-catalogue JSON file into the
    model registry.

    Called automatically at module import with the shipped
    ``harness/model_prices.json``. Can be called again with a custom path
    to load user-defined overrides or an updated price snapshot.

    Args:
        prices_path: Path to the JSON catalogue. Defaults to the shipped
                     ``harness/model_prices.json``.
        override:    If True, overwrite any existing registry entry for a
                     model key. If False (default), existing entries win —
                     this means a user's .harness_config.json entry always
                     takes precedence over the shipped catalogue.

    Returns:
        Number of models successfully loaded.
    """
    path = prices_path or _PRICES_JSON_PATH
    if not os.path.isfile(path):
        logger.debug("[gateway] Model prices file not found at %s; skipping.", path)
        return 0
    try:
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = _json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("[gateway] Could not load model prices from %s: %s", path, exc)
        return 0

    count = 0
    for model_key, spec_dict in raw.items():
        if model_key.startswith("_"):
            continue  # skip comment/metadata keys
        if not isinstance(spec_dict, dict):
            continue
        if not override and model_key in _MODEL_REGISTRY:
            continue  # user config already registered this key — don't clobber
        try:
            spec = ModelSpec(
                provider=spec_dict.get("provider", model_key.split(":")[0] if ":" in model_key else "unknown"),
                model_id=spec_dict.get("model_id", model_key),
                context_window=int(spec_dict.get("context_window", 131072)),
                input_cost_per_1m=float(spec_dict.get("input_cost_per_1m", 0.0)),
                output_cost_per_1m=float(spec_dict.get("output_cost_per_1m", 0.0)),
                cached_input_cost_per_1m=float(spec_dict.get("cached_input_cost_per_1m", 0.0)),
                cache_creation_cost_per_1m=float(spec_dict.get("cache_creation_cost_per_1m", 0.0)),
                api_base_url=spec_dict.get("api_base_url", ""),
                api_key=spec_dict.get("api_key", ""),
                supports_thinking=bool(spec_dict.get("supports_thinking", False)),
                supports_cache=bool(spec_dict.get("supports_cache", False)),
                anthropic_version=spec_dict.get("anthropic_version", "2023-06-01"),
                thinking_budget_tokens=int(spec_dict.get("thinking_budget_tokens", 8000)),
            )
            _MODEL_REGISTRY[model_key] = spec
            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gateway] Skipping malformed entry '%s' in %s: %s", model_key, path, exc)

    if count:
        logger.debug("[gateway] Loaded %d model(s) from %s.", count, path)
    return count


def get_model_spec(model_key: str) -> Optional[ModelSpec]:
    """
    Look up a model specification by its canonical key.

    Returns None if the model is not registered. The registry is pre-seeded
    with the shipped ``harness/model_prices.json`` catalogue; additional
    models can be registered via .harness_config.json or register_model().
    """
    return _MODEL_REGISTRY.get(model_key)


def register_model(model_key: str, spec: ModelSpec) -> None:
    """
    Register a model specification in the global registry.

    Args:
        model_key: Canonical key (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet-4').
        spec: The ModelSpec with provider details, costs, and context window.
    """
    _MODEL_REGISTRY[model_key] = spec
    logger.info("[gateway] Registered model '%s' (provider=%s, ctx=%d).", model_key, spec.provider, spec.context_window)


# ---------------------------------------------------------------------------
# Auto-seed the registry from the shipped price catalogue at import time.
# This is a lightweight file read (~2ms). It runs before any user config
# is loaded; register_models_from_config() called later will override these
# defaults since it calls register_model() which always wins.
# ---------------------------------------------------------------------------
load_model_prices()


def register_models_from_config(config_dict: dict[str, Any]) -> int:
    """
    Batch-register models from a .harness_config.json 'models' section.

    Expected config format:
        {
          "models": {
            "openai:gpt-4o": {
              "provider": "openai",
              "model_id": "gpt-4o",
              "context_window": 128000,
              "input_cost_per_1m": 2.50,
              "output_cost_per_1m": 10.00,
              "cached_input_cost_per_1m": 1.25,
              "api_base_url": "https://api.openai.com/v1"
            }
          }
        }

    Args:
        config_dict: Parsed config dictionary from .harness_config.json.

    Returns:
        Number of models registered.
    """
    models_section = config_dict.get("models", {})
    count = 0
    for model_key, spec_dict in models_section.items():
        if not isinstance(spec_dict, dict):
            logger.warning("[gateway] Skipping invalid model spec for '%s': not a dict.", model_key)
            continue
        try:
            # Merge over the catalogue baseline if it exists, so users only
            # need to specify the keys they want to override (e.g. api_key).
            baseline = _MODEL_REGISTRY.get(model_key)
            merged: dict[str, Any] = {}
            if baseline is not None:
                merged.update({
                    "provider": baseline.provider,
                    "model_id": baseline.model_id,
                    "context_window": baseline.context_window,
                    "input_cost_per_1m": baseline.input_cost_per_1m,
                    "output_cost_per_1m": baseline.output_cost_per_1m,
                    "cached_input_cost_per_1m": baseline.cached_input_cost_per_1m,
                    "cache_creation_cost_per_1m": baseline.cache_creation_cost_per_1m,
                    "api_base_url": baseline.api_base_url,
                    "api_key": baseline.api_key,
                    "supports_thinking": baseline.supports_thinking,
                    "supports_cache": baseline.supports_cache,
                    "anthropic_version": baseline.anthropic_version,
                    "thinking_budget_tokens": baseline.thinking_budget_tokens,
                })
            merged.update(spec_dict)  # user config wins over catalogue baseline

            spec = ModelSpec(
                provider=merged.get("provider", model_key.split(":")[0] if ":" in model_key else "unknown"),
                model_id=merged.get("model_id", model_key),
                context_window=int(merged.get("context_window", 131072)),
                input_cost_per_1m=float(merged.get("input_cost_per_1m", 0.0)),
                output_cost_per_1m=float(merged.get("output_cost_per_1m", 0.0)),
                cached_input_cost_per_1m=float(merged.get("cached_input_cost_per_1m", 0.0)),
                cache_creation_cost_per_1m=float(merged.get("cache_creation_cost_per_1m", 0.0)),
                api_base_url=merged.get("api_base_url", ""),
                api_key=merged.get("api_key", ""),
                supports_thinking=bool(merged.get("supports_thinking", False)),
                supports_cache=bool(merged.get("supports_cache", False)),
                anthropic_version=merged.get("anthropic_version", "2023-06-01"),
                thinking_budget_tokens=int(merged.get("thinking_budget_tokens", 8000)),
            )
            register_model(model_key, spec)
            count += 1
        except Exception as exc:
            logger.warning("[gateway] Failed to register model '%s': %s", model_key, exc)
    if count > 0:
        logger.info("[gateway] Registered %d model(s) from config.", count)
    return count


# ---------------------------------------------------------------------------
# 2. BaseLLM Abstract Interface
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    Abstract base for all LLM provider clients.

    Each provider (DeepSeek, Anthropic, OpenAI, Ollama) implements:
        - chat_completion(messages, **kwargs) → LLMResponse
        - extract_usage(raw_response) → TokenUsage
        - compute_cost(usage) → float
    """

    def __init__(
        self,
        spec: ModelSpec,
        api_key: Optional[str] = None,
        ssl_verify: Union[bool, str] = True,
    ):
        self.spec = spec
        # Resolution order: explicit arg → env var → config file → empty
        self.api_key = api_key or os.environ.get(f"{spec.provider.upper()}_API_KEY", "") or spec.api_key
        self.ssl_verify = ssl_verify
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> str:
        return self.spec.provider

    @property
    def model_name(self) -> str:
        return self.spec.model_id

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse an httpx AsyncClient."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.spec.api_base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
                headers=self._build_headers(),
                verify=self.ssl_verify,
            )
        return self._client

    def _build_headers(self) -> dict[str, str]:
        """Construct provider-specific HTTP headers."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return a standardized response."""
        ...

    @abstractmethod
    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        """Parse token usage metadata from the provider's raw response JSON."""
        ...

    @abstractmethod
    def compute_cost(self, usage: TokenUsage) -> float:
        """Compute USD cost based on token counts and model pricing rates."""
        ...


def _parse_json_response(response: Any) -> dict[str, Any]:
    """
    Parse a JSON response body, converting JSONDecodeError into an
    HTTPStatusError-like exception so retry_with_backoff treats it as
    a retryable server error rather than letting it escape uncaught.
    """
    try:
        return response.json()  # type: ignore[no-any-return]
    except Exception as exc:
        raise httpx.HTTPStatusError(
            f"Malformed JSON in response body: {exc}",
            request=response.request,
            response=response,
        ) from exc


# ---------------------------------------------------------------------------
# 3. DeepSeek Provider Implementation
# ---------------------------------------------------------------------------

class DeepSeekProvider(BaseLLM):
    """DeepSeek API client using OpenAI-compatible /v1/chat/completions endpoint."""

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking and self.spec.supports_thinking:
            payload["thinking"] = {"type": "enabled"}

        logger.debug("[deepseek] Sending completion request. model=%s tokens_est=%d", self.spec.model_id, len(messages))

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
        )

    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        usage_block = raw_response.get("usage", {})
        # DeepSeek returns prompt_tokens_details.cached_tokens when cache hits occur
        prompt_details = usage_block.get("prompt_tokens_details", {})
        return TokenUsage(
            input_tokens=usage_block.get("prompt_tokens", 0),
            output_tokens=usage_block.get("completion_tokens", 0),
            cached_tokens=prompt_details.get("cached_tokens", 0),
            model_name=self.spec.model_id,
        )

    def compute_cost(self, usage: TokenUsage) -> float:
        spec = self.spec
        # Cache-hit tokens are billed at the lower cached rate
        cached = usage.cached_tokens
        uncached_input = max(0, usage.input_tokens - cached)

        input_cost = (uncached_input / 1_000_000) * spec.input_cost_per_1m
        cached_cost = (cached / 1_000_000) * spec.cached_input_cost_per_1m
        output_cost = (usage.output_tokens / 1_000_000) * spec.output_cost_per_1m

        return input_cost + cached_cost + output_cost


# ---------------------------------------------------------------------------
# 4. Anthropic Provider Implementation
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLM):
    """Anthropic (Claude) API client using /v1/messages endpoint."""

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        # Anthropic uses x-api-key header instead of Authorization Bearer
        headers["x-api-key"] = self.api_key
        # Version is per-model: newer models / features may require a newer
        # date. Pulled from ModelSpec.anthropic_version so .harness_config.json
        # can bump it without code changes.
        headers["anthropic-version"] = self.spec.anthropic_version or "2023-06-01"
        # Remove the Bearer header since Anthropic doesn't use it
        headers.pop("Authorization", None)
        return headers

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()

        # Anthropic requires a system prompt separated from the messages array.
        # Extract system message(s) and pass them as the top-level 'system' field.
        system_content: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_content.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_content.append(block.get("text", ""))
                        elif isinstance(block, dict) and block.get("type") not in ("text", None):
                            logger.warning(
                                "[anthropic] Dropping non-text system block of type %r "
                                "— Anthropic's top-level system field supports text only.",
                                block.get("type"),
                            )
            else:
                # Map to Anthropic message format
                anthropic_msg: dict[str, Any] = {"role": role, "content": msg.get("content", "")}
                anthropic_messages.append(anthropic_msg)

        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_content:
            # Anthropic expects a single string or list of text blocks for system
            payload["system"] = "\n\n".join(system_content)

        # Extended thinking: must be opted in per request, and Anthropic requires
        # temperature=1.0 with thinking enabled. budget_tokens must be < max_tokens.
        if thinking and self.spec.supports_thinking:
            budget = max(1024, min(self.spec.thinking_budget_tokens, max_tokens - 512))
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1.0
            # Ensure max_tokens accommodates the thinking budget + visible reply
            if max_tokens <= budget:
                payload["max_tokens"] = budget + 1024

        logger.debug("[anthropic] Sending completion request. model=%s thinking=%s",
                     self.spec.model_id, thinking and self.spec.supports_thinking)

        response = await client.post("/messages", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        # Anthropic returns content as a list of blocks; extract text
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        content = "\n".join(text_parts)

        finish_reason = data.get("stop_reason", "stop")

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
        )

    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        # Anthropic reports cache reads and cache creations separately from
        # input_tokens — they are NOT included in input_tokens. Keep them
        # distinct so compute_cost can price each correctly.
        usage_block = raw_response.get("usage", {})
        return TokenUsage(
            input_tokens=usage_block.get("input_tokens", 0),
            output_tokens=usage_block.get("output_tokens", 0),
            cached_tokens=usage_block.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage_block.get("cache_creation_input_tokens", 0),
            model_name=self.spec.model_id,
        )

    def compute_cost(self, usage: TokenUsage) -> float:
        # Anthropic billing tiers:
        #   - input_tokens:          full input_cost_per_1m (already excludes cache hits)
        #   - cache_read tokens:     cached_input_cost_per_1m (~10% of input)
        #   - cache_creation tokens: cache_creation_cost_per_1m (~125% of input)
        #     Falls back to 1.25x input rate when the spec doesn't carry an
        #     explicit creation rate (matches Anthropic's published surcharge).
        spec = self.spec
        creation_rate = (
            spec.cache_creation_cost_per_1m
            if spec.cache_creation_cost_per_1m > 0
            else spec.input_cost_per_1m * 1.25
        )
        input_cost = (usage.input_tokens / 1_000_000) * spec.input_cost_per_1m
        cache_read_cost = (usage.cached_tokens / 1_000_000) * spec.cached_input_cost_per_1m
        cache_creation_cost = (usage.cache_creation_tokens / 1_000_000) * creation_rate
        output_cost = (usage.output_tokens / 1_000_000) * spec.output_cost_per_1m
        return input_cost + cache_read_cost + cache_creation_cost + output_cost


# ---------------------------------------------------------------------------
# 5. OpenAI Provider Implementation
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLM):
    """OpenAI API client using /v1/chat/completions endpoint."""

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        logger.debug("[openai] Sending completion request. model=%s", self.spec.model_id)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
        )

    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        usage_block = raw_response.get("usage", {})
        prompt_details = usage_block.get("prompt_tokens_details", {})
        return TokenUsage(
            input_tokens=usage_block.get("prompt_tokens", 0),
            output_tokens=usage_block.get("completion_tokens", 0),
            cached_tokens=prompt_details.get("cached_tokens", 0),
            model_name=self.spec.model_id,
        )

    def compute_cost(self, usage: TokenUsage) -> float:
        spec = self.spec
        # OpenAI's prompt_tokens_details.cached_tokens are billed at the
        # discounted cached rate; subtract them from input_tokens so they
        # aren't double-charged at the full rate.
        cached = usage.cached_tokens
        uncached_input = max(0, usage.input_tokens - cached)
        input_cost = (uncached_input / 1_000_000) * spec.input_cost_per_1m
        cached_cost = (cached / 1_000_000) * spec.cached_input_cost_per_1m
        output_cost = (usage.output_tokens / 1_000_000) * spec.output_cost_per_1m
        return input_cost + cached_cost + output_cost


# ---------------------------------------------------------------------------
# 6. Ollama (Local) Provider Implementation
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLM):
    """Ollama local inference server using OpenAI-compatible /v1/chat/completions endpoint."""

    def _build_headers(self) -> dict[str, str]:
        # Ollama doesn't require an API key; skip Authorization header
        return {"Content-Type": "application/json"}

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        logger.debug("[ollama] Sending completion request. model=%s", self.spec.model_id)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)

        usage = self.extract_usage(data)
        usage.cost_usd = 0.0  # Local inference is free

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
        )

    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        usage_block = raw_response.get("usage", {})
        return TokenUsage(
            input_tokens=usage_block.get("prompt_tokens", 0),
            output_tokens=usage_block.get("completion_tokens", 0),
            cached_tokens=0,
            model_name=self.spec.model_id,
        )

    def compute_cost(self, usage: TokenUsage) -> float:
        return 0.0  # Local models incur no API cost


# ---------------------------------------------------------------------------
# 7. Provider Factory
# ---------------------------------------------------------------------------

_provider_classes: dict[str, type[BaseLLM]] = {
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def create_provider(
    model_key: str,
    api_key: Optional[str] = None,
    ssl_verify: Union[bool, str] = True,
) -> BaseLLM:
    """
    Factory: create the correct BaseLLM provider for a given model key.

    Args:
        model_key: Canonical model key (e.g., 'openai:gpt-4o').
        api_key: Optional API key override. Falls back to environment variable.
        ssl_verify: TLS verification setting passed to httpx. Pass a CA bundle
                    path (str) for corporate proxies, or False for air-gapped
                    environments (not recommended for production).

    Returns:
        A configured BaseLLM provider instance.

    Raises:
        ValueError: If the model is not registered or the provider is unrecognized.
    """
    spec = get_model_spec(model_key)
    if spec is None:
        raise ValueError(
            f"Model '{model_key}' is not registered. "
            f"Register it via .harness_config.json 'models' section or gateway.register_model()."
        )
    provider_name = spec.provider
    cls = _provider_classes.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider_name}' for model '{model_key}'. "
            f"Supported providers: {list(_provider_classes.keys())}"
        )
    return cls(spec, api_key=api_key or spec.api_key, ssl_verify=ssl_verify)


# ---------------------------------------------------------------------------
# 8. Token Counting Utility (Pre-flight Context Window Guard)
# ---------------------------------------------------------------------------

def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """
    Fast heuristic token estimation for pre-flight context window checks.

    Uses a simple character-to-token ratio (~4 chars per token for English text)
    plus overhead for message formatting. Not exact, but fast and sufficient for
    the 85% guardrail threshold check.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block))
        total_chars += 50  # Overhead per message for role markers, formatting, etc.
    return max(1, total_chars // 4)  # ~4 chars per token is a common heuristic


# ---------------------------------------------------------------------------
# 9. Prefix Caching Anchor Utility
# ---------------------------------------------------------------------------

def ensure_prefix_cache_anchor(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Guarantee that the immutable system prompt is anchored at messages[0].

    This is critical for provider prompt caching (DeepSeek and Anthropic both
    offer discounted rates for repeated prefix content). The system prompt
    must never be moved, modified, or truncated — it stays at position 0 always.

    If the first message is not a system message, this utility logs a warning
    but does not reorder (to avoid destroying conversation semantics).
    """
    if not messages:
        return messages

    first = messages[0]
    if first.get("role") != "system":
        logger.warning(
            "[gateway] messages[0] is not a system message (role='%s'). "
            "Prefix caching discounts may not apply.",
            first.get("role"),
        )

    # Compute a content hash of the system prompt for cache-hit tracking
    if first.get("role") == "system":
        content = first.get("content", "")
        content_hash = hashlib.sha256(
            content.encode("utf-8") if isinstance(content, str) else json.dumps(content, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        logger.debug("[gateway] System prompt anchor hash: %s", content_hash)

    return messages


# ---------------------------------------------------------------------------
# 10. Context Window Guardrail & Truncation
# ---------------------------------------------------------------------------

async def check_context_window(
    messages: list[dict[str, Any]],
    spec: ModelSpec,
    threshold_pct: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Pre-flight context window guardrail.

    If the estimated token count exceeds `threshold_pct` of the model's
    context window, aggressively truncate older non-system messages until
    the payload fits within the threshold.

    Truncation strategy:
        1. Always keep messages[0] (the system prompt anchor).
        2. Always keep the last user message (the current request).
        3. Drop the oldest non-system, non-current messages first.
        4. If still over threshold after dropping all trimmable messages, raise.

    Args:
        messages: The full conversation messages array.
        spec: The target model's specification (context_window limit).
        threshold_pct: Fraction of context window at which to start truncating.

    Returns:
        A (possibly truncated) messages list.

    Raises:
        ValueError: If the payload cannot be reduced below the threshold.
    """
    max_tokens = spec.context_window
    threshold = int(max_tokens * threshold_pct)
    estimated = estimate_token_count(messages)

    if estimated <= threshold:
        logger.debug("[gateway] Token estimate %d within threshold %d/%d.", estimated, threshold, max_tokens)
        return messages

    logger.warning(
        "[gateway] Token estimate %d exceeds %d%% threshold (%d/%d). Truncating conversation.",
        estimated,
        int(threshold_pct * 100),
        threshold,
        max_tokens,
    )

    if len(messages) <= 2:
        # Only system prompt + current user message; can't truncate further
        raise ValueError(
            f"Cannot reduce payload below {estimate_token_count(messages)} tokens. "
            f"Model context window: {max_tokens}. Consider splitting the task."
        )

    # Core strategy: keep system prompt [0] and last message [-1]
    preserved = [messages[0], messages[-1]]
    preserved_count = estimate_token_count(preserved)

    # If even just the system prompt + last message exceeds threshold, fail
    if preserved_count > threshold:
        raise ValueError(
            f"System prompt + current message alone exceed the context threshold "
            f"({preserved_count} > {threshold}). Reduce the system prompt size or split the task."
        )

    # Build truncated list: system + most recent N messages that fit
    truncated = [messages[0]]
    available_budget = threshold - estimate_token_count(truncated) - estimate_token_count([messages[-1]])

    # Fill from the end (most recent first) excluding system[0] and last[-1]
    middle_messages = messages[1:-1]
    insertion_point = 1  # After system prompt

    for msg in reversed(middle_messages):
        msg_estimate = estimate_token_count([msg])
        if msg_estimate <= available_budget:
            truncated.insert(insertion_point, msg)
            available_budget -= msg_estimate
        # Don't break — a subsequent (older, smaller) message may still fit.

    truncated.append(messages[-1])

    final_estimate = estimate_token_count(truncated)
    logger.info(
        "[gateway] Truncation complete. %d → %d messages, %d → ~%d tokens.",
        len(messages),
        len(truncated),
        estimated,
        final_estimate,
    )
    return truncated


# ---------------------------------------------------------------------------
# 11. Exponential Backoff with Jitter
# ---------------------------------------------------------------------------

async def retry_with_backoff(
    fn: Callable[..., Awaitable[LLMResponse]],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> LLMResponse:
    """
    Execute an async LLM call with exponential backoff + random jitter.

    Handles HTTP 429 (rate limit), 5xx (server errors), and connection errors.
    After max_retries, re-raises the last exception.

    Backoff formula: min(max_delay, base_delay * 2^attempt) * (0.5 + random * 0.5)
    This gives a jitter range of 50%-100% of the exponential base.
    """
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                delay = _delay_from_rate_limit_headers(exc.response.headers, base_delay, attempt)
                logger.warning("[gateway] Rate limited (429). Attempt %d/%d. Delay=%.2fs",
                                attempt + 1, max_retries + 1, delay)
            elif status >= 500:
                delay = base_delay * (2 ** attempt)
                logger.warning("[gateway] Server error (%d). Attempt %d/%d.", status, attempt + 1, max_retries + 1)
            else:
                raise  # Non-retryable HTTP error (4xx except 429)
            last_exception = exc
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
            delay = base_delay * (2 ** attempt)
            logger.warning("[gateway] Connection error. Attempt %d/%d. %s", attempt + 1, max_retries + 1, exc)
            last_exception = exc

        if attempt < max_retries:
            # Apply jitter: 50%-100% of computed delay
            jittered = delay * (0.5 + random.random() * 0.5)
            jittered = min(jittered, max_delay)
            logger.debug("[gateway] Backing off for %.2fs before retry.", jittered)
            await asyncio.sleep(jittered)

    raise last_exception  # type: ignore[misc]


def _delay_from_rate_limit_headers(
    headers: Any, base_delay: float, attempt: int
) -> float:
    """
    Compute the retry delay from common rate-limit response headers.

    Recognized (in priority order):
      - ``Retry-After``: seconds (numeric) or HTTP-date (RFC 7231)
      - ``anthropic-ratelimit-tokens-reset`` / ``-requests-reset``: ISO 8601 datetime
      - ``X-RateLimit-Reset``: epoch seconds (OpenAI-style)
      - ``RateLimit-Reset``: seconds-from-now (RFC 9651)

    Falls back to exponential backoff if no header is parseable.
    """
    from datetime import datetime, timezone

    # 1. Retry-After (numeric seconds or HTTP-date)
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(retry_after)
                delta = (dt - datetime.now(timezone.utc)).total_seconds()
                if delta > 0:
                    return delta
            except (TypeError, ValueError):
                pass

    # 2. Anthropic-specific reset timestamps (ISO 8601 UTC)
    for key in ("anthropic-ratelimit-tokens-reset", "anthropic-ratelimit-requests-reset"):
        reset = headers.get(key)
        if reset:
            try:
                dt = datetime.fromisoformat(reset.replace("Z", "+00:00"))
                delta = (dt - datetime.now(timezone.utc)).total_seconds()
                if delta > 0:
                    return delta
            except (TypeError, ValueError):
                pass

    # 3. OpenAI X-RateLimit-Reset (epoch seconds)
    x_reset = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset-requests")
    if x_reset:
        try:
            target = float(x_reset)
            now = datetime.now(timezone.utc).timestamp()
            # Heuristic: if value > now, it's epoch; else seconds-from-now
            delta = target - now if target > now else target
            if delta > 0:
                return delta
        except ValueError:
            pass

    # 4. RFC 9651 RateLimit-Reset (seconds-from-now)
    rl_reset = headers.get("RateLimit-Reset")
    if rl_reset:
        try:
            return max(0.0, float(rl_reset))
        except ValueError:
            pass

    # Fallback: exponential backoff
    return base_delay * (2 ** attempt)


# ---------------------------------------------------------------------------
# 12. Gateway Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class GatewayConfig:
    """Runtime configuration for the LLM gateway, parsed from .harness_config.json.

    All model keys default to empty strings. Users must configure model routing
    via .harness_config.json. No default models are bundled.
    """
    planning_primary: str = ""
    planning_mode: str = "thinking_max"
    planning_fallback: str = ""
    patching_primary: str = ""
    patching_mode: str = "non_thinking"
    repair_primary: str = ""
    repair_fallback: str = ""
    repair_mode: str = "thinking"
    # Doc reviewer — fully independent of code reviewer. Empty primary == disabled.
    doc_reviewer_primary: str = ""
    doc_reviewer_mode: str = "thinking"
    doc_reviewer_fallback: str = ""
    max_doc_review_cycles: int = 1
    # Code reviewer — fully independent of doc reviewer. Empty primary == disabled.
    code_reviewer_primary: str = ""
    code_reviewer_mode: str = "thinking"
    code_reviewer_fallback: str = ""
    max_code_review_cycles: int = 1
    # Hard ceiling on rounds of the discovery interview loop. Without this,
    # a confused user (or hostile LLM) can loop indefinitely on follow-up
    # questions, burning budget. Clamped to [1, 30] at config load.
    max_discovery_iterations: int = 10
    # Hard ceiling on build → repair → compile retries after the initial
    # patching pass. After this many failed repair attempts the router
    # diverts to HITL instead of looping forever. Clamped to [1, 10] at
    # config load. Wired from node_throttle.max_patch_repair_iterations.
    max_patch_repair_iterations: int = 3
    ollama_local_model: str = ""
    ollama_local_backup: str = ""
    force_local_only: bool = False
    hard_cap_usd: float = 2.00
    context_window_threshold_pct: float = 0.85
    max_retries: int = 5
    base_delay: float = 1.0
    # TLS: set to a CA bundle path (str) for corporate proxies, or False to
    # disable verification in air-gapped envs (not recommended for production).
    ssl_verify: Union[bool, str] = True
    # Per-call max_tokens ceiling. Used by Gateway._max_tokens_for(role).
    # max_tokens_default is the fallback when a role isn't listed in
    # max_tokens_per_role. Both are clamped to [256, 32768] in
    # validate_config_strict so the gateway can trust the values here.
    # For reasoning-mode models (deepseek-v4-pro) the ceiling is shared
    # between the hidden thinking trace and the visible content — bumping
    # repair to 8192 is the recommended baseline so the thinking trace +
    # patch blocks both fit.
    max_tokens_default: int = 4096
    max_tokens_per_role: dict[str, int] = field(default_factory=dict)


class Gateway:
    """
    Central orchestrator for model-agnostic LLM dispatching.

    Responsibilities:
        - Route calls to the correct provider based on NodeRole and config.
        - Enforce token budget (rejects calls when budget_remaining_usd <= 0).
        - Apply prefix caching anchor at messages[0].
        - Run pre-flight context window guardrail checks.
        - Aggregate token usage into the LangGraph state token_tracker.
        - Handle retry with exponential backoff.
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        # Provider cache: lazily instantiated per unique model_key
        self._providers: dict[str, BaseLLM] = {}
        # 429/503 circuit breaker (P1.9). When too many rate-limit / server
        # failures pile up in a short window, fall the next call back to
        # local Ollama instead of burning retries with no chance of
        # success. Kept in-memory only — resets when the process restarts.
        from collections import deque as _deque
        self._rate_limit_failures: "_deque[float]" = _deque(maxlen=64)
        self._circuit_window_seconds: float = 300.0   # 5-minute rolling window
        self._circuit_failure_threshold: int = 3       # open after 3 hits in-window
        self._circuit_open_until: float = 0.0           # epoch seconds; 0 = closed

    async def _get_provider(self, model_key: str) -> BaseLLM:
        """Get or create a cached provider instance."""
        if model_key not in self._providers:
            self._providers[model_key] = create_provider(
                model_key, ssl_verify=self.config.ssl_verify
            )
        return self._providers[model_key]

    def _circuit_is_open(self) -> bool:
        """Return True when recent 429/503 failures should force a fall-back
        to local Ollama for the next call. The breaker auto-closes after
        ``_circuit_window_seconds`` of cool-down."""
        import time as _t
        now = _t.monotonic()
        if self._circuit_open_until and now < self._circuit_open_until:
            return True
        # Trim failures outside the rolling window.
        window_start = now - self._circuit_window_seconds
        while self._rate_limit_failures and self._rate_limit_failures[0] < window_start:
            self._rate_limit_failures.popleft()
        if len(self._rate_limit_failures) >= self._circuit_failure_threshold:
            # Open the circuit for the rest of the window plus a short buffer.
            self._circuit_open_until = now + max(60.0, self._circuit_window_seconds / 2)
            logger.warning(
                "[gateway] Rate-limit circuit breaker OPEN: %d failures in last %ds. "
                "Forcing local fallback until %.0fs from now.",
                len(self._rate_limit_failures),
                int(self._circuit_window_seconds),
                self._circuit_open_until - now,
            )
            return True
        return False

    def _record_rate_limit_failure(self) -> None:
        """Record a 429/503 failure so the circuit breaker can detect bursts."""
        import time as _t
        self._rate_limit_failures.append(_t.monotonic())

    async def close(self) -> None:
        """Close all open provider HTTP clients."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

    def select_model(self, role: NodeRole, force_local: bool = False) -> str:
        """
        Select the appropriate model for a given node role based on config.

        Args:
            role: The graph node making the request.
            force_local: If True (or config.force_local_only), use local Ollama only.

        Returns:
            The canonical model key to use.
        """
        if force_local or self.config.force_local_only:
            return f"ollama:{self.config.ollama_local_model}"

        if role == NodeRole.PLANNING:
            return self.config.planning_primary
        elif role == NodeRole.PATCHING:
            return self.config.patching_primary
        elif role == NodeRole.REPAIR:
            return self.config.repair_primary
        elif role == NodeRole.DOC_REVIEWER:
            # No silent fallback: empty means reviewer is not configured. The
            # caller (spec_review_node) must check this and skip the call.
            return self.config.doc_reviewer_primary
        elif role == NodeRole.CODE_REVIEWER:
            return self.config.code_reviewer_primary
        else:
            return self.config.patching_primary  # Default

    def should_use_thinking(self, role: NodeRole) -> bool:
        """Determine if thinking/reasoning mode should be enabled for this role."""
        if role == NodeRole.PLANNING:
            return self.config.planning_mode.lower() == "thinking" or self.config.planning_mode.lower() == "thinking_max"
        elif role == NodeRole.PATCHING:
            return self.config.patching_mode.lower() == "thinking" or self.config.patching_mode.lower() == "thinking_max"
        elif role == NodeRole.REPAIR:
            return self.config.repair_mode.lower() == "thinking" or self.config.repair_mode.lower() == "thinking_max"
        elif role == NodeRole.DOC_REVIEWER:
            return self.config.doc_reviewer_mode.lower() in ("thinking", "thinking_max")
        elif role == NodeRole.CODE_REVIEWER:
            return self.config.code_reviewer_mode.lower() in ("thinking", "thinking_max")
        return False

    def _max_tokens_for(self, role: NodeRole) -> int:
        """Resolve the per-call max_tokens ceiling for ``role``.

        Looks up ``llm_dispatch.max_tokens_per_role.<role>`` from config;
        falls back to ``llm_dispatch.max_tokens_default``. Roles absent
        from the map (e.g. a future NodeRole addition shipped before the
        operator updates config) inherit the default — they don't crash.
        validate_config_strict already clamped both values into
        [256, 32768], so the result is always usable.
        """
        per_role = self.config.max_tokens_per_role or {}
        value = per_role.get(role.value)
        if isinstance(value, int) and value > 0:
            return value
        return self.config.max_tokens_default

    def _get_models_with_keys(self) -> list[tuple[str, ModelSpec]]:
        """Return all registered non-Ollama models that have a resolvable API key."""
        keyed: list[tuple[str, ModelSpec]] = []
        for model_key, spec in _MODEL_REGISTRY.items():
            if spec.provider == "ollama":
                continue
            # Check resolution order: env var → config api_key
            key = os.environ.get(f"{spec.provider.upper()}_API_KEY", "") or spec.api_key
            if key:
                keyed.append((model_key, spec))
        return keyed

    def _get_ollama_models(self) -> list[str]:
        """Return all registered Ollama model keys."""
        return [key for key, spec in _MODEL_REGISTRY.items() if spec.provider == "ollama"]

    async def dispatch(
        self,
        *,
        messages: list[dict[str, Any]],
        role: NodeRole,
        budget_remaining_usd: float,
        force_local: bool = False,
        model_override: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> tuple[LLMResponse, float]:
        """
        Dispatch an LLM call with full guardrails.

        Args:
            messages: The conversation messages array.
            role: Which graph node is making the call.
            budget_remaining_usd: Current remaining budget. If <= 0, the call is refused.
            force_local: If True, force local Ollama inference.
            model_override: Optional model key to use for this call only,
                bypassing `select_model(role)`. Use this for one-shot
                escalation (e.g., repair attempt 3 → reasoning model) instead
                of mutating gateway.config, which would race in concurrent
                dispatches and leak state on exception.
            **llm_kwargs: Additional parameters passed to the provider's chat_completion.

        Returns:
            A tuple of (LLMResponse, new_budget_remaining_usd).

        Raises:
            RuntimeError: If the budget is exhausted.
        """
        # Financial guardrail
        if budget_remaining_usd <= 0.0:
            try:
                from harness.observability import log_failure
                log_failure(
                    "token_budget_exhausted",
                    hard_cap_usd=self.config.hard_cap_usd,
                    budget_remaining_usd=budget_remaining_usd,
                    role=role.name if hasattr(role, "name") else str(role),
                )
            except Exception:  # noqa: BLE001 — telemetry must not mask the guardrail
                pass
            raise RuntimeError(
                f"[GUARDRAIL EXHAUSTED]: Active session hit the ${self.config.hard_cap_usd:.2f} threshold. "
                f"Budget remaining: ${budget_remaining_usd:.4f}"
            )

        # P1.9: 429/503 circuit breaker — if recent rate-limit bursts have
        # piled up, divert this call to local Ollama instead of burning
        # retries against a degraded provider. We do this BEFORE model
        # selection so explicit overrides also get diverted (the override
        # would have hit the same wall).
        if not force_local and not self.config.force_local_only and self._circuit_is_open():
            logger.warning(
                "[gateway] Rate-limit circuit OPEN — diverting role=%s to local Ollama.",
                role.value,
            )
            force_local = True

        # Select model + provider — explicit override wins over role-based routing.
        if model_override and not force_local:
            model_key = model_override
        else:
            model_key = self.select_model(role, force_local=force_local)
        thinking = self.should_use_thinking(role)

        # If budget is low and not forcing local, fall back to ollama to preserve budget
        if budget_remaining_usd < 0.05 and not force_local and not self.config.force_local_only:
            logger.info(
                "[gateway] Budget low ($%.4f). Switching to local Ollama to preserve remaining budget.",
                budget_remaining_usd,
            )
            model_key = f"ollama:{self.config.ollama_local_model}"
            force_local = True
            thinking = False

        provider = await self._get_provider(model_key)

        # --- Smart API Key Resolution ---
        if provider.spec.provider != "ollama" and not provider.api_key:
            # The configured model has no key. Scan all registered models for keys.
            keyed_models = self._get_models_with_keys()
            ollama_models = self._get_ollama_models()

            if len(keyed_models) == 1:
                # Exactly one model has a key — auto-consolidate all roles to it
                auto_model, auto_spec = keyed_models[0]
                logger.warning(
                    "[gateway] Configured model '%s' has no API key. "
                    "Only one model with a key found ('%s'). Auto-consolidating all roles to it.",
                    model_key, auto_model,
                )
                model_key = auto_model
                provider = await self._get_provider(auto_model)
            elif len(keyed_models) > 1:
                # Multiple models have keys — tell user which one to configure
                keyed_names = [name for name, _ in keyed_models]
                raise RuntimeError(
                    f"[API KEY MISSING]: No API key configured for '{model_key}'.\n"
                    f"  However, {len(keyed_models)} other model(s) already have keys configured: {', '.join(keyed_names)}.\n"
                    f"  To use '{model_key}', add its API key to <myharness_root>/config/config.json or set the "
                    f"{provider.spec.provider.upper()}_API_KEY environment variable.\n"
                    f"  To use a different model that already has a key, update 'model_routing' in your config to "
                    f"point to one of: {', '.join(keyed_names)}."
                )
            elif len(ollama_models) > 0:
                # No remote models have keys, but Ollama is available — suggest it
                raise RuntimeError(
                    f"[API KEY MISSING]: No API key configured for '{model_key}', and no other remote "
                    f"models have keys either.\n"
                    f"  However, local Ollama model(s) are registered: {', '.join(ollama_models)}.\n"
                    f"  To use local inference, set \"force_local_only\": true in your config's "
                    f"model_routing section, or add an API key as described below.\n"
                    f"\n"
                    f"  To add an API key, either:\n"
                    f"  1. Set the {provider.spec.provider.upper()}_API_KEY environment variable\n"
                    f"  2. Add \"api_key\" to the model entry in <myharness_root>/config/config.json"
                )
            else:
                # No models have keys at all — standard error
                env_var = f"{provider.spec.provider.upper()}_API_KEY"
                raise RuntimeError(
                    f"[API KEY MISSING]: No API key configured for '{model_key}'. "
                    f"No other registered models have API keys either.\n"
                    f"\n"
                    f"  To fix this, you have two options:\n"
                    f"\n"
                    f"  1. Set the {env_var} environment variable:\n"
                    f"     export {env_var}=\"your-api-key-here\"\n"
                    f"\n"
                    f"  2. Add \"api_key\" to the model entry in <myharness_root>/config/config.json:\n"
                    f"     {{\n"
                    f"       \"models\": {{\n"
                    f"         \"{model_key}\": {{\n"
                    f"           \"provider\": \"{provider.spec.provider}\",\n"
                    f"           \"model_id\": \"{provider.spec.model_id}\",\n"
                    f"           \"api_key\": \"your-api-key-here\",\n"
                    f"           ...\n"
                    f"         }}\n"
                    f"       }}\n"
                    f"     }}"
                )

        spec = provider.spec

        # --- Redact secrets from messages before transmission ---
        # Fail-closed: if the redactor cannot be loaded (missing module,
        # syntax error in the file, broken install), refuse to send rather
        # than ship raw messages — silent skip would mean secrets in
        # outbound API calls. The redactor is a hard dependency of the
        # gateway's security contract.
        try:
            from harness.redactor import redact_messages
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Refusing to dispatch: secret redactor unavailable ({e!r}). "
                f"Fix harness.redactor before retrying."
            ) from e
        messages = redact_messages(messages)

        # Anchor system prompt at messages[0] for prefix caching
        messages = ensure_prefix_cache_anchor(list(messages))

        # Pre-flight context window guardrail
        messages = await check_context_window(
            messages,
            spec,
            threshold_pct=self.config.context_window_threshold_pct,
        )

        # Pre-flight budget estimate (P1.4). The post-call guard at the top
        # of dispatch refuses calls when the budget is already <= 0, but it
        # doesn't catch the "budget is positive but the next call costs more
        # than it" case — a single big planning call could overspend by its
        # own cost. We compute a rough projected cost from message length
        # plus a reserve for the response and refuse early if it would push
        # us past the hard cap. Estimation is intentionally pessimistic
        # (chars/4 + 4k output reserve) — better to refuse a borderline
        # call than overspend the cap silently.
        try:
            est_input_chars = sum(
                len(m.get("content", "")) if isinstance(m.get("content", ""), str) else 0
                for m in messages
            )
            est_input_tokens = max(1, est_input_chars // 4)
            est_output_tokens = 4000  # pessimistic reserve for the response
            est_cost = (
                (est_input_tokens / 1_000_000.0) * spec.input_cost_per_1m
                + (est_output_tokens / 1_000_000.0) * spec.output_cost_per_1m
            )
            if est_cost > budget_remaining_usd:
                raise BudgetTooLowError(
                    "Pre-flight estimate $%.4f exceeds remaining budget $%.4f "
                    "for role=%s model=%s (est_input_tokens=%d). Aborting "
                    "before dispatch to keep the hard cap honest."
                    % (est_cost, budget_remaining_usd, role.value, model_key, est_input_tokens)
                )
            # Early warning when we land within 20% of the cap. Helps the
            # operator notice they're approaching the wall before HITL fires.
            if budget_remaining_usd > 0 and est_cost > 0.8 * budget_remaining_usd:
                logger.warning(
                    "[gateway] Pre-flight estimate $%.4f is within 20%% of remaining "
                    "budget $%.4f (role=%s, model=%s). Consider raising the cap or "
                    "switching to a cheaper model.",
                    est_cost, budget_remaining_usd, role.value, model_key,
                )
        except BudgetTooLowError:
            raise
        except Exception as exc:  # noqa: BLE001 — estimate must never block valid calls
            logger.debug("[gateway] Pre-flight cost estimate failed: %s", exc)

        # Per-role max_tokens ceiling. Inject only when the caller hasn't
        # passed one explicitly, so one-shot call sites can still override
        # (e.g. summarizer prompts that want a hard 1024 cap regardless of
        # global config). The default per-role values come from
        # llm_dispatch.max_tokens_per_role in config.json; see
        # Gateway._max_tokens_for for resolution order.
        if "max_tokens" not in llm_kwargs:
            llm_kwargs["max_tokens"] = self._max_tokens_for(role)

        # Execute with retry/backoff
        logger.info(
            "[gateway] Dispatching to %s (role=%s, thinking=%s, max_tokens=%d).",
            model_key, role.value, thinking, llm_kwargs["max_tokens"],
        )

        import time as _time
        _dispatch_start = _time.monotonic()

        async def _call() -> LLMResponse:
            return await provider.chat_completion(
                messages=messages,
                thinking=thinking,
                **llm_kwargs,
            )

        # P1.9: instrument the retry path so a 429/503 burst that exhausts
        # retries gets recorded for the circuit breaker. Non-rate-limit
        # exceptions propagate unchanged.
        try:
            response = await retry_with_backoff(
                _call,
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay,
            )
        except httpx.HTTPStatusError as exc:
            try:
                status = int(exc.response.status_code)
            except Exception:  # noqa: BLE001
                status = 0
            if status == 429 or status >= 500:
                self._record_rate_limit_failure()
            raise

        # Empty-content guard (P1.5). retry_with_backoff handles transport
        # failures (429 / 5xx / connection) but not "200 OK with empty
        # content body" — that surface as a silent success. Retry up to two
        # extra times on a fresh dispatch before giving up; if still empty,
        # raise EmptyLLMResponseError so the caller (repair / HITL router)
        # can short-circuit to a clear operator message instead of wasting
        # three repair iterations.
        empty_retry_attempts = 2
        while (
            response.content is None
            or (isinstance(response.content, str) and not response.content.strip())
        ) and empty_retry_attempts > 0:
            logger.warning(
                "[gateway] Provider returned empty content (model=%s role=%s). "
                "Retrying (%d remaining).",
                response.model, role.value, empty_retry_attempts,
            )
            empty_retry_attempts -= 1
            try:
                response = await retry_with_backoff(
                    _call,
                    max_retries=self.config.max_retries,
                    base_delay=self.config.base_delay,
                )
            except Exception:  # noqa: BLE001 — let the empty path below raise cleanly
                break

        if response.content is None or (
            isinstance(response.content, str) and not response.content.strip()
        ):
            try:
                from harness.observability import log_failure
                log_failure(
                    "llm_empty_response",
                    role=role.value if hasattr(role, "value") else str(role),
                    model=getattr(response, "model", ""),
                )
            except Exception:  # noqa: BLE001
                pass
            raise EmptyLLMResponseError(
                f"Provider returned empty content for role={role.value} model="
                f"{getattr(response, 'model', '?')} after empty-retry exhaustion. "
                f"This commonly indicates a content filter, an exhausted token "
                f"budget on the provider side, or a malformed prompt. Surface to "
                f"HITL rather than looping."
            )

        # Deduct cost from budget
        cost = response.usage.cost_usd
        new_budget = budget_remaining_usd - cost
        elapsed_ms = round((_time.monotonic() - _dispatch_start) * 1000)

        logger.info(
            "[gateway] Response received. model=%s tokens_in=%d tokens_out=%d cache_hit=%d cost=$%.6f budget_left=$%.4f",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cached_tokens,
            cost,
            new_budget,
        )

        try:
            from harness.observability import emit_event
            emit_event(
                "llm_call",
                model=response.model,
                role=role.value,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                cached_tokens=response.usage.cached_tokens,
                cache_creation_tokens=response.usage.cache_creation_tokens,
                cost_usd=cost,
                budget_remaining_usd=new_budget,
                elapsed_ms=elapsed_ms,
                finish_reason=response.finish_reason,
            )
        except Exception:  # noqa: BLE001 — observability must never break dispatch
            pass

        return response, new_budget

    def aggregate_tokens(
        self,
        tracker: dict[str, Any],
        usage: TokenUsage,
    ) -> dict[str, Any]:
        """
        Merge token usage from a single LLM call into the cumulative tracker.

        Args:
            tracker: The current token_tracker dict from AgentState.
            usage: The TokenUsage from a single LLMResponse.

        Returns:
            Updated tracker dict.
        """
        tracker["total_input_tokens"] = tracker.get("total_input_tokens", 0) + usage.input_tokens
        tracker["total_output_tokens"] = tracker.get("total_output_tokens", 0) + usage.output_tokens
        tracker["total_cached_tokens"] = tracker.get("total_cached_tokens", 0) + usage.cached_tokens
        tracker["total_cache_creation_tokens"] = (
            tracker.get("total_cache_creation_tokens", 0) + usage.cache_creation_tokens
        )
        tracker["total_cost_usd"] = tracker.get("total_cost_usd", 0.0) + usage.cost_usd

        # Per-model breakdown
        per_model: dict[str, dict[str, Any]] = tracker.setdefault("per_model", {})
        model_key = f"{usage.model_name}"
        if model_key not in per_model:
            per_model[model_key] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "cache_creation_tokens": 0,
                "cost_usd": 0.0,
            }
        per_model[model_key]["input_tokens"] += usage.input_tokens
        per_model[model_key]["output_tokens"] += usage.output_tokens
        per_model[model_key]["cached_tokens"] += usage.cached_tokens
        per_model[model_key].setdefault("cache_creation_tokens", 0)
        per_model[model_key]["cache_creation_tokens"] += usage.cache_creation_tokens
        per_model[model_key]["cost_usd"] += usage.cost_usd

        return tracker


# ---------------------------------------------------------------------------
# 13. Gateway Factory from Config
# ---------------------------------------------------------------------------

def _validate_routing_keys(
    model_routing: dict[str, Any],
    registered_models: set[str],
) -> None:
    """
    Validate that all model routing keys reference models that exist in the registry.

    If a routing key references a non-existent model, provide a helpful error message
    suggesting the closest matching registered model (fuzzy matching for typo detection).

    Args:
        model_routing: The model_routing section from config.
        registered_models: Set of model keys currently in the registry.

    Raises:
        ValueError: If any routing key references an unregistered model.
    """
    routing_keys = [
        ("planning_primary", model_routing.get("planning_primary", "")),
        ("planning_fallback", model_routing.get("planning_fallback", "")),
        ("patching_primary", model_routing.get("patching_primary", "")),
        ("repair_primary", model_routing.get("repair_primary", "")),
        ("repair_fallback", model_routing.get("repair_fallback", "")),
        ("doc_reviewer_primary", model_routing.get("doc_reviewer_primary", "")),
        ("doc_reviewer_fallback", model_routing.get("doc_reviewer_fallback", "")),
        ("code_reviewer_primary", model_routing.get("code_reviewer_primary", "")),
        ("code_reviewer_fallback", model_routing.get("code_reviewer_fallback", "")),
        ("ollama_local_model", model_routing.get("ollama_local_model", "")),
        ("ollama_local_backup", model_routing.get("ollama_local_backup", "")),
    ]

    errors: list[str] = []
    for routing_field, model_key in routing_keys:
        if not model_key:
            continue  # Empty string means not configured — skip
        if model_key not in registered_models:
            # Find closest matching registered model using character-level similarity
            suggestion = _find_closest_match(model_key, registered_models)
            if suggestion:
                errors.append(
                    f"  - {routing_field}: '{model_key}' is not registered. "
                    f"Did you mean '{suggestion}'?"
                )
            else:
                errors.append(
                    f"  - {routing_field}: '{model_key}' is not registered, "
                    f"and no similar models were found in the registry."
                )

    if errors:
        registered_list = "\n    ".join(sorted(registered_models)) if registered_models else "(none)"
        ollama_model = model_routing.get("ollama_local_model", "")

        if ollama_model:
            # Auto-fall back: point all misconfigured routing keys to local Ollama
            logger.error(
                "Model routing references %d unregistered model(s):\n%s\n\n"
                "  Registered models:\n    %s\n\n"
                "  Auto-falling back to local Ollama model '%s' for these roles. "
                "Fix the typos in .harness_config.json to restore remote model routing.",
                len(errors),
                "\n".join(errors),
                registered_list,
                ollama_model,
            )
            # Return — no exception. The caller will proceed with Ollama as ultimate fallback.
            return

        # No Ollama configured — log the error but don't crash;
        # individual dispatch calls will handle missing providers gracefully.
        logger.error(
            "Model routing references %d unregistered model(s):\n%s\n\n"
            "  Registered models:\n    %s\n\n"
            "  No ollama_local_model configured for auto-fallback. "
            "Dispatch will fail at runtime. Fix the typos in .harness_config.json.",
            len(errors),
            "\n".join(errors),
            registered_list,
        )


def _find_closest_match(target: str, candidates: set[str]) -> Optional[str]:
    """
    Find the closest matching string from a set of candidates using a simple
    similarity heuristic (shared prefix length + character overlap ratio).

    Returns the best match if the similarity score is above a threshold, else None.
    """
    if not candidates:
        return None

    target_lower = target.lower()
    best_match: Optional[str] = None
    best_score = 0.0

    for candidate in candidates:
        candidate_lower = candidate.lower()

        # Compute a simple similarity score:
        # 1. Shared prefix bonus (up to min length of both strings)
        prefix_len = 0
        for a, b in zip(target_lower, candidate_lower):
            if a == b:
                prefix_len += 1
            else:
                break

        # 2. Character set overlap (Jaccard-like)
        target_chars = set(target_lower)
        candidate_chars = set(candidate_lower)
        intersection = len(target_chars & candidate_chars)
        union = len(target_chars | candidate_chars)
        jaccard = intersection / union if union > 0 else 0.0

        # 3. Shared substring bonus (sliding window of length 3)
        substring_score = 0
        window = 3
        target_substrings = {target_lower[i:i+window] for i in range(max(0, len(target_lower) - window + 1))}
        candidate_substrings = {candidate_lower[i:i+window] for i in range(max(0, len(candidate_lower) - window + 1))}
        if target_substrings:
            substring_score = len(target_substrings & candidate_substrings) / len(target_substrings)

        # Combined score: prefix gets highest weight, then substring overlap, then Jaccard
        max_prefix = max(len(target_lower), len(candidate_lower))
        prefix_ratio = prefix_len / max_prefix if max_prefix > 0 else 0.0
        score = (prefix_ratio * 0.5) + (substring_score * 0.3) + (jaccard * 0.2)

        if score > best_score:
            best_score = score
            best_match = candidate

    # Only suggest if similarity is reasonably high (> 0.4)
    if best_score >= 0.4 and best_match is not None:
        return best_match
    return None


def create_gateway_from_config(config_dict: dict[str, Any]) -> Gateway:
    """
    Build a Gateway instance from a .harness_config.json dictionary.

    Also registers any models defined in the 'models' section of the config,
    and validates that all routing keys reference registered models.

    Args:
        config_dict: Parsed JSON dict from .harness_config.json.

    Returns:
        Configured Gateway instance.

    Raises:
        ValueError: If routing keys reference unregistered models.
    """
    # Register models from the 'models' section
    register_models_from_config(config_dict)

    model_routing = config_dict.get("model_routing", {})

    # Validate routing keys against registered models (catches typos early)
    _validate_routing_keys(model_routing, set(_MODEL_REGISTRY.keys()))
    token_budget = config_dict.get("token_budget", {})
    node_throttle = config_dict.get("node_throttle", {})

    # Clamp review-cycle caps to [0, 5] so a misconfigured value cannot blow
    # past the safety budget. 0 is a valid runtime opt-out (suspends the
    # reviewer without clearing the model slot).
    def _clamp_cycles(raw: Any, default: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        if value < 0:
            logger.warning("Review cycle cap %d < 0; clamping to 0.", value)
            return 0
        if value > 5:
            logger.warning("Review cycle cap %d > 5; clamping to 5.", value)
            return 5
        return value

    def _clamp_repair_iterations(raw: Any) -> int:
        """Clamp ``node_throttle.max_patch_repair_iterations`` to [1, 10].

        1 is the floor: the operator wants a single repair attempt before
        HITL — anything less would mean "no repair loop at all," which
        defeats the point of the node. 10 is the ceiling: past that the
        graph is just burning budget on hopeless retries.
        """
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 3
        if value < 1:
            logger.warning(
                "max_patch_repair_iterations %d < 1; clamping to 1 (single attempt).",
                value,
            )
            return 1
        if value > 10:
            logger.warning(
                "max_patch_repair_iterations %d > 10; clamping to 10.",
                value,
            )
            return 10
        return value

    def _clamp_discovery_iterations(raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 10
        if value < 1:
            logger.warning(
                "max_discovery_iterations %d < 1; clamping to 1 (one pass only).",
                value,
            )
            return 1
        if value > 30:
            logger.warning(
                "max_discovery_iterations %d > 30; clamping to 30.", value,
            )
            return 30
        return value

    def _clamp_max_tokens(raw: Any, fallback: int) -> int:
        """Clamp llm_dispatch.max_tokens_* into [256, 32768].

        validate_config_strict already enforces this range when the
        section is present in config.json, but this clamp is the second
        line of defense for programmatic callers that hand-build a
        config dict (tests, embed-in-pipeline use cases).
        """
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return fallback
        if value < 256:
            logger.warning("max_tokens %d < 256; clamping to 256.", value)
            return 256
        if value > 32768:
            logger.warning("max_tokens %d > 32768; clamping to 32768.", value)
            return 32768
        return value

    llm_dispatch = config_dict.get("llm_dispatch", {}) or {}
    max_tokens_default = _clamp_max_tokens(
        llm_dispatch.get("max_tokens_default", 4096), 4096
    )
    raw_per_role = llm_dispatch.get("max_tokens_per_role", {}) or {}
    max_tokens_per_role: dict[str, int] = {}
    if isinstance(raw_per_role, dict):
        for role_name, role_mt in raw_per_role.items():
            if not isinstance(role_name, str) or not role_name.strip():
                continue
            max_tokens_per_role[role_name] = _clamp_max_tokens(
                role_mt, max_tokens_default
            )

    gateway_config = GatewayConfig(
        planning_primary=model_routing.get("planning_primary", ""),
        planning_mode=model_routing.get("planning_mode", "thinking_max"),
        planning_fallback=model_routing.get("planning_fallback", ""),
        patching_primary=model_routing.get("patching_primary", ""),
        patching_mode=model_routing.get("patching_mode", "non_thinking"),
        repair_primary=model_routing.get("repair_primary", ""),
        repair_fallback=model_routing.get("repair_fallback", ""),
        repair_mode=model_routing.get("repair_mode", "thinking"),
        doc_reviewer_primary=model_routing.get("doc_reviewer_primary", ""),
        doc_reviewer_mode=model_routing.get("doc_reviewer_mode", "thinking"),
        doc_reviewer_fallback=model_routing.get("doc_reviewer_fallback", ""),
        max_doc_review_cycles=_clamp_cycles(node_throttle.get("max_doc_review_cycles", 1), 1),
        code_reviewer_primary=model_routing.get("code_reviewer_primary", ""),
        code_reviewer_mode=model_routing.get("code_reviewer_mode", "thinking"),
        code_reviewer_fallback=model_routing.get("code_reviewer_fallback", ""),
        max_code_review_cycles=_clamp_cycles(node_throttle.get("max_code_review_cycles", 1), 1),
        max_discovery_iterations=_clamp_discovery_iterations(
            node_throttle.get("max_discovery_iterations", 10)
        ),
        max_patch_repair_iterations=_clamp_repair_iterations(
            node_throttle.get("max_patch_repair_iterations", 3)
        ),
        ollama_local_model=model_routing.get("ollama_local_model", ""),
        ollama_local_backup=model_routing.get("ollama_local_backup", ""),
        force_local_only=model_routing.get("force_local_only", False),
        hard_cap_usd=token_budget.get("hard_cap_usd", 2.00),
        context_window_threshold_pct=token_budget.get("context_window_threshold_pct", 0.85),
        ssl_verify=config_dict.get("ssl_verify", True),
        max_tokens_default=max_tokens_default,
        max_tokens_per_role=max_tokens_per_role,
    )
    return Gateway(gateway_config)
