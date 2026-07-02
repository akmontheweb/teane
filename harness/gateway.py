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
    # Auxiliary judgment calls (patcher-rejection diagnosis, HITL
    # escalation summary, autofix classification). They reuse the cheap
    # repair model but ship a tiny prompt with no shared system message,
    # so binding them to REPAIR's cache-drift key flips the recorded
    # prefix hash back and forth every call and forces auto-cache
    # misses on the real repair-loop dispatch. Distinct role → distinct
    # ``(session, role)`` drift bucket.
    JUDGMENT = "judgment"


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


class _SkipDriftDetection(Exception):
    """Internal control-flow signal: bail out of the prefix-drift block
    without logging a warning. Used to skip drift tracking for roles
    (JUDGMENT) whose prompts are intentionally one-shot.
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
    # B6 — structured tool-use bridge. Providers that support native
    # function/tool calling (Anthropic Messages API, OpenAI / DeepSeek /
    # Ollama OpenAI-compat) populate this list when the LLM emitted
    # ``tool_use`` / ``tool_calls`` blocks instead of (or alongside)
    # text patches. Shape: ``[{"name": str, "input": dict[str, Any],
    # "id": str | None}]``. Empty list means no tool calls — caller
    # should fall back to parsing ``content`` as the text DSL. Today
    # only the schema exists; provider wiring is gated behind
    # ``GatewayConfig.use_structured_tools`` and added in a follow-up.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Reasoning-model chain-of-thought. Populated when the provider
    # surfaces internal reasoning tokens separately from final ``content``
    # (OpenAI-compat: ``message.reasoning_content`` or ``message.thinking``;
    # Anthropic: ``type="thinking"`` content blocks). These tokens are
    # billed in ``usage.output_tokens`` even though they don't appear in
    # ``content`` — without this field they show up as "tokens charged
    # but no visible response", which makes debugging silently-thinking
    # models impossible. Default empty string for non-reasoning models.
    reasoning_content: str = ""


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
    # B6 capability flag — true for models that accept native function/tool
    # calling on their wire format. Anthropic 3.x+/4.x all support it;
    # OpenAI gpt-4+/o-series support it; DeepSeek v3+ supports it; Ollama
    # is per-model (llama 3.1+, qwen 2.5+, mistral nemo). When false, the
    # gateway refuses to attach PATCH_TOOLS to this model and the patching
    # path falls back to the text DSL automatically.
    supports_tools: bool = False
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
                supports_tools=bool(spec_dict.get("supports_tools", False)),
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
                    "supports_tools": baseline.supports_tools,
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
                supports_tools=bool(merged.get("supports_tools", False)),
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
                timeout=httpx.Timeout(300.0, connect=10.0),
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
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return a standardized response.

        ``tools`` is the canonical ``[{name, description, input_schema}, ...]``
        list from :mod:`harness.tool_schemas`. Providers translate it to their
        native wire format (Anthropic raw, OpenAI ``[{type: function, function:
        {...}}]``) and populate ``LLMResponse.tool_calls`` with parsed
        ``{name, input, id}`` dicts when the model emits tool-use blocks.
        ``None`` (default) keeps the legacy text-DSL behaviour.
        """
        ...

    @abstractmethod
    def extract_usage(self, raw_response: dict[str, Any]) -> TokenUsage:
        """Parse token usage metadata from the provider's raw response JSON."""
        ...

    @abstractmethod
    def compute_cost(self, usage: TokenUsage) -> float:
        """Compute USD cost based on token counts and model pricing rates."""
        ...


class ProviderEmbeddedError(RuntimeError):
    """Raised when a provider returns HTTP 200 but the JSON body
    carries a structured error object (Azure OpenAI quota path,
    intermediate proxies, some self-hosted servers). Audit §4.7."""

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.payload = payload or {}


def _parse_json_response(response: Any) -> dict[str, Any]:
    """
    Parse a JSON response body, converting JSONDecodeError into a
    synthesised HTTP 502 so retry_with_backoff treats it as a retryable
    server error rather than letting a real 200 + malformed body
    propagate as a non-retryable error. Audit §4.3.
    """
    try:
        return response.json()
    except Exception as exc:
        # Forge a 502 (Bad Gateway) so the existing 5xx retry path
        # picks this up — a transient proxy error page mid-stream
        # used to escape as a non-retryable 200 (audit §4.3).
        try:
            response.status_code = 502
        except Exception:  # noqa: BLE001 — best-effort
            pass
        raise httpx.HTTPStatusError(
            f"Malformed JSON in response body: {exc}",
            request=response.request,
            response=response,
        ) from exc


def _check_provider_embedded_error(data: Any) -> None:
    """Inspect a parsed provider response and raise ProviderEmbeddedError
    if the body carries an ``{"error": {...}}`` envelope despite a
    200-OK status. Audit §4.7."""
    if not isinstance(data, dict):
        return
    err = data.get("error")
    if not isinstance(err, dict):
        return
    msg = str(err.get("message") or err.get("type") or "provider error")
    raise ProviderEmbeddedError(msg, payload=err)


def _normalize_messages_for_openai_tools(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic-style typed-block tool turns into OpenAI's
    ``role=tool`` / ``tool_calls`` shape.

    The harness uses Anthropic's typed-block representation
    (``content=[{"type": "tool_use", ...}]`` for the assistant turn,
    ``content=[{"type": "tool_result", "tool_use_id": ..., "content":
    ...}]`` for the follow-up user turn) as the *canonical* in-memory
    format. OpenAI / DeepSeek / Ollama OpenAI-compat want a different
    shape; this helper converts at the provider boundary so the
    canonical format stays one thing.

    Pass-through for any message that doesn't contain tool blocks —
    plain text messages survive unchanged.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        try:
                            args = json.dumps(block.get("input") or {})
                        except (TypeError, ValueError):
                            args = "{}"
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": args,
                            },
                        })
                if tool_calls:
                    msg_out: dict[str, Any] = {
                        "role": "assistant",
                        "content": ("\n".join(text_parts) or None),
                        "tool_calls": tool_calls,
                    }
                    out.append(msg_out)
                    continue
            elif role == "user":
                # Tool results land as one message per result with
                # role=tool in OpenAI's shape. Interleave any text blocks
                # as a follow-up user message so narration survives.
                tool_results: list[tuple[str, str]] = []
                trailing_text: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "tool_result":
                        tr_content = block.get("content", "")
                        if isinstance(tr_content, list):
                            # Nested content blocks → flatten to text.
                            tr_content = "\n".join(
                                str(b.get("text", "")) for b in tr_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        tool_results.append((
                            str(block.get("tool_use_id", "")),
                            str(tr_content),
                        ))
                    elif btype == "text":
                        trailing_text.append(str(block.get("text", "")))
                if tool_results:
                    for tool_use_id, tr_content in tool_results:
                        out.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tr_content,
                        })
                    if trailing_text:
                        out.append({
                            "role": "user",
                            "content": "\n".join(trailing_text),
                        })
                    continue
        out.append(msg)
    return out


def _extract_openai_compat_reasoning(message: dict[str, Any]) -> str:
    """Pull a reasoning-model's hidden chain-of-thought out of an
    OpenAI-compat chat-completions response message.

    Different reasoning models surface CoT under different keys on the
    same wire shape:
      - ``message.reasoning_content``: DeepSeek-Reasoner family, some
        OpenAI-compat self-hosts.
      - ``message.thinking``: Qwen3-thinking / DeepSeek-R1 over Ollama,
        some vLLM deployments.
      - ``message.reasoning``: OpenAI o-series via the legacy
        chat-completions shim (newer Responses API uses a different shape
        and goes through a different code path).

    All variants live under ``choices[0].message`` and are billed in
    ``completion_tokens`` even when ``content`` is empty. Returns the
    first non-empty string found, or ``""`` if the model didn't surface
    reasoning. Provider-agnostic — same helper for every OpenAI-shape
    backend so adding a new reasoning model is a price-catalogue edit,
    not a code change.
    """
    for key in ("reasoning_content", "thinking", "reasoning"):
        val = message.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _parse_openai_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate an OpenAI-compat ``message.tool_calls`` list into the
    harness's canonical ``[{name, input, id}, ...]`` shape.

    OpenAI ships ``arguments`` as a *JSON-encoded string* (not a dict),
    which is one of the most common pitfalls when wiring native tool-use.
    Each call gets a JSON.loads + a fallback that drops malformed calls
    rather than poisoning the patcher with garbage arguments.

    Used by all three OpenAI-shape providers (OpenAI, DeepSeek, Ollama-
    via-OpenAI-compat). Anthropic has its own typed-block parser inline
    in :meth:`AnthropicProvider.chat_completion`.
    """
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        args_raw = fn.get("arguments")
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                logger.warning(
                    "[gateway] Dropping tool_call '%s' with malformed JSON "
                    "arguments: %r",
                    name, args_raw[:200],
                )
                continue
            if not isinstance(parsed, dict):
                logger.warning(
                    "[gateway] Dropping tool_call '%s' — arguments parsed "
                    "to %s, expected object.",
                    name, type(parsed).__name__,
                )
                continue
            args = parsed
        else:
            args = {}
        out.append({
            "name": name,
            "input": args,
            "id": str(call.get("id") or ""),
        })
    return out


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
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        if tools:
            messages = _normalize_messages_for_openai_tools(messages)
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking and self.spec.supports_thinking:
            payload["thinking"] = {"type": "enabled"}
        if tools:
            from harness.tool_schemas import to_openai_tools
            payload["tools"] = to_openai_tools(tools)

        logger.debug("[deepseek] Sending completion request. model=%s tokens_est=%d", self.spec.model_id, len(messages))

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)
        _check_provider_embedded_error(data)  # audit §4.7

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        reasoning_content = _extract_openai_compat_reasoning(message)
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls = _parse_openai_tool_calls(message)

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
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
        tools: Optional[list[dict[str, Any]]] = None,
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

        # B6 native tool-use. Anthropic accepts the raw
        # ``{name, description, input_schema}`` shape directly via
        # ``tool_schemas.to_anthropic_tools``. When prompt caching is on,
        # the tool array is part of the cacheable prefix — anchor a
        # ``cache_control: ephemeral`` marker on the LAST tool so the
        # whole array caches together.
        if tools:
            from harness.tool_schemas import to_anthropic_tools
            anthropic_tools = to_anthropic_tools(tools)
            if (
                bool(getattr(self, "prompt_cache_enabled", True))
                and bool(self.spec.supports_cache)
                and anthropic_tools
            ):
                # Mutate a copy so the global PATCH_TOOLS list is untouched.
                anthropic_tools[-1] = {
                    **anthropic_tools[-1],
                    "cache_control": {"type": "ephemeral"},
                }
            payload["tools"] = anthropic_tools

        # Prompt caching. When the model declares ``supports_cache`` and the
        # gateway has not disabled it via ``prompt_cache_enabled=False``,
        # rewrite the system block into list-of-blocks form with a
        # ``cache_control: ephemeral`` marker. Optionally attach a second
        # breakpoint on the first user message when it carries the
        # immutable preamble (impact analysis / READ_FILE results /
        # planning blueprint). Anthropic allows up to 4 breakpoints;
        # 2 is enough for the harness's stable prefix shape.
        cache_enabled = (
            bool(getattr(self, "prompt_cache_enabled", True))
            and bool(self.spec.supports_cache)
        )
        if system_content:
            joined = "\n\n".join(system_content)
            if cache_enabled:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": joined,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = joined
        if cache_enabled and anthropic_messages:
            # Mark the first user message as a second cache breakpoint when
            # it's substantial (≥ ~4096 chars ≈ ~1024 tokens — Anthropic's
            # minimum block size for the ephemeral cache). Anything smaller
            # would be ignored by the server, so we keep the legacy string
            # form to avoid noise.
            first = anthropic_messages[0]
            first_content = first.get("content", "")
            if isinstance(first_content, str) and len(first_content) >= 4096:
                first["content"] = [
                    {
                        "type": "text",
                        "text": first_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

        # Extended thinking: must be opted in per request, and Anthropic requires
        # temperature=1.0 with thinking enabled. budget_tokens must be < max_tokens.
        if thinking and self.spec.supports_thinking:
            budget = max(1024, min(self.spec.thinking_budget_tokens, max_tokens - 512))
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1.0
            # Ensure max_tokens accommodates the thinking budget + visible reply
            if max_tokens <= budget:
                # Audit §4.16: log when we silently rewrite a caller-supplied
                # max_tokens upward to fit the thinking budget so cost-control
                # surprise doesn't go undetected.
                rewritten = budget + 1024
                logger.warning(
                    "[anthropic] thinking mode: caller-supplied max_tokens=%d "
                    "is below the %d-token thinking budget; rewriting to %d "
                    "(model=%s). Reduce models.%s.thinking_budget_tokens or "
                    "the per-role max_tokens cap to avoid this.",
                    max_tokens, budget, rewritten,
                    self.spec.model_id, self.spec.model_id,
                )
                payload["max_tokens"] = rewritten

        logger.debug("[anthropic] Sending completion request. model=%s thinking=%s",
                     self.spec.model_id, thinking and self.spec.supports_thinking)

        response = await client.post("/messages", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)
        _check_provider_embedded_error(data)  # audit §4.7

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        # Anthropic returns content as a list of typed blocks. Extract
        # the text parts AND the tool_use blocks separately — the model
        # can emit both in the same turn ("I'll start by reading the
        # file" + a tool_use call). Extended-thinking turns also surface
        # ``type=thinking`` (visible CoT) and ``type=redacted_thinking``
        # (opaque safety-classifier output) blocks; collect them into
        # ``reasoning_content`` so they parallel the OpenAI-shape
        # ``reasoning_content`` field instead of being silently dropped.
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thought = block.get("thinking") or block.get("text") or ""
                if thought:
                    reasoning_parts.append(thought)
            elif btype == "redacted_thinking":
                # Opaque ciphertext — we can't read it, but mark its
                # presence so debug dumps don't claim no reasoning happened.
                reasoning_parts.append("[redacted_thinking: opaque block from provider]")
            elif btype == "tool_use":
                tool_calls.append({
                    "name": block.get("name", ""),
                    "input": block.get("input") or {},
                    "id": block.get("id", ""),
                })
        content = "\n".join(text_parts)
        reasoning_content = "\n".join(reasoning_parts)

        finish_reason = data.get("stop_reason", "stop")

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
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
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        if tools:
            messages = _normalize_messages_for_openai_tools(messages)
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            from harness.tool_schemas import to_openai_tools
            payload["tools"] = to_openai_tools(tools)

        logger.debug("[openai] Sending completion request. model=%s", self.spec.model_id)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)
        _check_provider_embedded_error(data)  # audit §4.7

        usage = self.extract_usage(data)
        usage.cost_usd = self.compute_cost(usage)

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        reasoning_content = _extract_openai_compat_reasoning(message)
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls = _parse_openai_tool_calls(message)

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
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
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        if tools:
            messages = _normalize_messages_for_openai_tools(messages)
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            from harness.tool_schemas import to_openai_tools
            payload["tools"] = to_openai_tools(tools)

        logger.debug("[ollama] Sending completion request. model=%s", self.spec.model_id)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = _parse_json_response(response)
        _check_provider_embedded_error(data)  # audit §4.7

        usage = self.extract_usage(data)
        usage.cost_usd = 0.0  # Local inference is free

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        reasoning_content = _extract_openai_compat_reasoning(message)
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls = _parse_openai_tool_calls(message)

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.spec.model_id,
            finish_reason=finish_reason,
            raw_response=data,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
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


def hash_stable_prefix(
    messages: list[dict[str, Any]],
    n_stable: int = 2,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Hash the first ``n_stable`` messages of a request payload.

    Used by :class:`Gateway` to detect prefix drift across calls in the
    same (session, role) pair. OpenAI and DeepSeek auto-caches only fire
    when the request prefix is byte-identical; even one whitespace drift
    kills the hit. When this hash changes between consecutive calls for
    the same role, the gateway logs a warning + emits the
    ``cache_prefix_drift`` observability event so we can trace the leak
    back to the graph node that mutated a "should be stable" segment.

    ``tools`` is folded into the hash when native tool-use is on —
    Anthropic includes the tool array in its cacheable prefix, so a
    change to the tool definitions has the same cache-miss footprint as
    a change to the system prompt. We surface it through the same
    drift-detection channel.

    The hash is deterministic across processes (no Python ``hash()`` salt),
    so it's safe to compare across resumed sessions. SHA-256 is overkill
    for collision resistance here but matches what
    ``ensure_prefix_cache_anchor`` already uses and stays under 5 µs per
    call.
    """
    h = hashlib.sha256()
    for msg in messages[: max(0, n_stable)]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        h.update(role.encode("utf-8") if isinstance(role, str) else str(role).encode("utf-8"))
        h.update(b"|")
        if isinstance(content, str):
            h.update(content.encode("utf-8"))
        else:
            h.update(json.dumps(content, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\n---\n")
    if tools:
        h.update(b"tools:")
        h.update(json.dumps(tools, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _serialize_prefix_for_diff(
    messages: list[dict[str, Any]],
    n_stable: int = 2,
) -> str:
    """Phase 3(b) — return a stable text serialisation of the first
    ``n_stable`` messages so a subsequent call can diff against it.

    Uses the same shape rules as ``hash_stable_prefix`` (role|content
    blocks separated by ``\\n---\\n``) so the diff aligns with the hash
    boundaries the cache-drift detector saw. JSON content is serialised
    with sort_keys=True to ensure dict ordering doesn't cause spurious
    diffs.
    """
    parts: list[str] = []
    for msg in messages[: max(0, n_stable)]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            content_str = content
        else:
            content_str = json.dumps(content, sort_keys=True, default=str)
        parts.append(f"{role}|{content_str}")
    return "\n---\n".join(parts)


def _summarize_prefix_diff(
    *, old: str, new: str, max_excerpt_chars: int = 240,
) -> dict[str, Any]:
    """Phase 3(b) — produce a compact, observability-event-friendly
    summary of where two prefix snapshots first diverge.

    Returns a dict with:
      ``size_old`` / ``size_new`` — byte length of each snapshot
      ``first_diff_offset`` — character offset where they first differ
      ``first_diff_line`` — 1-indexed line where they first differ
      ``before_excerpt`` — short slice from the OLD prefix at the
        divergence point (so the post-mortem can see what was there)
      ``after_excerpt`` — short slice from the NEW prefix at the
        divergence point (so the post-mortem can see what changed it)

    When the snapshots are byte-identical (drift hash false-positive,
    rare but possible with collision-resistant SHA-256 we can rule it
    out), returns ``{"identical": True}``.
    """
    if old == new:
        return {"identical": True}
    n = min(len(old), len(new))
    first_diff = n  # default if one is a strict prefix of the other
    for i in range(n):
        if old[i] != new[i]:
            first_diff = i
            break
    line_no = old.count("\n", 0, first_diff) + 1
    half = max_excerpt_chars // 2
    before = old[max(0, first_diff - half): first_diff + half]
    after = new[max(0, first_diff - half): first_diff + half]
    return {
        "size_old": len(old),
        "size_new": len(new),
        "first_diff_offset": first_diff,
        "first_diff_line": line_no,
        "before_excerpt": before,
        "after_excerpt": after,
    }


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
    max_total_seconds: float = 300.0,
    rate_limit_observer: Optional[Callable[[], None]] = None,
    **kwargs: Any,
) -> LLMResponse:
    """
    Execute an async LLM call with exponential backoff + random jitter.

    Handles HTTP 429 (rate limit), 5xx (server errors), connection errors,
    AND read/connect/write timeouts (audit §4.1 — these all derive from
    ``httpx.TimeoutException`` which the earlier except clause omitted).

    After max_retries OR after ``max_total_seconds`` of cumulative sleep
    (audit §4.13), re-raises the last exception. Headers like
    ``Retry-After: 86400`` are now clamped to ``max_delay`` so a
    misconfigured provider can't make a single dispatch sleep for 24h.

    Jitter applied to the clamped delay (audit §4.12) so that long
    backoffs don't collapse to ~max_delay-floor with zero variance —
    that produced synchronized retry storms across concurrent dispatches.

    ``rate_limit_observer`` (audit §4.2): invoked once per 429 received,
    not just once per fully-exhausted dispatch — the gateway uses this
    to drive the circuit breaker on individual 429s.

    Backoff: ``delay = min(base_delay * 2^attempt, max_delay)``
             then jittered via ``delay * (0.5 + random * 0.5)``.
    """
    last_exception: Optional[Exception] = None
    cumulative_sleep = 0.0
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                if rate_limit_observer is not None:
                    try:
                        rate_limit_observer()
                    except Exception:  # noqa: BLE001
                        pass
                delay = _delay_from_rate_limit_headers(exc.response.headers, base_delay, attempt)
                logger.warning("[gateway] Rate limited (429). Attempt %d/%d. Delay=%.2fs",
                                attempt + 1, max_retries + 1, delay)
            elif status >= 500:
                delay = base_delay * (2 ** attempt)
                logger.warning("[gateway] Server error (%d). Attempt %d/%d.", status, attempt + 1, max_retries + 1)
            else:
                raise  # Non-retryable HTTP error (4xx except 429)
            last_exception = exc
        except httpx.TimeoutException as exc:
            # Audit §4.1: Connect / Read / Write / Pool timeouts all
            # derive from TimeoutException — catching them here makes
            # the most common transient failure mode retryable.
            delay = base_delay * (2 ** attempt)
            logger.warning("[gateway] Timeout (%s). Attempt %d/%d. %s",
                           type(exc).__name__, attempt + 1, max_retries + 1, exc)
            last_exception = exc
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            delay = base_delay * (2 ** attempt)
            logger.warning("[gateway] Connection error. Attempt %d/%d. %s", attempt + 1, max_retries + 1, exc)
            last_exception = exc

        if attempt < max_retries:
            # Clamp to max_delay BEFORE applying jitter so jitter range
            # stays meaningful at high attempts (audit §4.12).
            delay = min(delay, max_delay)
            jittered = delay * (0.5 + random.random() * 0.5)
            # Enforce overall budget on cumulative sleep (audit §4.13).
            if cumulative_sleep + jittered > max_total_seconds:
                logger.warning(
                    "[gateway] Backoff cumulative-sleep budget (%.0fs) "
                    "exhausted after %d attempt(s); giving up.",
                    max_total_seconds, attempt + 1,
                )
                break
            cumulative_sleep += jittered
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

    # 1. Retry-After (numeric seconds or HTTP-date). Clamp to a sane
    # ceiling so a misconfigured provider returning ``Retry-After: 86400``
    # can't make us sleep for 24h on a single dispatch (audit §4.13).
    _RA_MAX = 300.0
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), _RA_MAX))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(retry_after)
                delta = (dt - datetime.now(timezone.utc)).total_seconds()
                if delta > 0:
                    return min(delta, _RA_MAX)
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
                    # Audit §4.15: cap so a malformed future timestamp
                    # can't pin one dispatch for hours.
                    return min(delta, _RA_MAX)
            except (TypeError, ValueError):
                pass

    # 3. OpenAI X-RateLimit-Reset (epoch seconds). Audit §4.15: the
    # earlier ``target > now`` heuristic misclassified stale epoch
    # values that happen to be in the past, returning the raw value
    # (1.7 billion seconds) as the delay. Cap both interpretations to
    # ``_RA_MAX`` so a malformed header can never make us sleep
    # forever.
    x_reset = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset-requests")
    if x_reset:
        try:
            target = float(x_reset)
            now = datetime.now(timezone.utc).timestamp()
            # If value looks like a recent epoch (within ±30 days of now)
            # interpret as epoch; otherwise treat as seconds-from-now.
            month_seconds = 30 * 86400.0
            if abs(target - now) < month_seconds and target >= now:
                delta = target - now
            else:
                delta = max(0.0, target)
            if delta > 0:
                return min(delta, _RA_MAX)
        except ValueError:
            pass

    # 4. RFC 9651 RateLimit-Reset (seconds-from-now)
    rl_reset = headers.get("RateLimit-Reset")
    if rl_reset:
        try:
            return max(0.0, min(float(rl_reset), _RA_MAX))
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
    # Per-variant thinking mode for the fallback path. Defaults to the
    # primary's mode when unset (legacy semantics: one mode for both).
    planning_fallback_mode: str = ""
    patching_primary: str = ""
    patching_mode: str = "non_thinking"
    # Patching now supports a fallback model (added in the configure-page
    # overhaul). Empty primary stays disabled by ``_REQUIRED_ROUTING_FIELDS``.
    patching_fallback: str = ""
    patching_fallback_mode: str = ""
    repair_primary: str = ""
    repair_fallback: str = ""
    repair_mode: str = "thinking"
    repair_fallback_mode: str = ""
    # Doc reviewer — fully independent of code reviewer. Empty primary == disabled.
    doc_reviewer_primary: str = ""
    doc_reviewer_mode: str = "thinking"
    doc_reviewer_fallback: str = ""
    doc_reviewer_fallback_mode: str = ""
    max_doc_review_cycles: int = 1
    # Code reviewer — fully independent of doc reviewer. Empty primary == disabled.
    code_reviewer_primary: str = ""
    code_reviewer_mode: str = "thinking"
    code_reviewer_fallback: str = ""
    code_reviewer_fallback_mode: str = ""
    max_code_review_cycles: int = 1
    # Hard ceiling on rounds of the discovery interview loop. Without this,
    # a confused user (or hostile LLM) can loop indefinitely on follow-up
    # questions, burning budget. Clamped to [1, 30] at config load.
    max_discovery_iterations: int = 10
    # Hard ceiling on build → repair → compile retries after the initial
    # patching pass. After this many failed repair attempts the router
    # diverts to HITL instead of looping forever. Clamped to [1, 10] at
    # config load. Wired from node_throttle.max_patch_repair_iterations.
    # Default raised from 3 → 5: empirically the TaskDispatcher-shaped
    # multi-error tasks need >3 iterations to converge even when each
    # iteration lands real patches (see A5 in the post-mortem). The
    # speculative path also consumes part of this budget, so 3 leaves no
    # headroom for actual repair when speculative ends up salvaging.
    max_patch_repair_iterations: int = 5
    # Hard ceiling on consecutive repair rounds where the per-round
    # reflection LLM verdicts DISTRACTION or REGRESSION. The existing
    # ``max_patch_repair_iterations`` / ``no_progress_repairs`` gate
    # resets on any fingerprint-set shrinkage; an LLM that oscillates
    # the failing set (6 → 4 → 6 → 4) never trips it, because each
    # shrinkage credits the round as progress. This counter listens to
    # the reflection verdict directly: when the judgment says the
    # repair LLM isn't touching the real blocker N rounds in a row,
    # the router escalates to HITL. Clamped to [1, 10] at config load.
    max_consecutive_distraction_rounds: int = 3
    # Phase G — end-of-session regression repair cap. Caps the
    # repair → recompile loop the harness runs after security_scan
    # passes but before deployment. Read by
    # ``route_after_end_of_session_regression`` in harness/graph.py.
    max_end_of_session_regression_cycles: int = 3
    # Phase J — end-of-session repair authority. The default per-batch
    # repair shows the LLM up to 12 diagnostic-file slices + 50 inventory
    # files; at the EoS regression a security-scan repair may have
    # touched shared utilities so the failing test set can implicate
    # many more files. Raising the caps gives the senior reasoning model
    # enough surface to spot a cross-file cascade. Wired by
    # ``_repair_file_caps`` in harness/graph.py.
    end_of_session_repair_diagnostic_cap: int = 30
    end_of_session_repair_inventory_cap: int = 150
    # Phase J — when True, EoS regression repair jumps straight to the
    # reasoning model on the first attempt rather than burning
    # cheap-model rounds first. Off to fall back to the cycle-driven
    # escalation rule (escalates only on the LAST attempt before HITL).
    end_of_session_force_reasoning_model: bool = True
    ollama_local_model: str = ""
    ollama_local_backup: str = ""
    force_local_only: bool = False
    hard_cap_usd: float = 2.00
    # Optional per-stage soft budget allocation (C4 scaffold). Maps
    # NodeRole values to target fractions of hard_cap_usd. Today only
    # surfaces warnings when a stage exceeds its share; hard enforcement
    # is a follow-up. Empty dict = no per-stage warnings.
    stages: dict[str, float] = field(default_factory=dict)
    # Observability flag: when true, every LLM dispatch (across ALL roles)
    # writes its input messages + response to
    # ~/.harness/debug/<sid>_<seqno>_<role>_<model>.txt for ground-truth
    # debugging and post-mortem analysis. Wired from debug.dump_llm_calls
    # in config.json. The legacy debug.dump_repair_prompts flag is honoured
    # as an alias for backwards compatibility (see load_gateway_config).
    dump_llm_calls: bool = False
    # Cap on the number of .txt files kept under ~/.harness/debug. Oldest
    # (by mtime) are pruned on each write once exceeded. 0 disables pruning.
    # Wired from debug.dump_max_files in config.json. 5000 keeps roughly
    # the last 250 runs at ~20 dispatches/run, fitting easily in <100 MB.
    dump_max_files: int = 5000
    # B5: when true, the patcher rejects REPLACE_BLOCK / DELETE_BLOCK /
    # INSERT_AT_BLOCK against any file the LLM has not yet been shown this
    # turn (via pre-flight injection, READ_FILE resolution, or the patcher's
    # closest-match window). Mirrors Claude Code's Read-before-Edit
    # invariant. Default true: blind REPLACE_BLOCK on an unread file is the
    # most common patch-failure mode for web-app builds (lots of small
    # files), so the gate is on out of the box. Operators who depend on
    # the lax mode can set ``patcher.enforce_read_before_edit: false`` in
    # config.json. Drift detection (per-file sha256 comparison) runs
    # unconditionally whenever the host has recorded a hash for the file.
    enforce_read_before_edit: bool = True
    # B6: when true, providers that support native function/tool calling
    # (Anthropic Messages API, OpenAI / DeepSeek / Ollama OpenAI-compat)
    # pass the PATCH_TOOLS schema (see harness/tool_schemas.py) as
    # ``tools=...`` on their chat_completion call, and parse ``tool_use`` /
    # ``tool_calls`` responses into ``LLMResponse.tool_calls``. The
    # patching/repair nodes translate each tool call to a PatchBlock and
    # feed the existing apply pipeline. Falls back to text DSL parsing on
    # providers that don't support tool-use. Default false until the
    # per-provider wiring lands and is exercised in production.
    use_structured_tools: bool = False
    context_window_threshold_pct: float = 0.85
    max_retries: int = 5
    base_delay: float = 1.0
    # TLS: set to a CA bundle path (str) for corporate proxies, or False to
    # disable verification in air-gapped envs (not recommended for production).
    ssl_verify: Union[bool, str] = True
    # Per-call max_tokens ceiling. Used by Gateway._max_tokens_for(role).
    # max_tokens_default is the fallback when a role isn't listed in
    # max_tokens_per_role. Provided ints are clamped to [256, 32768] in
    # validate_config_strict. ``None`` means "no limit" — the gateway
    # omits the ``max_tokens`` kwarg from the provider call and lets the
    # provider's own per-request output cap take over. A blank per-role
    # entry overrides the default with "no limit" (it does NOT inherit
    # max_tokens_default); a missing per-role entry inherits the default.
    # For reasoning-mode models (deepseek-v4-pro) the ceiling is shared
    # between the hidden thinking trace and the visible content — bumping
    # repair to 8192 is the recommended baseline so the thinking trace +
    # patch blocks both fit.
    max_tokens_default: Optional[int] = None
    max_tokens_per_role: dict[str, Optional[int]] = field(default_factory=dict)
    # Prompt caching master switch. When True (default) and the selected
    # model carries ``supports_cache=True``, the gateway emits provider-
    # specific cache directives:
    #   - Anthropic: rewrites the system block to list-of-blocks form with
    #     ``cache_control: {"type": "ephemeral"}``; marks the first user
    #     message as a second breakpoint when it exceeds ~1024 tokens.
    #   - OpenAI / DeepSeek: nothing on the wire (server-side auto-cache);
    #     the prefix-stability hasher reports drift so we notice when an
    #     immutable prefix accidentally mutates and silently kills the
    #     cache hit.
    # Flip to False in ~/.harness/config.json (llm_dispatch.prompt_cache_enabled)
    # to fall back to the legacy string-form system payload — single-flag
    # rollback if any provider rejects the cache directives.
    prompt_cache_enabled: bool = True
    # Four cheap, opt-out LLM-judgment calls added on top of the deterministic
    # routers / autofix dispatcher. Each is on by default with an individual
    # kill switch, wired from the ``llm_judgment`` section of config.json:
    #   hitl_escalation_summary  — one-paragraph operator briefing emitted by
    #     human_intervention_node when a loop-stuck tripwire fires (replaces
    #     the bare trigger string).
    #   patcher_rejection_diagnosis — diagnoses why prior patches were
    #     rejected (allowlist miss vs stale context vs wrong file) and
    #     prepends actionable advice to the next repair prompt.
    #   preflight_autofix_judgment — on the first MISSING_DEP / DEP_RESOLUTION
    #     iteration, classifies each unique missing symbol as
    #     "manifest-fixable" vs "sandbox/toolchain mismatch" so futile
    #     autofix cycles are skipped and HITL is reached one round sooner.
    #   discovery_saturation_check — after each discovery round past the
    #     first, asks "given current answers + workspace evidence, is this
    #     section saturated?" and short-circuits the interview if yes.
    # All four reuse the repair role (cheap model + thinking-mode policy).
    # Disable any single one by setting llm_judgment.<name>: false; disable
    # the whole set by omitting the section and the repair_primary model
    # (the helper short-circuits when no model is routed).
    llm_judgment_hitl_escalation_summary: bool = True
    llm_judgment_patcher_rejection_diagnosis: bool = True
    llm_judgment_preflight_autofix: bool = True
    llm_judgment_discovery_saturation: bool = True
    # Phase 2.2 — per-round reflection. After each repair iteration
    # (from total_repairs >= 2 onward, when there's a prior round to
    # evaluate), a cheap LLM judges whether the previous round's patches
    # actually addressed the highest-priority error. If not, it names
    # the real blocker and injects that as a system message for the
    # current round's repair LLM. ~$0.001 per round. Off by setting
    # llm_judgment.repair_reflection=false.
    llm_judgment_repair_reflection: bool = True
    # Phase 4 — emit a structured JSON block of every diagnostic in the
    # repair prompt alongside the markdown summary, so the LLM has the
    # raw data and can override the harness's cascade ranking if it
    # disagrees. Capped at 25 items inside the formatter to bound token
    # cost. Off by setting repair_structured_diagnostic_payload=false in
    # config to fall back to markdown-only.
    repair_structured_diagnostic_payload: bool = True
    # Sixth touchpoint (added alongside the saturation check): on each
    # discovery follow-up round, picks the 3-5 sectors most worth
    # re-auditing this round and splices them into the prompt's
    # ``{FOCUS_SECTORS_BLOCK}`` slot. Saturation decides whether to keep
    # going; focus decides what to ask about next. Off by setting
    # llm_judgment.discovery_followup_focus=false — the follow-up prompt
    # then asks across every sector exactly as before.
    llm_judgment_discovery_followup_focus: bool = True
    # Fifth judgment touchpoint: one-sentence summary prefacing the
    # deterministic access-hint paragraph printed at exit by
    # installation_doc_node. Adds context the deterministic renderer
    # can't ("This is a Next.js storefront with a Stripe checkout flow
    # — open http://localhost:3000 and sign in as admin@example.com")
    # on top of the always-emitted URL/CLI hints. Off by setting
    # llm_judgment.app_usage_guide=false.
    llm_judgment_app_usage_guide: bool = True


# Filename pattern for universal LLM dumps written by Gateway._dump_llm_call_to_disk.
# Sequence-numbered, role-tagged, model-tagged so analysis tools can filter/sort.
_LLM_DUMP_GLOB_PREFIX = ""  # all .txt files in ~/.harness/debug are candidates


def _prune_debug_dumps(debug_dir: str, cap: int) -> None:
    """Keep ``debug_dir`` under ``cap`` files by deleting oldest entries.

    Operates on every ``.txt`` file in the directory — both the universal
    dumps and any legacy ``repair_<sid>_<iter>.txt`` files from prior
    sessions. Best-effort: silent on EPERM / file-races.
    """
    try:
        entries: list[tuple[float, str]] = []
        with os.scandir(debug_dir) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                if not ent.name.endswith(".txt"):
                    continue
                try:
                    entries.append((ent.stat().st_mtime, ent.path))
                except OSError:
                    continue
        if len(entries) <= cap:
            return
        entries.sort()  # oldest first
        excess = len(entries) - cap
        for _, path in entries[:excess]:
            try:
                os.remove(path)
            except OSError:
                continue
    except FileNotFoundError:
        return


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
        # Per-call dump counters (one monotonic seqno per session). asyncio.Lock
        # serializes increment so concurrent dispatches from speculative
        # variants get distinct filenames.
        import asyncio as _asyncio
        from itertools import count as _count
        self._dump_seqnos: dict[str, "_count[int]"] = {}
        self._dump_seqno_lock = _asyncio.Lock()
        # Defer the count() factory so the iter is created on first use
        self._count_factory = _count
        # Prefix-stability tracker. Maps (session_id, role) → last hash of
        # the first N "should be stable" messages. When the hash changes
        # between consecutive calls for the same key, we log a warning +
        # emit a ``cache_prefix_drift`` event so we can trace which graph
        # node accidentally mutated an immutable preamble (timestamps in
        # the planning blueprint, file mtimes in READ_FILE results, etc.).
        # Bounded in size: cleared when the gateway closes; in-memory only
        # so it doesn't survive process restarts (no value in resuming it).
        self._prefix_hashes: dict[tuple[str, str], str] = {}
        # Phase 3(b) — companion snapshot of the previous prefix as a
        # plain string per (session, role). When drift is detected, the
        # next call's prefix is diffed against this snapshot so the
        # observability event can carry "first changed line index" +
        # excerpt. Keeps a single string per key (~tens of KB max) so
        # the dict bounds match _prefix_hashes.
        self._prefix_snapshots: dict[tuple[str, str], str] = {}
        # Canonical session-wide token tracker. Every successful
        # ``dispatch`` call mutates this in place via ``aggregate_tokens``,
        # so it is structurally impossible to bypass: any caller that
        # reaches a provider goes through dispatch. The legacy
        # ``state["token_tracker"]`` mirror is left in place for
        # backward-compatibility, but this dict is the source of truth
        # for end-of-run / status / checkpoint displays.
        self.session_tracker: dict[str, Any] = {}

    def session_cost_summary(self) -> dict[str, Any]:
        """Defensive copy of the canonical session tracker.

        Use this — not ``state["token_tracker"]`` — when reporting cost
        to the operator. Returns a shallow-copied dict with deep-copied
        per_model / per_stage sub-dicts so callers can mutate freely
        without disturbing the live tracker.
        """
        import copy as _copy
        snapshot = dict(self.session_tracker)
        for key in ("per_model", "per_stage"):
            if key in snapshot:
                snapshot[key] = _copy.deepcopy(snapshot[key])
        return snapshot

    async def _get_provider(self, model_key: str) -> BaseLLM:
        """Get or create a cached provider instance."""
        if model_key not in self._providers:
            provider = create_provider(
                model_key, ssl_verify=self.config.ssl_verify
            )
            # Stamp the cache flag on the provider so it knows whether to
            # emit ``cache_control`` markers (Anthropic) or stay on the
            # legacy string-form system payload. Set as an attribute
            # (rather than threading through chat_completion kwargs) to
            # keep the provider signatures stable for the existing test
            # doubles (`_StubProvider` etc).
            provider.prompt_cache_enabled = bool(self.config.prompt_cache_enabled)  # type: ignore[attr-defined]
            self._providers[model_key] = provider
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

    async def _next_dump_seqno(self, session_id: str) -> int:
        """Allocate the next monotonic seqno for ``session_id`` dumps."""
        async with self._dump_seqno_lock:
            counter = self._dump_seqnos.get(session_id)
            if counter is None:
                counter = self._count_factory(1)
                self._dump_seqnos[session_id] = counter
            return next(counter)

    async def _dump_llm_call_to_disk(
        self,
        *,
        messages: list[dict[str, Any]],
        response: "LLMResponse",
        role: "NodeRole",
        cost_usd: float,
        elapsed_ms: int,
    ) -> None:
        """Persist a single LLM dispatch (input messages + response) to
        ``~/.harness/debug/<sid>_<seqno>_<role>_<model>.txt``.

        Gated by ``config.dump_llm_calls``. Best-effort: any I/O failure logs
        at debug and returns silently so dispatch is never blocked by dump
        problems. The file format mirrors the legacy
        ``_dump_repair_prompt_to_disk`` layout so existing tooling that reads
        ``repair_<sid>_<iter>.txt`` works on the unified files too.
        """
        if not bool(getattr(self.config, "dump_llm_calls", False)):
            return

        try:
            from harness.observability import get_active_session_id
            sid = get_active_session_id() or "unknown"
        except Exception:  # noqa: BLE001 — observability is best-effort here
            sid = "unknown"

        seqno = await self._next_dump_seqno(sid)
        sid_short = (sid.split("-")[0] if "-" in sid else sid) or "unknown"
        role_str = role.value if hasattr(role, "value") else str(role)
        model_short = (
            response.model.replace("/", "-").replace(":", "-")
            if isinstance(getattr(response, "model", None), str)
            else "unknown"
        )

        debug_dir = os.path.expanduser("~/.harness/debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(
            debug_dir,
            f"{sid_short}_{seqno:04d}_{role_str}_{model_short}.txt",
        )

        usage = response.usage
        header = (
            f"# LLM call {seqno}\n"
            f"# session: {sid}\n"
            f"# role: {role_str}  model: {response.model}  "
            f"finish: {response.finish_reason}\n"
            f"# tokens_in={usage.input_tokens}  "
            f"tokens_out={usage.output_tokens}  "
            f"cached={usage.cached_tokens}  "
            f"cost=${cost_usd:.6f}  elapsed_ms={elapsed_ms}\n"
        )
        sections = [header]
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            sections.append(
                f"\n---\n## input message {i}: role={msg.get('role', '?')}  "
                f"({len(content)} chars)\n---\n{content}\n"
            )
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        sections.append(
            f"\n---\n## response  ({len(response_text)} chars)\n---\n{response_text}\n"
        )
        # Reasoning-model chain-of-thought — invisible in `content` but
        # billed in `usage.output_tokens`. Without this section a 3000-token
        # response that looks like a 50-char stub on the wire is impossible
        # to debug. Always emit the section header so the dump shape is
        # stable; the body is empty for non-reasoning models.
        reasoning_text = getattr(response, "reasoning_content", "") or ""
        sections.append(
            f"\n---\n## reasoning_content  ({len(reasoning_text)} chars)\n---\n{reasoning_text}\n"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sections))

        logger.debug("[gateway] LLM call dumped: %s", path)

        # Retention: keep the dump dir under config.dump_max_files by
        # deleting the oldest files (mtime). Best-effort.
        cap = int(getattr(self.config, "dump_max_files", 0) or 0)
        if cap > 0:
            try:
                _prune_debug_dumps(debug_dir, cap)
            except Exception as exc:  # noqa: BLE001 — never block dispatch
                logger.debug("[gateway] dump retention skipped: %s", exc)

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
        elif role == NodeRole.JUDGMENT:
            # Auxiliary judgment calls (HITL summary, patcher-rejection
            # diagnosis, autofix classification) reuse the cheap repair
            # model — they're one-shot adviser calls, not part of the
            # repair loop itself. Distinct role keeps the gateway's
            # cache-drift bucket separate.
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
        elif role == NodeRole.JUDGMENT:
            # Adviser calls are cheap and short; thinking mode would
            # multiply their cost without changing the answer. Always off.
            return False
        elif role == NodeRole.DOC_REVIEWER:
            return self.config.doc_reviewer_mode.lower() in ("thinking", "thinking_max")
        elif role == NodeRole.CODE_REVIEWER:
            return self.config.code_reviewer_mode.lower() in ("thinking", "thinking_max")
        return False

    def _max_tokens_for(self, role: NodeRole) -> Optional[int]:
        """Resolve the per-call max_tokens ceiling for ``role``.

        Looks up ``llm_dispatch.max_tokens_per_role.<role>`` from config;
        falls back to ``llm_dispatch.max_tokens_default``. Returning
        ``None`` means "no limit" — the dispatch path omits the
        ``max_tokens`` kwarg and the provider's own per-request cap
        applies.

        A per-role entry whose key is present but value is blank
        (``None``) overrides the default with "no limit" rather than
        inheriting it. A missing key falls through to the default. Roles
        absent from the map (e.g. a future NodeRole addition shipped
        before the operator updates config) inherit the default — they
        don't crash. validate_config_strict already clamped provided
        ints into [256, 32768], so non-None results are always usable.
        """
        per_role = self.config.max_tokens_per_role or {}
        if role.value in per_role:
            value = per_role[role.value]
            return value if isinstance(value, int) and value > 0 else None
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
        tools: Optional[list[dict[str, Any]]] = None,
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
            # Audit §4.8: skip the rewrite when ``ollama_local_model`` isn't
            # configured. Without the guard, ``model_key = "ollama:"``
            # crashed in ``_get_provider`` with a cryptic ValueError exactly
            # when the operator most needed graceful degradation. Let the
            # dispatch continue with the originally-selected model and
            # surface a clear BudgetTooLowError if it ultimately fails.
            if not (self.config.ollama_local_model or "").strip():
                logger.warning(
                    "[gateway] Budget low ($%.4f) but ollama_local_model "
                    "is unset — staying on %s rather than crashing.",
                    budget_remaining_usd, model_key,
                )
            else:
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
                    f"  To use '{model_key}', add its API key to <teane_root>/config/config.json or set the "
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
                    f"  2. Add \"api_key\" to the model entry in <teane_root>/config/config.json"
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
                    f"  2. Add \"api_key\" to the model entry in <teane_root>/config/config.json:\n"
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

        # --- B6 native tool-use gate ---
        # ``tools`` only flows through to the provider when (a) the gateway
        # is configured for structured tools and (b) the routed model
        # actually supports them. If either is false we drop the tool
        # array silently and let the caller's text-DSL parsing handle the
        # response. This keeps an operator who flips ``use_structured_tools``
        # but routes patching to an Ollama model without tool support from
        # hard-failing every patching turn.
        effective_tools: Optional[list[dict[str, Any]]] = None
        if tools and self.config.use_structured_tools and spec.supports_tools:
            effective_tools = list(tools)
        elif tools and not (self.config.use_structured_tools and spec.supports_tools):
            logger.debug(
                "[gateway] tools= passed but suppressed (use_structured_tools=%s, "
                "model=%s supports_tools=%s).",
                self.config.use_structured_tools, model_key, spec.supports_tools,
            )

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

        # Prefix-stability drift detection. The first 2 messages of every
        # request should be byte-stable across calls in the same role
        # (system prompt + immutable preamble). When the hash changes,
        # OpenAI/DeepSeek auto-caches miss silently — surface it as an
        # observability event so we can chase the leak.
        #
        # JUDGMENT calls (HITL summary, patcher-rejection diagnosis,
        # autofix classifier) intentionally ship one-shot user-only
        # prompts that vary by purpose, so prefix drift is the norm and
        # tracking it just produces noise warnings. Skip the whole
        # detector for that role.
        try:
            from harness.observability import get_active_session_id, emit_event
            _sid = get_active_session_id() or "unknown"
        except Exception:  # noqa: BLE001 — telemetry is best-effort
            _sid = "unknown"
            emit_event = None  # type: ignore[assignment]
        try:
            if role == NodeRole.JUDGMENT:
                raise _SkipDriftDetection
            _prefix_hash = hash_stable_prefix(
                messages, n_stable=2, tools=effective_tools,
            )
            _drift_key = (_sid, role.value)
            _last_hash = self._prefix_hashes.get(_drift_key)
            if _last_hash is not None and _last_hash != _prefix_hash:
                logger.warning(
                    "[gateway] cache prefix drift role=%s session=%s prev=%s now=%s "
                    "— auto-cache will miss this call.",
                    role.value, _sid[:8] if _sid else "?",
                    _last_hash[:8], _prefix_hash[:8],
                )
                # Phase 3(b) — diff the prefix against the prior snapshot
                # so post-mortems can SEE what changed. The hash-only
                # event lets you grep for drift but tells you nothing
                # about the cause — so most teams have learned to
                # ignore it. Capture the first changed line index + a
                # short snippet of the new prefix at that offset.
                # All-best-effort: any error here downgrades silently
                # to the hash-only emit below.
                _diff_payload: dict[str, Any] = {}
                try:
                    _prev_snapshot = self._prefix_snapshots.get(_drift_key)
                    if _prev_snapshot is not None:
                        new_snapshot = _serialize_prefix_for_diff(
                            messages, n_stable=2,
                        )
                        _diff_payload = _summarize_prefix_diff(
                            old=_prev_snapshot, new=new_snapshot,
                        )
                except Exception:  # noqa: BLE001
                    _diff_payload = {}
                if emit_event is not None:
                    try:
                        emit_event(
                            "cache_prefix_drift",
                            role=role.value,
                            session_id=_sid,
                            prev_hash=_last_hash[:8],
                            now_hash=_prefix_hash[:8],
                            **_diff_payload,
                        )
                    except Exception:  # noqa: BLE001
                        pass
            self._prefix_hashes[_drift_key] = _prefix_hash
            # Stash a string snapshot of the new prefix so the NEXT
            # call can diff against it without re-serialising both
            # sides.
            try:
                self._prefix_snapshots[_drift_key] = (
                    _serialize_prefix_for_diff(messages, n_stable=2)
                )
            except Exception:  # noqa: BLE001
                pass
        except _SkipDriftDetection:
            pass  # JUDGMENT role intentionally skips drift tracking.
        except Exception as exc:  # noqa: BLE001 — drift telemetry must never block dispatch
            logger.debug("[gateway] prefix drift check skipped: %s", exc)

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
        # Gateway._max_tokens_for for resolution order. When the resolver
        # returns None ("no limit"), we skip injection entirely so the
        # provider's own per-request output cap applies.
        if "max_tokens" not in llm_kwargs:
            resolved_max_tokens = self._max_tokens_for(role)
            if resolved_max_tokens is not None:
                llm_kwargs["max_tokens"] = resolved_max_tokens

        # Execute with retry/backoff
        logger.info(
            "[gateway] Dispatching to %s (role=%s, thinking=%s, max_tokens=%s).",
            model_key, role.value, thinking,
            llm_kwargs.get("max_tokens", "unlimited"),
        )

        import time as _time
        _dispatch_start = _time.monotonic()

        async def _call() -> LLMResponse:
            return await provider.chat_completion(
                messages=messages,
                thinking=thinking,
                tools=effective_tools,
                **llm_kwargs,
            )

        # P1.9: instrument the retry path so a 429/503 burst that exhausts
        # retries gets recorded for the circuit breaker. Non-rate-limit
        # exceptions propagate unchanged.
        #
        # Audit §4.2: count EVERY 429 we encounter, not just full
        # exhaustion. The rate_limit_observer hook fires per-429 inside
        # retry_with_backoff so the breaker can trip on much shorter
        # bursts than the old (3 fully-exhausted dispatches) threshold
        # ever permitted (effectively ~18 actual 429s with max_retries=5).
        try:
            response = await retry_with_backoff(
                _call,
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay,
                rate_limit_observer=self._record_rate_limit_failure,
            )
        except httpx.HTTPStatusError as exc:
            try:
                status = int(exc.response.status_code)
            except Exception:  # noqa: BLE001
                status = 0
            if status == 429 or status >= 500:
                # _record_rate_limit_failure may have already been called
                # via the observer hook above; one extra record on full
                # exhaustion is fine — the deque is bounded.
                self._record_rate_limit_failure()
            raise

        # Empty-content guard (P1.5). retry_with_backoff handles transport
        # failures (429 / 5xx / connection) but not "200 OK with empty
        # content body" — that surface as a silent success. Retry up to two
        # extra times on a fresh dispatch before giving up; if still empty,
        # raise EmptyLLMResponseError so the caller (repair / HITL router)
        # can short-circuit to a clear operator message instead of wasting
        # three repair iterations.
        #
        # B6 caveat: in native tool-use mode the model can legitimately
        # emit a tool-only turn (zero text, one or more ``tool_use``
        # blocks). ``response.tool_calls`` carries those — treat the
        # response as non-empty even when ``content`` is blank.
        def _is_empty(r: LLMResponse) -> bool:
            if r.tool_calls:
                return False
            return r.content is None or (
                isinstance(r.content, str) and not r.content.strip()
            )

        # Track cumulative cost across empty-retries (audit §4.4). The
        # provider charges for every attempt server-side; the earlier
        # code deducted only the LAST response's cost so up to 2x of
        # the cost-per-call slipped past the local budget enforcer.
        accumulated_cost = float(getattr(response.usage, "cost_usd", 0.0) or 0.0)
        empty_retry_attempts = 2
        last_retry_exc: Optional[BaseException] = None
        while _is_empty(response) and empty_retry_attempts > 0:
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
                    rate_limit_observer=self._record_rate_limit_failure,
                )
                accumulated_cost += float(getattr(response.usage, "cost_usd", 0.0) or 0.0)
            except Exception as exc:  # noqa: BLE001 — captured below so the
                # operator sees the real failure (rate-limit / 5xx) rather
                # than the downstream "empty content" symptom that masked it.
                last_retry_exc = exc
                logger.warning(
                    "[gateway] Empty-retry call raised %s: %s — falling through to empty handler.",
                    type(exc).__name__, exc,
                )
                break

        if _is_empty(response):
            try:
                from harness.observability import log_failure
                log_failure(
                    "llm_empty_response",
                    role=role.value if hasattr(role, "value") else str(role),
                    model=getattr(response, "model", ""),
                    underlying_error=type(last_retry_exc).__name__ if last_retry_exc else "",
                )
            except Exception:  # noqa: BLE001
                pass
            if last_retry_exc is not None:
                # Surface the actual transport / rate-limit failure
                # rather than the downstream "empty content" symptom.
                # Chaining preserves the symptom for debugging while
                # presenting the real cause as the proximate exception.
                raise EmptyLLMResponseError(
                    f"Provider returned empty content for role={role.value} "
                    f"after retry raised {type(last_retry_exc).__name__}: "
                    f"{last_retry_exc}. Surface to HITL rather than looping."
                ) from last_retry_exc
            raise EmptyLLMResponseError(
                f"Provider returned empty content for role={role.value} model="
                f"{getattr(response, 'model', '?')} after empty-retry exhaustion. "
                f"This commonly indicates a content filter, an exhausted token "
                f"budget on the provider side, or a malformed prompt. Surface to "
                f"HITL rather than looping."
            )

        # Deduct cost from budget — use the accumulated sum across any
        # empty-retries (audit §4.4) so the operator's hard cap isn't
        # silently breached by partial-failure tails. Clamp the post-call
        # budget at zero rather than letting it go negative — negative
        # values used to surface in the dashboard as a negative dollar
        # figure even though the next dispatch correctly refuses at <=0.
        cost = accumulated_cost or response.usage.cost_usd
        new_budget = max(0.0, budget_remaining_usd - cost)
        elapsed_ms = round((_time.monotonic() - _dispatch_start) * 1000)

        # Canonical session tracker (impossible to bypass — every dispatch
        # lands here). Stamp the billed total onto response.usage first so
        # the per-model/per-stage rollups reflect the true charge across
        # any empty-retry tail rather than just the last call's cost.
        response.usage.cost_usd = cost
        self.aggregate_tokens(self.session_tracker, response.usage, role=role)

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

        # Universal per-call debug dump: persist input messages + response
        # text to ~/.harness/debug/<sid>_<seqno>_<role>_<model>.txt when
        # debug.dump_llm_calls is true. Captures EVERY role (planning,
        # patching, repair, doc/code review, test gen, discovery, etc.) so
        # future analysis has the exact bytes the LLM saw and produced.
        # See _dump_llm_call_to_disk for the format. Best-effort.
        try:
            await self._dump_llm_call_to_disk(
                messages=messages,
                response=response,
                role=role,
                cost_usd=cost,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001 — dump must never break dispatch
            logger.debug("[gateway] LLM-call dump skipped: %s", exc)

        return response, new_budget

    def aggregate_tokens(
        self,
        tracker: dict[str, Any],
        usage: TokenUsage,
        role: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Merge token usage from a single LLM call into the cumulative tracker.

        Args:
            tracker: The current token_tracker dict from AgentState.
            usage: The TokenUsage from a single LLMResponse.
            role: Optional NodeRole (or string) identifying the stage that
                spent this. When provided, accumulates into
                ``tracker["per_stage"][role]``. This is the observability
                substrate for per-stage budget sub-pools (C4): hard
                enforcement is a follow-up; today we emit per-stage
                tallies + soft warnings so the operator can see where
                budget actually goes. Callers that don't pass role get the
                same historical behaviour (no per-stage update).

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

        # Per-stage breakdown (additive — caller opt-in via the role param).
        if role is not None:
            role_key = role.value if hasattr(role, "value") else str(role)
            per_stage: dict[str, dict[str, Any]] = tracker.setdefault("per_stage", {})
            if role_key not in per_stage:
                per_stage[role_key] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                }
            per_stage[role_key]["input_tokens"] += usage.input_tokens
            per_stage[role_key]["output_tokens"] += usage.output_tokens
            per_stage[role_key]["cached_tokens"] += usage.cached_tokens
            per_stage[role_key]["cost_usd"] += usage.cost_usd
            per_stage[role_key]["calls"] += 1

        return tracker

    def track_embedding_call(
        self,
        model_key: str,
        prompt_tokens: int,
    ) -> float:
        """Account a ``/v1/embeddings`` call into the session tracker.

        Embeddings bypass ``dispatch`` (the call shape is different —
        no chat history, no role, no tool use), so without this helper
        their spend is invisible to every cost surface. We mirror the
        accounting tail of ``dispatch``: look up the model rate, build
        a ``TokenUsage``, aggregate into ``session_tracker`` (so the
        end-of-run summary and ``teane status`` see it), and emit an
        ``embedding_call`` observability event (so the JSONL replay
        in metrics.py / dashboard.py picks it up too).

        Returns the cost in USD. If the model isn't in the registry
        the call accounts at $0 and logs a warning rather than raising
        — embeddings must never break the index build.
        """
        spec = _MODEL_REGISTRY.get(model_key)
        if spec is None:
            logger.warning(
                "[gateway] No price spec for embedding model '%s' — "
                "accounting at $0. Add an entry to model_prices.json or "
                "register via config.models to track cost.",
                model_key,
            )
            cost = 0.0
        else:
            cost = (prompt_tokens / 1_000_000.0) * float(spec.input_cost_per_1m)

        usage = TokenUsage(
            input_tokens=int(prompt_tokens),
            output_tokens=0,
            model_name=model_key,
            cost_usd=cost,
        )
        # role=None — embeddings aren't a graph node; we still want the
        # totals and per_model rollups but not a per_stage bucket.
        self.aggregate_tokens(self.session_tracker, usage, role=None)

        try:
            from harness.observability import emit_event
            emit_event(
                "embedding_call",
                model=model_key,
                tokens_in=int(prompt_tokens),
                tokens_out=0,
                cached_tokens=0,
                cache_creation_tokens=0,
                cost_usd=cost,
            )
        except Exception:  # noqa: BLE001 — telemetry must not block work
            pass

        return cost


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
        substring_score = 0.0
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

    def _resolve_dump_llm_calls(cfg: dict[str, Any]) -> bool:
        """Resolve the universal LLM-call dump flag with legacy alias.

        Honors ``debug.dump_llm_calls`` (current name). When that key is
        absent but the legacy ``debug.dump_repair_prompts`` is present and
        truthy, treats it as opt-in and emits a one-time deprecation log so
        operators know to migrate. Default: False (callers must opt in).
        """
        debug = cfg.get("debug", {}) or {}
        if "dump_llm_calls" in debug:
            return bool(debug.get("dump_llm_calls"))
        legacy = debug.get("dump_repair_prompts")
        if legacy:
            logger.warning(
                "[gateway] debug.dump_repair_prompts is deprecated; "
                "honouring it as debug.dump_llm_calls=true. Migrate by "
                "renaming the key in your config.json."
            )
            return True
        return False

    def _resolve_dump_max_files(cfg: dict[str, Any]) -> int:
        """Resolve ``debug.dump_max_files`` with a sane default and clamp."""
        debug = cfg.get("debug", {}) or {}
        raw = debug.get("dump_max_files", 5000)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 5000
        if value < 0:
            return 0
        # Soft ceiling — anything above 100k is almost certainly a typo.
        return min(value, 100000)

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

    def _clamp_distraction_rounds(raw: Any) -> int:
        """Clamp ``node_throttle.max_consecutive_distraction_rounds`` to [1, 10].

        1 is the floor: at minimum, the operator wants the harness to
        escalate after a single DISTRACTION verdict (effectively
        trusting the judgment LLM). 10 is the ceiling: past that, the
        counter never trips before the existing ``max_patch_repair_iterations``
        ceiling fires, defeating its purpose.
        """
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 3
        if value < 1:
            logger.warning(
                "max_consecutive_distraction_rounds %d < 1; clamping to 1.",
                value,
            )
            return 1
        if value > 10:
            logger.warning(
                "max_consecutive_distraction_rounds %d > 10; clamping to 10.",
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

    def _resolve_max_tokens(raw: Any) -> Optional[int]:
        """Resolve an llm_dispatch.max_tokens_* value.

        Returns ``None`` (meaning "no limit") for blank inputs:
        missing key (None), empty string, or zero. validate_config_strict
        already rejects garbage strings / wrong types when the config
        comes from a config.json file, so the type-coercion branch here
        is the second line of defense for programmatic callers that
        hand-build a config dict (tests, embed-in-pipeline use cases).

        Provided positive ints are clamped to [256, 32768] for defense
        in depth.
        """
        if raw is None or raw == "" or raw == 0:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        if value < 256:
            logger.warning("max_tokens %d < 256; clamping to 256.", value)
            return 256
        if value > 32768:
            logger.warning("max_tokens %d > 32768; clamping to 32768.", value)
            return 32768
        return value

    llm_dispatch = config_dict.get("llm_dispatch", {}) or {}
    max_tokens_default = _resolve_max_tokens(
        llm_dispatch.get("max_tokens_default")
    )
    raw_per_role = llm_dispatch.get("max_tokens_per_role", {}) or {}
    max_tokens_per_role: dict[str, Optional[int]] = {}
    if isinstance(raw_per_role, dict):
        for role_name, role_mt in raw_per_role.items():
            if not isinstance(role_name, str) or not role_name.strip():
                continue
            max_tokens_per_role[role_name] = _resolve_max_tokens(role_mt)

    # Resolve per-variant thinking modes. The legacy ``<role>_mode`` key
    # supplies the primary's mode AND — when ``<role>_fallback_mode`` is
    # unset — the fallback's mode, preserving the old "one knob applies
    # to both" semantics. The new explicit fallback-mode key lets the
    # operator diverge the two paths from the configure page.
    def _mode(role: str, primary_default: str) -> str:
        return str(model_routing.get(f"{role}_mode", primary_default) or primary_default)

    def _fallback_mode(role: str, primary_default: str) -> str:
        explicit = str(model_routing.get(f"{role}_fallback_mode", "") or "")
        return explicit if explicit else _mode(role, primary_default)

    gateway_config = GatewayConfig(
        planning_primary=model_routing.get("planning_primary", ""),
        planning_mode=_mode("planning", "thinking_max"),
        planning_fallback=model_routing.get("planning_fallback", ""),
        planning_fallback_mode=_fallback_mode("planning", "thinking_max"),
        patching_primary=model_routing.get("patching_primary", ""),
        patching_mode=_mode("patching", "non_thinking"),
        patching_fallback=model_routing.get("patching_fallback", ""),
        patching_fallback_mode=_fallback_mode("patching", "non_thinking"),
        repair_primary=model_routing.get("repair_primary", ""),
        repair_fallback=model_routing.get("repair_fallback", ""),
        repair_mode=_mode("repair", "thinking"),
        repair_fallback_mode=_fallback_mode("repair", "thinking"),
        doc_reviewer_primary=model_routing.get("doc_reviewer_primary", ""),
        doc_reviewer_mode=_mode("doc_reviewer", "thinking"),
        doc_reviewer_fallback=model_routing.get("doc_reviewer_fallback", ""),
        doc_reviewer_fallback_mode=_fallback_mode("doc_reviewer", "thinking"),
        max_doc_review_cycles=_clamp_cycles(node_throttle.get("max_doc_review_cycles", 1), 1),
        code_reviewer_primary=model_routing.get("code_reviewer_primary", ""),
        code_reviewer_mode=_mode("code_reviewer", "thinking"),
        code_reviewer_fallback=model_routing.get("code_reviewer_fallback", ""),
        code_reviewer_fallback_mode=_fallback_mode("code_reviewer", "thinking"),
        max_code_review_cycles=_clamp_cycles(node_throttle.get("max_code_review_cycles", 1), 1),
        max_discovery_iterations=_clamp_discovery_iterations(
            node_throttle.get("max_discovery_iterations", 10)
        ),
        max_patch_repair_iterations=_clamp_repair_iterations(
            node_throttle.get("max_patch_repair_iterations", 5)
        ),
        max_consecutive_distraction_rounds=_clamp_distraction_rounds(
            node_throttle.get("max_consecutive_distraction_rounds", 3)
        ),
        # Phase G + Phase J — end-of-session repair / regression knobs.
        # Clamp the cycle / cap fields to sane ranges so a bogus
        # operator value (e.g. cycles=0 or cap=999999) doesn't silently
        # disable the gate or blow up prompt sizes.
        max_end_of_session_regression_cycles=max(1, min(10, int(
            node_throttle.get("max_end_of_session_regression_cycles", 3) or 3
        ))),
        end_of_session_repair_diagnostic_cap=max(1, min(200, int(
            node_throttle.get(
                "end_of_session_repair_diagnostic_cap", 30,
            ) or 30
        ))),
        end_of_session_repair_inventory_cap=max(1, min(1000, int(
            node_throttle.get(
                "end_of_session_repair_inventory_cap", 150,
            ) or 150
        ))),
        end_of_session_force_reasoning_model=bool(node_throttle.get(
            "end_of_session_force_reasoning_model", True,
        )),
        ollama_local_model=model_routing.get("ollama_local_model", ""),
        ollama_local_backup=model_routing.get("ollama_local_backup", ""),
        force_local_only=model_routing.get("force_local_only", False),
        hard_cap_usd=token_budget.get("hard_cap_usd", 2.00),
        stages={
            str(k): float(v)
            for k, v in (token_budget.get("stages") or {}).items()
            if isinstance(v, (int, float))
        },
        dump_llm_calls=_resolve_dump_llm_calls(config_dict),
        dump_max_files=_resolve_dump_max_files(config_dict),
        enforce_read_before_edit=bool(
            (config_dict.get("patcher", {}) or {}).get(
                "enforce_read_before_edit", True,
            )
        ),
        use_structured_tools=bool(
            (config_dict.get("patcher", {}) or {}).get(
                "use_structured_tools", False,
            )
        ),
        context_window_threshold_pct=token_budget.get("context_window_threshold_pct", 0.85),
        ssl_verify=config_dict.get("ssl_verify", True),
        max_tokens_default=max_tokens_default,
        max_tokens_per_role=max_tokens_per_role,
        prompt_cache_enabled=bool(
            llm_dispatch.get("prompt_cache_enabled", True)
        ),
        llm_judgment_hitl_escalation_summary=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "hitl_escalation_summary", True,
            )
        ),
        llm_judgment_patcher_rejection_diagnosis=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "patcher_rejection_diagnosis", True,
            )
        ),
        llm_judgment_preflight_autofix=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "preflight_autofix_judgment", True,
            )
        ),
        llm_judgment_discovery_saturation=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "discovery_saturation_check", True,
            )
        ),
        llm_judgment_repair_reflection=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "repair_reflection", True,
            )
        ),
        repair_structured_diagnostic_payload=bool(
            (config_dict.get("repair", {}) or {}).get(
                "structured_diagnostic_payload", True,
            )
        ),
        llm_judgment_discovery_followup_focus=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "discovery_followup_focus", True,
            )
        ),
        llm_judgment_app_usage_guide=bool(
            (config_dict.get("llm_judgment", {}) or {}).get(
                "app_usage_guide", True,
            )
        ),
    )
    return Gateway(gateway_config)
