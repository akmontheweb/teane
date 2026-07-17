"""Regression tests for the B6 native tool-use wiring.

Covers:
    - Provider parse round-trips: Anthropic typed-block ``tool_use``
      and OpenAI-shape ``tool_calls`` (with JSON-encoded ``arguments``)
      land in ``LLMResponse.tool_calls`` in the canonical
      ``{name, input, id}`` shape.
    - Capability detection: when the routed model's ``supports_tools``
      is False, ``Gateway.dispatch`` drops ``tools=`` silently so the
      patching path falls back to the text DSL automatically.
    - Mixed-mode responses: text + tool_use in the same turn populate
      both ``content`` and ``tool_calls``.
    - Message-shape converter normalises Anthropic-style typed-block
      tool turns into OpenAI's ``role=tool`` / ``tool_calls`` shape.
    - ``hash_stable_prefix`` includes the tools array, so swapping
      tool definitions surfaces as drift.
    - ``apply_patch_blocks`` applies pre-built PatchBlocks via the same
      pipeline as ``process_llm_patch_output``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness.gateway import (
    AnthropicProvider,
    Gateway,
    GatewayConfig,
    LLMResponse,
    ModelSpec,
    NodeRole,
    OpenAIProvider,
    TokenUsage,
    _normalize_messages_for_openai_tools,
    _parse_openai_tool_calls,
    hash_stable_prefix,
    register_model,
)
from harness.tool_schemas import (
    PATCH_TOOLS,
    to_anthropic_tools,
    to_openai_tools,
    tool_calls_to_patch_blocks,
)


# ---------------------------------------------------------------------------
# Stub HTTP plumbing
# ---------------------------------------------------------------------------

class _StubHttpResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.status_code = 200
        self.request = None
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _RecordingClient:
    def __init__(self, response_payload: dict[str, Any]):
        self.last_payload: dict[str, Any] | None = None
        self._response_payload = response_payload

    async def post(self, _path: str, json: dict[str, Any]) -> _StubHttpResponse:
        self.last_payload = json
        return _StubHttpResponse(self._response_payload)


def _make_anthropic_provider(
    response_payload: dict[str, Any],
    *,
    supports_tools: bool = True,
    supports_cache: bool = True,
) -> tuple[AnthropicProvider, _RecordingClient]:
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-test",
        context_window=200_000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        cached_input_cost_per_1m=0.30,
        cache_creation_cost_per_1m=3.75,
        api_base_url="https://api.anthropic.com/v1",
        api_key="x",
        supports_cache=supports_cache,
        supports_tools=supports_tools,
    )
    provider = AnthropicProvider(spec, api_key="x")
    provider.prompt_cache_enabled = supports_cache  # type: ignore[attr-defined]
    client = _RecordingClient(response_payload)
    provider._client = client  # type: ignore[assignment]
    return provider, client


def _make_openai_provider(
    response_payload: dict[str, Any],
    *,
    supports_tools: bool = True,
) -> tuple[OpenAIProvider, _RecordingClient]:
    spec = ModelSpec(
        provider="openai",
        model_id="gpt-test",
        context_window=128_000,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        api_base_url="https://api.openai.com/v1",
        api_key="x",
        supports_tools=supports_tools,
    )
    provider = OpenAIProvider(spec, api_key="x")
    client = _RecordingClient(response_payload)
    provider._client = client  # type: ignore[assignment]
    return provider, client


# ---------------------------------------------------------------------------
# 1. Provider response parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_parses_tool_use_blocks():
    """tool_use blocks in the typed-block response land in LLMResponse.tool_calls."""
    provider, client = _make_anthropic_provider({
        "content": [
            {"type": "text", "text": "I'll edit the file."},
            {"type": "tool_use", "id": "toolu_01ABC", "name": "edit_file",
             "input": {"file_path": "src/a.py",
                       "old_string": "foo", "new_string": "bar"}},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 12,
                  "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
        "stop_reason": "tool_use",
    })
    response = await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
        ],
        tools=PATCH_TOOLS,
    )
    assert response.content == "I'll edit the file."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "edit_file"
    assert call["id"] == "toolu_01ABC"
    assert call["input"]["file_path"] == "src/a.py"


@pytest.mark.asyncio
async def test_anthropic_attaches_tools_payload_with_cache_control():
    provider, client = _make_anthropic_provider({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
        "stop_reason": "end_turn",
    })
    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools=PATCH_TOOLS,
    )
    assert client.last_payload is not None
    tools = client.last_payload["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "read_file"
    # Last tool carries the cache_control marker so the whole tools
    # array participates in the cacheable prefix.
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_anthropic_pure_text_response_has_empty_tool_calls():
    provider, client = _make_anthropic_provider({
        "content": [{"type": "text", "text": "just words"}],
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
        "stop_reason": "end_turn",
    })
    response = await provider.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.content == "just words"
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_openai_parses_tool_calls_with_json_arguments():
    """OpenAI/DeepSeek ship arguments as a JSON-encoded string; the
    parser must decode it back to a dict."""
    provider, _ = _make_openai_provider({
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_42",
                    "type": "function",
                    "function": {
                        "name": "create_file",
                        "arguments": json.dumps({
                            "file_path": "app/main.py",
                            "content": "print('hi')\n",
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    })
    response = await provider.chat_completion(
        messages=[{"role": "user", "content": "scaffold"}],
        tools=PATCH_TOOLS,
    )
    assert response.content == ""  # null content normalises to ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "create_file"
    assert call["input"]["file_path"] == "app/main.py"
    assert call["input"]["content"] == "print('hi')\n"


def test_parse_openai_tool_calls_drops_malformed_json():
    """Per-call JSON parse failures are dropped rather than poisoning
    the patcher with garbage arguments."""
    message = {
        "tool_calls": [
            {"id": "1", "function": {"name": "edit_file", "arguments": "not json{"}},
            {"id": "2", "function": {"name": "create_file", "arguments": "{\"file_path\":\"a\",\"content\":\"\"}"}},
        ],
    }
    out = _parse_openai_tool_calls(message)
    # Malformed call dropped; valid one survives.
    assert len(out) == 1
    assert out[0]["name"] == "create_file"


def test_parse_openai_tool_calls_drops_missing_name():
    message = {"tool_calls": [{"id": "1", "function": {"arguments": "{}"}}]}
    assert _parse_openai_tool_calls(message) == []


# ---------------------------------------------------------------------------
# 2. Message-shape converter
# ---------------------------------------------------------------------------

def test_normalize_messages_for_openai_passes_text_through():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back"},
    ]
    assert _normalize_messages_for_openai_tools(msgs) == msgs


def test_normalize_messages_converts_assistant_tool_use_blocks():
    msgs = [
        {"role": "user", "content": "edit"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Sure."},
            {"type": "tool_use", "id": "tu_1", "name": "edit_file",
             "input": {"file_path": "a", "old_string": "x", "new_string": "y"}},
        ]},
    ]
    out = _normalize_messages_for_openai_tools(msgs)
    # First message passes through unchanged.
    assert out[0] == {"role": "user", "content": "edit"}
    # Assistant turn flattened.
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "Sure."
    assert len(out[1]["tool_calls"]) == 1
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "edit_file"
    assert json.loads(tc["function"]["arguments"]) == {
        "file_path": "a", "old_string": "x", "new_string": "y",
    }


def test_normalize_messages_converts_tool_result_to_role_tool():
    msgs = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "file bytes"},
        ]},
    ]
    out = _normalize_messages_for_openai_tools(msgs)
    assert len(out) == 1
    assert out[0] == {
        "role": "tool", "tool_call_id": "tu_1", "content": "file bytes",
    }


def test_normalize_messages_handles_assistant_with_only_tool_use():
    """An assistant turn that's *purely* tool_use (no narration) emits
    content=None to satisfy OpenAI's schema."""
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"file_path": "a"}},
        ]},
    ]
    out = _normalize_messages_for_openai_tools(msgs)
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] is None
    assert len(out[0]["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# 2b. Tool-less dispatch flatten (lumina 019f6e13 regression)
# ---------------------------------------------------------------------------

def test_flatten_renders_tool_blocks_as_text():
    # Regression: nodes that dispatch WITHOUT tools (test_generation,
    # reviewers, discovery) inherit the tool loop's typed-block history;
    # the raw blocks shipped to the wire and DeepSeek 400'd the request
    # ("unknown variant `tool_use`, expected `text`"), which the
    # gateway's 4xx handling escalated to a run-aborting
    # HarnessConfigError.
    from harness.gateway import _flatten_tool_turns_for_plain_dispatch
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"file_path": "a.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "print('hi')"},
        ]},
        {"role": "user", "content": "now write tests"},
    ]
    out = _flatten_tool_turns_for_plain_dispatch(msgs)
    # Roles preserved, count preserved, plain messages untouched.
    assert [m["role"] for m in out] == ["system", "assistant", "user", "user"]
    assert out[0] == msgs[0]
    assert out[3] == msgs[3]
    # Every content is now a plain string — nothing typed remains.
    assert all(isinstance(m["content"], str) for m in out)
    assert "read_file" in out[1]["content"]
    assert "Let me look." in out[1]["content"]
    assert "print('hi')" in out[2]["content"]
    flat = json.dumps(out)
    assert "tool_use" not in flat
    assert "tool_result" not in flat


def test_flatten_passes_plain_history_through_identically():
    from harness.gateway import _flatten_tool_turns_for_plain_dispatch
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back"},
        # Anthropic text-block list without tool blocks stays as-is
        # (the cache-control path depends on the block shape).
        {"role": "user", "content": [{"type": "text", "text": "block"}]},
    ]
    assert _flatten_tool_turns_for_plain_dispatch(msgs) == msgs


def test_flatten_handles_nested_tool_result_content():
    from harness.gateway import _flatten_tool_turns_for_plain_dispatch
    msgs = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_9",
         "content": [{"type": "text", "text": "part1"},
                     {"type": "text", "text": "part2"}]},
    ]}]
    out = _flatten_tool_turns_for_plain_dispatch(msgs)
    assert "part1" in out[0]["content"] and "part2" in out[0]["content"]


def test_flatten_carries_anti_mimicry_note_exactly_once():
    # Regression (lumina 019f7109): the first flatten rendering used an
    # imperative "[called tool X with arguments: ...]" shape; the next
    # tool-less dispatcher (test_generation) adopted it as a tool syntax
    # and zero-emitted three responses in it — one containing a complete
    # valid test file the parser ignored. The rendering must be
    # narrative AND the first flattened message must carry an explicit
    # this-is-not-a-tool-interface note — once, not per message.
    from harness.gateway import _flatten_tool_turns_for_plain_dispatch
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "read_file",
             "input": {"file_path": "x.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "b", "name": "glob",
             "input": {"pattern": "**/*.py"}},
        ]},
    ]
    out = _flatten_tool_turns_for_plain_dispatch(msgs)
    joined = "\n".join(m["content"] for m in out if isinstance(m.get("content"), str))
    assert joined.count("NOT a tool interface") == 1
    # The note leads the FIRST flattened message.
    assert out[1]["content"].startswith("[NOTE:")
    # No imperative bracket form anywhere — nothing inviting imitation.
    assert "[called tool" not in joined
    assert "(history: invoked read_file" in joined


# ---------------------------------------------------------------------------
# 2c. read_file resolver path tolerance (lumina 019f7109)
# ---------------------------------------------------------------------------

class TestReadFilePathTolerance:
    def _call(self, path, ws):
        import asyncio
        from harness.graph import _resolve_read_file_call
        result = _resolve_read_file_call(
            {"input": {"file_path": path}}, str(ws),
        )
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return result

    def test_absolute_path_inside_workspace_is_served(self, tmp_path):
        # Absolute anchors leak into prompts (pytest assertion-rewrite
        # File lines); the model echoes them and the old guard refused
        # perfectly legitimate in-workspace reads — three wasted tool
        # rounds per turn in lumina 019f7109.
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "SPEC.md").write_text("spec body\n")
        out = self._call(str(tmp_path / "docs" / "SPEC.md"), tmp_path)
        assert "spec body" in out
        assert not out.startswith("Error:")

    def test_absolute_path_outside_workspace_refused(self, tmp_path):
        out = self._call("/etc/passwd", tmp_path)
        assert out.startswith("Error: refused absolute / traversal path")

    def test_traversal_refused(self, tmp_path):
        out = self._call("../secrets.txt", tmp_path)
        assert out.startswith("Error: refused absolute / traversal path")

    def test_prefix_sibling_dir_refused(self, tmp_path):
        # /ws-sibling must not pass a startswith("/ws") check.
        sibling = tmp_path.parent / (tmp_path.name + "-sibling")
        sibling.mkdir(exist_ok=True)
        (sibling / "f.txt").write_text("outside\n")
        out = self._call(str(sibling / "f.txt"), tmp_path)
        assert out.startswith("Error: refused absolute / traversal path")

    def test_relative_path_still_works(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        out = self._call("a.py", tmp_path)
        assert "x = 1" in out


# ---------------------------------------------------------------------------
# 3. Capability detection in Gateway.dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gateway_drops_tools_when_use_structured_tools_is_off():
    """When the operator hasn't opted in, ``tools=`` passed by a caller
    must NOT reach the provider — the gateway silently drops it so the
    text-DSL path stays in charge."""
    register_model("stub:caps-off", ModelSpec(
        provider="stub", model_id="caps", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x", supports_tools=True,
    ))
    cfg = GatewayConfig(
        planning_primary="stub:caps-off",
        patching_primary="stub:caps-off",
        repair_primary="stub:caps-off",
        use_structured_tools=False,
    )
    gw = Gateway(cfg)

    seen_tools: list[Any] = []

    class _Stub:
        spec = ModelSpec(
            provider="stub", model_id="caps", context_window=64_000,
            input_cost_per_1m=0.1, output_cost_per_1m=0.2,
            api_base_url="", api_key="x", supports_tools=True,
        )
        api_key = "x"

        async def chat_completion(self, *, tools=None, **_kwargs: Any) -> LLMResponse:
            seen_tools.append(tools)
            return LLMResponse(
                content="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1,
                                  model_name="stub:caps", cost_usd=0.0),
                model="stub:caps",
            )

        async def close(self) -> None:
            return None

    gw._providers["stub:caps-off"] = _Stub()  # type: ignore[assignment]
    await gw.dispatch(
        messages=[{"role": "user", "content": "go"}],
        role=NodeRole.PATCHING,
        budget_remaining_usd=1.0,
        tools=PATCH_TOOLS,
    )
    assert seen_tools == [None]


@pytest.mark.asyncio
async def test_gateway_drops_tools_when_model_does_not_support_them():
    """Even with ``use_structured_tools=True``, a model declared
    ``supports_tools=False`` must NOT receive the tool array — that
    would 400 the request."""
    register_model("stub:no-tools", ModelSpec(
        provider="stub", model_id="no-tools", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x", supports_tools=False,
    ))
    cfg = GatewayConfig(
        planning_primary="stub:no-tools",
        patching_primary="stub:no-tools",
        repair_primary="stub:no-tools",
        use_structured_tools=True,
    )
    gw = Gateway(cfg)

    seen_tools: list[Any] = []

    class _Stub:
        spec = ModelSpec(
            provider="stub", model_id="no-tools", context_window=64_000,
            input_cost_per_1m=0.1, output_cost_per_1m=0.2,
            api_base_url="", api_key="x", supports_tools=False,
        )
        api_key = "x"

        async def chat_completion(self, *, tools=None, **_kwargs: Any) -> LLMResponse:
            seen_tools.append(tools)
            return LLMResponse(
                content="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1,
                                  model_name="stub:no-tools", cost_usd=0.0),
                model="stub:no-tools",
            )

        async def close(self) -> None:
            return None

    gw._providers["stub:no-tools"] = _Stub()  # type: ignore[assignment]
    await gw.dispatch(
        messages=[{"role": "user", "content": "go"}],
        role=NodeRole.PATCHING,
        budget_remaining_usd=1.0,
        tools=PATCH_TOOLS,
    )
    assert seen_tools == [None]


@pytest.mark.asyncio
async def test_gateway_threads_tools_through_when_caps_match():
    register_model("stub:caps-on", ModelSpec(
        provider="stub", model_id="caps-on", context_window=64_000,
        input_cost_per_1m=0.1, output_cost_per_1m=0.2,
        api_base_url="", api_key="x", supports_tools=True,
    ))
    cfg = GatewayConfig(
        planning_primary="stub:caps-on",
        patching_primary="stub:caps-on",
        repair_primary="stub:caps-on",
        use_structured_tools=True,
    )
    gw = Gateway(cfg)

    seen_tools: list[Any] = []

    class _Stub:
        spec = ModelSpec(
            provider="stub", model_id="caps-on", context_window=64_000,
            input_cost_per_1m=0.1, output_cost_per_1m=0.2,
            api_base_url="", api_key="x", supports_tools=True,
        )
        api_key = "x"

        async def chat_completion(self, *, tools=None, **_kwargs: Any) -> LLMResponse:
            seen_tools.append(tools)
            return LLMResponse(
                content="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1,
                                  model_name="stub:caps-on", cost_usd=0.0),
                model="stub:caps-on",
            )

        async def close(self) -> None:
            return None

    gw._providers["stub:caps-on"] = _Stub()  # type: ignore[assignment]
    await gw.dispatch(
        messages=[{"role": "user", "content": "go"}],
        role=NodeRole.PATCHING,
        budget_remaining_usd=1.0,
        tools=PATCH_TOOLS,
    )
    assert len(seen_tools) == 1
    assert seen_tools[0] is not None
    assert seen_tools[0][0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# 4. hash_stable_prefix folds tools into the hash
# ---------------------------------------------------------------------------

def test_hash_stable_prefix_changes_when_tools_change():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "p"},
    ]
    h_no_tools = hash_stable_prefix(msgs, n_stable=2)
    h_with_tools = hash_stable_prefix(msgs, n_stable=2, tools=PATCH_TOOLS)
    h_with_other_tools = hash_stable_prefix(msgs, n_stable=2, tools=[
        {"name": "fake", "description": "x", "input_schema": {"type": "object"}},
    ])
    assert h_no_tools != h_with_tools
    assert h_with_tools != h_with_other_tools


# ---------------------------------------------------------------------------
# 5. Tool-call → PatchBlock translator integration
# ---------------------------------------------------------------------------

def test_tool_calls_to_patch_blocks_partitions_reads_and_patches():
    calls = [
        {"name": "read_file", "id": "1", "input": {"file_path": "a.py"}},
        {"name": "edit_file", "id": "2", "input": {
            "file_path": "a.py", "old_string": "x", "new_string": "y",
        }},
        {"name": "create_file", "id": "3", "input": {
            "file_path": "b.py", "content": "pass",
        }},
    ]
    blocks, reads = tool_calls_to_patch_blocks(calls)
    assert len(blocks) == 2
    assert {b.file for b in blocks} == {"a.py", "b.py"}
    assert len(reads) == 1
    assert reads[0]["input"]["file_path"] == "a.py"


def test_tool_call_to_patch_block_handles_insert_at_line():
    """Forward-compat dispatch — PATCH_TOOLS does not yet expose
    ``insert_at_line`` as a tool, but the translator must round-trip
    it cleanly when added so we never repeat the autofix._apply_block
    silent-drop bug from 2026-06-25."""
    from harness.patcher import OperationType
    from harness.tool_schemas import tool_call_to_patch_block

    block = tool_call_to_patch_block({
        "name": "insert_at_line",
        "id": "x",
        "input": {
            "file_path": "Dockerfile",
            "line": 5,
            "content": "USER 1000:1000",
            "expected_file_hash": "deadbeef",
        },
    })
    assert block is not None
    assert block.operation == OperationType.INSERT_AT_LINE
    assert block.file == "Dockerfile"
    assert block.line == 5
    assert block.content == "USER 1000:1000"
    assert block.expected_file_hash == "deadbeef"


def test_tool_call_to_patch_block_handles_replace_line_range():
    from harness.patcher import OperationType
    from harness.tool_schemas import tool_call_to_patch_block

    block = tool_call_to_patch_block({
        "name": "replace_line_range",
        "id": "y",
        "input": {
            "file_path": "Dockerfile",
            "line": 2,
            "end_line": 4,
            "content": "RUN pip install flask==3.0.3",
        },
    })
    assert block is not None
    assert block.operation == OperationType.REPLACE_LINE_RANGE
    assert block.file == "Dockerfile"
    assert block.line == 2
    assert block.end_line == 4
    assert "flask==3.0.3" in block.content


# ---------------------------------------------------------------------------
# 6. apply_patch_blocks end-to-end (uses the same pipeline as text DSL)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_patch_blocks_writes_create_file_to_disk(tmp_path):
    from harness.patcher import apply_patch_blocks
    blocks, _reads = tool_calls_to_patch_blocks([{
        "name": "create_file", "id": "1", "input": {
            "file_path": "src/hello.py",
            "content": "print('hi')\n",
        },
    }])
    results, modified = await apply_patch_blocks(
        blocks, str(tmp_path), [], allowed_paths=["src/"],
    )
    assert len(results) == 1
    assert results[0].success, results[0].error
    assert "src/hello.py" in modified
    written = (tmp_path / "src" / "hello.py").read_text()
    # The patcher normalises the trailing newline; the test just cares
    # that the body lands.
    assert written.startswith("print('hi')")


@pytest.mark.asyncio
async def test_apply_patch_blocks_rejects_outside_allowlist(tmp_path):
    from harness.patcher import apply_patch_blocks
    blocks, _reads = tool_calls_to_patch_blocks([{
        "name": "create_file", "id": "1", "input": {
            "file_path": "secrets/passwords.txt",
            "content": "no",
        },
    }])
    results, modified = await apply_patch_blocks(
        blocks, str(tmp_path), [], allowed_paths=["src/"],
    )
    assert len(results) == 1
    assert not results[0].success
    assert "not in skill allowlist" in (results[0].error or "")
    assert modified == []


@pytest.mark.asyncio
async def test_apply_patch_blocks_refuses_harness_config_at_root(tmp_path):
    """``.harness_config.json`` is harness-internal — patches to it must
    fail with a precise diagnostic (not the generic allowlist message)
    so the repair LLM stops proposing them."""
    from harness.patcher import apply_patch_blocks
    blocks, _reads = tool_calls_to_patch_blocks([{
        "name": "create_file", "id": "1", "input": {
            "file_path": ".harness_config.json",
            "content": "{}",
        },
    }])
    results, _ = await apply_patch_blocks(
        blocks, str(tmp_path), [], allowed_paths=None,
    )
    assert len(results) == 1
    assert not results[0].success
    assert "harness-internal" in (results[0].error or "")


@pytest.mark.asyncio
async def test_apply_patch_blocks_refuses_harness_config_in_subdir(tmp_path):
    """A subdir copy like ``tests/.harness_config.json`` is dead weight
    (the runtime only reads the root file). Reject too."""
    from harness.patcher import apply_patch_blocks
    blocks, _reads = tool_calls_to_patch_blocks([{
        "name": "create_file", "id": "1", "input": {
            "file_path": "tests/.harness_config.json",
            "content": "{}",
        },
    }])
    results, _ = await apply_patch_blocks(
        blocks, str(tmp_path), [],
        allowed_paths=["tests/"],  # subdir IS in allowlist
    )
    assert len(results) == 1
    assert not results[0].success
    assert "harness-internal" in (results[0].error or "")


# ---------------------------------------------------------------------------
# 7. Adapter shape sanity
# ---------------------------------------------------------------------------

def test_to_openai_tools_wraps_in_function_envelope():
    out = to_openai_tools(PATCH_TOOLS)
    assert all(t["type"] == "function" for t in out)
    assert {t["function"]["name"] for t in out} == {
        t["name"] for t in PATCH_TOOLS
    }


def test_to_anthropic_tools_keeps_raw_shape():
    out = to_anthropic_tools(PATCH_TOOLS)
    assert {t["name"] for t in out} == {t["name"] for t in PATCH_TOOLS}
    assert "input_schema" in out[0]
