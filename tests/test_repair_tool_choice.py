"""Repair must not be able to answer in prose instead of a patch.

lumina run 13, from the per-call dumps: eight repair rounds, each

    finish: length  tokens_out=32768  tool_calls=0  content_chars=~123000
    elapsed_ms=302950

— 303 seconds and $0.036 apiece, no patch written. The prompts already said
"no prose outside blocks" and "Only patch blocks, no prose"; the model ignored
them eight times running. Prompt pressure is spent, so the answer is
structural: with ``tool_choice="required"`` the model cannot emit commentary
in place of a tool call.

Why this is safe on the default routing, which is the part worth getting
right: the documented wire quirk is that deepseek-v4-pro 400s when a
**thinking** request also carries a compelled ``tool_choice`` — hence
``thinking_rejects_forced_tool_choice``. Repair never asks for thinking, and
that model is configured ``supports_thinking: false``, so the conflict cannot
arise. And ``resolve_tool_choice_for_thinking`` DOWNGRADES rather than raises,
so even a provider that refuses forcing degrades to ``"auto"`` instead of
failing the round.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import harness.graph as g
from harness.tool_schemas import (
    TOOL_CHOICE_AUTO,
    TOOL_CHOICE_REQUIRED,
    resolve_tool_choice_for_thinking,
)

CONFIG = json.loads(
    (pathlib.Path(__file__).parent.parent / "config" / "config.json").read_text()
)


class TestTheResolver:
    def test_it_defaults_to_required(self):
        assert g._resolve_repair_tool_choice({}) == TOOL_CHOICE_REQUIRED

    def test_the_shipped_config_compels_a_tool_call(self):
        state = {"llm_dispatch_config": CONFIG["llm_dispatch"]}
        assert g._resolve_repair_tool_choice(state) == TOOL_CHOICE_REQUIRED

    def test_an_operator_can_restore_auto(self):
        """The documented escape hatch, for a provider that rejects forcing
        outright rather than letting the gateway downgrade it."""
        state = {"llm_dispatch_config": {"repair_tool_choice": "auto"}}
        assert g._resolve_repair_tool_choice(state) == TOOL_CHOICE_AUTO

    def test_an_empty_value_falls_back_to_the_default(self):
        state = {"llm_dispatch_config": {"repair_tool_choice": "   "}}
        assert g._resolve_repair_tool_choice(state) == TOOL_CHOICE_REQUIRED

    def test_a_missing_config_section_is_survivable(self):
        assert g._resolve_repair_tool_choice(
            {"llm_dispatch_config": None}
        ) == TOOL_CHOICE_REQUIRED


class TestTheBridgeThreadsItThrough:
    """``_dispatch_tools_as_dsl`` is the only path repair's tool calls take,
    and it previously had no way to carry a tool_choice at all."""

    @pytest.mark.asyncio
    async def test_the_choice_reaches_the_gateway(self):
        seen: dict = {}

        class _GW:
            class config:
                scoped_spec_context = False

            async def dispatch(self, **kw):
                seen.update(kw)

                class _R:
                    content = "<<<READ_FILE>>>\nfile: a.py\n<<<END_READ_FILE>>>"
                    tool_calls = None
                    finish_reason = "stop"

                    class usage:
                        cost_usd = 0.0
                        input_tokens = 1
                        output_tokens = 1
                        cached_tokens = 0
                        model = "stub"
                return _R(), kw.get("budget_remaining_usd", 1.0)

            def aggregate_tokens(self, t, _u, role=None):
                return t

        await g._dispatch_tools_as_dsl(
            gateway=_GW(), messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f", tools=[],
            workspace="/ws", state={}, node_label="repair_node",
            tool_choice=TOOL_CHOICE_REQUIRED,
        )
        assert seen.get("tool_choice") == TOOL_CHOICE_REQUIRED

    @pytest.mark.asyncio
    async def test_omitting_it_sends_nothing(self):
        """A caller that does not compel must not start sending a key the
        provider may reject."""
        seen: dict = {}

        class _GW:
            class config:
                scoped_spec_context = False

            async def dispatch(self, **kw):
                seen.update(kw)

                class _R:
                    content = "done"
                    tool_calls = None
                    finish_reason = "stop"

                    class usage:
                        cost_usd = 0.0
                        input_tokens = 1
                        output_tokens = 1
                        cached_tokens = 0
                        model = "stub"
                return _R(), 1.0

            def aggregate_tokens(self, t, _u, role=None):
                return t

        await g._dispatch_tools_as_dsl(
            gateway=_GW(), messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f", tools=[],
            workspace="/ws", state={}, node_label="repair_node",
        )
        assert "tool_choice" not in seen


class TestTheThinkingConflictCannotBite:
    """The reason this is safe rather than a 400 on every repair dispatch."""

    def test_the_model_repair_routes_to_does_not_ask_for_thinking(self):
        spec = CONFIG["models"]["deepseek:deepseek-v4-pro"]
        assert spec["supports_thinking"] is False, (
            "if this ever becomes true, forcing may start conflicting with "
            "thinking on this model — see thinking_rejects_forced_tool_choice"
        )

    def test_forcing_survives_when_thinking_is_off(self):
        assert resolve_tool_choice_for_thinking(
            TOOL_CHOICE_REQUIRED, thinking=False, rejects_forced=True,
        ) == TOOL_CHOICE_REQUIRED

    def test_forcing_downgrades_rather_than_raises_when_thinking_is_on(self):
        """The fail-open posture: losing the guarantee beats losing the
        request, so a thinking role that compels degrades to auto."""
        assert resolve_tool_choice_for_thinking(
            TOOL_CHOICE_REQUIRED, thinking=True, rejects_forced=True,
        ) == TOOL_CHOICE_AUTO

    def test_a_model_without_the_quirk_keeps_the_force(self):
        assert resolve_tool_choice_for_thinking(
            TOOL_CHOICE_REQUIRED, thinking=True, rejects_forced=False,
        ) == TOOL_CHOICE_REQUIRED


class TestTheWiring:
    def test_repair_passes_the_resolved_choice(self):
        """The resolver existing proves nothing if the call site ignores it —
        which is exactly how my first attempt at the deliberation fix went."""
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert "tool_choice=_resolve_repair_tool_choice(state)," in src

    def test_it_is_on_the_tool_path_only(self):
        """The text-DSL fallback has no tool call to compel, so the caller's
        parse fallback must stay live."""
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        at = src.index("tool_choice=_resolve_repair_tool_choice(state),")
        assert "if _repair_use_tools:" in src[max(0, at - 2000):at]

    def test_the_config_key_is_registered_and_typed(self):
        from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA

        assert "repair_tool_choice" in _KNOWN_NESTED_KEYS["llm_dispatch"]
        assert _TYPE_SCHEMA["llm_dispatch.repair_tool_choice"] == (str,)

    def test_it_is_documented_with_its_escape_hatch(self):
        doc = (
            pathlib.Path(__file__).parent.parent / "config" / "config_help.md"
        ).read_text()
        assert "repair_tool_choice" in doc
        assert 'Set\nto `"auto"`' in doc or 'to `"auto"` to restore' in doc
