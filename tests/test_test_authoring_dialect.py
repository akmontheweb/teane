"""Test authoring was told to call tools it was never given.

From the lumina debug dumps, a test-authoring prompt (call 31, session
01a18134):

    ## input message 0: role=system  (199424 chars)
    …
    ## How to emit changes — NATIVE TOOL-USE MODE (authoritative)
    Native tool-use is ENABLED. Emit every change by CALLING a tool. Never
    write tool syntax as text.

    # tool_calls=0  content_chars=36919

The banner appears FIVE times in that one prompt — message 0 plus four
patching turns replayed in the history — and the dispatch supplied no
``tools`` array at all. The only thing that could possibly work was
writing tool syntax as text, so the model had to disobey its instructions
to accomplish anything.

It did, every measured time: all nine zero-tool-call authoring rounds
emitted correct ``<<<CREATE_FILE>>>`` DSL. That is the model guessing
against its own prompt, not a design. ``_dispatch_tools_as_dsl``'s
docstring lists the occasions the same contradiction produced a WRONG
guess elsewhere — Anthropic XML on runs 5 and 14, DeepSeek DSML and bare
``<read_file><path>`` on lumina-run24, five rounds lost and two
zero_patch_loop HITLs.

The root cause is that the prompt's dialect decision and each node's
decision to supply tools were two separate conditions.
``_prompt_teaches_native_tools`` is now the single one, and
:class:`TestTheOnePredicate` is the guard that keeps them from drifting
apart again.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

import harness.graph as g
import harness.test_generation as tg
from harness.patcher import parse_patch_blocks

CONFIG = json.loads(
    (pathlib.Path(__file__).parent.parent / "config" / "config.json").read_text()
)


class TestTheOnePredicate:
    """The prompt and the dispatch must answer "tools or DSL?" identically,
    for every config, or the contradiction is simply back."""

    @pytest.mark.parametrize("uses_tools", [True, False])
    @pytest.mark.parametrize("can_call", [True, False])
    def test_it_agrees_with_the_prompt_builder(
        self, monkeypatch, uses_tools, can_call,
    ):
        import harness.gateway as gw

        monkeypatch.setattr(
            gw, "can_roles_call_tools", lambda _c, _r: can_call,
        )
        cfg = {"patcher": {"use_structured_tools": uses_tools}}
        native = g._prompt_teaches_native_tools(cfg)
        block = g._build_patch_interface_block(uses_tools, cfg)
        taught_native = "NATIVE TOOL-USE MODE" in block
        assert native is taught_native

    def test_the_shipped_config_is_native(self):
        assert g._prompt_teaches_native_tools(CONFIG) is True

    def test_an_empty_config_is_native_as_documented(self):
        assert g._prompt_teaches_native_tools({}) is True

    def test_structured_tools_off_means_dsl(self):
        assert g._prompt_teaches_native_tools(
            {"patcher": {"use_structured_tools": False}},
        ) is False


@dataclasses.dataclass
class _Usage:
    """A dataclass because the multi-round path sums ``response.usage``
    field by field via ``dataclasses.fields`` — the gateway's real usage
    object is one too."""

    cost_usd: float = 0.0
    input_tokens: int = 1
    output_tokens: int = 1
    cached_tokens: int = 0
    model: str = "stub"

    @property
    def prompt_tokens(self) -> int:
        """Mirrors gateway.TokenUsage. A property, not a field, so the
        dataclasses.fields() summing above still sees only the four real
        counters and does not try to add a derived total to itself."""
        return self.input_tokens + self.cached_tokens


class _Resp:
    def __init__(self, content="", tool_calls=None, finish_reason="stop"):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.usage = _Usage()


class _GW:
    class config:
        scoped_spec_context = False
        use_structured_tools = True

    def __init__(self, *responses):
        self._responses = list(responses) or [_Resp("ok")]
        self.calls: list[dict] = []

    async def dispatch(self, **kw):
        self.calls.append(kw)
        r = self._responses.pop(0) if len(self._responses) > 1 \
            else self._responses[0]
        return r, kw["budget_remaining_usd"] - 0.01

    def aggregate_tokens(self, t, _u, role=None):
        return t


CREATE = {
    "id": "c1", "name": "create_file",
    "input": {
        "file_path": "server/tests/test_config.py",
        "content": '"""Tests."""\n\n# @tests: server/app/config.py\n',
    },
}

NATIVE_STATE = {"harness_config": CONFIG, "llm_dispatch_config": {}}
DSL_STATE = {
    "harness_config": {"patcher": {"use_structured_tools": False}},
    "llm_dispatch_config": {},
}


async def _author(gw, state, budget=1.0):
    return await tg._dispatch_test_author(
        gateway=gw,
        messages=[{"role": "system", "content": "s"},
                  {"role": "user", "content": "write tests"}],
        budget=budget,
        cache_family="patching:test_authoring",
        state=state,
        workspace_path="/ws",
    )


class TestNativeModeSuppliesTools:
    @pytest.mark.asyncio
    async def test_the_tools_array_is_sent(self):
        gw = _GW(_Resp("", [CREATE]))
        await _author(gw, NATIVE_STATE)
        names = {t["name"] for t in gw.calls[0]["tools"]}
        assert {"create_file", "rewrite_file", "edit_file"} <= names

    @pytest.mark.asyncio
    async def test_the_escape_tools_are_NOT_offered_here(self):
        """``declare_unsatisfiable_test`` is repair's exit from a
        protected-test blocker. Test authoring has none to exit."""
        gw = _GW(_Resp("", [CREATE]))
        await _author(gw, NATIVE_STATE)
        names = {t["name"] for t in gw.calls[0]["tools"]}
        assert "declare_unsatisfiable_test" not in names

    @pytest.mark.asyncio
    async def test_a_tool_call_comes_back_as_parseable_patch_dsl(self):
        """Every call site consumes ``response.content`` through
        ``process_llm_patch_output``, so the fold has to hand back DSL or
        the round writes nothing."""
        gw = _GW(_Resp("", [CREATE]))
        resp, _b = await _author(gw, NATIVE_STATE)
        blocks = parse_patch_blocks(resp.content)
        assert [b.file for b in blocks] == ["server/tests/test_config.py"]

    @pytest.mark.asyncio
    async def test_the_default_choice_does_not_compel(self):
        """Unlike repair. Test authoring's zero-tool-call rounds were
        terminal wrap-ups after the files had landed, and compelling a call
        on a wrap-up turn buys a gratuitous write."""
        gw = _GW(_Resp("", [CREATE]))
        await _author(gw, NATIVE_STATE)
        assert gw.calls[0].get("tool_choice") == "auto"

    @pytest.mark.asyncio
    async def test_an_operator_can_compel_it(self):
        gw = _GW(_Resp("", [CREATE]))
        state = dict(NATIVE_STATE)
        state["llm_dispatch_config"] = {"test_authoring_tool_choice": "required"}
        await _author(gw, state)
        assert gw.calls[0]["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_a_plain_text_dsl_answer_still_passes_through(self):
        """The model may keep writing DSL out of habit; the fold must not
        eat a response that has no tool calls."""
        text = (
            "<<<CREATE_FILE>>>\nfile: server/tests/test_x.py\n"
            "content:\nassert True\n<<<END_CREATE_FILE>>>"
        )
        gw = _GW(_Resp(text, None))
        resp, _b = await _author(gw, NATIVE_STATE)
        assert parse_patch_blocks(resp.content)


class TestDslModeIsUntouched:
    @pytest.mark.asyncio
    async def test_no_tools_are_sent(self):
        """A provider that cannot call tools gets the DSL prompt, and
        sending it a tools array would be the mirror of the bug."""
        gw = _GW(_Resp("ok"))
        await _author(gw, DSL_STATE)
        assert "tools" not in gw.calls[0]

    @pytest.mark.asyncio
    async def test_no_tool_choice_is_sent(self):
        gw = _GW(_Resp("ok"))
        await _author(gw, DSL_STATE)
        assert "tool_choice" not in gw.calls[0]

    @pytest.mark.asyncio
    async def test_the_response_passes_straight_through(self):
        gw = _GW(_Resp("<<<CREATE_FILE>>>\nfile: a.py\ncontent:\nx\n"
                       "<<<END_CREATE_FILE>>>"))
        resp, _b = await _author(gw, DSL_STATE)
        assert parse_patch_blocks(resp.content)

    @pytest.mark.asyncio
    async def test_a_missing_state_is_survivable(self):
        """``state=None`` resolves to the documented native default rather
        than raising — this helper runs inside a node that must not crash
        the graph."""
        gw = _GW(_Resp("", [CREATE]))
        resp, _b = await _author(gw, None)
        assert resp is not None


class TestReadsAreAnsweredByTheDispatcher:
    """Test authoring has no READ_FILE cycle of its own. Repair does, which
    is why the two pass different values for ``resolve_reads``."""

    @pytest.mark.asyncio
    async def test_a_read_is_resolved_and_the_turn_continues(self, tmp_path):
        (tmp_path / "app.py").write_text("VALUE = 1\n")
        read = {"id": "r1", "name": "read_file",
                "input": {"file_path": "app.py"}}
        gw = _GW(_Resp("", [read]), _Resp("", [CREATE]))
        resp, _b = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="patching", cache_family="f",
            tools=g._patch_tools_for_state({}), workspace=str(tmp_path),
            state={}, node_label="test_generation_node", resolve_reads=True,
        )
        assert len(gw.calls) == 2, "the read was not answered and re-dispatched"
        assert "VALUE = 1" in json.dumps(gw.calls[1]["messages"])
        assert parse_patch_blocks(resp.content)

    @pytest.mark.asyncio
    async def test_without_the_flag_the_read_is_handed_to_the_caller(
        self, tmp_path,
    ):
        """Repair's behaviour, unchanged: the read is rendered as DSL for a
        pipeline that knows how to resolve it."""
        read = {"id": "r1", "name": "read_file",
                "input": {"file_path": "app.py"}}
        gw = _GW(_Resp("", [read]))
        resp, _b = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f",
            tools=g._patch_tools_for_state({}), workspace=str(tmp_path),
            state={}, node_label="repair_node",
        )
        assert len(gw.calls) == 1
        assert "<<<READ_FILE>>>" in resp.content

    @pytest.mark.asyncio
    async def test_retrieval_calls_are_still_resolved_either_way(
        self, monkeypatch, tmp_path,
    ):
        import harness.retrieval_tools as rt

        async def _resolve(_c, _ws, config=None):
            return "grep: 3 matches"

        monkeypatch.setattr(rt, "resolve_retrieval_call", _resolve)
        grep = {"id": "g1", "name": "grep", "input": {"pattern": "x"}}
        gw = _GW(_Resp("", [grep]), _Resp("", [CREATE]))
        await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="patching", cache_family="f",
            tools=g._patch_tools_for_state({}), workspace=str(tmp_path),
            state={}, node_label="test_generation_node", resolve_reads=True,
        )
        assert len(gw.calls) == 2
        assert "grep: 3 matches" in json.dumps(gw.calls[1]["messages"])


class TestTheWiring:
    """Every authoring call site must go through the helper; one left
    behind keeps the contradiction alive on that path."""

    def test_no_authoring_site_dispatches_raw(self):
        src = (
            pathlib.Path(__file__).parent.parent / "harness"
            / "test_generation.py"
        ).read_text()
        for family in (
            "patching:test_authoring",
            "patching:test_contradiction_reprompt",
            "patching:test_spec_review_reprompt",
        ):
            at = src.index(f"cache_family='{family}'")
            window = src[max(0, at - 400):at]
            assert "_dispatch_test_author(" in window, family
            assert "gateway.dispatch(" not in window, family

    def test_the_helper_asks_the_shared_predicate(self):
        src = (
            pathlib.Path(__file__).parent.parent / "harness"
            / "test_generation.py"
        ).read_text()
        assert "_prompt_teaches_native_tools" in src
        assert "resolve_reads=True" in src

    def test_regeneration_is_deliberately_untouched(self):
        """``build_regeneration_messages`` builds its OWN system message
        and never sees the native-tool banner, so its DSL prompt and its
        tool-less dispatch already agree. Changing it would be the bug."""
        src = (
            pathlib.Path(__file__).parent.parent / "harness"
            / "test_regeneration.py"
        ).read_text()
        assert "_dispatch_test_author" not in src
        assert "_REGEN_SYSTEM" in src


class TestTheConfigKey:
    def test_it_is_known_and_typed(self):
        from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA

        assert "test_authoring_tool_choice" in _KNOWN_NESTED_KEYS["llm_dispatch"]
        assert _TYPE_SCHEMA["llm_dispatch.test_authoring_tool_choice"] == (str,)

    def test_the_shipped_config_sets_auto(self):
        assert CONFIG["llm_dispatch"]["test_authoring_tool_choice"] == "auto"

    def test_it_is_documented(self):
        doc = (
            pathlib.Path(__file__).parent.parent / "config" / "config_help.md"
        ).read_text()
        assert "test_authoring_tool_choice" in doc
        assert '"required"' in doc
