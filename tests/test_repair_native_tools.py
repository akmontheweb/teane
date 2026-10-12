"""Repair is dispatched with the native tools its prompt says it has.

The shared system prompt carries "Native tool-use is ENABLED… call the
tools" whenever ``patcher.use_structured_tools`` is on, but repair_node
dispatched with no tools at all. A model told to call tools it was never
given writes the calls as text in its trained syntax. lumina-run24-
20260929-2327 lost five repair rounds that way (DeepSeek DSML, bare
``<read_file><path>``), and two zero_patch_loop HITLs with them; runs 5
and 14 had lost rounds to Anthropic XML before. Each was fixed as a new
dialect while the contradiction stayed.

Repair now dispatches with the tools and folds the calls back into the
DSL its pipeline is built on (``_dispatch_tools_as_dsl``).
"""

from __future__ import annotations

import os

import pytest

from harness import graph
from harness.gateway import LLMResponse, NodeRole, TokenUsage
from harness.patcher import (
    OperationType,
    PatchBlock,
    Placement,
    parse_patch_blocks,
    parse_read_blocks,
    render_patch_block,
)
from harness.tool_schemas import tool_call_to_dsl, tool_call_to_patch_block


def _resp(content: str = "", calls=None, out: int = 10) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=TokenUsage(input_tokens=100, output_tokens=out, cost_usd=0.01),
        model="stub",
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=list(calls or []),
    )


# ---------------------------------------------------------------------------
# render_patch_block — the inverse of the parser
# ---------------------------------------------------------------------------


class TestRenderRoundTrip:
    @pytest.mark.parametrize("block", [
        PatchBlock(OperationType.REPLACE_BLOCK, "a.py",
                   search="    def f(self):\n        return 1",
                   replace="    def f(self):\n        return 2"),
        PatchBlock(OperationType.REPLACE_BLOCK, "a.py", search="x", replace="",
                   count="all"),
        PatchBlock(OperationType.CREATE_FILE, "pkg/new.py", content="x = 1\n"),
        PatchBlock(OperationType.REWRITE_FILE, "a.py", content="print(1)"),
        PatchBlock(OperationType.DELETE_BLOCK, "a.py", search="dead()",
                   count="first"),
        PatchBlock(OperationType.INSERT_AT_BLOCK, "a.py", anchor="class A",
                   placement=Placement.BEFORE, content="import os"),
        PatchBlock(OperationType.INSERT_AT_LINE, "a.py", line=3,
                   content="y = 2", expected_file_hash="abc123"),
        PatchBlock(OperationType.REPLACE_LINE_RANGE, "a.py", line=2,
                   end_line=4, content="z = 3"),
    ])
    def test_every_operation_parses_back_to_itself(self, block):
        text = render_patch_block(block)
        assert text is not None
        [back] = parse_patch_blocks(text)
        assert back.operation is block.operation
        assert back.file == block.file
        assert back.search == block.search.rstrip()
        assert back.replace == block.replace.rstrip()
        assert back.content == block.content.rstrip()
        assert back.anchor == block.anchor
        assert back.placement is block.placement
        assert back.count == (block.count or "unique")
        assert (back.line, back.end_line) == (block.line, block.end_line)
        assert back.expected_file_hash == block.expected_file_hash

    def test_indentation_at_the_start_of_a_body_survives(self):
        # A search string's leading indentation is what makes it match.
        b = PatchBlock(OperationType.REPLACE_BLOCK, "a.py",
                       search="        return x", replace="        return y")
        [back] = parse_patch_blocks(render_patch_block(b))
        assert back.search == "        return x"

    def test_a_multi_line_anchor_is_refused_not_mangled(self):
        b = PatchBlock(OperationType.INSERT_AT_BLOCK, "a.py",
                       anchor="class A:\n    pass", content="x")
        assert render_patch_block(b) is None

    def test_a_body_holding_a_block_marker_is_refused(self):
        b = PatchBlock(OperationType.CREATE_FILE, "doc.md",
                       content="use <<<CREATE_FILE>>> like this")
        assert render_patch_block(b) is None


class TestToolCallToDsl:
    def test_an_edit_means_what_the_native_path_means(self):
        call = {"name": "edit_file", "input": {
            "file_path": "server/app/services/rate_limiter.py",
            "old_string": "limit: int = 30", "new_string": "limit: int = 60"}}
        [back] = parse_patch_blocks(tool_call_to_dsl(call))
        native = tool_call_to_patch_block(call)
        assert (back.operation, back.file, back.search, back.replace) == (
            native.operation, native.file, native.search, native.replace)

    def test_a_read_becomes_a_read_file_block_with_its_range(self):
        dsl = tool_call_to_dsl({"name": "read_file", "input": {
            "file_path": "server/app/db.py", "start_line": 10, "end_line": 40}})
        assert parse_read_blocks(dsl) == [("server/app/db.py", (10, 40))]

    def test_unknown_tools_and_pathless_calls_have_no_dsl(self):
        assert tool_call_to_dsl({"name": "run_terminal_cmd",
                                 "input": {"command": "ls"}}) is None
        assert tool_call_to_dsl({"name": "edit_file", "input": {
            "old_string": "a", "new_string": "b"}}) is None
        assert tool_call_to_dsl({"name": "read_file", "input": {}}) is None


# ---------------------------------------------------------------------------
# _dispatch_tools_as_dsl
# ---------------------------------------------------------------------------


class ScriptedGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.calls.append({"messages": list(messages), **kw})
        return self.responses.pop(0), budget_remaining_usd - 0.01


async def _fold(gw, tmp_path, messages=None):
    return await graph._dispatch_tools_as_dsl(
        gateway=gw, messages=messages or [{"role": "user", "content": "fix"}],
        budget=1.0, role=NodeRole.REPAIR, cache_family="repair",
        tools=[{"name": "edit_file"}], workspace=str(tmp_path), state={},
        node_label="repair_node",
    )


class TestDispatchToolsAsDsl:
    @pytest.mark.asyncio
    async def test_tools_are_sent_and_calls_come_back_as_dsl(self, tmp_path):
        gw = ScriptedGateway([_resp("", [
            {"id": "c1", "name": "edit_file", "input": {
                "file_path": "a.py", "old_string": "x", "new_string": "y"}},
            {"id": "c2", "name": "read_file", "input": {"file_path": "b.py"}},
        ])])
        resp, budget = await _fold(gw, tmp_path)
        assert gw.calls[0]["tools"] == [{"name": "edit_file"}]
        assert resp.tool_calls == []
        [blk] = parse_patch_blocks(resp.content)
        assert (blk.file, blk.search, blk.replace) == ("a.py", "x", "y")
        assert parse_read_blocks(resp.content) == [("b.py", None)]
        assert budget == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_text_the_model_wrote_is_kept(self, tmp_path):
        # UNSATISFIABLE_TEST and PROMOTE_DEFERRED are text-only declarations.
        gw = ScriptedGateway([_resp("UNSATISFIABLE_TEST: t.py — reason", [])])
        resp, _ = await _fold(gw, tmp_path)
        assert resp.content == "UNSATISFIABLE_TEST: t.py — reason"

    @pytest.mark.asyncio
    async def test_a_search_turn_is_resolved_then_the_edit_folded(self, tmp_path):
        (tmp_path / "a.py").write_text("limit = 30\n")
        gw = ScriptedGateway([
            _resp("", [{"id": "g1", "name": "grep",
                        "input": {"pattern": "limit"}}], out=5),
            _resp("", [{"id": "e1", "name": "edit_file", "input": {
                "file_path": "a.py", "old_string": "limit = 30",
                "new_string": "limit = 31"}}], out=7),
        ])
        msgs = [{"role": "user", "content": "fix"}]
        resp, _ = await _fold(gw, tmp_path, msgs)
        assert len(gw.calls) == 2
        second = gw.calls[1]["messages"]
        assert second[-1]["content"][0]["type"] == "tool_result"
        assert second[-1]["content"][0]["tool_use_id"] == "g1"
        # The caller's messages never carry the private tool exchange.
        assert msgs == [{"role": "user", "content": "fix"}]
        assert parse_patch_blocks(resp.content)[0].replace == "limit = 31"
        # One aggregation by the caller must cover both dispatches.
        assert resp.usage.output_tokens == 12
        assert resp.usage.cost_usd == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_an_untranslatable_call_is_dropped_loudly(self, tmp_path, caplog):
        gw = ScriptedGateway([_resp("", [
            {"id": "x", "name": "run_terminal_cmd", "input": {"command": "ls"}},
        ])])
        with caplog.at_level("WARNING"):
            resp, _ = await _fold(gw, tmp_path)
        assert resp.tool_calls == [] and resp.content == ""
        assert "run_terminal_cmd" in caplog.text


# ---------------------------------------------------------------------------
# repair_node end to end
# ---------------------------------------------------------------------------


class _ToolGateway:
    class config:
        use_structured_tools = True
        repair_fallback = ""
        planning_fallback = ""

    def __init__(self, workspace):
        self.workspace = workspace
        self.calls: list[dict] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.calls.append({"messages": list(messages), **kw})
        return _resp("", [{"id": "e1", "name": "edit_file", "input": {
            "file_path": "app/limits.py",
            "old_string": "LIMIT = 30", "new_string": "LIMIT = 31"}}]), \
            budget_remaining_usd

    def aggregate_tokens(self, tracker, usage, role=None):
        return tracker or {}


class TestRepairNodeUsesTheTools:
    @pytest.mark.asyncio
    async def test_a_native_edit_lands_and_the_prompt_teaches_no_markers(self, tmp_path):
        os.makedirs(tmp_path / "app")
        target = tmp_path / "app" / "limits.py"
        # Over the screen's 20-char "empty file" floor, like real code.
        target.write_text('"""Mutation limits."""\n\nLIMIT = 30\n')
        gw = _ToolGateway(str(tmp_path))
        graph.set_gateway(gw)
        try:
            await graph.repair_node({
                "workspace_path": str(tmp_path),
                "compiler_errors": [{
                    "file": "app/limits.py", "line": 1, "column": 0,
                    "severity": "error", "error_code": "AssertionError",
                    "message": "AssertionError: assert 30 == 31",
                    "semantic_context": "",
                }],
                "loop_counter": {},
                "messages": [],
                "modified_files": [],
                "budget_remaining_usd": 1.0,
            })
        finally:
            graph.set_gateway(None)

        assert gw.calls, "repair must dispatch"
        assert gw.calls[0].get("tools"), "repair must be given the tools its prompt promises"
        last = gw.calls[0]["messages"][-1]["content"]
        assert "NATIVE TOOL-USE MODE" in last
        assert "<<<REPLACE_BLOCK>>>" not in last
        assert target.read_text() == '"""Mutation limits."""\n\nLIMIT = 31\n'
