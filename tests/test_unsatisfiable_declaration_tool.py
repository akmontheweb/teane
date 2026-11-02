"""The escape ladder's rung 2 must survive a compelled tool call.

``ef4968b`` set ``tool_choice="required"`` on repair's native-tool path to
stop the run-13 pathology: eight rounds, ``finish: length``,
``tokens_out=32768``, ``tool_calls=0``, ~123,000 characters of prose, no
patch. It worked on that, and it collided with something else.

Repair's one model-DECLARED exit is the ``UNSATISFIABLE_TEST: <path> —
<reason>`` line, scraped out of message TEXT by
``_parse_unsatisfiable_test_declaration``. Compelling a tool call and
asking for a prose declaration are mutually exclusive requests: the model
can satisfy one. Measured on the lumina dumps, NINE repair rounds took that
exit in pure prose with zero tool calls and ``finish: stop`` (session
01a19a9c, calls 73/89/91/94/96/98/100/102/104), so the collision is a live
path, not a hypothetical. If the provider returns ``content: null`` when it
forces a call — which OpenAI-compat providers commonly do — the declaration
has nowhere to go, repair keeps patching a test it has proven unsatisfiable,
and the only remaining exit is the declined-offer floor at 3 declines.

The fix is not to stop compelling; it is to stop making structure and
declaration compete. ``declare_unsatisfiable_test`` folds back to the exact
line the existing regex parses, so the offer-set validation, the decline
counter and ``_apply_unsat_offer_floor`` all run unchanged.

``PROMOTE_DEFERRED`` is repair's second prose-only channel and gets the same
treatment, for the same reason — losing it degrades diagnostic ranking
rather than deadlocking the loop, but the defect is identical and fixing
one of two would mean fixing the symptom twice.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import harness.graph as g
from harness.patcher import parse_promote_deferred_blocks
from harness.tool_schemas import (
    DECLARE_UNSATISFIABLE_TEST_SCHEMA,
    PATCH_TOOLS,
    PROMOTE_DEFERRED_SCHEMA,
    REPAIR_ESCAPE_TOOLS,
    TOOL_CHOICE_REQUIRED,
    tool_call_to_dsl,
    tool_call_to_patch_block,
)

TEST_PATH = "server/tests/test_birthday_repo.py"
REASON = (
    "asserts SQLite WAL journal mode on an in-memory connection, which "
    "cannot support WAL"
)


def _decl(**kw) -> dict:
    args = {"file_path": TEST_PATH, "reason": REASON}
    args.update(kw)
    return {"id": "t1", "name": "declare_unsatisfiable_test", "input": args}


class TestTheDeclarationRoundTrips:
    """The load-bearing property: what the tool renders is what the
    existing parser reads. Nothing downstream was rewritten, so if the
    round trip breaks, the escape silently stops working."""

    def test_the_rendered_line_parses_back(self):
        dsl = tool_call_to_dsl(_decl())
        assert dsl is not None
        assert g._parse_unsatisfiable_test_declaration(dsl) == (
            TEST_PATH, REASON,
        )

    def test_it_parses_when_folded_after_other_content(self):
        """The fold joins the model's prose and the rendered calls with
        blank lines; the regex is MULTILINE and needs the line to start a
        line of its own."""
        folded = "I have exhausted the production options.\n\n" + str(
            tool_call_to_dsl(_decl())
        )
        assert g._parse_unsatisfiable_test_declaration(folded) == (
            TEST_PATH, REASON,
        )

    def test_a_multiline_reason_is_collapsed_to_one_line(self):
        """The parser's reason group stops at end-of-line. An un-collapsed
        reason would be truncated mid-sentence — and worse, its tail would
        sit in the content as a stray line."""
        dsl = tool_call_to_dsl(_decl(reason="line one\nline two\n  line three"))
        assert dsl is not None
        assert dsl.count("\n") == 0
        assert g._parse_unsatisfiable_test_declaration(dsl) == (
            TEST_PATH, "line one line two line three",
        )

    def test_a_missing_reason_still_declares(self):
        """The path is the load-bearing part — the parser documents that
        the reason may be empty."""
        dsl = tool_call_to_dsl(_decl(reason=""))
        assert g._parse_unsatisfiable_test_declaration(dsl) == (TEST_PATH, "")

    def test_backticks_around_the_path_are_stripped(self):
        dsl = tool_call_to_dsl(_decl(file_path=f"`{TEST_PATH}`"))
        assert g._parse_unsatisfiable_test_declaration(dsl) == (
            TEST_PATH, REASON,
        )


class TestItRefusesWhatWouldMisParse:
    """A dropped call costs a round. A MIS-parsed one quarantines the wrong
    test, so these refuse rather than render something half-readable."""

    @pytest.mark.parametrize("path", [
        "server/tests/my test.py",   # \\S* would stop at the space
        "  ",
        "",
        "-server/tests/test_x.py",   # eaten by the separator alternation
        ":server/tests/test_x.py",
        "—dash-first",
    ])
    def test_a_path_the_parser_would_mangle_is_refused(self, path):
        assert tool_call_to_dsl(_decl(file_path=path)) is None

    def test_a_non_dict_input_is_survivable(self):
        assert tool_call_to_dsl(
            {"name": "declare_unsatisfiable_test", "input": "oops"},
        ) is None

    def test_a_refused_render_cannot_half_declare(self):
        """Belt and braces: whatever is refused must not leave a parseable
        fragment behind."""
        for path in ("server/tests/my test.py", "-x.py"):
            assert tool_call_to_dsl(_decl(file_path=path)) is None


class TestItIsNotMistakenForAPatch:
    def test_a_declaration_is_not_a_patch_block(self):
        """It must not reach the apply pipeline — and ``_any_real_patch``
        decides whether the declaration is honoured at all, so a
        declaration counted as a patch would veto itself."""
        assert tool_call_to_patch_block(_decl()) is None

    def test_promote_deferred_is_not_a_patch_block_either(self):
        assert tool_call_to_patch_block({
            "name": "promote_deferred_diagnostics",
            "input": {"codes": ["TS2769"]},
        }) is None


class TestPromoteDeferred:
    def _call(self, codes):
        return {
            "id": "p1",
            "name": "promote_deferred_diagnostics",
            "input": {"codes": codes},
        }

    def test_the_rendered_block_parses_back(self):
        dsl = tool_call_to_dsl(self._call(["TS2769", "TS2353"]))
        assert parse_promote_deferred_blocks(str(dsl)) == ["TS2769", "TS2353"]

    def test_a_bare_string_is_accepted(self):
        """The predictable mis-call. Dropping the signal over the model's
        choice of container would be a worse outcome than accepting it."""
        dsl = tool_call_to_dsl(self._call("F401"))
        assert parse_promote_deferred_blocks(str(dsl)) == ["F401"]

    def test_empty_codes_render_nothing(self):
        assert tool_call_to_dsl(self._call([])) is None
        assert tool_call_to_dsl(self._call(["", "  "])) is None

    def test_a_non_list_is_survivable(self):
        assert tool_call_to_dsl(self._call(42)) is None

    def test_the_marker_is_not_reported_as_stray_patch_dsl(self):
        """``_note_dsl_in_native_mode`` warns about text patch DSL on a
        tools-enabled turn. The fold emits this block by design, so it must
        already be excused."""
        assert "PROMOTE_DEFERRED" in g._NON_PATCH_DSL_MARKERS


class TestWhoIsOfferedTheTools:
    def test_repair_gets_the_escape_tools(self):
        names = {t["name"] for t in g._repair_tools_for_state({})}
        assert "declare_unsatisfiable_test" in names
        assert "promote_deferred_diagnostics" in names

    def test_patching_does_not(self):
        """No protected-test blocker to escape from in prod generation, and
        an escape offered where it cannot apply is a way to skip work."""
        names = {t["name"] for t in g._patch_tools_for_state({})}
        assert "declare_unsatisfiable_test" not in names
        assert "promote_deferred_diagnostics" not in names

    def test_repair_still_gets_every_patch_tool(self):
        names = {t["name"] for t in g._repair_tools_for_state({})}
        for t in PATCH_TOOLS:
            assert t["name"] in names

    def test_the_escape_set_is_exactly_the_two_prose_channels(self):
        assert REPAIR_ESCAPE_TOOLS == [
            DECLARE_UNSATISFIABLE_TEST_SCHEMA, PROMOTE_DEFERRED_SCHEMA,
        ]

    def test_repair_honours_retrieval_being_disabled(self):
        state = {"retrieval_tools_config": {"enabled": False}}
        names = {t["name"] for t in g._repair_tools_for_state(state)}
        assert "grep" not in names
        assert "declare_unsatisfiable_test" in names

    def test_the_schemas_are_closed_enough_to_be_offered(self):
        for schema in REPAIR_ESCAPE_TOOLS:
            assert schema["input_schema"]["type"] == "object"
            assert schema["input_schema"]["required"]
            assert schema["description"].strip()


class TestTheBannerAsksForWhatTheRoundCanGive:
    """Both PROTECTED TEST LOCATION banners said "emit ZERO patch blocks
    and exactly one line of the form: UNSATISFIABLE_TEST: …". Under a
    compelled tool_choice that instruction is unsatisfiable by
    construction."""

    def test_native_mode_names_the_tool(self):
        out = g._unsat_escape_directive(True)
        assert "declare_unsatisfiable_test" in out

    def test_native_mode_does_not_ask_for_a_prose_line(self):
        out = g._unsat_escape_directive(True)
        assert "UNSATISFIABLE_TEST:" not in out

    def test_dsl_mode_keeps_the_prose_line_verbatim(self):
        """A provider that cannot call tools still reaches this banner, and
        that path was working."""
        out = g._unsat_escape_directive(False)
        assert "UNSATISFIABLE_TEST: <workspace-relative test path>" in out
        assert "declare_unsatisfiable_test" not in out

    def test_both_banners_use_the_helper(self):
        """Testing the helper proves nothing if a banner still hard-codes
        the prose form — the mistake this suite already caught once on the
        deliberation fix."""
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert src.count(
            "+ _unsat_escape_directive(_repair_use_tools)"
        ) == 2

    def test_the_native_reminder_no_longer_sends_declarations_to_text(self):
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        at = src.index("[REPAIR — NATIVE TOOL-USE MODE]")
        block = src[at:at + 1400]
        assert "declare_unsatisfiable_test" in block
        assert "Nothing belongs in your message text" in block

    def test_repair_dispatches_its_own_tool_list(self):
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert "tools=_repair_tools_for_state(state)," in src


class _Usage:
    cost_usd = 0.0
    input_tokens = 1
    output_tokens = 1
    cached_tokens = 0
    model = "stub"


class _Resp:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = "tool_calls"
        self.usage = _Usage()


class _GW:
    """Returns one forced declaration call and NO content, which is the
    provider behaviour that would otherwise lose the declaration."""

    class config:
        scoped_spec_context = False

    def __init__(self, content, calls):
        self._content = content
        self._calls = calls
        self.seen: dict = {}

    async def dispatch(self, **kw):
        self.seen.update(kw)
        return _Resp(self._content, list(self._calls)), kw["budget_remaining_usd"]

    def aggregate_tokens(self, t, _u, role=None):
        return t


class TestEndToEndThroughTheFold:
    """The test that would have caught the regression: a compelled round
    whose only tool call is the declaration must still reach the parser."""

    @pytest.mark.asyncio
    async def test_a_forced_declaration_with_null_content_still_declares(self):
        gw = _GW(None, [_decl()])
        resp, _budget = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "fix it"}],
            budget=1.0, role="repair", cache_family="f",
            tools=g._repair_tools_for_state({}), workspace="/ws", state={},
            node_label="repair_node", tool_choice=TOOL_CHOICE_REQUIRED,
        )
        assert gw.seen.get("tool_choice") == TOOL_CHOICE_REQUIRED
        assert g._parse_unsatisfiable_test_declaration(resp.content) == (
            TEST_PATH, REASON,
        )

    @pytest.mark.asyncio
    async def test_the_response_leaves_carrying_no_tool_calls(self):
        """Repair's pipeline is DSL-based; a leftover tool_use id would be
        an unanswered call on the next dispatch."""
        gw = _GW(None, [_decl()])
        resp, _b = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f",
            tools=g._repair_tools_for_state({}), workspace="/ws", state={},
            node_label="repair_node", tool_choice=TOOL_CHOICE_REQUIRED,
        )
        assert not resp.tool_calls

    @pytest.mark.asyncio
    async def test_a_declaration_alongside_a_patch_keeps_both(self):
        """The hedge case. ``_any_real_patch`` is what decides which wins,
        and it cannot decide if the fold dropped one of them."""
        patch_call = {
            "id": "c1", "name": "create_file",
            "input": {"file_path": "a.py", "content": "x = 1\n"},
        }
        gw = _GW("", [patch_call, _decl()])
        resp, _b = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f",
            tools=g._repair_tools_for_state({}), workspace="/ws", state={},
            node_label="repair_node", tool_choice=TOOL_CHOICE_REQUIRED,
        )
        assert "<<<CREATE_FILE>>>" in resp.content
        assert g._parse_unsatisfiable_test_declaration(resp.content) is not None

    @pytest.mark.asyncio
    async def test_a_forced_promotion_reaches_its_parser_too(self):
        gw = _GW(None, [{
            "id": "p1", "name": "promote_deferred_diagnostics",
            "input": {"codes": ["TS2769"]},
        }])
        resp, _b = await g._dispatch_tools_as_dsl(
            gateway=gw, messages=[{"role": "user", "content": "x"}],
            budget=1.0, role="repair", cache_family="f",
            tools=g._repair_tools_for_state({}), workspace="/ws", state={},
            node_label="repair_node", tool_choice=TOOL_CHOICE_REQUIRED,
        )
        assert parse_promote_deferred_blocks(resp.content) == ["TS2769"]


class TestTheShippedConfigIsUnchangedHere:
    def test_repair_still_compels(self):
        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent / "config" / "config.json")
            .read_text()
        )
        assert cfg["llm_dispatch"]["repair_tool_choice"] == "required"
