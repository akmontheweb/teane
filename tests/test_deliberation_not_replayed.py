"""Deliberation must not be replayed, and must not be continued.

lumina run 13, read from the per-call debug dumps rather than the log:

    role: repair  finish: length
    tokens_out=32768  tool_calls=0  content_chars=123945  elapsed_ms=302950

Eight rounds like that. The 32,768-token output contained **no patch block
and no tool call** — 123,945 characters of "But wait… Actually, let me
reconsider… Hmm, let me reconsider", running out of budget mid-thought after
303 seconds. The prompt that produced it was 467,483 characters, of which
121,174 was message 3: *the previous round's blockless deliberation, replayed
verbatim*. Input grew 75,616 → 94,351 tokens across the eight rounds.

Two defects, both covered here:

* the DISTRACTION rebuild selects the last assistant turn and replays it, so
  a round of circular reasoning is handed back as "here is what you tried" —
  inflating every later prompt AND showing the model that deliberating is
  what happens here;
* ``_continue_on_length`` would happily continue such a turn, re-sending the
  whole argument and asking for more of it.

The distinction both fixes turn on, and the one that matters most: a
truncated turn that DID contain blocks is half-finished work and must always
be replayed and continued. Losing it is a worse failure than the one being
fixed. :class:`TestRealWorkIsNeverTouched` pins that on both paths.
"""

from __future__ import annotations

import pytest

import harness.graph as g

PATCH = (
    "<<<REPLACE_BLOCK>>>\n"
    "file: server/app/services/birthday_service.py\n"
    "search:\n"
    "    return []\n"
    "replace:\n"
    "    return rows\n"
    "<<<END_REPLACE_BLOCK>>>"
)
READ = "<<<READ_FILE>>>\nfile: server/app/main.py\n<<<END_READ_FILE>>>"

#: Run 13's shape, at run 13's scale — and deliberately NOT repetitive.
#: My first fixture repeated one line 3,000 times, which scores 1.0 on
#: _repetition_ratio against a 0.7 threshold, so the EXISTING
#: _refuse_degenerate guard caught it and the new gate never ran. The test
#: passed with the fix neutered. Real deliberation varies line to line, which
#: is exactly why it slips past the repetition check.
DELIBERATION = (
    "Now I can see the full picture. The bug is clear.\n\n"
    + "".join(
        f"But wait - reconsidering step {i}: the repository returns rows for "
        f"owner {i}, so list_all({i}) cannot be empty unless the seed in "
        f"fixture {i} never committed. Hmm, let me check case {i}.\n"
        for i in range(1200)
    )
)


def _assert_not_degenerate() -> None:
    """Guards the fixture itself: if it ever becomes repetitive, every test
    below silently starts exercising the old guard instead of the new one."""
    assert g._repetition_ratio(DELIBERATION) < g._DEGENERATE_REPEAT_RATIO


class TestThePriorTurnIsTrimmed:
    def test_blockless_deliberation_is_cut_to_its_head(self):
        msg = {"role": "assistant", "content": DELIBERATION}
        out = g._trimmed_prior_turn(msg)
        assert len(out["content"]) < len(DELIBERATION) / 10
        assert len(out["content"]) <= g._DELIBERATION_HEAD_CHARS + 200

    def test_the_opening_diagnosis_is_kept(self):
        """A blockless turn's one piece of value is what it concluded before
        it started arguing with itself."""
        out = g._trimmed_prior_turn(
            {"role": "assistant", "content": DELIBERATION}
        )
        assert "Now I can see the full picture" in out["content"]

    def test_it_says_why_the_rest_is_gone(self):
        """Silently shortening the model's own turn would be its own kind of
        confusing — the notice tells it what happened and what to do."""
        out = g._trimmed_prior_turn(
            {"role": "assistant", "content": DELIBERATION}
        )
        assert "deliberation with no patch block" in out["content"]
        assert "Emit a patch block this round" in out["content"]

    def test_the_input_message_is_not_mutated(self):
        """The caller's list is shared state; trimming in place would edit a
        turn other code may still read."""
        msg = {"role": "assistant", "content": DELIBERATION}
        g._trimmed_prior_turn(msg)
        assert msg["content"] == DELIBERATION

    def test_run_13s_actual_reduction(self):
        """121,174 chars of msg 3 against a 467,483-char prompt — this fix
        alone removes about a quarter of the input."""
        msg = {"role": "assistant", "content": "x" * 121174}
        out = g._trimmed_prior_turn(msg)
        assert len(out["content"]) < 5000


class TestTheFixtureAndTheWiring:
    def test_the_fixture_is_not_degenerate_by_repetition(self):
        _assert_not_degenerate()

    def test_the_distraction_rebuild_trims_the_prior_turn(self):
        """Testing the helper proves nothing if the call site ignores it —
        both of my first-cut tests passed with the wiring removed."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert "_focused.append(_trimmed_prior_turn(_last_assistant_msg))" in src
        assert "_focused.append(_last_assistant_msg)" not in src

    def test_the_continuation_gate_is_live(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert (
            "if not _has_attempt and len(_partial) > "
            "_DELIBERATION_HEAD_CHARS:" in src
        )


class TestRealWorkIsNeverTouched:
    """The failure that would be worse than the one being fixed."""

    def test_a_turn_with_a_patch_block_is_replayed_whole(self):
        content = "Here is the fix.\n" + PATCH + "\n" + "prose\n" * 3000
        out = g._trimmed_prior_turn(
            {"role": "assistant", "content": content}
        )
        assert out["content"] == content

    def test_a_turn_with_a_read_block_is_replayed_whole(self):
        content = READ + "\n" + "prose\n" * 3000
        out = g._trimmed_prior_turn(
            {"role": "assistant", "content": content}
        )
        assert out["content"] == content

    def test_a_short_turn_is_untouched(self):
        msg = {"role": "assistant", "content": "No patch needed."}
        assert g._trimmed_prior_turn(msg) is msg

    def test_an_empty_turn_is_survivable(self):
        msg = {"role": "assistant", "content": ""}
        assert g._trimmed_prior_turn(msg) is msg

    def test_a_parser_fault_replays_the_turn_rather_than_cutting_it(
        self, monkeypatch,
    ):
        """On doubt, keep the work. A trim that fires on a parser error would
        silently discard a real patch set."""
        import harness.patcher as patcher

        def _boom(_text):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(patcher, "parse_patch_blocks", _boom)
        msg = {"role": "assistant", "content": DELIBERATION}
        assert g._trimmed_prior_turn(msg) is msg


class _Resp:
    def __init__(self, content, finish_reason="length", tool_calls=None):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls


async def _never_called(_messages, _budget):  # pragma: no cover
    raise AssertionError("continuation must not dispatch here")


async def _one_more(messages, budget):
    return _Resp("continued " + PATCH, finish_reason="stop"), budget - 0.01


class TestContinuationRefusesABlocklessTurn:
    @pytest.mark.asyncio
    async def test_a_deliberating_turn_is_not_continued(self):
        """Continuing re-sends the whole argument and asks for more of it —
        the same trap the json_output gate already documents."""
        msgs: list = [{"role": "system", "content": "s"}]
        resp, budget, chunks = await g._continue_on_length(
            initial_response=_Resp(DELIBERATION),
            initial_budget=5.0,
            messages=msgs,
            dispatch=_never_called,
            continue_prompt="continue",
            enabled=True,
            role_label="repair_node",
        )
        assert budget == 5.0, "no dispatch, no spend"
        assert chunks == [DELIBERATION]
        assert len(msgs) == 1, "the argument must not be appended either"

    @pytest.mark.asyncio
    async def test_a_truncated_turn_WITH_a_block_still_continues(self):
        """The case that must keep working: half-finished work is completed,
        not abandoned. This is the whole reason continuation exists."""
        msgs: list = [{"role": "system", "content": "s"}]
        resp, budget, chunks = await g._continue_on_length(
            initial_response=_Resp("Fixing it.\n" + PATCH + "\nand then"),
            initial_budget=5.0,
            messages=msgs,
            dispatch=_one_more,
            continue_prompt="continue",
            enabled=True,
            role_label="repair_node",
        )
        assert budget < 5.0, "it must have dispatched"
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_a_truncated_turn_with_a_read_block_still_continues(self):
        msgs: list = [{"role": "system", "content": "s"}]
        _r, budget, _c = await g._continue_on_length(
            initial_response=_Resp(READ + "\npartial"),
            initial_budget=5.0,
            messages=msgs,
            dispatch=_one_more,
            continue_prompt="continue",
            enabled=True,
            role_label="repair_node",
        )
        assert budget < 5.0

    @pytest.mark.asyncio
    async def test_a_short_blockless_turn_still_continues(self):
        """A brief truncated turn may simply have been cut off before it got
        going; the refusal is for a round that spent its budget arguing."""
        msgs: list = [{"role": "system", "content": "s"}]
        _r, budget, _c = await g._continue_on_length(
            initial_response=_Resp("Let me look at the service."),
            initial_budget=5.0,
            messages=msgs,
            dispatch=_one_more,
            continue_prompt="continue",
            enabled=True,
            role_label="repair_node",
        )
        assert budget < 5.0

    @pytest.mark.asyncio
    async def test_a_normally_finished_turn_is_not_continued(self):
        msgs: list = [{"role": "system", "content": "s"}]
        _r, budget, _c = await g._continue_on_length(
            initial_response=_Resp(DELIBERATION, finish_reason="stop"),
            initial_budget=5.0,
            messages=msgs,
            dispatch=_never_called,
            continue_prompt="continue",
            enabled=True,
            role_label="repair_node",
        )
        assert budget == 5.0


class TestTheCapsGateIsDeliberate:
    """I widened this gate and a test caught it — recorded so it is not
    widened again."""

    def test_a_normally_finished_blockless_response_is_NOT_capped(self):
        """Repair parses prose SIGNALS out of a completed response — the
        UNSATISFIABLE_TEST declaration above all — and truncating at
        head_chars would cut one that lands late. A response that STOPPED
        said what it meant to say."""
        r = _Resp(DELIBERATION, finish_reason="stop")
        assert g._cap_runaway_deliberation(
            r, head_chars=4000, node_label="t",
        ) is False
        assert r.content == DELIBERATION

    def test_a_truncated_blockless_response_IS_capped(self):
        r = _Resp(DELIBERATION, finish_reason="length")
        assert g._cap_runaway_deliberation(
            r, head_chars=4000, node_label="t",
        ) is True
        assert len(r.content) < len(DELIBERATION)
