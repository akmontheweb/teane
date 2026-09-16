"""A response that never reaches a block must not go on costing money.

lumina-run7-20260915-1345 measured both halves of the cost. Repair call
0064 returned 140,425 characters of deliberation, spent the full
32,768-token output cap at $0.0288 (a normal repair round on that run cost
~$0.006), and the very next call's input then ballooned from a ~63k-token
baseline to 94,714 tokens at $0.0206. Call 0047 did the same with 141,172
characters.

Call 0064's first paragraph had already reached the right answer -- "the
only way to make this pass without modifying the test is to make the API
route use a fixed date. But that breaks production behavior" -- which is
exactly the UNSATISFIABLE_TEST case. The remaining 140,000 characters were
the model talking itself out of it. So the head is kept and the tail
dropped, rather than the whole response discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.graph import (
    _cap_runaway_deliberation,
    _resolve_deliberation_head_chars,
)


@dataclass
class _Resp:
    content: str = ""
    finish_reason: str = "stop"
    _extra: dict = field(default_factory=dict)


PROSE = "Wait -- actually, let me reconsider this. " * 5000
PATCH = (
    "<<<REPLACE_BLOCK>>>\nfile: server/app/main.py\nsearch:\na\nreplace:\nb\n"
    "<<<END_REPLACE_BLOCK>>>"
)
READ = "<<<READ_FILE>>>\nfile: server/app/main.py\n<<<END_READ_FILE>>>"


class TestLeavesGoodRoundsAlone:
    """The cap must never touch a round that produced something usable."""

    def test_normal_completion_untouched(self) -> None:
        r = _Resp(content=PROSE, finish_reason="stop")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is False
        assert r.content == PROSE

    def test_truncated_but_carries_a_patch_block(self) -> None:
        """Truncation with a usable block is the continuation path's job."""
        r = _Resp(content=PROSE + PATCH, finish_reason="length")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is False
        assert PATCH in r.content

    def test_truncated_but_carries_a_read_block(self) -> None:
        r = _Resp(content=PROSE + READ, finish_reason="length")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is False
        assert READ in r.content

    def test_short_truncated_response_untouched(self) -> None:
        """Below the head budget there is nothing to reclaim."""
        r = _Resp(content="short thought", finish_reason="length")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is False
        assert r.content == "short thought"


class TestCapsTheRunaway:
    def test_caps_and_keeps_the_head(self) -> None:
        head = "Root cause 1: the route uses date.today(), which the test "
        r = _Resp(content=head + PROSE, finish_reason="length")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is True
        assert r.content.startswith(head), "the diagnosis must survive"
        assert len(r.content) < len(head + PROSE)

    def test_tells_the_model_what_happened_and_what_to_do(self) -> None:
        r = _Resp(content=PROSE, finish_reason="length")
        _cap_runaway_deliberation(r, head_chars=4000, node_label="t")
        assert "output token cap" in r.content
        assert "UNSATISFIABLE_TEST" in r.content

    def test_is_idempotent(self) -> None:
        """A second pass must not re-cap an already-capped response."""
        r = _Resp(content=PROSE, finish_reason="length")
        assert _cap_runaway_deliberation(
            r, head_chars=4000, node_label="t") is True
        first = r.content
        _cap_runaway_deliberation(r, head_chars=4000, node_label="t")
        assert r.content == first


class TestHeadCharsResolution:
    def test_default(self) -> None:
        assert _resolve_deliberation_head_chars({}) == 4000

    def test_reads_config(self) -> None:
        assert _resolve_deliberation_head_chars(
            {"llm_dispatch_config": {"deliberation_head_chars": 8000}}
        ) == 8000

    def test_clamps_and_tolerates_junk(self) -> None:
        for raw, want in ((10, 500), (999999, 40000), ("nope", 4000)):
            assert _resolve_deliberation_head_chars(
                {"llm_dispatch_config": {"deliberation_head_chars": raw}}
            ) == want
