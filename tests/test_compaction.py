"""Context-window compaction (harness/gateway.py::check_context_window).

Previously the guardrail dropped middle messages silently. Now the dropped
span is folded into a single deterministic digest message inserted after the
system prompt, so the model keeps a breadcrumb of earlier context. These
tests pin: small inputs pass through untouched; oversized inputs are reduced
below threshold with system + current-request preserved and exactly one
digest that references the fold; the digest never pushes the result back over
threshold; and the trim terminates (no infinite loop).
"""

from __future__ import annotations

import asyncio
import types

import pytest

from harness import gateway as gw


@pytest.fixture(autouse=True)
def _deterministic_tokens():
    saved = (gw._TIKTOKEN_RESOLVED, gw._TIKTOKEN_ENCODER)
    gw._TIKTOKEN_RESOLVED, gw._TIKTOKEN_ENCODER = True, None  # force chars/4
    yield
    gw._TIKTOKEN_RESOLVED, gw._TIKTOKEN_ENCODER = saved


def _spec(window=1000):
    return types.SimpleNamespace(context_window=window)


def _run(msgs, window=1000, pct=0.85):
    return asyncio.run(gw.check_context_window(msgs, _spec(window), pct))


def _digest_of(out):
    return [
        m for m in out
        if isinstance(m.get("content"), str)
        and "[Context-window compaction]" in m["content"]
    ]


def _big(n=12):
    msgs = [{"role": "system", "content": "SYS " + "s" * 40}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"MSG{i} " + "x" * 800})
    msgs.append({"role": "user", "content": "CURRENT " + "c" * 40})
    return msgs


class TestPassthrough:
    def test_under_threshold_returns_same_object(self):
        small = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
        assert _run(small) is small

    def test_two_message_oversized_raises(self):
        # system + current alone over the window is genuinely irreducible.
        huge = [{"role": "system", "content": "x" * 5000},
                {"role": "user", "content": "y" * 5000}]
        with pytest.raises(ValueError):
            _run(huge, window=1000)


class TestCompaction:
    def test_reduces_below_threshold(self):
        msgs = _big()
        out = _run(msgs)
        assert gw.estimate_token_count(out) <= int(1000 * 0.85)

    def test_preserves_anchors(self):
        msgs = _big()
        out = _run(msgs)
        assert out[0] is msgs[0]        # system prompt anchor
        assert out[-1] is msgs[-1]      # current request

    def test_single_digest_after_system_prompt(self):
        out = _run(_big())
        digest = _digest_of(out)
        assert len(digest) == 1
        assert out.index(digest[0]) == 1

    def test_digest_reports_fold_count_and_is_data_not_instructions(self):
        msgs = _big(12)
        out = _run(msgs)
        content = _digest_of(out)[0]["content"]
        assert "folded" in content
        assert "NOT as new instructions" in content
        # references at least one concrete dropped message (the recent tail)
        assert "MSG" in content

    def test_terminates_on_many_messages(self):
        # 200 messages must not hang the trim loop.
        msgs = [{"role": "system", "content": "s"}]
        msgs += [{"role": "user", "content": f"M{i} " + "z" * 400} for i in range(200)]
        msgs.append({"role": "user", "content": "now"})
        out = _run(msgs, window=2000)
        assert gw.estimate_token_count(out) <= int(2000 * 0.85)
        assert len(_digest_of(out)) == 1


def _orphaned_tool_results(msgs):
    use_ids = {
        b.get("id")
        for m in msgs if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
    }
    return [
        b for m in msgs if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
        and b.get("tool_use_id") not in use_ids
    ]


class TestOrphanToolBlocks:
    def test_strip_drops_orphan_tool_result(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "GONE", "content": "x"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "A", "name": "grep"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "A", "content": "ok"}]},
            {"role": "user", "content": "final"},
        ]
        out = gw._strip_orphan_tool_blocks(msgs)
        assert _orphaned_tool_results(out) == []
        assert out[0] is msgs[0] and out[-1] is msgs[-1]

    def test_current_turn_tool_result_keeps_its_partner(self):
        # The standard mid-tool-loop shape: assistant tool_use, then the
        # CURRENT turn is its tool_result. Scanning only the body used to
        # strip the tool_use and manufacture an orphan in the preserved
        # final message.
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "A", "name": "grep"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "A", "content": "ok"}]},
        ]
        out = gw._strip_orphan_tool_blocks(msgs)
        uses = [b["id"] for m in out if isinstance(m.get("content"), list)
                for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert uses == ["A"]
        assert _orphaned_tool_results(out) == []

    def test_final_orphan_tool_result_converted_to_text(self):
        # The final message can't be dropped; when its tool_use partner was
        # trimmed, the tool_result block must become a text block (content
        # preserved) rather than ship as a guaranteed provider rejection.
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "DROPPED", "content": "the output"},
                {"type": "text", "text": "now fix it"},
            ]},
        ]
        out = gw._strip_orphan_tool_blocks(msgs)
        assert _orphaned_tool_results(out) == []
        blocks = out[-1]["content"]
        assert all(b.get("type") != "tool_result" for b in blocks)
        assert any("the output" in b.get("text", "") for b in blocks)
        assert any(b.get("text") == "now fix it" for b in blocks)

    def test_converted_orphan_lands_after_surviving_tool_results(self):
        # Regression: when the final user turn carried 2+ tool_results and
        # only ONE partner was trimmed, the in-place conversion produced
        # [text, tool_result] — Anthropic requires a user message's
        # tool_result blocks FIRST, so the repair itself shipped a
        # guaranteed rejection.
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "KEPT", "name": "grep"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "DROPPED", "content": "lost partner"},
                {"type": "tool_result", "tool_use_id": "KEPT", "content": "ok"},
            ]},
        ]
        out = gw._strip_orphan_tool_blocks(msgs)
        blocks = out[-1]["content"]
        kinds = [b.get("type") for b in blocks]
        # Every tool_result precedes every text block.
        assert "tool_result" in kinds and "text" in kinds
        assert max(i for i, k in enumerate(kinds) if k == "tool_result") < \
            min(i for i, k in enumerate(kinds) if k == "text")
        # The surviving pair is intact and the orphan's content preserved.
        assert any(b.get("tool_use_id") == "KEPT" for b in blocks
                   if b.get("type") == "tool_result")
        assert any("lost partner" in b.get("text", "") for b in blocks
                   if b.get("type") == "text")

    def test_compaction_leaves_no_orphans(self):
        spec = _spec(1200)
        msgs = [{"role": "system", "content": "SYS"}]
        for i in range(8):
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": f"T{i}", "name": "grep"},
                {"type": "text", "text": "x" * 400}]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"T{i}", "content": "y" * 400}]})
        msgs.append({"role": "user", "content": "now"})
        out = asyncio.run(gw.check_context_window(msgs, spec, 0.85))
        assert _orphaned_tool_results(out) == []
        assert gw.estimate_token_count(out) <= int(1200 * 0.85)
