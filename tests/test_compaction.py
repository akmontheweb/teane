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
