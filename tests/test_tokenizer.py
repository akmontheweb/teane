"""Preflight/context token estimation (harness/gateway.py).

count_text_tokens uses a real BPE tokenizer (tiktoken) when installed and
falls back to the historical chars/4 heuristic otherwise. These tests pin
both paths via a mock encoder so they're env-independent, and confirm the
fallback is byte-identical to the old estimate (no behaviour change when
tiktoken is absent).
"""

from __future__ import annotations

import pytest

from harness import gateway as gw


@pytest.fixture(autouse=True)
def _reset_encoder():
    # Save/restore the module-level tokenizer cache around each test.
    saved = (gw._TIKTOKEN_RESOLVED, gw._TIKTOKEN_ENCODER)
    yield
    gw._TIKTOKEN_RESOLVED, gw._TIKTOKEN_ENCODER = saved


def _force_fallback():
    gw._TIKTOKEN_RESOLVED = True
    gw._TIKTOKEN_ENCODER = None


class _FakeEnc:
    """1 token per whitespace-delimited word."""
    def encode(self, text, disallowed_special=()):
        return list(range(len(text.split())))


def _force_tiktoken():
    gw._TIKTOKEN_RESOLVED = True
    gw._TIKTOKEN_ENCODER = _FakeEnc()


class TestCountTextTokens:
    def test_empty_is_zero(self):
        assert gw.count_text_tokens("") == 0

    def test_fallback_is_chars_over_four(self):
        _force_fallback()
        assert gw.count_text_tokens("x" * 400) == 100
        assert gw.count_text_tokens("abc") == max(1, 3 // 4)  # >=1

    def test_uses_tokenizer_when_present(self):
        _force_tiktoken()
        assert gw.count_text_tokens("one two three four") == 4

    def test_tokenizer_exception_falls_back(self):
        class _Boom:
            def encode(self, text, disallowed_special=()):
                raise RuntimeError("boom")
        gw._TIKTOKEN_RESOLVED = True
        gw._TIKTOKEN_ENCODER = _Boom()
        assert gw.count_text_tokens("x" * 40) == 10  # fell back to chars/4


class TestEstimateTokenCount:
    def test_sums_contents_plus_overhead_fallback(self):
        _force_fallback()
        msgs = [
            {"role": "user", "content": "x" * 400},   # 100
            {"role": "system", "content": "y" * 40},  # 10
        ]
        # 100 + 10 + 12 overhead per message (2)
        assert gw.estimate_token_count(msgs) == 100 + 10 + 12 + 12

    def test_handles_list_content_blocks(self):
        _force_fallback()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "z" * 40}]}]
        # str(block) is longer than 40 chars, but must be counted + overhead
        assert gw.estimate_token_count(msgs) > 12
