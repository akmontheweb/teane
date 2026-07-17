"""Tests for repair-loop message pruning.

Finsearch session 156032347 STORY-042 spent 10+ repair rounds arguing
with its own past 17k-char assistant messages — a `del _store[k]`
branch the LLM hallucinated in round 4 was still in context through
round 12. The pruner keeps only the immutable prefix + recent tail
once total_repairs crosses the threshold.
"""

from __future__ import annotations

import asyncio

import harness.repair_context as rc
from harness.repair_context import (
    DEFAULT_KEEP_TAIL,
    DEFAULT_PRUNE_AFTER_ROUND,
    condense_repair_messages,
    prune_repair_messages,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


class TestNoOpBeforeThreshold:
    """First N repair rounds keep the full history — the LLM has every
    chance to converge with maximum context."""

    def test_round_1_passes_through(self) -> None:
        msgs = [_msg("system", "s"), _msg("user", "u"), _msg("assistant", "a")]
        assert prune_repair_messages(msgs, total_repairs=1) == msgs

    def test_round_3_boundary_passes_through(self) -> None:
        msgs = [_msg("system", f"m{i}") for i in range(20)]
        # exactly at prune_after_round → still full pass-through
        result = prune_repair_messages(
            msgs, total_repairs=DEFAULT_PRUNE_AFTER_ROUND,
        )
        assert result == msgs

    def test_short_history_never_pruned(self) -> None:
        # Even past threshold, a small array (head + tail already covers
        # everything) is passed through — nothing to drop.
        msgs = [_msg("system", "s"), _msg("user", "u")]
        result = prune_repair_messages(msgs, total_repairs=10)
        assert result == msgs

    def test_result_is_a_copy_not_alias(self) -> None:
        # Callers mutate their working list; pruner must not hand back
        # the caller's own list.
        msgs = [_msg("system", "s"), _msg("user", "u")]
        result = prune_repair_messages(msgs, total_repairs=1)
        assert result == msgs
        assert result is not msgs


class TestPruneAfterThreshold:
    """Past the threshold, only the head (msg[0..2]) + tail survive."""

    def _long_history(self, n: int) -> list[dict[str, str]]:
        return (
            [_msg("system", "system-prompt"), _msg("user", "initial-task")]
            + [_msg("assistant", f"round-{i}-attempt") for i in range(n)]
        )

    def test_round_4_prunes_middle(self) -> None:
        msgs = self._long_history(15)  # 2 head + 15 assistant = 17
        result = prune_repair_messages(msgs, total_repairs=4)
        # head 2 + tail 6 = 8 total
        assert len(result) == 2 + DEFAULT_KEEP_TAIL
        # Immutable prefix intact
        assert result[0]["content"] == "system-prompt"
        assert result[1]["content"] == "initial-task"
        # Tail is the LAST keep_tail messages, not the first
        assert result[-1]["content"] == "round-14-attempt"
        # Middle rounds gone
        contents = [m["content"] for m in result]
        assert "round-4-attempt" not in contents
        assert "round-0-attempt" not in contents

    def test_prefix_survives_across_rounds(self) -> None:
        # The prefix-cache anchor MUST stay byte-identical every round
        # — otherwise the cache is evicted and cost skyrockets.
        msgs_r4 = self._long_history(15)
        msgs_r10 = self._long_history(30)
        r4 = prune_repair_messages(msgs_r4, total_repairs=4)
        r10 = prune_repair_messages(msgs_r10, total_repairs=10)
        assert r4[:2] == r10[:2]

    def test_tail_length_is_bounded(self) -> None:
        # No matter how deep the repair loop goes, the array stays
        # bounded at 2 (head) + keep_tail.
        msgs = self._long_history(100)
        result = prune_repair_messages(msgs, total_repairs=50)
        assert len(result) == 2 + DEFAULT_KEEP_TAIL


class TestOverrides:
    def test_prune_after_round_override(self) -> None:
        msgs = [_msg("system", "s"), _msg("user", "u")] + [
            _msg("assistant", f"a{i}") for i in range(10)
        ]
        # Aggressive: prune from round 1
        result = prune_repair_messages(
            msgs, total_repairs=2, prune_after_round=1, keep_tail=3,
        )
        assert len(result) == 5

    def test_keep_tail_override(self) -> None:
        msgs = [_msg("system", "s"), _msg("user", "u")] + [
            _msg("assistant", f"a{i}") for i in range(10)
        ]
        result = prune_repair_messages(
            msgs, total_repairs=4, keep_tail=2,
        )
        # 2 head + 2 tail
        assert len(result) == 4
        assert result[-1]["content"] == "a9"
        assert result[-2]["content"] == "a8"


def _history(n: int, prefix: str = "task") -> list[dict[str, str]]:
    return (
        [_msg("system", "system-prompt"), _msg("user", f"{prefix}-initial")]
        + [_msg("assistant", f"round-{i}-attempt") for i in range(n)]
    )


class TestCondense:
    """LLM-condensing variant: dropped middle becomes a digest message."""

    def setup_method(self) -> None:
        rc._CONDENSE_CACHE.clear()

    def _run(self, msgs, judgment_call, total_repairs=4, **kw):
        return asyncio.run(condense_repair_messages(
            msgs, total_repairs=total_repairs,
            judgment_call=judgment_call, **kw,
        ))

    def test_below_threshold_passes_through(self) -> None:
        calls: list[str] = []

        async def judge(prompt: str):
            calls.append(prompt)
            return "SUMMARY"

        msgs = _history(15)
        result = self._run(msgs, judge, total_repairs=DEFAULT_PRUNE_AFTER_ROUND)
        assert result == msgs
        assert calls == []  # no LLM spend before the threshold

    def test_digest_inserted_after_immutable_prefix(self) -> None:
        async def judge(prompt: str):
            return "tried X in round 1; dead end: patching foo.py"

        msgs = _history(15)
        result = self._run(msgs, judge)
        # head 2 + digest + tail
        assert len(result) == 2 + 1 + DEFAULT_KEEP_TAIL
        assert result[0]["content"] == "system-prompt"  # cache anchor intact
        assert result[1]["content"] == "task-initial"
        assert result[2]["role"] == "user"
        assert "Repair-history digest" in result[2]["content"]
        assert "dead end: patching foo.py" in result[2]["content"]
        assert result[-1]["content"] == "round-14-attempt"

    def test_incremental_only_delta_summarized(self) -> None:
        prompts: list[str] = []

        async def judge(prompt: str):
            prompts.append(prompt)
            return f"summary-v{len(prompts)}"

        # Round 4: middle = rounds 0..8 (15 turns, tail keeps last 6).
        self._run(_history(15), judge)
        # Round 5: history grew by 3 turns; only the newly-dropped turns
        # plus the previous summary go to the LLM — not rounds 0..8 again.
        result = self._run(_history(18), judge, total_repairs=5)
        assert len(prompts) == 2
        assert "summary-v1" in prompts[1]          # previous summary folded in
        assert "round-0-attempt" not in prompts[1]  # already-covered turns not resent
        assert "summary-v2" in result[2]["content"]

    def test_no_delta_reuses_cached_summary_without_llm_call(self) -> None:
        prompts: list[str] = []

        async def judge(prompt: str):
            prompts.append(prompt)
            return "stable-summary"

        self._run(_history(15), judge)
        result = self._run(_history(15), judge, total_repairs=5)
        assert len(prompts) == 1  # second round: cache hit, no spend
        assert "stable-summary" in result[2]["content"]

    def test_none_from_llm_falls_back_to_plain_prune(self) -> None:
        async def judge(prompt: str):
            return None

        msgs = _history(15)
        result = self._run(msgs, judge)
        assert result == prune_repair_messages(msgs, total_repairs=4)

    def test_exception_from_llm_falls_back_to_plain_prune(self) -> None:
        async def judge(prompt: str):
            raise RuntimeError("provider down")

        msgs = _history(15)
        result = self._run(msgs, judge)
        assert result == prune_repair_messages(msgs, total_repairs=4)

    def test_missing_judgment_call_behaves_like_prune(self) -> None:
        msgs = _history(15)
        result = self._run(msgs, None)
        assert result == prune_repair_messages(msgs, total_repairs=4)

    def test_stale_summary_survives_llm_outage(self) -> None:
        healthy = True

        async def judge(prompt: str):
            return "good-summary" if healthy else None

        self._run(_history(15), judge)
        healthy = False
        # LLM down this round: previous summary still inserted (stale
        # beats absent), and the uncovered delta stays pending.
        result = self._run(_history(18), judge, total_repairs=5)
        assert "good-summary" in result[2]["content"]

    def test_stale_summary_survives_llm_raise(self) -> None:
        # Regression: the except path set summary = "" — a RAISING
        # provider (vs one returning None) discarded the cached digest,
        # contradicting the docstring's "stale beats absent" contract.
        healthy = True

        async def judge(prompt: str):
            if healthy:
                return "good-summary"
            raise RuntimeError("provider down")

        self._run(_history(15), judge)
        healthy = False
        result = self._run(_history(18), judge, total_repairs=5)
        assert "good-summary" in result[2]["content"]

    def test_distinct_sessions_do_not_share_summaries(self) -> None:
        async def judge_a(prompt: str):
            return "summary-for-a"

        async def judge_b(prompt: str):
            return "summary-for-b"

        ra = self._run(_history(15, prefix="alpha"), judge_a)
        rb = self._run(_history(15, prefix="beta"), judge_b)
        assert "summary-for-a" in ra[2]["content"]
        assert "summary-for-b" in rb[2]["content"]

    def test_summary_clipped_to_max_chars(self) -> None:
        async def judge(prompt: str):
            return "x" * 10_000

        result = self._run(_history(15), judge, max_summary_chars=500)
        digest_body = result[2]["content"].split(":\n", 1)[1]
        assert len(digest_body) == 500
