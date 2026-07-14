"""Tests for repair-loop message pruning.

Finsearch session 156032347 STORY-042 spent 10+ repair rounds arguing
with its own past 17k-char assistant messages — a `del _store[k]`
branch the LLM hallucinated in round 4 was still in context through
round 12. The pruner keeps only the immutable prefix + recent tail
once total_repairs crosses the threshold.
"""

from __future__ import annotations

from harness.repair_context import (
    DEFAULT_KEEP_TAIL,
    DEFAULT_PRUNE_AFTER_ROUND,
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
