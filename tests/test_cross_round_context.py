"""Regression tests for the always-on cross-round workspace context
injection into the repair-node prompt.

Ciod session 523e86a7's batch 28 stalled 90+ minutes in repair — the
LLM produced 1 patch/round without cross-round awareness because
memory cleanse strips inter-round context. The workspace-context
enrichment (file modification history + symbol registry + reverse
import map) already existed but was gated behind the persistent-
blocker banner, whose trigger requires the SAME file/line to be
targeted 2+ rounds running. Batch 28 never tripped that gate — the
judge named a different file each round.

The fix widens the trigger: cross-round context is now injected on
every repair round for files that:
  * appear in the current diagnostic set, OR
  * were modified by the LLM in the last 2 rounds

Capped at ``_CROSS_ROUND_CTX_CAP`` files so the prompt stays
bounded.
"""

from __future__ import annotations


class TestRecentModifiedFilesSelection:
    """Replica of the inline selection loop from ``repair_node`` —
    kept as a helper here so future refactors that extract the
    production loop into a callable can share this suite."""

    def _select_recent(self, history, current_round, window=2):
        recent = set()
        for f, entries in history.items():
            if not entries:
                continue
            last_entry = entries[-1]
            if not isinstance(last_entry, (list, tuple)) or not last_entry:
                continue
            try:
                last_round = int(last_entry[0])
            except (TypeError, ValueError):
                continue
            if current_round - last_round <= window:
                recent.add(f)
        return recent

    def _pick_context_files(self, diag_files, recent_modified, cap):
        picked = []
        seen = set()
        for f in list(diag_files) + sorted(recent_modified):
            if f in seen:
                continue
            seen.add(f)
            picked.append(f)
            if len(picked) >= cap:
                break
        return picked

    def test_recent_window_catches_last_two_rounds(self):
        # ``window=2`` means "rounds N, N-1, N-2 all qualify" — the
        # LLM's mental model of a file it patched two rounds ago is
        # still fresh enough to be worth surfacing, but three rounds
        # back is beyond the mid-loop cleanse horizon.
        history = {
            "a.py": [[5, "replace_block", True, False, ""]],  # current
            "b.py": [[4, "replace_block", True, False, ""]],  # -1
            "c.py": [[3, "replace_block", True, False, ""]],  # -2 (still in)
            "d.py": [[2, "replace_block", True, False, ""]],  # -3 (out)
            "e.py": [[1, "replace_block", True, False, ""]],  # far out
        }
        recent = self._select_recent(history, current_round=5, window=2)
        assert recent == {"a.py", "b.py", "c.py"}

    def test_diag_files_prioritised_over_recent_modified(self):
        # If both diag_files and recent_modified name the same path,
        # dedupe keeps only one entry. Diag files come first in the
        # prompt because they're the immediate target.
        picked = self._pick_context_files(
            diag_files=["diag_file.py"],
            recent_modified={"diag_file.py", "other.py"},
            cap=3,
        )
        assert picked == ["diag_file.py", "other.py"]

    def test_cap_bounds_selection(self):
        # 5 diag files + 5 recently modified, cap=3 → exactly 3.
        picked = self._pick_context_files(
            diag_files=[f"d{i}.py" for i in range(5)],
            recent_modified={f"r{i}.py" for i in range(5)},
            cap=3,
        )
        assert len(picked) == 3
        # Diag files fill the cap first.
        assert all(f.startswith("d") for f in picked)

    def test_empty_history_and_no_diag_returns_nothing(self):
        picked = self._pick_context_files(
            diag_files=[],
            recent_modified=set(),
            cap=3,
        )
        assert picked == []

    def test_recent_modified_alone_still_injected(self):
        # No diagnostics on a file this round, but the LLM patched it
        # last round → cross-round context should still surface so the
        # LLM sees its own recent history.
        picked = self._pick_context_files(
            diag_files=[],
            recent_modified={"stale.py"},
            cap=3,
        )
        assert picked == ["stale.py"]


class TestCrossRoundCtxCap:
    def test_module_cap_is_bounded(self):
        # Empirical: 3 files at 300 lines / 6 KB each caps the
        # enrichment at ~18 KB of prompt real estate — enough to
        # cover the diagnostic hot-spots without starving downstream
        # sections (diagnostics, patch failures, allowlist).
        from harness.graph import _CROSS_ROUND_CTX_CAP
        assert 1 <= _CROSS_ROUND_CTX_CAP <= 10
