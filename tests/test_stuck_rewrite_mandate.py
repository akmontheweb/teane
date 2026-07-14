"""Unit tests for the stuck-target REWRITE recovery mandate.

The mandate is a strong prompt block injected by ``repair_node`` on the
recovery round — the one round the router hands back to repair before
escalating to HITL when a file has crossed the REPLACE_BLOCK stuck-target
limit for the first time in the current batch. Distinct from the softer
"REWRITE_FILE UNLOCKED" grant at >= 2 misses:

  * The grant is advisory — the LLM MAY use REWRITE_FILE.
  * The mandate is directive — the LLM MUST use REWRITE_FILE on the
    listed files and MUST NOT emit surgical ops against them this round.

Added post-finsearch-156032347 (5 of 10 uncovered HITL fires from that
session were stuck REPLACE_BLOCK).
"""

from __future__ import annotations

from harness.graph import (
    _decide_stuck_rewrite_signal,
    _format_stuck_rewrite_mandate,
)


class TestDecideStuckRewriteSignal:
    """Pure-function tests for the "one recovery shot per file per
    batch" decision logic. The ``repair_node`` counter-tick block
    delegates to this helper, so any regression here would land in
    production without an async / gateway roundtrip needed to detect
    it. Semantics:

      * A file at or above ``stuck_limit`` and NOT already in the
        recovery ledger → returned in ``newly_stuck`` AND added to
        the ledger for future rounds.
      * A file at or above ``stuck_limit`` AND already in the ledger
        → NOT returned (the file already used its shot; router will
        HITL on the next pass).
      * A file whose miss counter has been CLEARED (not in
        ``rb_misses``) → dropped from the ledger, eligible again.
    """

    def test_no_stuck_files_returns_empty(self):
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"a.py": 1, "b.py": 2},
            attempted_prev=[],
            stuck_limit=3,
        )
        assert newly_stuck == []
        assert attempted_next == []

    def test_first_time_stuck_arms_recovery(self):
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"a.py": 3},
            attempted_prev=[],
            stuck_limit=3,
        )
        assert newly_stuck == ["a.py"]
        assert attempted_next == ["a.py"]

    def test_second_time_stuck_does_not_re_arm(self):
        """The recovery is one-shot per file per batch: if the ledger
        already contains the file, we do NOT stamp it again — the
        router will see an empty signal list and HITL."""
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"a.py": 5},          # still stuck, went higher
            attempted_prev=["a.py"],         # already used recovery
            stuck_limit=3,
        )
        assert newly_stuck == []
        assert attempted_next == ["a.py"]

    def test_cleared_counter_drops_ledger_entry(self):
        """A file whose miss counter has been reset (real patch success
        elsewhere between rounds) drops off the ledger — it earns back
        its recovery shot for a hypothetical future stuck event in the
        same batch. Without this filter, a self-healing file would
        permanently veto its own recovery."""
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"b.py": 1},          # a.py's counter no longer here
            attempted_prev=["a.py"],
            stuck_limit=3,
        )
        assert newly_stuck == []
        # a.py dropped because it's not in rb_misses anymore.
        assert attempted_next == []

    def test_multiple_new_stuck_files_all_armed_and_sorted(self):
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"c.py": 4, "a.py": 3, "b.py": 5},
            attempted_prev=[],
            stuck_limit=3,
        )
        assert newly_stuck == ["a.py", "b.py", "c.py"]
        assert attempted_next == ["a.py", "b.py", "c.py"]

    def test_mix_of_new_and_previously_attempted(self):
        """Ledger + fresh stuck event coexist: fresh files get armed,
        previously-attempted ones stay on the ledger without re-arm."""
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"a.py": 3, "b.py": 3},
            attempted_prev=["b.py"],
            stuck_limit=3,
        )
        assert newly_stuck == ["a.py"]
        assert attempted_next == ["a.py", "b.py"]

    def test_non_int_counter_values_are_ignored(self):
        """Defensive: a corrupted checkpoint might have a non-int under
        the per-file key. The helper must treat it as not-stuck
        (skip) rather than crash the whole tick block."""
        newly_stuck, attempted_next = _decide_stuck_rewrite_signal(
            rb_misses={"a.py": "3", "b.py": None, "c.py": 3},  # type: ignore[dict-item]
            attempted_prev=[],
            stuck_limit=3,
        )
        assert newly_stuck == ["c.py"]
        assert attempted_next == ["c.py"]


class TestFormatStuckRewriteMandate:

    def test_empty_signal_returns_empty_string(self, tmp_path):
        """No signal paths → no injection. Caller can unconditionally
        concatenate the return value."""
        assert _format_stuck_rewrite_mandate([], str(tmp_path)) == ""

    def test_mandate_includes_full_disk_content(self, tmp_path):
        f = tmp_path / "target.py"
        body = "def foo():\n    return 42\n"
        f.write_text(body)

        out = _format_stuck_rewrite_mandate(["target.py"], str(tmp_path))

        # Directive markers — the LLM's next-round parser hunts for these.
        assert "MANDATORY RECOVERY" in out
        assert "REWRITE_FILE" in out
        # The file body must appear verbatim so the LLM doesn't have to
        # re-READ. This is the whole point of the mandate — save the
        # extra READ_FILE round-trip on the LAST-chance shot.
        assert body in out
        # The relative path (as provided) is what the LLM should key on.
        assert "`target.py`" in out

    def test_mandate_lists_every_signal_path(self, tmp_path):
        (tmp_path / "a.py").write_text("A = 1\n")
        (tmp_path / "b.py").write_text("B = 2\n")

        out = _format_stuck_rewrite_mandate(["a.py", "b.py"], str(tmp_path))

        assert "`a.py`" in out
        assert "`b.py`" in out
        assert "A = 1" in out
        assert "B = 2" in out

    def test_mandate_bans_surgical_ops_explicitly(self, tmp_path):
        (tmp_path / "x.py").write_text("pass\n")
        out = _format_stuck_rewrite_mandate(["x.py"], str(tmp_path))
        # The LLM often defaults to REPLACE_BLOCK even under the softer
        # grant; the mandate must call it out by name so the ban is
        # unambiguous.
        assert "REPLACE_BLOCK" in out
        assert "MUST NOT" in out

    def test_missing_file_falls_back_to_placeholder(self, tmp_path):
        """The caller can't guarantee the path exists on disk (e.g. a
        checkpoint was resumed after the file was renamed). The mandate
        must NOT crash — it must emit a placeholder and let the LLM
        decide what to do."""
        out = _format_stuck_rewrite_mandate(
            ["never_existed.py"], str(tmp_path),
        )
        assert "never_existed.py" in out
        assert "unable to read current content" in out

    def test_absolute_path_bypasses_workspace_join(self, tmp_path):
        """When the caller passes an absolute path (rare — normally the
        harness threads relative paths — but should be tolerated),
        os.path.join must not double-anchor it into a garbage location."""
        f = tmp_path / "abs.py"
        f.write_text("ABS = True\n")
        out = _format_stuck_rewrite_mandate([str(f)], "/some/other/root")
        # Content is there → the abs-path branch worked.
        assert "ABS = True" in out

    def test_large_file_truncated_with_marker(self, tmp_path):
        """A 200 KB fixture would blow the context on the recovery round.
        Truncation must be signaled so the LLM doesn't preserve the
        truncation artifact in its REWRITE."""
        big = "line\n" * 20_000  # ~100 KB
        (tmp_path / "big.py").write_text(big)
        out = _format_stuck_rewrite_mandate(["big.py"], str(tmp_path))
        # The truncation marker is the contract with the LLM; without it
        # the model would faithfully write a file that ENDS mid-way.
        assert "truncated" in out
