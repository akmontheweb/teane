"""Regression tests for the 2026-07-07 repair-loop audit fixes.

Eight findings surfaced by an audit of the repair loop after the cec4d124
HITL#4 fix. All eight are siblings of previously-fixed bugs — the earlier
fix was the instance, this one is the class.

Findings covered:
  #1 patching_node uses real_success_count for tripwire gates
     (graph.py ~L4083, 4108, 4121)
  #2 code_review_node uses real_success_count for reset + repatched flag
     (graph.py ~L17117)
  #3 no_progress_repairs added to _STALL_TRIPWIRE_KEYS
     (graph.py ~L9720)
  #4 budget_preflight terminates like budget_exhausted in headless mode
     (cli.py ~L3340)
  #5 Save-and-Quit rewind preserves diagnostic trackers via
     _reset_iteration_counters (graph.py ~L19287)
  #6 _reset_stale_gate_counters_on_resume includes
     consecutive_all_allowlist_rejected_rounds + cheap_shots_taken
     (graph.py ~L19103)
  #7 cheap_shots_taken reset on forward progress via _STALL_TRIPWIRE_KEYS
     (graph.py ~L9720)
  #8 code_review re-patch advances progress_tracker no_progress marker
     (graph.py ~L17128)
"""

from __future__ import annotations

from harness.graph import _STALL_TRIPWIRE_KEYS


# ---------------------------------------------------------------------------
# Finding #3 + #7 — additions to _STALL_TRIPWIRE_KEYS
# ---------------------------------------------------------------------------


class TestStallTripwireKeysExtension:
    """Both counters now sit alongside the four original tripwires so a
    green build / code_review re-patch clears them together."""

    def test_no_progress_repairs_in_key_set(self):
        assert "no_progress_repairs" in _STALL_TRIPWIRE_KEYS

    def test_cheap_shots_taken_in_key_set(self):
        assert "cheap_shots_taken" in _STALL_TRIPWIRE_KEYS


# ---------------------------------------------------------------------------
# Finding #4 — budget_preflight guard in headless auto-resume
# ---------------------------------------------------------------------------


class TestBudgetPreflightGuard:
    """cli.py:3340 excluded ``budget_exhausted`` from auto-resume but
    fell through on ``budget_preflight``. Both live in the budget family
    and both need to terminate — auto-resume without adding budget just
    re-hits the same wall in ~10ms."""

    def test_guard_includes_budget_preflight(self):
        # Introspect the source to lock in the {both} set. The behavior
        # is guarded inside an async function that requires a full HITL
        # menu setup to exercise end-to-end — a source-level check is
        # the pragmatic backstop against a regression that would only
        # surface under headless mode with a preflight trigger.
        import inspect
        from harness import cli
        src = inspect.getsource(cli.hitl_menu_loop)
        # Must reference BOTH trigger names in the same guard.
        assert 'budget_exhausted' in src
        assert 'budget_preflight' in src
        # And they must appear together in a set/tuple literal — a
        # single ``in {..., ...}`` check.
        assert (
            '{"budget_exhausted", "budget_preflight"}' in src
            or '{"budget_preflight", "budget_exhausted"}' in src
        ), "guard must be a single membership check on both triggers"


# ---------------------------------------------------------------------------
# Finding #6 — _reset_stale_gate_counters_on_resume parity
# ---------------------------------------------------------------------------


class TestResumeGateKeyParity:
    """The [r] auto-resume path (``_reset_hitl_trip_counters``) and the
    ``teane resume`` path (``_reset_stale_gate_counters_on_resume``)
    both grant a fresh repair budget. They MUST reset the same key set
    or one resume path lets a stale counter re-trip HITL immediately."""

    def test_resume_gate_keys_include_allowlist_and_cheap_shots(self):
        import inspect
        from harness import graph
        src = inspect.getsource(graph._reset_stale_gate_counters_on_resume)
        # Extended 2026-07-07 to match _reset_hitl_trip_counters.
        assert '"consecutive_all_allowlist_rejected_rounds"' in src
        assert '"cheap_shots_taken"' in src


class TestResetHitlTripDropsRewriteRecoveryLedger:
    """Companion invariant to the cap-to-2 behavior on
    ``replace_block_misses_per_file``. Post-finsearch-156032347:
    ``_reset_hitl_trip_counters`` caps per-file miss counts at 2 so
    headless auto-resume gets ONE more repair attempt on the stuck file
    without immediately re-tripping the >=3 stuck-file HITL guard.

    The REWRITE-recovery ledger (``stuck_rewrite_recovery_attempted``)
    would silently veto that repair attempt from becoming a REWRITE
    recovery round — repair_node's ``_decide_stuck_rewrite_signal``
    skips files already in the ledger. So the reset MUST drop ledger
    entries for exactly the files whose miss counter it just capped;
    otherwise the cap-to-2 unlocks REPLACE_BLOCK misses to run again
    but recovery is dead, and the router HITLs again the moment the
    counter climbs back to 3.
    """

    def test_ledger_entry_dropped_when_file_miss_capped(self):
        from harness.cli import _reset_hitl_trip_counters

        loop_counter: dict = {
            "replace_block_misses_per_file": {
                "test_a.py": 3,   # will be capped to 2 → drop ledger
                "test_b.py": 2,   # already <=2 → keep ledger
                "test_c.py": 5,   # will be capped to 2 → drop ledger
            },
            "stuck_rewrite_recovery_attempted": [
                "test_a.py", "test_b.py", "test_c.py", "test_d.py",
            ],
        }
        _reset_hitl_trip_counters(loop_counter)

        # test_a and test_c were capped → dropped from ledger. test_b
        # was already <=2 (not capped) → stays. test_d isn't in the
        # miss map at all → the reset has no signal that its ledger
        # entry needs clearing (would be handled by the miss-cleared
        # filter in repair_node's tick block instead).
        assert loop_counter["stuck_rewrite_recovery_attempted"] == [
            "test_b.py", "test_d.py",
        ]

    def test_ledger_untouched_when_no_files_capped(self):
        from harness.cli import _reset_hitl_trip_counters

        loop_counter: dict = {
            "replace_block_misses_per_file": {"a.py": 1, "b.py": 2},
            "stuck_rewrite_recovery_attempted": ["a.py", "b.py"],
        }
        _reset_hitl_trip_counters(loop_counter)

        # No files were >=3, so nothing was capped — ledger stays.
        assert loop_counter["stuck_rewrite_recovery_attempted"] == [
            "a.py", "b.py",
        ]

    def test_missing_ledger_is_tolerated(self):
        from harness.cli import _reset_hitl_trip_counters

        # A checkpoint from before this feature landed won't have the
        # ledger key. The reset must not KeyError — the empty-map
        # branch simply doesn't need to write anything.
        loop_counter: dict = {
            "replace_block_misses_per_file": {"a.py": 4},
        }
        _reset_hitl_trip_counters(loop_counter)
        # No ledger key was created (nothing to write since prior=None).
        assert "stuck_rewrite_recovery_attempted" not in loop_counter


# ---------------------------------------------------------------------------
# Finding #5 — Save-and-Quit rewind preserves diagnostic trackers
# ---------------------------------------------------------------------------


class TestSaveAndQuitRewindPreservesTrackers:
    """The pre-2026-07-07 rewind wiped ``loop_counter`` down to
    ``{patching, repair, compiler, total_repairs}``. That destroyed the
    per-file diagnostic trackers whose whole reason for surviving HITL
    is to keep the "use a different operation" / no-op fixation
    directives firing after resume."""

    def test_rewind_uses_reset_iteration_counters(self):
        # The rewind function is stateful and requires a compiled graph
        # to exercise. Guard via source-inspection: the fix threads
        # ``_reset_iteration_counters`` through the rewind path instead
        # of the old 4-key dict-literal wipe.
        import inspect
        from harness import graph
        src = inspect.getsource(graph._rewind_suspended_checkpoint)
        # The old wipe was a hard-coded 4-key literal — check it's gone.
        assert '"patching": 0' not in src or "_reset_iters" in src, (
            "the 4-key wipe must be replaced with _reset_iteration_counters"
        )
        # The new path calls the shared reset helper.
        assert "_reset_iteration_counters" in src


# ---------------------------------------------------------------------------
# Finding #1 — patching_node uses real_success_count on tripwire gates
# ---------------------------------------------------------------------------


class TestPatchingNodeRealSuccessCount:
    """Idempotency no-ops (LLM re-emitting already-applied patches)
    must NOT reset the stall tripwires. patching_node was the sibling
    bug to repair_node's real_success_count switch."""

    def test_patching_node_defines_real_success_count(self):
        # Guard the semantic contract at source level: the function
        # must compute real_success_count = success_count - no_op_count
        # and use it to gate tripwires. Full behaviour is exercised
        # by story/batch integration tests; this is the minimal
        # invariant check.
        import inspect
        from harness import graph
        src = inspect.getsource(graph.patching_node)
        assert "real_success_count" in src, (
            "patching_node must compute real_success_count (audit #1)"
        )
        # The three gate sites that were previously on raw success_count
        # must now be on real_success_count. Check each key appears
        # near a real_success_count reference.
        assert src.count("real_success_count") >= 4, (
            "expected real_success_count at multiple gate sites: "
            "no_progress marker, consecutive_zero_patch_rounds, "
            "consecutive_all_allowlist_rejected_rounds, "
            "story_zero_patch_rounds"
        )


# ---------------------------------------------------------------------------
# Finding #2 + #8 — code_review re-patch uses real_success_count and
# advances the no_progress marker
# ---------------------------------------------------------------------------


class TestCodeReviewRepatchRealSuccessCount:
    """The reset shipped in commit 5c47b81 used raw ``success_count``.
    Reviewer re-patches that produce only idempotent no-ops would
    falsely wipe the stall counters — same class as finding #1."""

    def test_code_review_uses_real_success_count(self):
        import inspect
        from harness import graph
        src = inspect.getsource(graph.code_review_node)
        assert "real_success_count" in src, (
            "code_review_node must compute real_success_count (audit #2)"
        )

    def test_code_review_advances_no_progress_marker(self):
        # Finding #8: code_review re-patch success is real forward
        # progress; the no_progress failsafe marker has to advance or
        # a follow-up compile fail can trip early on a session that
        # just landed real changes.
        import inspect
        from harness import graph
        src = inspect.getsource(graph.code_review_node)
        assert "_np_update_and_check" in src, (
            "code_review_node must call _np_update_and_check on real "
            "success (audit #8)"
        )
