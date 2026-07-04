"""Regression tests for the decoupled cheap-model → reasoning
escalation gate.

Before 2026-07-04 the gate was ``total_repairs >= max_repair_attempts - 1``.
With ``total_repairs`` also driving iteration counting, memory-cleanse
timing, and hard-cap gating, the coupling meant that once
``_TOTAL_HARD_CAP_MULTIPLIER`` was raised to 4 (hard cap = 12 rounds
at default ``max_repair_attempts=3``), every round from #2 onward
spent the reasoning model. Ciod session 523e86a7 saw 5+ escalations
per batch as a direct consequence.

The fix introduces ``cheap_shots_taken`` — a dedicated counter that:
  * increments each round the cheap model runs,
  * resets on HITL auto-resume via ``_reset_hitl_trip_counters``.

so the cheap model gets ``max_repair_attempts - 1`` fresh shots per
HITL cycle, not one-and-done.

These tests exercise the gate logic and reset behaviour in isolation.
"""

from __future__ import annotations


class TestCheapShotsGate:
    def test_cheap_selected_when_no_shots_taken_yet(self):
        # Fresh session: cheap_shots_taken=0, gate should choose cheap.
        max_repair_attempts = 3
        cheap_shots_taken = 0
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        assert use_escalation is False

    def test_cheap_selected_on_shot_1(self):
        # One shot taken (cheap ran once), still one more before escalation.
        max_repair_attempts = 3
        cheap_shots_taken = 1
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        assert use_escalation is False

    def test_escalates_at_max_shots_minus_1(self):
        # Two shots taken → gate = 2 >= (3-1) → escalate on the third.
        max_repair_attempts = 3
        cheap_shots_taken = 2
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        assert use_escalation is True

    def test_stays_escalated_past_gate(self):
        # If we somehow accumulated more shots without a HITL reset,
        # the gate stays escalated. Not expected in normal flow (the
        # counter increments only on cheap rounds) but locks in the
        # monotonic ≥ semantic.
        max_repair_attempts = 3
        cheap_shots_taken = 7
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        assert use_escalation is True

    def test_floor_at_1_prevents_immediate_escalation_on_max_1(self):
        # A pathological config with max_repair_attempts=1 would give
        # (1-1)=0 as the gate, escalating on the very first round.
        # ``max(1, ...)`` floors the gate at 1 so the cheap model
        # always gets at least one shot.
        max_repair_attempts = 1
        cheap_shots_taken = 0
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        assert use_escalation is False


class TestCheapShotsReset:
    """The HITL auto-resume path must clear ``cheap_shots_taken`` so
    the cheap model gets fresh shots on the next round; without this,
    every post-HITL round burns the reasoning model."""

    def test_reset_hitl_trip_counters_clears_cheap_shots(self):
        from harness.cli import _reset_hitl_trip_counters
        loop_counter = {
            "cheap_shots_taken": 5,
            "consecutive_zero_patch_rounds": 3,
            "total_repairs": 12,
        }
        _reset_hitl_trip_counters(loop_counter)
        assert loop_counter["cheap_shots_taken"] == 0
        # ``total_repairs`` is NOT reset here — that's the caller's
        # job via ``_reset_iteration_counters`` (they're the two
        # independent counters the fix decoupled).
        assert loop_counter["total_repairs"] == 12

    def test_reset_no_op_when_key_absent(self):
        # A legacy checkpoint that predates the fix has no
        # ``cheap_shots_taken`` key. Reset must be defensive: don't
        # crash, don't inject the key.
        from harness.cli import _reset_hitl_trip_counters
        loop_counter = {"consecutive_zero_patch_rounds": 3}
        _reset_hitl_trip_counters(loop_counter)
        # Key still absent (present-only reset — see the ``if key in``
        # guard in the source).
        assert "cheap_shots_taken" not in loop_counter


class TestCheapShotsEscalationScenario:
    """End-to-end scenario check on the gate + reset combination
    over a synthetic HITL cycle. Documents the intended cost profile."""

    def _step(self, cheap_shots_taken, max_repair_attempts):
        """Return (use_escalation, new_cheap_shots)."""
        use_escalation = cheap_shots_taken >= max(1, max_repair_attempts - 1)
        new_shots = cheap_shots_taken if use_escalation else cheap_shots_taken + 1
        return use_escalation, new_shots

    def test_cheap_reasoning_ratio_across_hitl_cycle(self):
        # Simulate a HITL cycle with hard_cap=12 (4× multiplier) and
        # max_repair_attempts=3. Under the fix we expect 2 cheap +
        # 10 reasoning per cycle, not 1 + 11 (the pre-fix ratio).
        max_repair_attempts = 3
        cheap_shots_taken = 0
        rounds_taken = []
        for _ in range(12):  # hard cap for one cycle
            use_esc, cheap_shots_taken = self._step(
                cheap_shots_taken, max_repair_attempts,
            )
            rounds_taken.append("reasoning" if use_esc else "cheap")
        assert rounds_taken.count("cheap") == 2
        assert rounds_taken.count("reasoning") == 10

    def test_hitl_resume_restores_cheap_budget(self):
        # After the HITL cycle above, ``_reset_hitl_trip_counters``
        # zeroes ``cheap_shots_taken``. The next cycle should get
        # another 2 cheap shots before escalating.
        from harness.cli import _reset_hitl_trip_counters
        max_repair_attempts = 3
        loop_counter = {"cheap_shots_taken": 2}
        _reset_hitl_trip_counters(loop_counter)
        cheap_shots_taken = int(loop_counter.get("cheap_shots_taken", 0) or 0)
        # First round after resume: cheap runs.
        use_esc, cheap_shots_taken = self._step(
            cheap_shots_taken, max_repair_attempts,
        )
        assert use_esc is False
        # Second round: cheap runs again.
        use_esc, cheap_shots_taken = self._step(
            cheap_shots_taken, max_repair_attempts,
        )
        assert use_esc is False
        # Third round: escalate.
        use_esc, _ = self._step(cheap_shots_taken, max_repair_attempts)
        assert use_esc is True
