"""Single source of truth for ``loop_counter`` key groups that MUST
stay in sync across multiple reset sites.

Why this module exists
----------------------
``loop_counter`` is a flat dict threaded through the whole graph. Certain
keys are per-batch caps that gate a HITL escalation once they reach a
threshold — e.g. ``test_generation`` trips ``test_generation_max_iterations``
at 5. Three separate call sites are responsible for zeroing them:

  1. ``story_loop._batch_commit_node``       — batch-boundary reset
  2. ``cli._reset_iteration_counters``       — Save & Quit / [r] resume
  3. ``cli._reset_hitl_trip_counters``       — headless auto-resume

Historically each site kept its own hand-maintained tuple. That is drift
bait: a new cap counter added to one place (e.g. ``test_generation_zero_emit``)
silently persists across every recovery event that forgets to list it,
and the *next* batch trips the cap on its first entry with no real
iteration attempted. Finsearch session 156032347 batch 110 died from
exactly this pattern.

The invariant
-------------
Any counter listed in ``PER_BATCH_CAP_COUNTERS`` MUST be reset by all
three sites above. ``tests/test_reset_registry.py`` seeds every key to a
non-zero value and asserts each reset path zeros it, so a new counter
added to the tuple triggers a failing test until the reset sites are
updated.

Adding a new capped counter
---------------------------
1. Increment/read the counter in whichever node owns the cap.
2. Add its name to ``PER_BATCH_CAP_COUNTERS`` below.
3. The three reset sites pick it up automatically via ``for k in
   PER_BATCH_CAP_COUNTERS: base[k] = 0``.
4. The regression test in ``tests/test_reset_registry.py`` verifies the
   wiring didn't regress.

If a new counter should reset only on batch boundary (not on resume),
it belongs in a different tuple — introduce one alongside this one
rather than special-casing this list.
"""

from __future__ import annotations


PER_BATCH_CAP_COUNTERS: tuple[str, ...] = (
    # Repair-loop iteration counters. Per-batch budgets — a new batch
    # gets a fresh N shots per role.
    "patching",
    "repair",
    "compiler",
    "review_spec",
    "review_code",
    # Test-generation caps. ``test_generation`` is the real-attempt
    # budget; ``test_generation_zero_emit`` is the separate zero-emit
    # sub-cap that fires when the LLM emits no patch blocks. Both trip
    # HITL via ``llm_behavior_symbol`` triggers, so both MUST clear on
    # every recovery event or the next batch tripsthe cap without ever
    # entering the node's real work (finsearch 156032347 batch 110).
    "test_generation",
    "test_generation_zero_emit",
)


STALL_TRIPWIRE_KEYS: tuple[str, ...] = (
    # "N bad rounds in a row" tripwires that ``route_after_compiler`` /
    # ``route_after_patching`` consult to short-circuit to HITL. Shared
    # by three reset sites that must never drift apart:
    #
    #   1. ``graph._reset_stall_tripwires_on_progress`` — zeroed on any
    #      real forward progress (green build, code_review re-patch).
    #   2. ``cli._reset_hitl_trip_counters``  — zeroed on headless
    #      auto-resume (nothing outside the harness changed; a counter
    #      left at its cap re-fires the same trigger within seconds).
    #   3. ``cli._reset_iteration_counters``  — stepped to one below
    #      their current value on human [r]/[e] resume. Zeroing would
    #      discard directive-shaping signal the repair prompts rely on,
    #      but preserving them verbatim is a guaranteed dead end when
    #      the HITL trigger IS one of these counters: the router gates
    #      on them BEFORE repair_node can run, and the only resets
    #      (PROGRESS verdict, green build) are unreachable. Session
    #      22471c0c re-fired reflection_distraction_loop:3 twenty
    #      seconds after every [r] with zero repair turns in between.
    "consecutive_zero_patch_rounds",
    "consecutive_all_allowlist_rejected_rounds",
    "consecutive_distraction_rounds",
    "consecutive_low_signal_rounds",
    "no_progress_repairs",
    "cheap_shots_taken",
)
