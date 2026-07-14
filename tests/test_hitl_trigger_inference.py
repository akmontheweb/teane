"""Unit tests for harness.graph._infer_hitl_trigger.

This helper is consulted by ``human_intervention_node`` before it
hands off to the cli's ``hitl_menu_loop`` — it turns the state
snapshot at HITL-time into a precise, operator-friendly trigger
label. Routers escalate to HITL via conditional edges (no state
mutation), so the inference layer is the only place these signals
get a readable name.

The 2026-06-25 incident that motivated these tests: a security run
hit HITL with ``Trigger: unknown`` because the inference table
didn't cover the security_fix_limit path. Operators had to dig
through the log to figure out why the run paused.
"""

from __future__ import annotations

from harness.graph import _infer_hitl_trigger


def _state(**overrides) -> dict:
    """Construct a minimal AgentState-shaped dict for inference."""
    base: dict = {
        "loop_counter": {},
        "node_state": {},
        "budget_remaining_usd": 5.00,
        "exit_code": 0,
        "security_scan_config": {},
    }
    base.update(overrides)
    return base


def test_unknown_when_no_signal_present():
    assert _infer_hitl_trigger(_state(), max_repair=3) == "unknown"


def test_env_misconfig_with_symbol():
    out = _infer_hitl_trigger(
        _state(node_state={"env_misconfig": True, "env_misconfig_symbol": "pytest"}),
        max_repair=3,
    )
    assert out == "env_misconfig:pytest"


def test_env_misconfig_without_symbol():
    out = _infer_hitl_trigger(
        _state(node_state={"env_misconfig": True}), max_repair=3,
    )
    assert out == "env_misconfig"


def test_budget_exhausted_takes_priority_over_loop_signals():
    """Budget at $0 short-circuits even when other failure flags fire,
    because no other branch can productively rerun without money."""
    out = _infer_hitl_trigger(
        _state(
            budget_remaining_usd=0.0,
            loop_counter={
                "security": 99,
                "consecutive_zero_patch_rounds": 99,
                "total_repairs": 99,
            },
            exit_code=1,
        ),
        max_repair=3,
    )
    assert out == "budget_exhausted"


def test_no_progress_failsafe_trips_when_tracker_marked():
    """Layer-3 no-progress failsafe must produce the canonical label
    so an operator knows the run wasn't just slow — it was bleeding
    budget without producing patches."""
    out = _infer_hitl_trigger(
        _state(loop_counter={
            "progress_tracker": {
                "budget_at_last_progress": 10.0,
                "tripped": True,
            },
        }),
        max_repair=3,
    )
    assert out == "no_progress_failsafe"


def test_security_fix_limit_with_attempts_label():
    """The label includes the attempt count so the operator sees how
    many rounds were burned before HITL."""
    out = _infer_hitl_trigger(
        _state(
            loop_counter={"security": 2},
            security_scan_config={"max_security_fix_attempts": 2},
        ),
        max_repair=3,
    )
    assert out == "security_fix_limit:2/2"


def test_security_fix_limit_respects_config_override():
    out = _infer_hitl_trigger(
        _state(
            loop_counter={"security": 5},
            security_scan_config={"max_security_fix_attempts": 5},
        ),
        max_repair=3,
    )
    assert out == "security_fix_limit:5/5"


def test_zero_patch_loop_label_includes_count():
    out = _infer_hitl_trigger(
        _state(loop_counter={"consecutive_zero_patch_rounds": 3}),
        max_repair=10,
    )
    assert out == "zero_patch_loop:3"


def test_low_signal_verdict_loop_trips_at_default_cap():
    # Bug B (2026-07-04): ciod session 54f4eaf2 accumulated 21 consecutive
    # PROGRESS+"insufficient data" verdicts without any counter firing —
    # the reset branch in repair_node cleared the streak on every
    # PROGRESS. The route gate must fire at the default cap of 5 so the
    # loop no longer grinds silently.
    out = _infer_hitl_trigger(
        _state(loop_counter={"consecutive_low_signal_rounds": 5}),
        max_repair=10,
    )
    assert out == "low_signal_verdict_loop:5"


def test_low_signal_verdict_loop_holds_under_cap():
    # Below the cap the label must NOT switch to low-signal — the loop
    # is still allowed to try one more round of the low-signal
    # escalation prompt (build-output tail injected at streak >=2).
    out = _infer_hitl_trigger(
        _state(loop_counter={
            "consecutive_low_signal_rounds": 4,
            "total_repairs": 2,
        }),
        max_repair=10,
    )
    assert out != "low_signal_verdict_loop:4"


def test_distraction_loop_takes_priority_over_low_signal():
    # Both counters can co-exist — a DISTRACTION streak is a stronger
    # signal (judge IS grounding but the LLM is ignoring it). Make sure
    # ordering surfaces the more actionable label.
    out = _infer_hitl_trigger(
        _state(loop_counter={
            "consecutive_distraction_rounds": 3,
            "consecutive_low_signal_rounds": 8,
        }),
        max_repair=10,
    )
    assert out == "reflection_distraction_loop:3"


def test_repair_loop_limit_at_cap():
    out = _infer_hitl_trigger(
        _state(loop_counter={"total_repairs": 3}, exit_code=1),
        max_repair=3,
    )
    assert out == "repair_loop_limit"


def test_persistent_build_failure_when_nothing_more_specific():
    """exit_code != 0 with no other signal — last branch."""
    out = _infer_hitl_trigger(_state(exit_code=1), max_repair=3)
    assert out == "persistent_build_failure"


def test_specific_security_label_wins_over_persistent_build_failure():
    """Regression test for the 2026-06-25 incident: with exit_code=0
    (semgrep itself succeeded) and security attempts at the cap, the
    label MUST be ``security_fix_limit``, not ``unknown`` or
    ``persistent_build_failure``."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=0,
            loop_counter={"security": 2},
            security_scan_config={"max_security_fix_attempts": 2},
        ),
        max_repair=3,
    )
    assert out == "security_fix_limit:2/2"


def test_traceability_block_wins_over_persistent_build_failure():
    """Phase 7 BUG #6 regression: when installation_doc_node sets
    traceability_blocked + exit_code=1, the trigger MUST be
    ``traceability_block``, not ``persistent_build_failure`` — the
    HITL UX needs to route to coverage-gap advice, not
    open-failing-files advice."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            node_state={"traceability_blocked": True},
        ),
        max_repair=3,
    )
    assert out == "traceability_block"


def test_traceability_block_when_combined_with_other_signals():
    """If traceability AND other signals coexist, traceability wins —
    it's the most specific reason the run is at HITL."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            node_state={"traceability_blocked": True},
            loop_counter={"total_repairs": 99},
        ),
        max_repair=3,
    )
    assert out == "traceability_block"


# ---------------------------------------------------------------------------
# Post-finsearch-156032347 additions. The router escalates for several
# distinct reasons (stuck REPLACE_BLOCK per-file, no_progress_repairs cap,
# hard total-iteration ceiling, same-MISSING_DEP recurrence,
# build_command_blocked) but the pre-change ``_infer_hitl_trigger`` fell
# through to the generic ``repair_loop_limit`` / ``persistent_build_failure``
# fallbacks for each of them. Result: 6 of 10 finsearch HITL fires were
# mislabeled, and post-mortem learning attached rules under the wrong
# hypothesis. These tests pin the specific labels.
# ---------------------------------------------------------------------------


def test_build_command_blocked_wins_over_persistent_build_failure():
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            node_state={
                "build_command_blocked": True,
                "build_command_blocked_rule": "cd_not_allowed",
            },
        ),
        max_repair=3,
    )
    assert out == "build_command_blocked:cd_not_allowed"


def test_build_command_blocked_without_rule_still_labeled():
    out = _infer_hitl_trigger(
        _state(exit_code=1, node_state={"build_command_blocked": True}),
        max_repair=3,
    )
    assert out == "build_command_blocked"


def test_replace_block_stuck_single_file():
    """A file at or above the router's stuck_target_limit surfaces its
    path in the label so operator/dashboards see WHICH file is stuck."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={
                "replace_block_misses_per_file": {
                    "server/app/tests/test_rate_limit.py": 3,
                },
                # Ensure the label doesn't fall through to repair_loop_limit
                # (which also matches here at total_repairs >= max_repair).
                "total_repairs": 11,
            },
        ),
        max_repair=3,
    )
    assert out == "replace_block_stuck:server/app/tests/test_rate_limit.py"


def test_replace_block_stuck_multiple_files_labels_first_and_suffix():
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={
                "replace_block_misses_per_file": {
                    "b.py": 4,
                    "a.py": 3,
                    "c.py": 5,
                },
                "total_repairs": 11,
            },
        ),
        max_repair=3,
    )
    # Sorted alphabetically → "a.py" is head, +2 more.
    assert out == "replace_block_stuck:a.py+2"


def test_replace_block_stuck_below_limit_does_not_fire():
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={
                "replace_block_misses_per_file": {"a.py": 2},
                "total_repairs": 11,
            },
        ),
        max_repair=3,
    )
    # Below the default limit (3) — falls through to repair_loop_limit.
    assert out == "repair_loop_limit"


def test_no_progress_repairs_label_beats_repair_loop_limit():
    """finsearch STORY-NFR-* fire (14:04:45): total_repairs=11 but the
    real trip was no_progress_repairs at 3/3."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={
                "no_progress_repairs": 3,
                "total_repairs": 11,
            },
        ),
        max_repair=3,
    )
    assert out == "no_progress_repairs:3/3"


def test_hard_iteration_ceiling_label_beats_repair_loop_limit():
    """finsearch 05:53:09 / 06:07:59: total_repairs hit the 12/12 hard
    cap while per-round progress signals kept the earlier caps from
    tripping. The label must reflect the hard-cap identity, not the
    generic repair_loop_limit."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={"total_repairs": 12},
        ),
        max_repair=3,
    )
    # max_repair=3 × default multiplier=4 → hard cap = 12.
    assert out == "hard_iteration_ceiling:12/12"


def test_hard_iteration_ceiling_below_cap_falls_back():
    out = _infer_hitl_trigger(
        _state(exit_code=1, loop_counter={"total_repairs": 3}),
        max_repair=3,
    )
    assert out == "repair_loop_limit"


def test_same_missing_dep_label_with_symbol():
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                {"error_code": "MISSING_DEP", "message": "pip"},
            ],
            loop_counter={
                "missing_dep_consecutive_same": 3,
                "missing_dep_last_symbol": "pip",
                # Even at the max_repair cap, the more specific label wins.
                "total_repairs": 3,
            },
        ),
        max_repair=3,
    )
    assert out == "same_missing_dep:pip"


def test_same_missing_dep_ignored_when_current_diags_have_no_missing_dep():
    """Stale-counter defense: ``missing_dep_consecutive_same`` persists
    across rounds — if the dep cascade resolved but the counter didn't
    reset AND we're now at HITL for a DIFFERENT reason (e.g., no_progress
    on downstream failures), the label must NOT be ``same_missing_dep``.
    The router's real gate requires ``has_autofixable``; this mirrors
    that by requiring current MISSING_DEP diagnostics."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                {"error_code": "TS2769", "message": "downstream failure"},
            ],
            loop_counter={
                "missing_dep_consecutive_same": 3,
                "missing_dep_last_symbol": "pip",  # stale
                "no_progress_repairs": 3,
                "total_repairs": 5,
            },
        ),
        max_repair=3,
    )
    # no_progress_repairs is the correct label — the missing_dep counter
    # is stale and must not shadow it.
    assert out == "no_progress_repairs:3/3"


def test_no_progress_beats_same_missing_dep_when_both_apply():
    """Ordering (matches router gate order at L17836 vs L17890): a
    session that has BOTH signals should surface no_progress_repairs
    since that's the gate that trips FIRST in route_after_compiler.
    Requires a NON-autofixable diagnostic to be present — if every
    error were autofixable, the router would route to the autofix
    bypass instead of HITL, so no_progress could not have fired."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                # Mixed: has MISSING_DEP so same_missing_dep is legal,
                # AND has a non-autofixable so no_progress is legal.
                {"error_code": "MISSING_DEP", "message": "pip"},
                {"error_code": "TS2769", "message": "downstream"},
            ],
            loop_counter={
                "no_progress_repairs": 3,
                "missing_dep_consecutive_same": 3,
                "missing_dep_last_symbol": "pip",
                "total_repairs": 5,
            },
        ),
        max_repair=3,
    )
    assert out == "no_progress_repairs:3/3"


def test_no_progress_repairs_ignored_when_all_diagnostics_are_autofixable():
    """Router at L17836 requires ``not has_autofixable`` before firing
    the no_progress gate — the autofix bypass will land the fix
    without a repair round. Inference must mirror. If we're at HITL via
    a different path (e.g. security branch) with a stale no_progress
    counter AND current diagnostics are all autofixable, the label
    must NOT be no_progress_repairs."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                {"error_code": "MISSING_DEP", "message": "pip"},
            ],
            loop_counter={
                "no_progress_repairs": 3,
                "total_repairs": 3,
            },
        ),
        max_repair=3,
    )
    # Falls through to repair_loop_limit since no_progress is masked.
    assert out == "repair_loop_limit"


def test_hard_iteration_ceiling_ignored_when_all_diagnostics_are_autofixable():
    """Same guard as no_progress — the router's L17864 hard-cap check
    also gates on ``not has_autofixable``. Inference must mirror."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                {"error_code": "MISSING_DEP", "message": "pip"},
                {"error_code": "DEP_RESOLUTION_CONFLICT", "message": "x"},
            ],
            loop_counter={"total_repairs": 12},
        ),
        max_repair=3,
    )
    # 12 hits the hard cap but every diagnostic is autofixable → skip
    # the hard-cap label. Falls through to repair_loop_limit.
    assert out == "repair_loop_limit"


def test_hard_iteration_ceiling_fires_when_mixed_diagnostics():
    """Router requires ALL diagnostics autofixable (not any). One
    non-autofixable in the set unblocks hard_cap escalation. Mirror."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            compiler_errors=[
                {"error_code": "MISSING_DEP", "message": "pip"},
                {"error_code": "TS2769", "message": "test failure"},
            ],
            loop_counter={"total_repairs": 12},
        ),
        max_repair=3,
    )
    assert out == "hard_iteration_ceiling:12/12"


def test_replace_block_stuck_beats_all_allowlist_rejected():
    """Ordering: replace_block_stuck is more specific than the generic
    all-allowlist-rejected fallback; both can co-exist in a wedged
    session but the stuck-file label is what the operator needs."""
    out = _infer_hitl_trigger(
        _state(
            exit_code=1,
            loop_counter={
                "replace_block_misses_per_file": {"x.py": 3},
                "consecutive_all_allowlist_rejected_rounds": 5,
            },
        ),
        max_repair=3,
    )
    assert out == "replace_block_stuck:x.py"
