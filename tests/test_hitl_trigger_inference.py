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
