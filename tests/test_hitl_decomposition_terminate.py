"""A4 — decomposition failures terminate the headless HITL immediately.

A decomposition failure produced zero stories; the repair HITL menu has no
action that fixes it, and ``route_after_hitl`` sends a decomposition
trigger's "resume" straight back to ``decomposition_node`` — which re-runs
the same planning LLM against the same spec and re-fails identically. So a
headless run used to auto-resume it up to the whole auto-resume cap (N full
planning dispatches, real money) before the cap-hit path terminated.

``hitl_menu_loop`` now terminates on the FIRST headless hit for these
triggers, mirroring the cap-hit clean-exit (sets ``hitl_abandon`` so
``route_after_hitl`` → END, and ``hitl_auto_resume_cap_hit`` so a later
``teane resume`` stays recoverable) with zero wasted re-runs.
"""

from __future__ import annotations

import pytest

from harness import cli


def _headless(monkeypatch) -> None:
    # Force the auto-approve (headless) path regardless of the test runner's
    # stdin, so the menu never blocks on input().
    monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")


def _state(trigger: str) -> dict:
    return {
        "node_state": {"hitl_trigger": trigger},
        "budget_remaining_usd": 1.0,
        "budget_initial_usd": 10.0,
        "loop_counter": {},
        "compiler_errors": [],
        "exit_code": 1,
        "modified_files": [],
        "workspace_path": "/tmp/teane-nonexistent-ws",
        "session_id": "test-a4",
    }


class TestDecompositionTriggerSet:
    def test_set_contents(self):
        assert cli._HEADLESS_TERMINAL_TRIGGERS == frozenset({
            "decomposition_validation_failed",
            "decomposition_missing",
        })

    def test_resumable_triggers_excluded(self):
        # Budget/repair/traceability triggers stay on their own paths.
        for other in (
            "budget_exhausted", "budget_preflight", "repair_loop_limit",
            "traceability_block", "zero_patch_loop:2",
        ):
            assert other not in cli._HEADLESS_TERMINAL_TRIGGERS


class TestHeadlessDecompositionTermination:
    @pytest.mark.parametrize(
        "trigger", ["decomposition_validation_failed", "decomposition_missing"],
    )
    def test_terminates_immediately_with_abandon_flags(self, trigger, monkeypatch):
        _headless(monkeypatch)
        out = cli.hitl_menu_loop(_state(trigger))
        ns = out["node_state"]
        # Clean exit: route_after_hitl → END via hitl_abandon.
        assert ns["hitl_abandon"] is True
        # Recoverable via `teane resume` (resume-rewind clears the pair).
        assert ns["hitl_auto_resume_cap_hit"] is True
        # Honest reason for observability.
        assert ns["hitl_terminated_reason"] == trigger
        assert ns["hitl_active"] is False
        assert ns["hitl_awaiting_input"] is False

    def test_no_auto_resume_accounting_touched(self, monkeypatch):
        # Immediate terminate must NOT record an auto-resume (it took none).
        _headless(monkeypatch)
        out = cli.hitl_menu_loop(_state("decomposition_validation_failed"))
        lc = out.get("loop_counter", {})
        assert int(lc.get("hitl_auto_resumes_taken", 0) or 0) == 0

    def test_banner_names_the_spec_fix(self, monkeypatch, capsys):
        _headless(monkeypatch)
        cli.hitl_menu_loop(_state("decomposition_missing"))
        err = capsys.readouterr().err
        assert "TERMINATED" in err
        assert "SPEC_REQUIREMENTS.md" in err
