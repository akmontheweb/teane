"""Tests for the loud termination banner + per-trigger frequency
tracking when the HITL auto-resume cap is exhausted in headless mode.

Before this fix (finsearch session 5f65a887, 2026-07-09), the cap-hit
path emitted a single WARNING line buried in verbose output. The
operator had to reconstruct "why did the process exit" from log
archaeology. This banner surfaces the exit reason with a visually
distinct stderr block, the trigger that finally exhausted the cap,
the frequency of each auto-resumed trigger this session, and copy-
pasteable recovery hints.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr
from typing import Any

import pytest

from harness import cli


def _minimal_state(*,
                   trigger: str = "zero_patch_loop",
                   resumes_taken: int = 3,
                   cap: int = 3,
                   per_trigger: dict[str, int] | None = None,
                   session_id: str = "test-sess-abcd1234",
                   total_repairs: int = 8,
                   budget_left: float = 4.20) -> dict[str, Any]:
    lc: dict[str, Any] = {
        "hitl_auto_resumes_taken": resumes_taken,
        "total_repairs": total_repairs,
    }
    if per_trigger is not None:
        lc["hitl_auto_resumes_per_trigger"] = dict(per_trigger)
    return {
        "session_id": session_id,
        "budget_remaining_usd": budget_left,
        "budget_initial_usd": 10.0,
        "hitl_auto_resume_cap": cap,
        "loop_counter": lc,
        "compiler_errors": [],
        "exit_code": 1,
        "modified_files": [],
        "workspace_path": "/tmp/x",
        "node_state": {
            "hitl_trigger": trigger,
            "hitl_active": True,
            "hitl_awaiting_input": True,
        },
    }


def _force_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``_gatekeeper_auto_approves`` returns True so the code
    takes the auto-resume branch."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
    # Also disable the repair HITL gate so _hitl_gate_enabled('repair')
    # returns False in the "not enabled OR auto-approve" branch — both
    # paths converge on the auto-resume block.
    monkeypatch.setattr(cli, "_HITL_FLAGS", {"repair": False})


class TestCapHitBannerAndState:
    def test_banner_printed_to_stderr(self, monkeypatch, capsys):
        _force_headless(monkeypatch)
        state = _minimal_state(
            trigger="zero_patch_loop",
            resumes_taken=3, cap=3,
            per_trigger={
                "zero_patch_loop": 2,
                "persistent_build_failure": 1,
            },
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = cli.hitl_menu_loop(state)
        stderr = buf.getvalue()
        # Banner is visually distinct.
        assert "=" * 78 in stderr
        assert "TERMINATED" in stderr
        assert "HITL auto-resume cap 3/3 exhausted" in stderr
        # Trigger context.
        assert "zero_patch_loop" in stderr
        assert "persistent_build_failure" in stderr
        # Session + numbers.
        assert "test-sess-abcd1234" in stderr
        assert "$4.20" in stderr or "4.2" in stderr
        # Recovery hints.
        assert "Recovery options" in stderr
        assert "--hitl-repair" in stderr
        assert "auto_resume_cap" in stderr
        # State reflects the abandon.
        ns = result.get("node_state", {})
        assert ns.get("hitl_auto_resume_cap_hit") is True
        assert ns.get("hitl_abandon") is True
        assert ns.get("hitl_active") is False

    def test_banner_survives_missing_per_trigger(self, monkeypatch):
        """Legacy states with no per-trigger dict still get a sane
        banner ("no per-trigger accounting recorded") — no KeyError."""
        _force_headless(monkeypatch)
        state = _minimal_state(
            trigger="persistent_build_failure",
            resumes_taken=3, cap=3,
            per_trigger=None,
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = cli.hitl_menu_loop(state)
        stderr = buf.getvalue()
        assert "TERMINATED" in stderr
        assert "no per-trigger accounting recorded" in stderr
        assert result["node_state"]["hitl_auto_resume_cap_hit"] is True


class TestPerTriggerFrequency:
    def test_auto_resume_increments_per_trigger_counter(self, monkeypatch):
        _force_headless(monkeypatch)
        state = _minimal_state(
            trigger="zero_patch_loop",
            resumes_taken=0, cap=3,
            per_trigger=None,
        )
        result = cli.hitl_menu_loop(state)
        lc = result["loop_counter"]
        assert lc.get("hitl_auto_resumes_taken") == 1
        per_trig = lc.get("hitl_auto_resumes_per_trigger", {})
        assert per_trig.get("zero_patch_loop") == 1

    def test_multiple_triggers_accumulate_independently(self, monkeypatch):
        _force_headless(monkeypatch)
        state = _minimal_state(
            trigger="persistent_build_failure",
            resumes_taken=1, cap=3,
            per_trigger={"zero_patch_loop": 1},
        )
        result = cli.hitl_menu_loop(state)
        per_trig = result["loop_counter"]["hitl_auto_resumes_per_trigger"]
        assert per_trig.get("zero_patch_loop") == 1
        assert per_trig.get("persistent_build_failure") == 1
