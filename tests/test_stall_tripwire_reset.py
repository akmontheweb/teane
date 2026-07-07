"""Regression tests for the 2026-07-07 stall-tripwire reset fix.

Session cec4d124 HITL#4 uncovered a false-positive escalation:
``consecutive_zero_patch_rounds`` carried its ``2`` across HITL#3 resume
→ green compile → code_review re-patch (3/3 files) → compile-fail →
``route_after_compiler`` saw the stale ``2`` and escalated to HITL
despite the intervening forward progress.

The fix wires two reset sites to a shared helper
:func:`harness.graph._reset_stall_tripwires_on_progress`:

  * ``compiler_node`` (``exit_code == 0`` branch)
  * ``code_review_node`` (``success_count > 0`` branch)

Both sites now zero every counter in ``_STALL_TRIPWIRE_KEYS`` so a
subsequent compile-fail cannot re-trip HITL on a counter that predates
the visible progress.
"""

from __future__ import annotations

import pytest

from harness.graph import (
    _STALL_TRIPWIRE_KEYS,
    _reset_stall_tripwires_on_progress,
    compiler_node,
    route_after_compiler,
)


# ---------------------------------------------------------------------------
# Direct helper tests — the reset key list is the invariant. If a future
# tripwire is added and someone forgets to include it here, the fix will
# only cover the old set and the same class of false positive will
# resurface. These tests make that regression loud.
# ---------------------------------------------------------------------------


class TestResetHelper:
    def test_wipes_every_tripwire_in_place(self):
        loop_counter: dict = {
            "consecutive_zero_patch_rounds": 4,
            "consecutive_all_allowlist_rejected_rounds": 2,
            "consecutive_distraction_rounds": 5,
            "consecutive_low_signal_rounds": 3,
            # Non-tripwire counters must NOT be touched — the fix is
            # scoped to "the loop is stuck" signals, not general
            # bookkeeping like total_repairs / cheap_shots_taken.
            "total_repairs": 7,
            "cheap_shots_taken": 2,
        }
        _reset_stall_tripwires_on_progress(loop_counter)

        for key in _STALL_TRIPWIRE_KEYS:
            assert loop_counter[key] == 0, f"tripwire {key} not reset"
        assert loop_counter["total_repairs"] == 7
        assert loop_counter["cheap_shots_taken"] == 2

    def test_key_set_covers_the_four_router_tripwires(self):
        # Explicit list guards future refactors: if the counter set
        # changes without a corresponding fix to the reset helper, this
        # test flags the drift.
        assert set(_STALL_TRIPWIRE_KEYS) == {
            "consecutive_zero_patch_rounds",
            "consecutive_all_allowlist_rejected_rounds",
            "consecutive_distraction_rounds",
            "consecutive_low_signal_rounds",
        }

    def test_absent_keys_are_added_as_zero(self):
        # A fresh state may not have every counter yet; the helper must
        # treat "missing" and "already zero" identically (set → 0).
        loop_counter: dict = {}
        _reset_stall_tripwires_on_progress(loop_counter)
        for key in _STALL_TRIPWIRE_KEYS:
            assert loop_counter[key] == 0


# ---------------------------------------------------------------------------
# compiler_node integration — reset on green build, preserve on fail.
# Reuses the stub-sandbox pattern from test_env_misconfig.py.
# ---------------------------------------------------------------------------


class _StubBuildResult:
    def __init__(self, exit_code: int, raw_output: str = "") -> None:
        self.exit_code = exit_code
        self.raw_output = raw_output
        self.diagnostics = []
        self.timed_out = False
        self.log_truncated = False
        self.elapsed_seconds = 0.1
        self.backend_name = "stub"


class _StubSandboxExecutor:
    canned: _StubBuildResult = _StubBuildResult(0, "")

    def __init__(self, **kwargs):
        pass

    async def run(self, build_command: str):
        return _StubSandboxExecutor.canned


@pytest.fixture
def stub_sandbox(monkeypatch, tmp_path):
    import harness.sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "SandboxExecutor", _StubSandboxExecutor)

    def _set(exit_code: int, raw_output: str = "") -> None:
        _StubSandboxExecutor.canned = _StubBuildResult(exit_code, raw_output)

    return _set


class TestCompilerNodeResetsOnGreenBuild:
    """The bug that motivated this whole test file: session cec4d124
    HITL#4. Elevated ``consecutive_zero_patch_rounds`` from a prior
    stall survived a green compile because the compiler_node success
    branch never touched it. Without the reset the very next compile
    fail (via code_review → recompile) re-tripped HITL immediately."""

    @pytest.mark.asyncio
    async def test_green_build_resets_every_tripwire(
        self, stub_sandbox, tmp_path,
    ):
        stub_sandbox(0, "")  # exit=0 → memory cleanse + reset branch

        state = {
            "workspace_path": str(tmp_path),
            "build_command": "true",
            "allow_network": False,
            "sandbox_config": {},
            "loop_counter": {
                "consecutive_zero_patch_rounds": 2,
                "consecutive_all_allowlist_rejected_rounds": 1,
                "consecutive_distraction_rounds": 3,
                "consecutive_low_signal_rounds": 4,
                "total_repairs": 2,
            },
            "messages": [],
        }
        result = await compiler_node(state)

        assert result["exit_code"] == 0
        for key in _STALL_TRIPWIRE_KEYS:
            assert result["loop_counter"][key] == 0, (
                f"tripwire {key} not reset after green build"
            )
        # Non-tripwire counters are preserved.
        assert result["loop_counter"]["total_repairs"] == 2

    @pytest.mark.asyncio
    async def test_failed_build_does_not_reset_tripwires(
        self, stub_sandbox, tmp_path,
    ):
        # Regression guard: the reset must be gated on exit=0. If it
        # ever fires on a failed build the router loses the very signal
        # it needs to escalate legitimately-stuck loops.
        stub_sandbox(1, "some compile error without env_misconfig pattern\n")

        elevated = {
            "consecutive_zero_patch_rounds": 2,
            "consecutive_all_allowlist_rejected_rounds": 1,
            "consecutive_distraction_rounds": 3,
            "consecutive_low_signal_rounds": 4,
        }
        state = {
            "workspace_path": str(tmp_path),
            "build_command": "false",
            "allow_network": False,
            "sandbox_config": {},
            "loop_counter": dict(elevated),
            "messages": [],
        }
        result = await compiler_node(state)

        assert result["exit_code"] == 1
        # None of the tripwires should have been zeroed on a failing
        # build — they're the router's evidence for the next decision.
        for key, expected in elevated.items():
            assert result["loop_counter"][key] == expected, (
                f"tripwire {key} was reset on failed build "
                f"(saw {result['loop_counter'][key]}, expected {expected})"
            )


# ---------------------------------------------------------------------------
# route_after_compiler behavioural contract — green build must not
# check tripwires, and after the reset the next fail must NOT stale-trip
# HITL. These lock in the observed session-cec4d124 fix at the router
# layer even if someone later refactors the reset into a different site.
# ---------------------------------------------------------------------------


class TestRouterHonoursResetFromPriorGreenBuild:

    def test_router_does_not_check_tripwires_on_green_build(self):
        # Sanity: on exit=0 the router routes to security_scan
        # unconditionally, without consulting stall counters.
        state = {
            "exit_code": 0,
            "compiler_errors": [],
            "loop_counter": {"consecutive_zero_patch_rounds": 99},
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        assert route_after_compiler(state) == "security_scan_node"

    def test_router_hitls_only_on_actual_stall_after_reset(self):
        # The false-positive scenario as seen post-fix:
        #   HITL#3 fires (counter=2 preserved across resume) →
        #   green compile (counter reset to 0 by compiler_node fix) →
        #   code_review re-patch → compile-fail (counter=1 from new
        #   zero-patch round) → router must NOT immediately re-fire
        #   HITL because 1 < 2.
        state = {
            "exit_code": 1,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                # Only 1 real consecutive zero-patch round since reset.
                "consecutive_zero_patch_rounds": 1,
                "total_repairs": 2,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        assert route_after_compiler(state) == "repair_node"

    def test_router_still_hitls_on_genuine_stall(self):
        # Regression guard on the OTHER side: after fixing the false
        # positive we must not accidentally have widened the tripwire.
        # Two consecutive zero-patch rounds with non-autofixable
        # diagnostics is a real stall and must still escalate.
        state = {
            "exit_code": 1,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "consecutive_zero_patch_rounds": 2,
                "total_repairs": 2,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        assert route_after_compiler(state) == "human_intervention_node"
