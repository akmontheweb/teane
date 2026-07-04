"""Regression tests for the deterministic CLI exit codes.

Prior to 2026-07-04 ``cmd_run`` / ``cmd_resume`` returned only
``0 if exit_code == 0 else 1``. That meant:

  * v12 shipped with 15/26 traceability coverage + 3 traceability
    block trips and still exited 0.
  * A ``teane build && teane deploy`` script would deploy runs that
    failed silently.
  * A CI consumer couldn't distinguish "spec drift, please fix
    inputs" from "gateway outage, please retry" from
    "everything clean, ship it."

The fix reserves five distinct exit codes and adds a resolver that
maps the graph's final state to the right one.
"""

from __future__ import annotations

from harness.cli import (
    EXIT_BUDGET_EXHAUSTED,
    EXIT_CLEAN,
    EXIT_CONFIG_ERROR,
    EXIT_INFRASTRUCTURE_FAILURE,
    EXIT_PARTIAL_SUCCESS,
    _resolve_cli_exit_code,
)


class TestExitCodeConstants:
    def test_codes_are_distinct(self):
        codes = {
            EXIT_CLEAN,
            EXIT_PARTIAL_SUCCESS,
            EXIT_CONFIG_ERROR,
            EXIT_BUDGET_EXHAUSTED,
            EXIT_INFRASTRUCTURE_FAILURE,
        }
        assert len(codes) == 5, "every reserved code must be distinct"

    def test_codes_within_shell_range(self):
        # POSIX shells reserve exit codes >125 for special meanings
        # (128+signal, 126=not executable, 127=not found). Our codes
        # stay in [0, 125] to avoid collision.
        for c in (
            EXIT_CLEAN, EXIT_PARTIAL_SUCCESS, EXIT_CONFIG_ERROR,
            EXIT_BUDGET_EXHAUSTED, EXIT_INFRASTRUCTURE_FAILURE,
        ):
            assert 0 <= c <= 125


class TestResolverPrecedence:
    def test_clean_run_returns_zero(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=0, final_state={"node_state": {}},
        ) == EXIT_CLEAN

    def test_budget_terminated_wins_over_infra(self):
        # A budget-exhausted run may also carry an env_misconfig
        # flag from the compile that noticed the missing binary;
        # budget termination is the more actionable label ("raise
        # the cap") so it takes precedence.
        state = {
            "node_state": {
                "budget_terminated": True,
                "env_misconfig": True,
            },
        }
        assert _resolve_cli_exit_code(
            graph_exit_code=1, final_state=state,
        ) == EXIT_BUDGET_EXHAUSTED

    def test_env_misconfig_is_infrastructure(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"env_misconfig": True}},
        ) == EXIT_INFRASTRUCTURE_FAILURE

    def test_llm_silent_is_infrastructure(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"llm_silent": True}},
        ) == EXIT_INFRASTRUCTURE_FAILURE

    def test_traceability_blocked_is_partial(self):
        # Ciod v12 exit-code regression — the build fired 3
        # traceability_block cycles, hit the cap, and returned 0.
        # After the fix, that same shape returns 1 so a deploy
        # pipeline can gate on it.
        assert _resolve_cli_exit_code(
            graph_exit_code=0,
            final_state={"node_state": {"traceability_blocked": True}},
        ) == EXIT_PARTIAL_SUCCESS

    def test_hitl_abandon_is_partial(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"hitl_abandon": True}},
        ) == EXIT_PARTIAL_SUCCESS

    def test_hitl_suspend_is_partial(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"hitl_suspend": True}},
        ) == EXIT_PARTIAL_SUCCESS

    def test_nonzero_graph_exit_without_flags_is_partial(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {}},
        ) == EXIT_PARTIAL_SUCCESS

    def test_empty_final_state_falls_back_to_partial(self):
        # A resume flow with no ``final_state`` shouldn't crash the
        # resolver — treat it as partial success (the graph made
        # it to termination but we can't verify its final state).
        assert _resolve_cli_exit_code(
            graph_exit_code=1, final_state={},
        ) == EXIT_PARTIAL_SUCCESS

    def test_none_node_state_treated_as_empty(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=0, final_state={"node_state": None},
        ) == EXIT_CLEAN


class TestBudgetTerminationFlag:
    """The HITL menu's headless auto-resume sets ``budget_terminated``
    on ``node_state`` when it aborts a budget-exhausted trigger
    instead of re-entering the loop. This test locks in that flag
    → exit code mapping."""

    def test_flag_maps_to_budget_exit(self):
        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"budget_terminated": True}},
        ) == EXIT_BUDGET_EXHAUSTED

    def test_flag_wins_over_graph_zero(self):
        # Corner case: graph reported 0 but the HITL abort fired on
        # a post-completion budget check. The flag still means
        # "budget-exhausted"; caller shouldn't treat this as clean.
        assert _resolve_cli_exit_code(
            graph_exit_code=0,
            final_state={"node_state": {"budget_terminated": True}},
        ) == EXIT_BUDGET_EXHAUSTED
