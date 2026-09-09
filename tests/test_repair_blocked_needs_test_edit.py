"""Never route to repair when the only remedy is a test edit.

``repair_node`` may edit production code and nothing else — the tamper guard
refuses test files. Yet two of the harness's own signals routinely tell it to
change a test:

  * the isolation signal ("install/remove overrides inside a fixture", "each
    module needs its own engine") when a file passes alone but fails in suite;
  * the reflection judge's recommendation ("…or update
    tests/acceptance/test_story_001_acceptance.py:162 to access
    body['birthdays']").

Both are unexecutable, so repair spends its budget on production code instead.
lumina 969f8e1c burned eight rounds on ``_extract_record_id`` under the first
and its whole distraction budget under the second.

A third case has the same shape: two tests demanding opposite things, where
the diagnostics oscillate A → B → A and every patch swaps the pair.
"""

from __future__ import annotations

from harness.graph import (
    _detect_contradictory_tests,
    _record_suite_order_pollution,
    _verdict_demands_test_edit,
)


class TestVerdictDemandsTestEdit:
    def test_detects_the_lumina_recommendation(self):
        verdict = {
            "verdict": "DISTRACTION",
            "recommendation": (
                "Edit the API endpoint that returns the birthday list to wrap "
                "the array in an 'items' key, or update "
                "tests/acceptance/test_story_001_acceptance.py:162 to access "
                "body['birthdays'] instead."
            ),
        }
        assert _verdict_demands_test_edit(verdict).startswith(
            "tests/acceptance/test_story_001_acceptance.py"
        )

    def test_merely_citing_a_test_is_not_a_demand_to_edit_it(self):
        # The normal case: the judge names the test as where the failure
        # SURFACES. That must still go to repair.
        verdict = {
            "real_blocker": (
                "The assertion at server/tests/test_main.py:21 fails because "
                "the endpoint returns 500 from a database error."
            ),
            "recommendation": "Fix the database session handling in db.py.",
        }
        assert _verdict_demands_test_edit(verdict) == ""

    def test_production_only_recommendation_is_ignored(self):
        verdict = {"recommendation": "Edit server/app/db.py to bind lazily."}
        assert _verdict_demands_test_edit(verdict) == ""

    def test_handles_junk_input(self):
        for junk in (None, "", [], {"recommendation": None}):
            assert _verdict_demands_test_edit(junk) == ""


class TestSuiteOrderPollutionRecording:
    def test_records_and_dedupes(self):
        lc: dict = {}
        _record_suite_order_pollution(lc, "server/tests/test_a.py")
        _record_suite_order_pollution(lc, "server/tests/test_a.py")
        _record_suite_order_pollution(lc, "server/tests/test_b.py")
        assert lc["suite_order_pollution_files"] == [
            "server/tests/test_a.py", "server/tests/test_b.py",
        ]

    def test_empty_path_is_ignored(self):
        lc: dict = {}
        _record_suite_order_pollution(lc, "")
        assert "suite_order_pollution_files" not in lc


class TestContradictionDetection:
    def test_a_b_a_oscillation_is_flagged(self):
        # items-vs-birthdays: fixing one breaks the other, forever.
        lc: dict = {}
        A = ["AssertionError::items not in body"]
        B = ["AssertionError::birthdays not in body"]
        assert _detect_contradictory_tests(lc, A) is False   # A
        assert _detect_contradictory_tests(lc, B) is False   # A B
        assert _detect_contradictory_tests(lc, A) is True    # A B A

    def test_steady_progress_is_not_flagged(self):
        lc: dict = {}
        for fps in (["a", "b", "c"], ["a", "b"], ["a"]):
            assert _detect_contradictory_tests(lc, fps) is False

    def test_a_stall_is_not_flagged(self):
        # Same shape every round is a stall, owned by the zero-patch and
        # no-progress guards — not a contradiction.
        lc: dict = {}
        A = ["AssertionError::same"]
        for _ in range(4):
            assert _detect_contradictory_tests(lc, A) is False

    def test_empty_diagnostics_never_flag(self):
        lc: dict = {}
        assert _detect_contradictory_tests(lc, []) is False
        assert _detect_contradictory_tests(lc, ["x"]) is False
        assert _detect_contradictory_tests(lc, []) is False

    def test_history_is_bounded(self):
        from harness.graph import _DIAG_HISTORY_DEPTH
        lc: dict = {}
        for i in range(12):
            _detect_contradictory_tests(lc, [f"fp{i}"])
        assert len(lc["_diag_fp_history"]) <= _DIAG_HISTORY_DEPTH


class TestRouterDivertsAwayFromRepair:
    """The behaviour that matters: these failures must no longer reach
    ``repair_node``."""

    @staticmethod
    def _state(tmp_path, **over):
        base = {
            "exit_code": 1,
            "budget_remaining_usd": 5.0,
            "workspace_path": str(tmp_path),
            "loop_counter": {"total_repairs": 1},
            "compiler_errors": [
                {"file": "server/tests/test_main.py", "line": 21,
                 "severity": "error", "error_code": "AssertionError",
                 "message": "assert 500 == 200"},
            ],
            "node_state": {},
            # test_regeneration enabled so the ladder's top rung is reachable
            "test_regeneration_config": {"enabled": True, "tier_b_auto": True,
                                         "max_attempts_per_test": 1},
        }
        base.update(over)
        return base

    def test_suite_order_pollution_does_not_reach_repair(self, tmp_path):
        from harness.graph import route_after_compiler
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / "server" / "tests" / "test_main.py").write_text(
            "def test_a():\n    assert 1\n", encoding="utf-8")
        state = self._state(tmp_path, loop_counter={
            "total_repairs": 1,
            "suite_order_pollution_files": ["server/tests/test_main.py"],
        })
        assert route_after_compiler(state) != "repair_node"

    def test_contradiction_does_not_reach_repair(self, tmp_path):
        from harness.graph import route_after_compiler
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / "server" / "tests" / "test_main.py").write_text(
            "def test_a():\n    assert 1\n", encoding="utf-8")
        state = self._state(tmp_path, node_state={"contradictory_tests": True})
        assert route_after_compiler(state) != "repair_node"

    def test_judge_recommending_a_test_edit_does_not_reach_repair(self, tmp_path):
        from harness.graph import route_after_compiler
        (tmp_path / "tests" / "acceptance").mkdir(parents=True)
        (tmp_path / "tests" / "acceptance" / "test_story.py").write_text(
            "def test_a():\n    assert 1\n", encoding="utf-8")
        state = self._state(tmp_path, loop_counter={
            "total_repairs": 1,
            "last_reflection_verdict": {
                "verdict": "DISTRACTION",
                "recommendation": (
                    "Edit the API endpoint, or update "
                    "tests/acceptance/test_story.py to access body['x']."
                ),
            },
        })
        assert route_after_compiler(state) != "repair_node"

    def test_an_ordinary_failure_still_goes_to_repair(self, tmp_path):
        # Regression guard: the common path must be untouched.
        from harness.graph import route_after_compiler
        state = self._state(tmp_path)
        assert route_after_compiler(state) == "repair_node"
