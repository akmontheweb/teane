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

All three are derived state that outlives the round it describes, so each
diversion must name a test the CURRENT build is failing
(lumina-run9-20260921-1758: a judge verdict already carried out drove two
zero-block regenerations of a test that had left the failing set).
"""

from __future__ import annotations

import pytest

from harness.graph import (
    _contradiction_flipping_fingerprints,
    _detect_contradictory_tests,
    _diagnostic_fingerprint,
    _record_suite_order_pollution,
    _update_contradiction_latch,
    _verdict_demands_test_edit,
    compiler_node,
    route_after_compiler,
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
        fp = _diagnostic_fingerprint(state["compiler_errors"][0])
        state["loop_counter"]["contradiction_pair"] = [[fp], ["other::shape"]]
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
        }, compiler_errors=[
            {"file": "tests/acceptance/test_story.py", "line": 3,
             "severity": "error", "error_code": "AssertionError",
             "message": "KeyError: 'x'"},
        ])
        assert route_after_compiler(state) != "repair_node"

    def test_an_ordinary_failure_still_goes_to_repair(self, tmp_path):
        # Regression guard: the common path must be untouched.
        from harness.graph import route_after_compiler
        state = self._state(tmp_path)
        assert route_after_compiler(state) == "repair_node"


class TestContradictionLatch:
    """The latch is a claim about one A/B pair and lives exactly as long as
    the loop stays inside that pair."""

    A = ["AssertionError::items not in body"]
    B = ["AssertionError::birthdays not in body"]

    def _latched(self):
        lc: dict = {}
        ns: dict = {}
        assert _update_contradiction_latch(lc, ns, self.A) is False
        assert _update_contradiction_latch(lc, ns, self.B) is False
        assert _update_contradiction_latch(lc, ns, self.A) is True
        return lc, ns

    def test_detection_records_the_pair(self):
        lc, ns = self._latched()
        assert ns["contradictory_tests"] is True
        assert sorted(lc["contradiction_pair"]) == sorted([self.A, self.B])

    def test_holds_while_the_loop_stays_inside_the_pair(self):
        # Regeneration that fails lands back on A: still the same conflict,
        # and re-detecting it would cost two more repair rounds.
        lc, ns = self._latched()
        for shape in (self.A, self.B, self.A):
            assert _update_contradiction_latch(lc, ns, shape) is False
            assert ns["contradictory_tests"] is True

    def test_a_new_shape_retires_it(self):
        lc, ns = self._latched()
        _update_contradiction_latch(lc, ns, ["TypeError::something else"])
        assert "contradictory_tests" not in ns
        assert "contradiction_pair" not in lc

    def test_a_green_build_retires_it(self):
        lc, ns = self._latched()
        _update_contradiction_latch(lc, ns, [])
        assert "contradictory_tests" not in ns
        assert "contradiction_pair" not in lc

    def test_a_flag_with_no_pair_is_retired(self):
        # A checkpoint from before the pair was recorded: nothing grounds it.
        ns = {"contradictory_tests": True}
        _update_contradiction_latch({}, ns, ["x::y"])
        assert "contradictory_tests" not in ns

    def test_flipping_fingerprints_exclude_bystanders(self):
        lc = {"contradiction_pair": [["a", "shared"], ["b", "shared"]]}
        assert _contradiction_flipping_fingerprints(lc) == {"a", "b"}
        assert _contradiction_flipping_fingerprints({}) == set()


class TestDivertsOnlyOnCurrentEvidence:
    """Each diversion must name a test the current build is failing."""

    _state = staticmethod(TestRouterDivertsAwayFromRepair._state)

    @staticmethod
    def _diag(path, message="assert 500 == 200"):
        return {"file": path, "line": 1, "severity": "error",
                "error_code": "AssertionError", "message": message}

    def test_run9_stale_verdict_does_not_divert(self, tmp_path):
        # lumina-run9-20260921-1758: the judge said to edit
        # test_birthday_repository.py; regeneration fixed it; the only
        # failure left was test_main.py. The router re-read the verdict and
        # regenerated the fixed file again, twice, with zero blocks.
        state = self._state(tmp_path, loop_counter={
            "total_repairs": 3,
            "last_reflection_verdict": {
                "verdict": "DISTRACTION",
                "recommendation": (
                    "Edit server/tests/test_birthday_repository.py line 118 "
                    "to remove the 'updated_at' keyword argument, and edit "
                    "server/app/main.py to fix the pagination logic."
                ),
            },
        }, compiler_errors=[self._diag(
            "server/tests/test_main.py", "assert 0 == 3")])
        assert route_after_compiler(state) == "repair_node"

    def test_contradiction_targets_a_flipping_test(self, tmp_path):
        (tmp_path / "tests" / "acceptance").mkdir(parents=True)
        (tmp_path / "tests" / "acceptance" / "test_story.py").write_text(
            "def test_a():\n    assert 1\n", encoding="utf-8")
        bystander = self._diag("server/tests/test_other.py", "bystander")
        flipping = self._diag("tests/acceptance/test_story.py",
                              "items not in body")
        state = self._state(
            tmp_path,
            compiler_errors=[bystander, flipping],
            node_state={"contradictory_tests": True},
            loop_counter={"total_repairs": 1, "contradiction_pair": [
                [_diagnostic_fingerprint(bystander),
                 _diagnostic_fingerprint(flipping)],
                [_diagnostic_fingerprint(bystander), "AssertionError::birthdays"],
            ]},
        )
        route_after_compiler(state)
        assert state["node_state"]["unsatisfiable_test"] == (
            "tests/acceptance/test_story.py"
        )

    def test_contradiction_with_no_flipping_test_does_not_shadow(self, tmp_path):
        # The conflict is between production-side diagnostics; no test is
        # one of the flipping expectations. It must neither regenerate an
        # unrelated test nor block the branches after it.
        state = self._state(
            tmp_path,
            node_state={"contradictory_tests": True},
            loop_counter={"total_repairs": 1,
                          "contradiction_pair": [["x::a"], ["x::b"]]},
        )
        assert route_after_compiler(state) == "repair_node"


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
def stub_sandbox(monkeypatch):
    import harness.sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "SandboxExecutor", _StubSandboxExecutor)

    def _set(exit_code: int, raw_output: str = "") -> None:
        _StubSandboxExecutor.canned = _StubBuildResult(exit_code, raw_output)

    return _set


class TestCompilerNodeRebuildsRoundState:
    @staticmethod
    def _state(tmp_path, loop_counter, node_state=None):
        return {
            "workspace_path": str(tmp_path),
            "build_command": "false",
            "allow_network": False,
            "sandbox_config": {},
            "loop_counter": loop_counter,
            "node_state": node_state or {},
            "messages": [],
        }

    @pytest.mark.asyncio
    async def test_pollution_from_an_earlier_round_is_dropped(
        self, stub_sandbox, tmp_path,
    ):
        stub_sandbox(1, "some compile error\n")
        result = await compiler_node(self._state(tmp_path, {
            "suite_order_pollution_files": ["server/tests/test_fixed_long_ago.py"],
        }))
        assert result["exit_code"] == 1
        assert not result["loop_counter"].get("suite_order_pollution_files")

    @pytest.mark.asyncio
    async def test_green_build_retires_the_contradiction_latch(
        self, stub_sandbox, tmp_path,
    ):
        stub_sandbox(0, "")
        result = await compiler_node(self._state(
            tmp_path,
            {"contradiction_pair": [["x::a"], ["x::b"]]},
            {"contradictory_tests": True},
        ))
        assert "contradictory_tests" not in result["node_state"]
        assert "contradiction_pair" not in result["loop_counter"]


class TestAnExhaustedLadderStopsDiverting:
    """lumina-run19-20260925-1313 spent its last four HITL trips in 35
    seconds on compiler -> router -> HITL -> resume -> compiler, with no
    repair round, no patch and no LLM call in between.

    The router kept diverting `server/tests/test_main.py` to the test-author
    ladder on the judge's recommendation; the ladder answered "already
    regenerated 2/2 and still unsatisfiable -> HITL" every time. The test was
    CORRECT — `route_paths` held only FastAPI's built-in docs routes, so the
    app genuinely had no router — and repair, the one component that could
    have fixed the production side, was never reached.
    """

    @staticmethod
    def _state(tmp_path, attempts, **over):
        (tmp_path / "server" / "tests").mkdir(parents=True, exist_ok=True)
        (tmp_path / "server" / "tests" / "test_main.py").write_text(
            "def test_a():\n    assert 1\n", encoding="utf-8")
        base = {
            "exit_code": 1,
            "budget_remaining_usd": 5.0,
            "workspace_path": str(tmp_path),
            "loop_counter": {
                "total_repairs": 1,
                "test_regen_attempts": {"server/tests/test_main.py": attempts},
                "last_reflection_verdict": {
                    "verdict": "DISTRACTION",
                    "recommendation": (
                        "Edit server/tests/test_main.py line 12 to extract "
                        "paths from route objects using getattr(...)."
                    ),
                },
            },
            "compiler_errors": [
                {"file": "server/tests/test_main.py", "line": 87,
                 "severity": "error", "error_code": "AssertionError",
                 "message": "assert '/api/birthdays' in ['/openapi.json']"},
            ],
            "node_state": {},
            "test_regeneration_config": {"enabled": True, "tier_b_auto": True,
                                         "max_attempts_per_test": 2},
        }
        base.update(over)
        return base

    def test_spent_attempts_go_to_repair_not_hitl(self, tmp_path):
        from harness.graph import route_after_compiler
        state = self._state(tmp_path, attempts=2)
        assert route_after_compiler(state) == "repair_node", (
            "repair cannot edit the test, but it CAN fix the production "
            "defect the test is reporting"
        )

    def test_attempts_remaining_still_divert(self, tmp_path):
        # Regression guard: the ladder must still be used while it has
        # something to offer.
        from harness.graph import route_after_compiler
        state = self._state(tmp_path, attempts=0)
        assert route_after_compiler(state) != "repair_node"

    def test_the_helper_reads_the_configured_cap(self, tmp_path):
        from harness.graph import _test_regen_exhausted
        s = self._state(tmp_path, attempts=1)
        assert _test_regen_exhausted(s, "server/tests/test_main.py") is False
        s["test_regeneration_config"]["max_attempts_per_test"] = 1
        assert _test_regen_exhausted(s, "server/tests/test_main.py") is True

    def test_regeneration_disabled_is_not_exhaustion(self, tmp_path):
        # With the feature off there is no ladder to spend; the other
        # branches keep their existing behaviour.
        from harness.graph import _test_regen_exhausted
        s = self._state(tmp_path, attempts=9)
        s["test_regeneration_config"]["enabled"] = False
        assert _test_regen_exhausted(s, "server/tests/test_main.py") is False

    def test_an_unrelated_file_is_unaffected(self, tmp_path):
        from harness.graph import _test_regen_exhausted
        s = self._state(tmp_path, attempts=2)
        assert _test_regen_exhausted(s, "server/tests/test_other.py") is False
