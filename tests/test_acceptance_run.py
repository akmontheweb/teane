"""Tests for the ADR-0006 Phase-1 acceptance-run engine (triage + orchestration).

Pure — the sandbox runner is a fake callable.
"""

from __future__ import annotations

from harness.acceptance_gen import ALTITUDE_E2E, ALTITUDE_INTEGRATION, CLASS_BACKEND, CLASS_UI, AcceptanceScenario
from harness import acceptance_run as ar
from harness.acceptance_run import (
    STATUS_ATTRIBUTABLE,
    STATUS_DEFERRED_DEP,
    STATUS_PASSED,
    STATUS_TEST_BUG,
    TestOutcome,
)


def _int_scen(ac: str, name: str) -> AcceptanceScenario:
    return AcceptanceScenario(ac, name, ALTITUDE_INTEGRATION, CLASS_BACKEND, "    assert client.get('/').status_code == 200")


# ---------------------------------------------------------------------------
# classify_acceptance_failure
# ---------------------------------------------------------------------------


class TestClassify:
    def test_assertion_is_attributable(self):
        assert ar.classify_acceptance_failure(
            "E   AssertionError: assert 500 == 201") == STATUS_ATTRIBUTABLE

    def test_server_error_is_attributable(self):
        assert ar.classify_acceptance_failure("got 500 Internal Server Error") == STATUS_ATTRIBUTABLE

    def test_import_error_is_dependency_blocked(self):
        assert ar.classify_acceptance_failure(
            "ModuleNotFoundError: No module named 'app.reports'") == STATUS_DEFERRED_DEP

    def test_404_is_dependency_blocked(self):
        assert ar.classify_acceptance_failure(
            "assert response.status_code == 200\nE assert 404 == 200") == STATUS_DEFERRED_DEP

    def test_missing_client_fixture_is_dependency_blocked(self):
        assert ar.classify_acceptance_failure("fixture 'client' not found") == STATUS_DEFERRED_DEP

    def test_undefined_name_is_test_bug(self):
        assert ar.classify_acceptance_failure("NameError: name 'foo' is not defined") == STATUS_TEST_BUG

    def test_unknown_defaults_to_deferred(self):
        assert ar.classify_acceptance_failure("something weird happened") == STATUS_DEFERRED_DEP

    def test_empty_defaults_to_deferred(self):
        assert ar.classify_acceptance_failure("") == STATUS_DEFERRED_DEP

    def test_dep_wins_over_assertion_when_both_present(self):
        # A 404 assertion must NOT be mistaken for an attributable code defect.
        msg = "AssertionError: assert 404 == 201"
        assert ar.classify_acceptance_failure(msg) == STATUS_DEFERRED_DEP


# ---------------------------------------------------------------------------
# select_runnable
# ---------------------------------------------------------------------------


class TestSelectRunnable:
    def test_excludes_e2e_and_already_passed(self):
        scen = [
            _int_scen("STORY-1.AC-1", "test_a"),
            _int_scen("STORY-1.AC-2", "test_b"),
            AcceptanceScenario("STORY-1.AC-3", "e2e", ALTITUDE_E2E, CLASS_UI, "expect(x)"),
        ]
        runnable = ar.select_runnable(scen, already_passed={"STORY-1.AC-1"})
        assert [s.verifies for s in runnable] == ["STORY-1.AC-2"]


# ---------------------------------------------------------------------------
# run_acceptance
# ---------------------------------------------------------------------------


def _runner(outcomes):
    def _r(paths, workspace):
        return outcomes
    return _r


class TestRunAcceptance:
    def test_pass_and_fail_mapping(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add"), _int_scen("STORY-1.AC-2", "test_edit")]
        outcomes = [
            TestOutcome("tests/acceptance/test_s1.py::test_add", True),
            TestOutcome("tests/acceptance/test_s1.py::test_edit", False, "AssertionError: assert 500 == 200"),
        ]
        res = ar.run_acceptance(scen, ["tests/acceptance/test_s1.py"], "/ws",
                                runner=_runner(outcomes), story_keys=["STORY-1"])
        assert {o.ac_key: o.status for o in res.outcomes} == {
            "STORY-1.AC-1": STATUS_PASSED,
            "STORY-1.AC-2": STATUS_ATTRIBUTABLE,
        }
        assert res.has_attributable()

    def test_nodeid_with_parametrization_suffix_maps(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add")]
        outcomes = [TestOutcome("f.py::test_add[case1]", True)]
        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_runner(outcomes))
        assert res.passed()[0].ac_key == "STORY-1.AC-1"

    def test_uncollected_test_is_deferred(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add")]
        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_runner([]))
        assert res.outcomes[0].status == STATUS_DEFERRED_DEP
        assert "not collected" in res.outcomes[0].detail

    def test_runner_crash_defers_whole_batch(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add")]

        def _boom(paths, ws):
            raise RuntimeError("sandbox exploded")

        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_boom)
        assert all(o.status == STATUS_DEFERRED_DEP for o in res.outcomes)
        assert res.has_attributable() is False

    def test_collection_error_signal_defers_as_collection_error(self):
        # Fix B3: a runner that could not run the suite at all raises the
        # typed signal → every AC is recorded as deferred:collection-error
        # (its own honest bucket), NOT the misleading blocked-by-dependency.
        scen = [
            _int_scen("STORY-1.AC-1", "test_add"),
            _int_scen("STORY-1.AC-2", "test_edit"),
        ]

        def _collect_err(paths, ws):
            raise ar.AcceptanceCollectionError("ModuleNotFoundError: fastapi")

        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_collect_err,
                                story_keys=["STORY-1"])
        assert {o.status for o in res.outcomes} == {
            ar.STATUS_DEFERRED_COLLECTION,
        }
        # Distinct from blocked-by-dependency, still a deferral (never a
        # hard failure), and the cause is preserved in the detail.
        assert res.outcomes[0].status != STATUS_DEFERRED_DEP
        assert all(o.is_deferred for o in res.outcomes)
        assert not res.has_attributable()
        assert "fastapi" in res.outcomes[0].detail

    def test_collection_error_status_is_a_deferral(self):
        # The new status must live in the deferred family so it never
        # manufactures a hard failure that stalls a headless run.
        o = ar.ACOutcome("STORY-1.AC-1", ar.STATUS_DEFERRED_COLLECTION)
        assert o.is_deferred is True

    def test_nothing_runnable_returns_not_ran(self):
        scen = [AcceptanceScenario("STORY-1.AC-1", "e2e", ALTITUDE_E2E, CLASS_UI, "expect(x)")]
        res = ar.run_acceptance(scen, [], "/ws", runner=_runner([]))
        assert res.ran is False
        assert res.outcomes == []

    def test_dependency_blocked_does_not_flag_attributable(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add")]
        outcomes = [TestOutcome("f.py::test_add", False, "ModuleNotFoundError: no module named 'x'")]
        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_runner(outcomes))
        assert res.outcomes[0].status == STATUS_DEFERRED_DEP
        assert res.has_attributable() is False


class TestParsePytestOutcomes:
    def test_parses_passed_and_failed_with_messages(self):
        out = """
============================= short test summary info ==============================
PASSED tests/acceptance/test_s1.py::test_add
FAILED tests/acceptance/test_s1.py::test_edit - AssertionError: assert 500 == 201
ERROR tests/acceptance/test_s1.py::test_del - fixture 'client' not found
======================== 1 passed, 1 failed, 1 error in 0.3s =======================
"""
        outcomes = ar.parse_pytest_outcomes(out)
        by = {o.nodeid.rsplit("::", 1)[-1]: o for o in outcomes}
        assert by["test_add"].passed is True
        assert by["test_edit"].passed is False
        assert "AssertionError" in by["test_edit"].message
        assert by["test_del"].passed is False
        assert "fixture 'client' not found" in by["test_del"].message

    def test_end_to_end_parse_then_run(self):
        scen = [_int_scen("STORY-1.AC-1", "test_add"), _int_scen("STORY-1.AC-2", "test_edit")]
        raw = (
            "PASSED tests/acceptance/test_s1.py::test_add\n"
            "FAILED tests/acceptance/test_s1.py::test_edit - AssertionError: assert 500 == 201\n"
        )
        outcomes = ar.parse_pytest_outcomes(raw)
        res = ar.run_acceptance(scen, ["tests/acceptance/test_s1.py"], "/ws",
                                runner=lambda p, w: outcomes)
        statuses = {o.ac_key: o.status for o in res.outcomes}
        assert statuses["STORY-1.AC-1"] == STATUS_PASSED
        assert statuses["STORY-1.AC-2"] == STATUS_ATTRIBUTABLE

    def test_xfail_counts_as_passed(self):
        outcomes = ar.parse_pytest_outcomes("XFAIL f.py::test_known_gap")
        assert outcomes[0].passed is True

    def test_ignores_non_summary_lines(self):
        assert ar.parse_pytest_outcomes("random text\n=== 0 passed ===") == []

    def test_backfills_message_from_failures_section(self):
        # Summary line has NO "- message" (multi-line assertion); the message must
        # be recovered from the FAILURES block so triage can classify it.
        raw = (
            "=================================== FAILURES ===================================\n"
            "___________________________ test_add_valid_contact ___________________________\n"
            "    def test_add_valid_contact(client):\n"
            ">       assert resp.status_code == 201\n"
            'E       AssertionError: {"detail":"Host port is required"}\n'
            "E       assert 403 == 201\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/acceptance/test_s.py::test_add_valid_contact\n"
        )
        outcomes = ar.parse_pytest_outcomes(raw)
        assert len(outcomes) == 1
        assert outcomes[0].passed is False
        assert "AssertionError" in outcomes[0].message
        # and it must now triage as attributable, not defaulted-to-deferred
        assert ar.classify_acceptance_failure(outcomes[0].message) == STATUS_ATTRIBUTABLE

    def test_error_at_setup_message_extracted(self):
        raw = (
            "==================================== ERRORS ====================================\n"
            "_________________ ERROR at setup of test_edit_contact _________________\n"
            "E       fixture 'client' not found\n"
            "=========================== short test summary info ============================\n"
            "ERROR tests/acceptance/test_s.py::test_edit_contact\n"
        )
        outcomes = ar.parse_pytest_outcomes(raw)
        assert "fixture 'client' not found" in outcomes[0].message

    def test_collection_error_detected(self):
        assert ar.is_collection_error("ERROR collecting tests/acceptance/test_s1.py") is True
        assert ar.is_collection_error("1 passed in 0.1s") is False


class TestSummarize:
    def test_counts_and_lists(self):
        scen = [_int_scen("STORY-1.AC-1", "test_a"), _int_scen("STORY-1.AC-2", "test_b")]
        outcomes = [
            TestOutcome("f.py::test_a", True),
            TestOutcome("f.py::test_b", False, "AssertionError"),
        ]
        res = ar.run_acceptance(scen, ["f.py"], "/ws", runner=_runner(outcomes))
        s = ar.summarize(res)
        assert s["total"] == 2
        assert s["counts"][STATUS_PASSED] == 1
        assert s["counts"][STATUS_ATTRIBUTABLE] == 1
        assert s["passed_acs"] == ["STORY-1.AC-1"]
