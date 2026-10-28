"""The batch must report its real scope, not just the part it could run.

``run_acceptance`` was given the RUNNABLE scenarios, so the run result -- and
therefore ``summarize`` and the ``acceptance_summary`` in node_state -- counted
only criteria with an integration test. A ui-only criterion (e2e scenario,
``integration: null``) or a not-runtime one existed solely as a row in the
deferral table and never entered the denominator.

lumina batch 88 reported ``total:10`` for a batch whose scope was 17 criteria,
and STORY-003 sealed ``done`` with 0 of its 4 criteria verified by anything at
all. Nothing in the log said so, which is the part that matters: the build was
not wrong to defer them -- a ui criterion needs a browser against the app
running on Docker, which is ``teane test``'s job -- it was wrong to report the
batch as though they had never been in scope.

So this is a REPORTING fix. No in-build e2e runner is added or wanted.
"""

from __future__ import annotations

from harness import acceptance_gen as ag
from harness import acceptance_run as ar

STORY = "STORY-003"


def _integration(ac_key: str, name: str) -> ag.AcceptanceScenario:
    return ag.AcceptanceScenario(
        ac_key, name, ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND,
        "assert True",
    )


def _ui(ac_key: str, name: str) -> ag.AcceptanceScenario:
    return ag.AcceptanceScenario(
        ac_key, name, ag.ALTITUDE_E2E, ag.CLASS_UI,
        "await page.goto('/');",
    )


class TestTheDenominator:
    SCEN = [
        _integration(f"{STORY}.AC-1", "test_directory_sorted"),
        _ui(f"{STORY}.AC-2", "test_directory_renders"),
        _ui(f"{STORY}.AC-3", "test_filter_chip_clears"),
    ]
    UNVERIFIABLE = {
        f"{STORY}.AC-2": (ar.STATUS_DEFERRED_UI, "ui-only"),
        f"{STORY}.AC-3": (ar.STATUS_DEFERRED_UI, "ui-only"),
    }

    def _run(self, *, unverifiable=None):
        outcomes = ar.parse_pytest_outcomes("PASSED t.py::test_directory_sorted")
        return ar.run_acceptance(
            self.SCEN, ["t.py"], "/ws", runner=lambda p, w: outcomes,
            unverifiable=unverifiable,
        )

    def test_the_regression_without_the_scope(self):
        """The premise: two of three criteria were simply absent."""
        res = self._run()
        assert len(res.outcomes) == 1
        assert ar.summarize(res)["total"] == 1

    def test_every_criterion_in_scope_is_reported(self):
        res = self._run(unverifiable=self.UNVERIFIABLE)
        assert {o.ac_key for o in res.outcomes} == {
            f"{STORY}.AC-1", f"{STORY}.AC-2", f"{STORY}.AC-3",
        }
        assert ar.summarize(res)["total"] == 3

    def test_the_summary_separates_verified_from_scope(self):
        """``total`` was read as "criteria verified" often enough to be worth
        ending."""
        summary = ar.summarize(self._run(unverifiable=self.UNVERIFIABLE))
        assert summary["total"] == 3
        assert summary["verified"] == 1
        assert summary["unverified"] == 2

    def test_the_ui_criteria_are_deferred_not_passed(self):
        res = self._run(unverifiable=self.UNVERIFIABLE)
        by_ac = {o.ac_key: o for o in res.outcomes}
        assert by_ac[f"{STORY}.AC-2"].status == ar.STATUS_DEFERRED_UI
        assert by_ac[f"{STORY}.AC-2"].is_deferred
        assert by_ac[f"{STORY}.AC-1"].status == ar.STATUS_PASSED

    def test_they_are_not_attributable_so_repair_is_not_entered(self):
        """Nothing the repair loop can do about a criterion needing a
        browser — it must not be handed one."""
        res = self._run(unverifiable=self.UNVERIFIABLE)
        assert res.attributable() == []

    def test_they_appear_in_the_deferred_list(self):
        summary = ar.summarize(self._run(unverifiable=self.UNVERIFIABLE))
        assert sorted(summary["deferred_acs"]) == [
            (f"{STORY}.AC-2", ar.STATUS_DEFERRED_UI),
            (f"{STORY}.AC-3", ar.STATUS_DEFERRED_UI),
        ]

    def test_story_003s_shape_is_now_visible(self):
        """Its actual numbers: four criteria, none verified in-build. The
        build may still seal — deferral is correct — but it can no longer do
        so while reporting an empty scope."""
        scen = [_ui(f"{STORY}.AC-{i}", f"test_ui_{i}") for i in range(1, 5)]
        unver = {
            f"{STORY}.AC-{i}": (ar.STATUS_DEFERRED_UI, "ui-only")
            for i in range(1, 5)
        }
        res = ar.run_acceptance(
            scen, [], "/ws", runner=lambda p, w: [], unverifiable=unver,
        )
        summary = ar.summarize(res)
        assert summary["total"] == 4
        assert summary["verified"] == 0
        assert summary["unverified"] == 4
        assert summary["ran"] is False, "nothing integration-shaped ran"


class TestTheOtherReturnPaths:
    """Every exit from the triage has to carry the scope, or the hole just
    moves to whichever path a run happens to take."""

    SCEN = [_integration(f"{STORY}.AC-1", "test_a")]
    UNVER = {f"{STORY}.AC-2": (ar.STATUS_DEFERRED_UI, "ui-only")}

    def test_a_collection_error_still_reports_scope(self):
        def _boom(paths, ws):
            raise ar.AcceptanceCollectionError("app did not import")

        res = ar.run_acceptance(
            self.SCEN, ["t.py"], "/ws", runner=_boom, unverifiable=self.UNVER,
        )
        assert {o.ac_key for o in res.outcomes} == {
            f"{STORY}.AC-1", f"{STORY}.AC-2",
        }

    def test_a_runner_crash_still_reports_scope(self):
        def _boom(paths, ws):
            raise RuntimeError("sandbox died")

        res = ar.run_acceptance(
            self.SCEN, ["t.py"], "/ws", runner=_boom, unverifiable=self.UNVER,
        )
        assert len(res.outcomes) == 2

    def test_nothing_runnable_still_reports_scope(self):
        """The ui-only-only batch — the case that reported nothing at all."""
        res = ar.run_acceptance(
            [_ui(f"{STORY}.AC-2", "test_ui")], [], "/ws",
            runner=lambda p, w: [], unverifiable=self.UNVER,
        )
        assert res.ran is False
        assert [o.ac_key for o in res.outcomes] == [f"{STORY}.AC-2"]

    def test_no_unverifiable_criteria_changes_nothing(self):
        res = ar.run_acceptance(
            self.SCEN, ["t.py"], "/ws",
            runner=lambda p, w: ar.parse_pytest_outcomes("PASSED t.py::test_a"),
        )
        assert len(res.outcomes) == 1


class TestTheNodesScopeComputation:
    def test_ui_only_excludes_a_criterion_that_also_has_integration(self):
        import harness.acceptance_node as an

        scen = [
            _ui(f"{STORY}.AC-1", "test_ui"),
            _integration(f"{STORY}.AC-1", "test_api"),
            _ui(f"{STORY}.AC-2", "test_ui_2"),
        ]
        assert an._ui_only_acs(scen) == {f"{STORY}.AC-2"}

    def test_it_folds_in_not_runtime_criteria(self):
        import harness.acceptance_node as an

        out = an._unverifiable_in_build(
            [_ui(f"{STORY}.AC-2", "test_ui")],
            {f"{STORY}.AC-5": "verified by the CI pipeline configuration"},
        )
        assert out[f"{STORY}.AC-2"][0] == ar.STATUS_DEFERRED_UI
        assert out[f"{STORY}.AC-5"][0] == ar.STATUS_DEFERRED_NOT_RUNTIME
        assert "CI pipeline" in out[f"{STORY}.AC-5"][1]

    def test_a_batch_with_nothing_undecidable_is_empty(self):
        import harness.acceptance_node as an

        assert an._unverifiable_in_build(
            [_integration(f"{STORY}.AC-1", "test_api")], {},
        ) == {}

    def test_the_two_paths_cannot_disagree_about_which_acs_are_ui_only(self):
        """``_persist_ui_only_deferrals`` and the reporting path now share
        ``_ui_only_acs``. They disagreed silently for as long as only the
        first existed."""
        import inspect

        import harness.acceptance_node as an

        assert "_ui_only_acs(scenarios)" in inspect.getsource(
            an._persist_ui_only_deferrals
        )
        assert "_ui_only_acs(scenarios)" in inspect.getsource(
            an._unverifiable_in_build
        )


class TestThePassthroughReportsScope:
    def test_a_ui_only_batch_reports_its_criteria(self):
        import harness.acceptance_node as an

        out = an._passthrough(
            {}, reason="no runnable integration ACs this batch",
            unverifiable={f"{STORY}.AC-2": (ar.STATUS_DEFERRED_UI, "ui-only")},
        )
        summary = out["node_state"]["acceptance_summary"]
        assert summary["total"] == 1
        assert summary["verified"] == 0
        assert summary["unverified"] == 1

    def test_a_genuine_skip_reports_no_summary(self):
        """acceptance.enabled is false — there is no scope to report, and
        inventing one would be its own lie."""
        import harness.acceptance_node as an

        out = an._passthrough({}, reason="acceptance.enabled is false")
        assert "acceptance_summary" not in out["node_state"]
        assert out["node_state"]["skipped"] is True

    def test_it_still_does_not_route_to_repair(self):
        import harness.acceptance_node as an

        out = an._passthrough(
            {}, reason="x",
            unverifiable={f"{STORY}.AC-2": (ar.STATUS_DEFERRED_UI, "ui")},
        )
        assert out["node_state"]["acceptance_attributable"] is False
        assert an.route_after_acceptance(out) == "code_review_node"
