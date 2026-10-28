"""A batch must not inherit the previous batch's parked failure.

ADR-0006 Phase 1 seals a batch on ``attributable`` acceptance failures by
design -- acceptance is advisory and never blocks a build. But ``_park_over_cap``
only recorded the deferral: the failing test stayed LIVE, so the next batch
collected it, failed it, and spent its own repair rounds rediscovering a defect
already written down. Batch 89 burnt 6 empty rounds and 2 resumes on batch 88's
``STORY-004.AC-1``.

The fix is a quarantine -- carry the test as an xfail -- and implementing it
surfaced a worse bug underneath, which is what most of this file covers:

    ``parse_pytest_outcomes`` counted XFAIL as **passed**.

A PASSED outcome calls ``link_test_to_ac(provenance="build")`` and clears the
criterion's deferral, so quarantining a test would have recorded an unverified
criterion as verified by a test that did not hold -- a silent false ``done``,
and strictly worse than inheriting the failure. XPASS was wrong in the other
direction: a test that ran and passed matched no failure fingerprint and landed
on ``deferred:blocked-by-dependency``.

:class:`TestTheFullLifecycle` is the class that matters: park, inherit nothing,
then clear on its own when the code catches up.
"""

from __future__ import annotations

import os

import pytest

from harness import acceptance_gen as ag
from harness import acceptance_run as ar

SUITE = '''# Acceptance for STORY-004. AUTO-GENERATED.
# integration altitude
from fastapi.testclient import TestClient


# @verifies: STORY-004.AC-1
def test_dashboard_returns_buckets(client):
    assert client.get("/api/v1/dash").status_code == 200


# @verifies: STORY-004.AC-2
def test_dashboard_groups_by_week(client):
    assert client.get("/api/v1/dash").json()["weeks"] == []
'''

AC1 = "STORY-004.AC-1"
AC2 = "STORY-004.AC-2"


class TestTheMarker:
    def test_it_marks_only_the_named_criterion(self):
        out = ag.quarantine_criterion(SUITE, AC1, "assert 200 == 500")
        assert out is not None
        assert "xfail" in out
        # AC-2's test is byte-for-byte untouched.
        assert "def test_dashboard_groups_by_week(client):\n    assert" in out
        assert out.count("xfail") == 1

    def test_the_verifies_markers_still_parse(self):
        """A decorator sits between the ``# @verifies:`` comment and the def.
        If that broke the marker parser, every criterion in the file would
        drop out of the mapping and the suite would stop being traceable."""
        out = ag.quarantine_criterion(SUITE, AC1, "boom")
        spans = ag.verifies_by_function(out)
        assert {k for k, _f, _l in spans.values()} == {AC1, AC2}

    def test_the_file_still_parses(self):
        out = ag.quarantine_criterion(SUITE, AC1, "boom")
        compile(out, "suite.py", "exec")

    def test_pytest_is_imported_when_it_was_not(self):
        out = ag.quarantine_criterion(SUITE, AC1, "boom")
        assert "import pytest" in out

    def test_pytest_is_not_imported_twice(self):
        src = SUITE.replace(
            "from fastapi.testclient import TestClient",
            "import pytest\nfrom fastapi.testclient import TestClient",
        )
        out = ag.quarantine_criterion(src, AC1, "boom")
        assert out.count("import pytest") == 1

    def test_it_is_idempotent(self):
        once = ag.quarantine_criterion(SUITE, AC1, "boom")
        assert ag.quarantine_criterion(once, AC1, "again") is None

    def test_an_unknown_criterion_changes_nothing(self):
        assert ag.quarantine_criterion(SUITE, "STORY-004.AC-9", "x") is None

    def test_an_unparseable_suite_changes_nothing(self):
        assert ag.quarantine_criterion("def (", AC1, "x") is None

    def test_the_reason_reaches_the_marker(self):
        out = ag.quarantine_criterion(SUITE, AC1, "assert 200 == 500")
        assert "assert 200 == 500" in out

    def test_a_quote_in_the_reason_cannot_break_the_decorator(self):
        out = ag.quarantine_criterion(SUITE, AC1, 'got "weeks" not weeks')
        compile(out, "suite.py", "exec")

    def test_a_long_reason_is_truncated(self):
        out = ag.quarantine_criterion(SUITE, AC1, "x" * 5000)
        compile(out, "suite.py", "exec")
        assert len(out) < len(SUITE) + 500

    def test_it_is_not_strict(self):
        """A strict marker reports an unexpected pass as FAILED with a
        non-zero exit, making the build red for the one event everyone
        wants. Non-strict reports XPASS, which the parser reads as a real
        pass."""
        out = ag.quarantine_criterion(SUITE, AC1, "boom")
        assert "strict=False" in out


class TestRemovingTheMarker:
    def test_it_removes_the_marker_it_added(self):
        marked = ag.quarantine_criterion(SUITE, AC1, "boom")
        out = ag.unquarantine_criterion(marked, AC1)
        assert out is not None
        assert "xfail" not in out
        compile(out, "suite.py", "exec")

    def test_it_removes_only_the_named_criterions_marker(self):
        marked = ag.quarantine_criterion(SUITE, AC1, "boom")
        marked = ag.quarantine_criterion(marked, AC2, "also boom")
        out = ag.unquarantine_criterion(marked, AC1)
        assert out.count("xfail") == 1
        assert "also boom" in out

    def test_it_is_a_noop_when_there_is_no_marker(self):
        assert ag.unquarantine_criterion(SUITE, AC1) is None

    def test_it_leaves_a_hand_written_xfail_alone(self):
        """A plain xfail in a generated suite is somebody's deliberate
        statement. Only the harness's own tagged marker is removed."""
        src = SUITE.replace(
            "# @verifies: STORY-004.AC-1\ndef",
            "# @verifies: STORY-004.AC-1\n"
            '@pytest.mark.xfail(reason="known upstream bug")\ndef',
        )
        assert ag.unquarantine_criterion(src, AC1) is None

    def test_an_unknown_criterion_changes_nothing(self):
        marked = ag.quarantine_criterion(SUITE, AC1, "boom")
        assert ag.unquarantine_criterion(marked, "STORY-004.AC-9") is None


class TestTheParserNoLongerCallsAnXfailAPass:
    """The bug underneath the feature."""

    def test_xfail_is_not_passed(self):
        o = ar.parse_pytest_outcomes("XFAIL f.py::test_gap")[0]
        assert o.passed is False
        assert o.pytest_status == "XFAIL"

    def test_xpass_is_passed(self):
        o = ar.parse_pytest_outcomes("XPASS f.py::test_fixed")[0]
        assert o.passed is True

    def test_a_strict_xpass_is_passed(self):
        o = ar.parse_pytest_outcomes(
            "FAILED f.py::test_fixed - [XPASS(strict)] teane-quarantine: x"
        )[0]
        assert o.passed is True
        assert o.pytest_status == "XPASS"

    def test_an_ordinary_failure_is_untouched(self):
        o = ar.parse_pytest_outcomes(
            "FAILED f.py::test_x - AssertionError: assert 200 == 500"
        )[0]
        assert o.passed is False
        assert o.pytest_status == "FAILED"

    def test_a_pass_is_untouched(self):
        o = ar.parse_pytest_outcomes("PASSED f.py::test_x")[0]
        assert o.passed is True
        assert o.pytest_status == "PASSED"


def _scenario(ac_key: str, name: str) -> ag.AcceptanceScenario:
    return ag.AcceptanceScenario(
        ac_key, name, ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND,
        "assert True",
    )


class TestTheQuarantinedStatus:
    """What the engine reports for a criterion carrying the marker."""

    SCEN = [
        _scenario(AC1, "test_dashboard_returns_buckets"),
        _scenario(AC2, "test_dashboard_groups_by_week"),
    ]

    def _run(self, raw: str):
        outcomes = ar.parse_pytest_outcomes(raw)
        res = ar.run_acceptance(
            self.SCEN, ["tests/acceptance/test_story_004_acceptance.py"],
            "/ws", runner=lambda p, w: outcomes,
        )
        return {o.ac_key: o for o in res.outcomes}

    def test_a_quarantined_criterion_is_not_passed(self):
        by_ac = self._run(
            "XFAIL t.py::test_dashboard_returns_buckets - "
            "teane-quarantine: assert 200 == 500\n"
            "PASSED t.py::test_dashboard_groups_by_week"
        )
        assert by_ac[AC1].status == ar.STATUS_DEFERRED_QUARANTINED
        assert by_ac[AC2].status == ar.STATUS_PASSED

    def test_it_is_a_deferred_status_so_it_never_hard_fails(self):
        """The never-hard-fail safety property of ADR-0006 is preserved: a
        quarantined criterion defers, it does not fail the batch."""
        by_ac = self._run("XFAIL t.py::test_dashboard_returns_buckets")
        assert by_ac[AC1].is_deferred
        assert ar.STATUS_DEFERRED_QUARANTINED in ar._DEFERRED

    def test_it_is_not_attributable_so_repair_is_not_re_entered(self):
        """The entire point. An attributable status sends the batch back
        into the repair loop for a defect already parked."""
        outcomes = ar.parse_pytest_outcomes(
            "XFAIL t.py::test_dashboard_returns_buckets"
        )
        res = ar.run_acceptance(
            self.SCEN, ["t.py"], "/ws", runner=lambda p, w: outcomes,
        )
        assert res.attributable() == []

    def test_a_quarantine_that_starts_passing_is_reported_passed(self):
        by_ac = self._run("XPASS t.py::test_dashboard_returns_buckets")
        assert by_ac[AC1].status == ar.STATUS_PASSED

    def test_the_detail_survives_for_the_summary(self):
        by_ac = self._run(
            "XFAIL t.py::test_dashboard_returns_buckets - "
            "teane-quarantine: assert 200 == 500"
        )
        assert "assert 200 == 500" in by_ac[AC1].detail

    def test_a_quarantined_criterion_with_no_reason_still_reads(self):
        by_ac = self._run("XFAIL t.py::test_dashboard_returns_buckets")
        assert by_ac[AC1].detail


class TestTheFullLifecycle:
    """Park it, inherit nothing, clear it when the code catches up —
    driven through the shipped functions on a real file."""

    def _suite(self, tmp_path) -> str:
        d = tmp_path / "tests" / "acceptance"
        d.mkdir(parents=True)
        p = d / "test_story_004_acceptance.py"
        p.write_text(SUITE)
        return str(p)

    SCEN = [
        _scenario(AC1, "test_dashboard_returns_buckets"),
        _scenario(AC2, "test_dashboard_groups_by_week"),
    ]

    def test_park_then_the_next_batch_inherits_nothing(self, tmp_path):
        import harness.acceptance_node as an

        path = self._suite(tmp_path)
        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE,
                          detail="assert 200 == 500")],
            [path],
        )
        on_disk = open(path).read()
        assert "xfail" in on_disk, "the parked test was left live"

        # Next batch: the same defect, now reported by pytest as an xfail.
        outcomes = ar.parse_pytest_outcomes(
            "XFAIL t.py::test_dashboard_returns_buckets - "
            "teane-quarantine: assert 200 == 500\n"
            "PASSED t.py::test_dashboard_groups_by_week"
        )
        res = ar.run_acceptance(
            self.SCEN, [path], str(tmp_path), runner=lambda p, w: outcomes,
        )
        assert res.attributable() == [], (
            "the next batch would re-enter repair for a parked defect"
        )
        assert [o.status for o in res.outcomes if o.ac_key == AC1] == [
            ar.STATUS_DEFERRED_QUARANTINED
        ], "and it must not be recorded as verified either"

    def test_a_db_failure_does_not_stop_the_quarantine(self, tmp_path):
        """``_park_over_cap`` opens the story DB first and returns early if
        it cannot. The marker is what stops the next batch paying, so it
        must not be collateral damage of an unavailable DB."""
        import harness.acceptance_node as an

        path = self._suite(tmp_path)
        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
            [path],
        )
        assert "xfail" in open(path).read()

    def test_passing_again_strips_the_marker(self, tmp_path):
        import harness.acceptance_node as an

        path = self._suite(tmp_path)
        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
            [path],
        )
        assert "xfail" in open(path).read()

        res = ar.AcceptanceRunResult(
            outcomes=[ar.ACOutcome(AC1, ar.STATUS_PASSED)]
        )
        an._unquarantine_passed(res, [path])
        on_disk = open(path).read()
        assert "xfail" not in on_disk, (
            "a stale marker would report a later regression as "
            "'quarantined, not verified' instead of the defect it is"
        )
        compile(on_disk, path, "exec")

    def test_a_still_failing_criterion_keeps_its_marker(self, tmp_path):
        import harness.acceptance_node as an

        path = self._suite(tmp_path)
        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
            [path],
        )
        an._unquarantine_passed(
            ar.AcceptanceRunResult(
                outcomes=[ar.ACOutcome(AC1, ar.STATUS_DEFERRED_QUARANTINED)]
            ),
            [path],
        )
        assert "xfail" in open(path).read()

    def test_a_missing_suite_file_is_survivable(self, tmp_path):
        import harness.acceptance_node as an

        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
            [str(tmp_path / "gone.py")],
        )  # must not raise

    def test_no_written_files_is_survivable(self, tmp_path):
        import harness.acceptance_node as an

        an._park_over_cap(
            "/nonexistent-workspace", "app",
            [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
            None,
        )  # must not raise

    def test_an_unwritable_suite_is_logged_not_raised(self, tmp_path):
        import harness.acceptance_node as an

        path = self._suite(tmp_path)
        os.chmod(path, 0o444)
        try:
            an._park_over_cap(
                "/nonexistent-workspace", "app",
                [ar.ACOutcome(AC1, ar.STATUS_ATTRIBUTABLE, detail="boom")],
                [path],
            )
        finally:
            os.chmod(path, 0o644)


class TestTheMarkerSurvivesARegeneration:
    """The acceptance suite is rewritten wholesale by its own author, so a
    quarantine is not permanent — and must not be, or a regenerated
    criterion would arrive pre-silenced."""

    def test_a_spliced_replacement_drops_the_marker(self):
        marked = ag.quarantine_criterion(SUITE, AC1, "boom")
        new = ag.splice_integration_scenarios(
            marked, {AC1: _scenario(AC1, "test_dashboard_returns_buckets")},
        )
        assert "xfail" not in new
        compile(new, "suite.py", "exec")

    @pytest.mark.parametrize("other", [AC2])
    def test_splicing_another_criterion_keeps_the_marker(self, other):
        marked = ag.quarantine_criterion(SUITE, AC1, "boom")
        new = ag.splice_integration_scenarios(
            marked, {other: _scenario(other, "test_dashboard_groups_by_week")},
        )
        assert "xfail" in new
