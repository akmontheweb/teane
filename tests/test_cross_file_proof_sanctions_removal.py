"""A provably-impossible unit test must be REMOVABLE.

The escape ladder exists to delete a test no production change can satisfy.
Two pieces of machinery stood between it and that outcome, and both were
single-file:

  * ``machine_unsatisfiable_reason`` reads ONE file, so a pair split across
    two files could reach regeneration only via the judge's behavioural
    ``oscillation_contradiction`` verdict -- i.e. after repair had already
    paid to oscillate.
  * ``proven_unsatisfiable_tests`` likewise, so the proof set arrived EMPTY
    for exactly that shape. ``coverage_nonregression_ok`` then held the
    rewrite to its one-function tolerance -- a tolerance that encodes "falsity
    arrives as a contradictory PAIR", which is merely the only shape the
    single-file prover can see -- and rejected the fix.

lumina 01a17704 (run 9) died this way: the router proved
``test_birthday_service.py`` unsatisfiable, regeneration produced the
corrected suite, and the gate threw it away for "test-function count dropped
18->16 (>1 removed)", twice, spending the file's whole budget. The
proof-sanctioning half of that was fixed then. This covers the other half:
the sanction list was empty for the cross-file case, which is the case that
dominated runs 5, 8, 9 and 10.

:class:`TestTheGateNowAcceptsTheFix` is the decisive class -- the same
rewrite, rejected without the cross-file proof and accepted with it.
"""

from __future__ import annotations

import os

import pytest

from harness.test_contradiction import (
    machine_unsatisfiable_reason,
    machine_unsatisfiable_reason_across,
    proven_unsatisfiable_tests,
    proven_unsatisfiable_tests_across,
)
from harness.test_regeneration import coverage_nonregression_ok

TARGET_REL = "server/tests/test_m.py"
SIBLING_REL = "server/tests/test_svc.py"

#: Four tests. The first two are each HALF of a cross-file contradiction.
TARGET = '''# @tests: app/m.py
import pytest
from app.m import normalise


def test_blank_is_rejected():
    with pytest.raises(ValueError):
        normalise("   ")


def test_dash_is_rejected():
    with pytest.raises(ValueError):
        normalise("-")


def test_trims_whitespace():
    assert normalise(" a ") == "a"


def test_keeps_case():
    assert normalise("Ab") == "Ab"
'''

#: The other halves: the same two calls, required to SUCCEED.
SIBLING = '''# @tests: app/svc.py
from app.m import normalise
from app.svc import save


def test_saves_blank_payload():
    value = normalise("   ")
    assert save(value) is not None


def test_saves_dash_payload():
    value = normalise("-")
    assert save(value) is not None
'''

#: The correct fix: the two impossible tests are gone, the rest kept.
REWRITE = '''# @tests: app/m.py
from app.m import normalise


def test_trims_whitespace():
    assert normalise(" a ") == "a"


def test_keeps_case():
    assert normalise("Ab") == "Ab"
'''

FILES = {TARGET_REL: TARGET, SIBLING_REL: SIBLING}


class TestTheRegressionItself:
    def test_the_single_file_prover_sees_none_of_this(self):
        """The premise. Neither half of the machinery could act, and the
        target file is not self-contradictory by any reading."""
        assert machine_unsatisfiable_reason(
            TARGET, filename=TARGET_REL
        ) is None
        assert proven_unsatisfiable_tests(
            TARGET, filename=TARGET_REL
        ) == frozenset()

    def test_the_gate_rejects_the_correct_fix_with_no_proof(self):
        """Run 9's failure, reproduced in miniature: the only available fix
        is refused for dropping more than one test function."""
        ok, detail = coverage_nonregression_ok(TARGET, REWRITE)
        assert not ok
        assert "dropped 4->2" in detail


class TestTheCrossFileProver:
    def test_it_names_this_files_side_of_each_pair(self):
        names = proven_unsatisfiable_tests_across(FILES, target=TARGET_REL)
        assert names == frozenset(
            {"test_blank_is_rejected", "test_dash_is_rejected"}
        )

    def test_it_names_the_other_files_side_when_that_is_the_target(self):
        """Symmetry: regeneration rewrites one file, so the sanction list is
        whichever side lives in the file being rewritten. Which side SHOULD
        go is the spec's call (spec_outranks_the_code), not this one's."""
        names = proven_unsatisfiable_tests_across(FILES, target=SIBLING_REL)
        assert names == frozenset(
            {"test_saves_blank_payload", "test_saves_dash_payload"}
        )

    def test_a_consistent_batch_proves_nothing(self):
        """No proof must ever read as licence to delete."""
        agreeing = SIBLING.replace(
            'value = normalise("   ")\n    assert save(value) is not None',
            'with pytest.raises(ValueError):\n        normalise("   ")',
        )
        names = proven_unsatisfiable_tests_across(
            {TARGET_REL: TARGET, SIBLING_REL: "import pytest\n" + agreeing},
            target=TARGET_REL,
        )
        assert "test_blank_is_rejected" not in names

    def test_a_file_not_in_the_batch_is_not_named(self):
        names = proven_unsatisfiable_tests_across(
            FILES, target="server/tests/test_unrelated.py",
        )
        assert names == frozenset()

    def test_a_non_python_target_is_declined(self):
        """jest/vitest are not this AST detector's dialect."""
        assert proven_unsatisfiable_tests_across(
            FILES, target="web/src/m.test.ts",
        ) == frozenset()
        assert machine_unsatisfiable_reason_across(
            FILES, target="web/src/m.test.ts",
        ) is None

    def test_a_same_named_file_in_another_tree_is_not_matched(self):
        """This repo has a recurring split-test-tree problem — server/tests/
        and tests/ holding files of the same name. Sanctioning a deletion in
        the wrong one would be the worst failure this gate can have, so the
        match is on the full normalised path, never the basename."""
        names = proven_unsatisfiable_tests_across(
            FILES, target="tests/test_m.py",
        )
        assert names == frozenset()

    def test_a_redundant_path_spelling_still_matches(self):
        names = proven_unsatisfiable_tests_across(
            FILES, target="server/./tests/test_m.py",
        )
        assert "test_blank_is_rejected" in names

    def test_an_intra_file_pair_names_both_sides(self):
        """Consistent with the single-file prover, so a caller holding only
        the batch map reaches the same answer.

        Note the success side is an ASSIGNMENT. A call written inline in an
        assert comparison — ``assert normalise("   ") == ""`` — is not
        counted as a success use by the scanner, so neither prover sees that
        shape; see ``TestTheScannerDoesNotSeeAnAssertExpression``.
        """
        both = '''import pytest
from app.m import normalise


def test_raises():
    with pytest.raises(ValueError):
        normalise("   ")


def test_succeeds():
    value = normalise("   ")
    assert value == ""
'''
        names = proven_unsatisfiable_tests_across(
            {"t.py": both}, target="t.py",
        )
        assert names == frozenset({"test_raises", "test_succeeds"})

    def test_an_unparseable_sibling_does_not_crash_the_prover(self):
        names = proven_unsatisfiable_tests_across(
            {TARGET_REL: TARGET, SIBLING_REL: SIBLING, "bad.py": "def ("},
            target=TARGET_REL,
        )
        assert "test_blank_is_rejected" in names


class TestTheScannerDoesNotSeeAnAssertExpression:
    """A separate gap, found by a wrong expectation of mine, pinned here so
    it is not rediscovered.

    A call written inline in an assert comparison — ``assert normalise(x)
    == y`` — is not counted as a SUCCESS use, so no contradiction involving
    it can be proven, by either prover. The same call assigned to a name
    first is. That matters because the inline form is how most of the false
    tests in the lumina runs were written: ``assert len(result) == 2``,
    ``assert [r["first_name"] for r in rows] == [...]``.

    Fixing it widens what the escape ladder can prove — upstream of both
    the routing decision and the sanction list — but it is a change to the
    detector's own semantics, not to this fix, and it would make more
    contradictions provable across the whole harness at once. Left
    deliberately, documented, measured by these tests.
    """

    INLINE = '''import pytest
from app.m import normalise


def test_raises():
    with pytest.raises(ValueError):
        normalise("   ")


def test_succeeds():
    assert normalise("   ") == ""
'''

    def test_the_inline_form_proves_nothing(self):
        assert proven_unsatisfiable_tests(
            self.INLINE, filename="t.py"
        ) == frozenset()
        assert proven_unsatisfiable_tests_across(
            {"t.py": self.INLINE}, target="t.py",
        ) == frozenset()

    def test_the_assigned_form_proves_the_same_contradiction(self):
        assigned = self.INLINE.replace(
            'assert normalise("   ") == ""',
            'value = normalise("   ")\n    assert value == ""',
        )
        assert proven_unsatisfiable_tests(
            assigned, filename="t.py"
        ) == frozenset({"test_raises", "test_succeeds"})


class TestTheCrossFileReason:
    def test_it_describes_the_pair_and_names_both_files(self):
        reason = machine_unsatisfiable_reason_across(
            FILES, target=TARGET_REL,
        )
        assert reason is not None
        assert TARGET_REL in reason and SIBLING_REL in reason
        assert "no production change can satisfy both" in reason

    def test_it_is_none_for_a_file_in_no_pair(self):
        assert machine_unsatisfiable_reason_across(
            FILES, target="server/tests/test_other.py",
        ) is None

    def test_a_single_file_batch_proves_nothing_across(self):
        assert machine_unsatisfiable_reason_across(
            {TARGET_REL: TARGET}, target=TARGET_REL,
        ) is None


class TestTheGateNowAcceptsTheFix:
    """The decisive class. Same old source, same rewrite — the only
    difference is whether the cross-file proof reached the gate."""

    def test_the_proof_sanctions_both_removals(self):
        names = proven_unsatisfiable_tests_across(FILES, target=TARGET_REL)
        ok, detail = coverage_nonregression_ok(
            TARGET, REWRITE, sanctioned_removals=names,
        )
        assert ok, detail
        assert "2 proof-sanctioned removal(s) discounted" in detail

    def test_the_tolerance_is_sanctioned_plus_one_not_sanctioned_times_two(
        self,
    ):
        """The exact boundary, because I guessed it wrong first.

        The floor is ``max(1, of - 1 - sanctioned)``, so a proof naming two
        removals licenses THREE: its two, plus the pre-existing
        one-function tolerance for the redundant half of an unproven pair.
        That tolerance predates any proof and is unchanged by one, so this
        is not a hole the sanctioning opened — but a reader comparing 4->1
        against "two were sanctioned" deserves to find it written down.
        """
        names = proven_unsatisfiable_tests_across(FILES, target=TARGET_REL)
        assert len(names) == 2
        one_left = '''# @tests: app/m.py
from app.m import normalise


def test_it_runs():
    assert normalise(" a ") == "a"
'''
        ok, _ = coverage_nonregression_ok(
            TARGET, one_left, sanctioned_removals=names,
        )
        assert ok, "2 sanctioned + 1 tolerated is exactly the floor"
        ok_none, _ = coverage_nonregression_ok(
            TARGET, one_left, sanctioned_removals=frozenset(),
        )
        assert not ok_none, "and without the proof it is well below it"

    def test_a_weakened_rewrite_is_still_refused(self):
        """The anti-reward-hack half. The assertion floor is measured on
        what lived OUTSIDE the sanctioned removals, so a proof buys no
        licence to hollow out the tests that remain."""
        rich = TARGET.replace(
            'def test_trims_whitespace():\n    assert normalise(" a ") == "a"',
            'def test_trims_whitespace():\n'
            '    assert normalise(" a ") == "a"\n'
            '    assert normalise("b ") == "b"\n'
            '    assert normalise(" c") == "c"\n'
            '    assert normalise("  d  ") == "d"',
        )
        hollow = '''# @tests: app/m.py
from app.m import normalise


def test_trims_whitespace():
    assert normalise(" a ") == "a"


def test_keeps_case():
    assert normalise("Ab") == "Ab"
'''
        names = proven_unsatisfiable_tests_across(
            {TARGET_REL: rich, SIBLING_REL: SIBLING}, target=TARGET_REL,
        )
        assert len(names) == 2
        ok, detail = coverage_nonregression_ok(
            rich, hollow, sanctioned_removals=names,
        )
        assert not ok, detail
        assert "weakened" in detail

    def test_a_proof_cannot_empty_the_file(self):
        """``max(1, ...)`` floor: even with every function sanctioned, a
        rewrite with no tests at all is not a fix."""
        names = frozenset(
            {"test_blank_is_rejected", "test_dash_is_rejected",
             "test_trims_whitespace", "test_keeps_case"}
        )
        ok, _ = coverage_nonregression_ok(
            TARGET, "# @tests: app/m.py\n", sanctioned_removals=names,
        )
        assert not ok

    def test_keeping_and_correcting_is_unaffected(self):
        """The ideal fix keeps both tests and corrects the assertion. A
        proof-named function the rewrite KEPT is not discounted, so this
        path is identical with and without the proof."""
        corrected = TARGET.replace(
            'with pytest.raises(ValueError):\n        normalise("   ")',
            'assert normalise("   ") == ""',
        )
        names = proven_unsatisfiable_tests_across(FILES, target=TARGET_REL)
        with_proof = coverage_nonregression_ok(
            TARGET, corrected, sanctioned_removals=names,
        )
        without = coverage_nonregression_ok(TARGET, corrected)
        assert with_proof[0] and without[0]


class TestTheBatchSourceGathering:
    """``_batch_test_sources`` — scoped to the batch's declared test files,
    deliberately not a workspace walk."""

    def _ws(self, tmp_path):
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / TARGET_REL).write_text(TARGET)
        (tmp_path / SIBLING_REL).write_text(SIBLING)
        return str(tmp_path)

    def test_it_gathers_the_siblings_the_batch_declared(self, tmp_path):
        import harness.graph as g

        ws = self._ws(tmp_path)
        out = g._batch_test_sources(
            {"workspace_path": ws, "generated_tests": [TARGET_REL, SIBLING_REL]},
            TARGET_REL, TARGET,
        )
        assert set(out) == {TARGET_REL, SIBLING_REL}

    def test_it_also_reads_test_files_from_modified_files(self, tmp_path):
        import harness.graph as g

        ws = self._ws(tmp_path)
        out = g._batch_test_sources(
            {"workspace_path": ws, "modified_files": [SIBLING_REL, "app/m.py"]},
            TARGET_REL, TARGET,
        )
        assert set(out) == {TARGET_REL, SIBLING_REL}, (
            "production files must not be scanned as tests"
        )

    def test_the_declared_source_is_used_not_re_read(self, tmp_path):
        """It is already in hand, and re-reading would race a rollback."""
        import harness.graph as g

        ws = self._ws(tmp_path)
        out = g._batch_test_sources(
            {"workspace_path": ws, "generated_tests": [TARGET_REL]},
            TARGET_REL, "# the in-memory source\n",
        )
        assert out[TARGET_REL] == "# the in-memory source\n"

    def test_an_unreadable_sibling_is_skipped(self, tmp_path):
        """A prover missing a file reaches a WEAKER conclusion, which is the
        safe direction — it must not take the router down."""
        import harness.graph as g

        ws = self._ws(tmp_path)
        out = g._batch_test_sources(
            {"workspace_path": ws,
             "generated_tests": [SIBLING_REL, "server/tests/test_gone.py"]},
            TARGET_REL, TARGET,
        )
        assert set(out) == {TARGET_REL, SIBLING_REL}

    def test_it_does_not_walk_the_workspace(self, tmp_path):
        """An undeclared test file on disk is NOT pulled in: routing cost
        must not grow with the workspace, and tests from unrelated stories
        must not be paired."""
        import harness.graph as g

        ws = self._ws(tmp_path)
        (tmp_path / "server" / "tests" / "test_stranger.py").write_text(
            SIBLING
        )
        out = g._batch_test_sources(
            {"workspace_path": ws, "generated_tests": [TARGET_REL]},
            TARGET_REL, TARGET,
        )
        assert set(out) == {TARGET_REL}

    def test_an_absolute_path_is_read(self, tmp_path):
        import harness.graph as g

        ws = self._ws(tmp_path)
        out = g._batch_test_sources(
            {"workspace_path": ws,
             "generated_tests": [os.path.join(ws, SIBLING_REL)]},
            TARGET_REL, TARGET,
        )
        assert len(out) == 2


class TestTheRouterReachesRegenerationOnTheASTAlone:
    """Before this, a cross-file pair reached regeneration only on the
    judge's behavioural ``oscillation_contradiction`` verdict — which means
    repair had already oscillated and paid for it."""

    def _state(self, tmp_path):
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / TARGET_REL).write_text(TARGET)
        (tmp_path / SIBLING_REL).write_text(SIBLING)
        return {
            "workspace_path": str(tmp_path),
            "generated_tests": [TARGET_REL, SIBLING_REL],
            "node_state": {"unsatisfiable_test": TARGET_REL},
            "loop_counter": {},
            "test_regeneration_config": {
                "enabled": True, "tier_b_auto": False,
                "max_attempts_per_test": 2,
            },
        }

    def test_it_routes_to_regeneration(self, tmp_path):
        import harness.graph as g

        state = self._state(tmp_path)
        assert g.route_after_unsatisfiable(state, TARGET_REL) == (
            "test_regeneration_node"
        )

    def test_it_seeds_the_proof_set_the_coverage_gate_reads(self, tmp_path):
        """The whole point: without this the rewrite is rejected."""
        import harness.graph as g

        state = self._state(tmp_path)
        g.route_after_unsatisfiable(state, TARGET_REL)
        assert state["node_state"]["unsatisfiable_proof_tests"] == [
            "test_blank_is_rejected", "test_dash_is_rejected",
        ]

    def test_without_the_sibling_declared_it_still_goes_to_hitl(
        self, tmp_path,
    ):
        """tier_b_auto off and nothing provable from one file — the pre-fix
        behaviour, unchanged. This is the vacuity check on the test above."""
        import harness.graph as g

        state = self._state(tmp_path)
        state["generated_tests"] = [TARGET_REL]
        assert g.route_after_unsatisfiable(state, TARGET_REL) == (
            "human_intervention_node"
        )

    def test_a_consistent_batch_is_not_routed_to_regeneration(self, tmp_path):
        import harness.graph as g

        state = self._state(tmp_path)
        (tmp_path / SIBLING_REL).write_text(
            "# @tests: app/svc.py\ndef test_ok():\n    assert True\n"
        )
        assert g.route_after_unsatisfiable(state, TARGET_REL) == (
            "human_intervention_node"
        )

    def test_the_attempt_cap_still_wins(self, tmp_path):
        """A provable defect does not buy unlimited attempts."""
        import harness.graph as g

        state = self._state(tmp_path)
        state["loop_counter"] = {"test_regen_attempts": {TARGET_REL: 2}}
        assert g.route_after_unsatisfiable(state, TARGET_REL) == (
            "human_intervention_node"
        )

    @pytest.mark.parametrize("flag", ["test_regen_stuck"])
    def test_the_earlier_guards_still_win(self, tmp_path, flag):
        import harness.graph as g

        state = self._state(tmp_path)
        state["node_state"][flag] = True
        assert g.route_after_unsatisfiable(state, TARGET_REL) == (
            "human_intervention_node"
        )
