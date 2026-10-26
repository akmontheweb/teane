"""A proof must outrank the anti-reward-hack heuristic.

Regression cover for lumina 01a17704 (run 9), which died with 0/16 while
the answer sat in its own log:

    [router] Test server/tests/test_birthday_service.py is PROVABLY
             unsatisfiable (...) — routing to test-author regeneration
    [test_regeneration_node] coverage_regression: test-function count
             dropped 18->16 (>1 removed) — deferring to ladder      x2

The router PROVED the test false by AST, regeneration produced the
corrected suite, and ``coverage_nonregression_ok`` threw it away for
dropping two functions instead of one. Both attempts went that way.

The gate's one-function tolerance encoded an assumption that falsity always
arrives as a contradictory PAIR -- which is merely the only shape
``find_contradictions`` can currently see. The gate exists to stop the MODEL
trading coverage for a green build; a deterministic proof is not the model's
opinion, so the removals it names are sanctioned.
"""

from __future__ import annotations

from harness.test_contradiction import (
    machine_unsatisfiable_reason,
    proven_unsatisfiable_tests,
)
from harness.test_regeneration import coverage_nonregression_ok

#: Two provably contradictory tests plus three uninvolved ones. The extra
#: tests are the point: they are what the floor must still protect.
MIXED = '''
import pytest
from app.models import ContactUpdate

def test_none_fields_allowed():
    obj = ContactUpdate(first_name=None)

def test_all_none_raises():
    with pytest.raises(ValidationError):
        ContactUpdate(first_name=None)

def test_innocent_one():
    assert ContactUpdate(first_name="a").first_name == "a"
    assert ContactUpdate(last_name="b").last_name == "b"

def test_innocent_two():
    assert ContactUpdate(email="e@x.com").email == "e@x.com"

def test_innocent_three():
    assert ContactUpdate(phone="1").phone == "1"
'''

#: The correct fix: drop exactly the two proven-false tests.
DROP_ONLY_PROVEN = '''
def test_innocent_one():
    assert ContactUpdate(first_name="a").first_name == "a"
    assert ContactUpdate(last_name="b").last_name == "b"

def test_innocent_two():
    assert ContactUpdate(email="e@x.com").email == "e@x.com"

def test_innocent_three():
    assert ContactUpdate(phone="1").phone == "1"
'''

#: Reward-hacking: delete the lot and leave something vacuous.
GUT_EVERYTHING = '''
def test_nothing():
    assert True
'''


class TestTheProver:
    def test_it_names_both_sides(self):
        assert proven_unsatisfiable_tests(MIXED, filename="t.py") == frozenset(
            {"test_none_fields_allowed", "test_all_none_raises"}
        )

    def test_it_agrees_with_the_reason_string(self):
        """The names and the prose must come from the same proof, or the gate
        would discount removals the router never sanctioned."""
        assert machine_unsatisfiable_reason(MIXED, filename="t.py") is not None
        assert proven_unsatisfiable_tests(MIXED, filename="t.py")

    def test_nothing_provable_means_no_names(self):
        clean = "def test_ok():\n    assert 1 == 1\n"
        assert proven_unsatisfiable_tests(clean, filename="t.py") == frozenset()

    def test_a_non_python_path_abstains(self):
        """A .ts suite is not this detector's business; abstaining must never
        read as 'delete freely'."""
        assert proven_unsatisfiable_tests(MIXED, filename="t.spec.ts") == (
            frozenset()
        )

    def test_unparseable_source_names_nothing(self):
        assert proven_unsatisfiable_tests("def (", filename="t.py") == frozenset()


class TestTheGateHonoursAProof:
    def test_run_9s_veto_is_lifted(self):
        """Dropping exactly the proven-false tests is now accepted."""
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        ok, detail = coverage_nonregression_ok(
            MIXED, DROP_ONLY_PROVEN, sanctioned_removals=proof,
        )
        assert ok, detail
        assert "proof-sanctioned" in detail

    def test_without_the_proof_it_is_still_vetoed(self):
        """Vacuity check: the SAME rewrite must fail with no proof in hand,
        or this test would pass against the broken code."""
        ok, detail = coverage_nonregression_ok(MIXED, DROP_ONLY_PROVEN)
        assert not ok, (
            "the rewrite must be refused without a proof — otherwise the "
            "anti-reward-hack floor is gone, not fixed"
        )

    def test_a_proof_licenses_only_what_it_names(self):
        """The whole safety argument. Deleting uninvolved tests is refused
        even with a proof in hand."""
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        ok, detail = coverage_nonregression_ok(
            MIXED, GUT_EVERYTHING, sanctioned_removals=proof,
        )
        assert not ok, detail
        assert "dropped 5->1" in detail

    def test_a_proof_name_the_rewrite_KEPT_is_not_discounted(self):
        """Keeping and correcting both halves is the ideal fix; it must not
        silently earn a coverage discount it does not need."""
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        kept = MIXED.replace(
            "    with pytest.raises(ValidationError):\n"
            "        ContactUpdate(first_name=None)",
            "    assert ContactUpdate(first_name=None).first_name is None",
        )
        ok, detail = coverage_nonregression_ok(
            MIXED, kept, sanctioned_removals=proof,
        )
        assert ok, detail
        assert "proof-sanctioned" not in detail, (
            "nothing was removed, so nothing should be discounted"
        )

    def test_an_empty_rewrite_is_refused_whatever_the_proof_says(self):
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        ok, _ = coverage_nonregression_ok(
            MIXED, "", sanctioned_removals=proof,
        )
        assert not ok

    def test_an_assertionless_rewrite_is_refused(self):
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        ok, detail = coverage_nonregression_ok(
            MIXED, "def test_x():\n    pass\n", sanctioned_removals=proof,
        )
        assert not ok
        assert "no assertions" in detail or "dropped" in detail

    def test_no_sanctioned_set_behaves_exactly_as_before(self):
        """Backwards compatibility: the default must be the old behaviour."""
        a = coverage_nonregression_ok(MIXED, DROP_ONLY_PROVEN)
        b = coverage_nonregression_ok(
            MIXED, DROP_ONLY_PROVEN, sanctioned_removals=None,
        )
        assert a == b

    def test_the_assertion_floor_is_discounted_too(self):
        """Removing a proven-false test removes its assertions, so the
        >50%-weakened floor has to discount them as well -- otherwise the
        gate vetoes the sanctioned removal via the assertion check instead
        of the function check, which is where this fix first fell short."""
        proof = proven_unsatisfiable_tests(MIXED, filename="t.py")
        ok, detail = coverage_nonregression_ok(
            MIXED, DROP_ONLY_PROVEN, sanctioned_removals=proof,
        )
        assert ok, f"assertion floor still vetoing: {detail}"


class TestTheWiring:
    def test_the_router_carries_the_proof_forward(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert "proven_unsatisfiable_tests" in src
        assert '"unsatisfiable_proof_tests"' in src

    def test_the_node_reads_it_and_passes_it_to_the_gate(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        assert 'node_state.get("unsatisfiable_proof_tests")' in src
        assert "sanctioned_removals=proof_tests" in src

    def test_a_harness_judgement_refunds_but_a_model_error_does_not(self):
        """The line drawn after the existing suite pushed back.

        ``test_node_rejects_stray_file`` asserts that aiming at another file
        DOES spend an attempt, and that is right: the model produced
        something invalid, so it is a fix that was tried and failed. A
        refund is for the other case -- the HARNESS refusing output that may
        well have been correct, which is exactly what the coverage gate did
        to a proven fix in run 9.
        """
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()

        at = src.index('_give_up("coverage_regression"')
        assert "_refund_attempt(" in src[max(0, at - 700):at], (
            "the coverage gate must refund — it refused a PROVEN fix twice "
            "in run 9 and spent the file's whole budget doing it"
        )

        for model_error in ("targeted_other_files", "no_code_linkage"):
            at = src.index(f'_give_up("{model_error}"')
            assert "_refund_attempt(" not in src[max(0, at - 400):at], (
                f"{model_error} is the model's own error; charging it is "
                "correct and a tested contract"
            )
