"""A confident judge must not divert a production defect into the test ladder.

The escape ladder's rungs are evidence-based — an AST contradiction, a
cross-file proof, an unparseable file. The judge's recommendation is the one
rung that is PROSE: ``_verdict_demands_test_edit`` regex-matches the verdict's
wording for "edit <test file>", and the only corroboration is
``_file_in_failing_set``, which asks whether that file is still failing. That
is a staleness check, not a correctness one.

So a confidently-worded verdict could route a real code gap to the
test-author ladder, whose only tool is rewriting a test that was already
right. lumina run 11 died of exactly this. Both terminal blockers were
production defects the judge reported as test edits:

  * ``STORY-008.AC-2`` — ``AssertionError: {"detail":"Method Not Allowed"}``.
    The PATCH endpoint did not exist; that body is Starlette's own 405.
  * ``STORY-006.AC-3`` — asserted 404 for another owner's record and got 200
    with the record. An authorization hole: the handler's docstring promised
    owner scoping the code never implemented.

Regeneration rewrote both tests, correctly, from their criteria — so they kept
demanding the right behaviour, kept failing, and charged ``unsatisfiable_test``
until the class cap terminated the run at 4/16 with $7 of $10 unspent. Repair,
the one component that could have closed either gap, was diverted away from
them every round.

Two halves here: the screen that refuses the diversion
(:class:`TestTheScreen`), and the judge-prompt clauses that stop the verdict
being produced in the first place (:class:`TestTheJudgeIsWarned`).
"""

from __future__ import annotations

import pytest

import harness.graph as g

TEST_FILE = "tests/acceptance/test_story_008_acceptance.py"
WS = "/ws"


def _state(message: str, *, file: str = TEST_FILE, **extra) -> dict:
    return {
        "workspace_path": WS,
        "compiler_errors": [
            {"file": file, "line": 56, "message": message, **extra},
        ],
    }


class TestTheScreen:
    """``_code_gap_evidence`` — executed against run 11's real messages."""

    def test_a_framework_405_body_is_a_code_gap(self):
        """STORY-008.AC-2, verbatim. Starlette emits this when no handler is
        registered, so the route does not exist and no test edit can change
        that."""
        why = g._code_gap_evidence(
            _state('AssertionError: {"detail":"Method Not Allowed"}'),
            TEST_FILE,
        )
        assert why
        assert "route does not exist" in why

    def test_a_framework_404_body_is_a_code_gap(self):
        why = g._code_gap_evidence(
            _state('AssertionError: {"detail":"Not Found"}'), TEST_FILE,
        )
        assert why

    def test_success_where_rejection_was_required_is_a_code_gap(self):
        """STORY-006.AC-3's shape: the app returned 200 where the criterion
        requires 404. Editing the test would delete the requirement — here, a
        data-leak check."""
        why = g._code_gap_evidence(
            _state("assert 200 == 404"), TEST_FILE,
        )
        assert why
        assert "delete the requirement" in why

    @pytest.mark.parametrize("msg", [
        "assert 200 == 404", "assert 201 == 400", "assert 204 == 403",
        "assert 200 == 422",
    ])
    def test_every_success_versus_rejection_pair_is_caught(self, msg):
        assert g._code_gap_evidence(_state(msg), TEST_FILE)

    def test_a_not_implemented_error_is_a_code_gap(self):
        why = g._code_gap_evidence(
            _state("NotImplementedError"), TEST_FILE,
        )
        assert "stub" in why

    def test_the_evidence_is_read_from_semantic_context_too(self):
        """The message is often a bare assertion; the shape frequently lands
        in the enriched context instead."""
        assert g._code_gap_evidence(
            _state(
                "AssertionError",
                semantic_context='response was {"detail": "Not Found"}',
            ),
            TEST_FILE,
        )


class TestWhatItMustNotCatch:
    """A screen that fires on an ordinary wrong test would disable the whole
    ladder — the thing that actually saves runs when a test IS false."""

    def test_an_ordinary_wrong_expectation_is_not_a_code_gap(self):
        assert g._code_gap_evidence(
            _state("assert 2 == 1"), TEST_FILE,
        ) == ""

    def test_a_rejection_the_app_correctly_made_is_not_a_code_gap(self):
        """The reverse direction: the test wanted success and got a 4xx.
        That may well be a code gap, but it is NOT the signature here —
        the test could equally be sending a malformed request. Left to the
        ladder."""
        assert g._code_gap_evidence(
            _state("assert 404 == 200"), TEST_FILE,
        ) == ""

    def test_a_contradictory_label_assertion_is_not_a_code_gap(self):
        """The dominant false-test shape across the lumina runs — a
        computed label the test gets wrong. The ladder must still reach it."""
        assert g._code_gap_evidence(
            _state("AssertionError: assert 'In 2 days' == 'This Saturday'"),
            TEST_FILE,
        ) == ""

    def test_a_literal_404_in_prose_is_not_enough(self):
        """Only the framework's own BODY counts, not the number 404 appearing
        somewhere in a message."""
        assert g._code_gap_evidence(
            _state("AssertionError: expected 404 NOT_FOUND code"), TEST_FILE,
        ) == ""

    def test_evidence_from_a_different_file_does_not_convict(self):
        state = _state(
            'AssertionError: {"detail":"Method Not Allowed"}',
            file="tests/acceptance/test_other_acceptance.py",
        )
        assert g._code_gap_evidence(state, TEST_FILE) == ""

    def test_no_diagnostics_is_no_evidence(self):
        assert g._code_gap_evidence(
            {"workspace_path": WS, "compiler_errors": []}, TEST_FILE,
        ) == ""

    def test_malformed_diagnostics_are_survivable(self):
        state = {
            "workspace_path": WS,
            "compiler_errors": ["not a dict", None, {}],
        }
        assert g._code_gap_evidence(state, TEST_FILE) == ""


class TestTheRouterRefusesTheDiversion:
    """The screen's consumer: the round must go to repair, not the ladder."""

    def _src(self) -> str:
        import inspect

        return inspect.getsource(g.route_after_compiler) if hasattr(
            g, "route_after_compiler"
        ) else ""

    def test_the_screen_runs_before_the_failing_set_check(self):
        """Order matters: a code gap must lose the diversion even when the
        named test IS still failing — which it always is."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        screen = src.index("_code_gap = (")
        honour = src.index(
            '_blocked_reason = "the judge\'s own recommendation is a test edit"'
        )
        assert screen < honour

    def test_a_code_gap_sets_no_blocked_reason(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        at = src.index("if _code_gap:")
        # End the window AT the next branch: the elif legitimately sets
        # _blocked_reason, and a window that runs into it proves nothing.
        end = src.index("elif _verdict_file and _file_in_failing_set", at)
        window = src[at:end]
        assert "_blocked_reason" not in window, (
            "the code-gap branch must not divert at all"
        )
        assert "Routing the round to " in window

    def test_the_refusal_is_logged_and_emitted(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        assert "router_test_edit_refused_code_gap" in src
        # A contiguous fragment: the full sentence spans a string-literal
        # break in the source, so asserting the whole phrase tests the line
        # wrapping rather than the message.
        assert "evidence says the defect is in PRODUCTION code" in src


class TestTheJudgeIsWarned:
    """Prevention: the verdict should not be produced in the first place.

    The prompt already states that a list-order difference is not evidence a
    test is wrong — the same house pattern of naming the shapes that do not
    convict a test. Run 11's two shapes were missing from it.
    """

    def _prompt(self) -> str:
        return g._build_repair_reflection_prompt(
            prior_diagnostics_count=3,
            current_diagnostics_count=3,
            resolved_fingerprints=[],
            persisted_fingerprints=["f1"],
            new_fingerprints=[],
            top_persisted_diagnostics=[{
                "file": TEST_FILE, "line": 56,
                "message": 'AssertionError: {"detail":"Method Not Allowed"}',
                "error_code": "PYTEST_ASSERT",
            }],
        )

    def test_the_framework_default_body_clause_is_present(self):
        p = self._prompt()
        assert "FRAMEWORK DEFAULT ERROR BODY IS NOT EVIDENCE" in p
        assert "Method Not Allowed" in p
        assert "endpoint does not exist yet" in p

    def test_the_success_versus_rejection_clause_is_present(self):
        p = self._prompt()
        assert "SUCCEEDING WHERE THE TEST EXPECTED A " in p
        assert "ownership or permission scoping" in p

    def test_both_clauses_tell_it_to_name_an_impl_module(self):
        """The hint's whole mechanism is ``impl_file``; a warning that does
        not redirect leaves the judge with nowhere to go."""
        p = self._prompt()
        at = p.index("FRAMEWORK DEFAULT ERROR BODY")
        window = p[at:at + 1800]
        assert "impl_file" in window or "Name the module" in window

    def test_the_existing_list_order_clause_still_stands(self):
        """Guards against clobbering the sibling rule these were modelled on."""
        p = self._prompt()
        assert "DIFFERENCE IN LIST ORDER" in p
