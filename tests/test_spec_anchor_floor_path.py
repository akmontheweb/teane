"""The spec anchor must fire on the path the runs actually take.

9c4eec2 flipped the rebuild anchor from the CODE to the SPEC when a test
contradicts the SRS -- and shipped dormant. ``reason_cites_spec_conflict``
needs a MODEL declaration whose prose blames a spec rule, and lumina run 10
produced **zero** model declarations: the floor took every escape after three
declined offers, and the floor synthesises its reason from exactly three
fixed labels (harness/graph.py, ``_blocked_reason``):

    "contradictory test expectations"
    "suite-order pollution"
    "the judge's own recommendation is a test edit"

None cites a spec, so the predicate correctly returned False every time and
every rebuild stayed code-anchored -- against code that had been written to
satisfy the very test being replaced. The floor path dominated runs 5, 8, 9
and 10.

The decisive case here is
:meth:`TestAProofIsEnoughOnItsOwn.test_a_proof_flips_the_anchor_with_no_prose`:
a proven contradiction demotes the code without any prose at all, because the
module satisfies at most ONE side of the pair and so cannot arbitrate.
"""

from __future__ import annotations

import pathlib

from harness.test_regeneration import (
    reason_cites_spec_conflict,
    spec_outranks_the_code,
)

#: Verbatim from harness/graph.py — the only reasons the floor ever mints.
FLOOR_REASONS = (
    "contradictory test expectations",
    "suite-order pollution",
    "the judge's own recommendation is a test edit",
)

SPEC_DECLARATION = (
    "the test asserts an unconditional 400 TIMEZONE_REQUIRED but ASM-009 in "
    "the SRS resolves a stored device-profile timezone first"
)

#: Run 10's actual judge verdict. A code gap, not a spec conflict.
JUDGE_CODE_GAP = (
    "recommendation: Edit server/tests/test_birthday_service.py:43 to call "
    "_effective_month_day with the required reference_year argument"
)

JUDGE_SPEC_CONFLICT = (
    "real blocker: test_quick_action_service.py:41 demands wa.me/+15551234567 "
    "but the SRS mandates the message URI without a leading +"
)


class TestTheRegressionItself:
    def test_no_floor_reason_cites_a_spec(self):
        """The root cause, pinned. If a future floor label happens to contain
        a spec marker this test fails and the fix below is redundant for it
        -- which is worth knowing either way."""
        for reason in FLOOR_REASONS:
            assert not reason_cites_spec_conflict(reason), reason

    def test_the_floor_reasons_are_still_what_the_router_mints(self):
        """Guards the premise. If graph.py stops minting these, the shape of
        this fix needs revisiting rather than silently covering nothing."""
        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        for reason in FLOOR_REASONS:
            assert f'_blocked_reason = "{reason}"' in src, reason

    def test_the_non_proof_floor_reasons_do_not_flip_on_their_own(self):
        """Without a proof or a judge verdict there is no evidence of a spec
        conflict, and inventing one would demote the code contract on every
        ordinary repair.

        "suite-order pollution" is an isolation defect, not a wrong
        expectation; the judge label carries no verdict of its own.
        """
        for reason in (
            "suite-order pollution",
            "the judge's own recommendation is a test edit",
        ):
            flip, why = spec_outranks_the_code(reason=reason)
            assert not flip, f"{reason} -> {why}"


class TestACrossFileContradictionIsProofToo:
    """Run 10's shape, and the gap that nearly made this fix dormant twice.

    ``proven_unsatisfiable_tests`` proves contradictions WITHIN one file. A
    cross-file pair -- ASM-007's directory order, where a sibling test in
    another file demanded the opposite sort -- cannot be proven that way, so
    the proof set comes back EMPTY for precisely the case that dominated the
    runs. The router's own label is the evidence there: it mints
    "contradictory test expectations" only when its deterministic detector
    found the pair.
    """

    def test_the_label_flips_the_anchor_with_an_empty_proof_set(self):
        flip, why = spec_outranks_the_code(
            reason="contradictory test expectations", proof_tests=(),
        )
        assert flip, why
        assert "arbitrate" in why

    def test_it_is_matched_case_insensitively(self):
        flip, _ = spec_outranks_the_code(
            reason="Contradictory Test Expectations",
        )
        assert flip

    def test_it_is_matched_inside_a_longer_reason(self):
        """The router capitalises and embeds it in prose before logging."""
        flip, _ = spec_outranks_the_code(
            reason="blocked on contradictory test expectations in test_x.py",
        )
        assert flip

    def test_suite_order_pollution_is_not_treated_as_a_contradiction(self):
        """The distinction the fix turns on. Flipping here would demote the
        module for a fixture bug and lose the contract the rewrite must
        compile against."""
        flip, why = spec_outranks_the_code(reason="suite-order pollution")
        assert not flip, why


class TestAProofIsEnoughOnItsOwn:
    def test_a_proof_flips_the_anchor_with_no_prose(self):
        """The whole point of the fix.

        A proven contradiction means two tests demand opposite outcomes for
        the identical call. Production satisfies at most one side, so the
        module was shaped by whichever side it happens to satisfy --
        rebuilding from it reproduces that side with a 50/50 chance of being
        the false one. Only the spec can arbitrate.
        """
        flip, why = spec_outranks_the_code(
            reason="contradictory test expectations",
            proof_tests=frozenset({"test_all_none_raises",
                                   "test_none_fields_allowed"}),
        )
        assert flip
        assert "proof" in why

    def test_the_reason_why_counts_the_proven_tests(self):
        """A post-mortem has to be able to tell a proof-driven rebuild from a
        prose-driven one; "spec leads" with no provenance was the state that
        made run 10 unreadable."""
        _, why = spec_outranks_the_code(
            reason="", proof_tests=("a", "b", "c"),
        )
        assert "3" in why

    def test_a_proof_outranks_an_empty_reason(self):
        flip, _ = spec_outranks_the_code(reason="", proof_tests=("x",))
        assert flip

    def test_no_proof_and_no_prose_does_not_flip(self):
        flip, why = spec_outranks_the_code(reason="", proof_tests=())
        assert not flip
        assert why == ""


class TestTheJudgeIsEvidenceToo:
    def test_a_judge_verdict_blaming_the_spec_flips(self):
        flip, why = spec_outranks_the_code(
            reason="the judge's own recommendation is a test edit",
            judge_diagnosis=JUDGE_SPEC_CONFLICT,
        )
        assert flip
        assert "judge" in why

    def test_a_judge_verdict_naming_a_code_gap_does_not(self):
        """Run 10's real verdict. A missing argument is a CODE defect; the
        module is the right anchor for it, and flipping would throw away the
        contract the rewrite has to compile against."""
        flip, why = spec_outranks_the_code(
            reason="the judge's own recommendation is a test edit",
            judge_diagnosis=JUDGE_CODE_GAP,
        )
        assert not flip, why

    def test_an_empty_judge_diagnosis_is_harmless(self):
        flip, _ = spec_outranks_the_code(reason="", judge_diagnosis="")
        assert not flip


class TestTheDeclarationPathIsUnchanged:
    def test_a_model_declaration_still_flips(self):
        flip, why = spec_outranks_the_code(reason=SPEC_DECLARATION)
        assert flip
        assert "declaration" in why

    def test_the_proof_is_reported_ahead_of_the_declaration(self):
        """Both present: the proof is the stronger evidence and should be
        what the log names."""
        _, why = spec_outranks_the_code(
            reason=SPEC_DECLARATION, proof_tests=("t",),
        )
        assert "proof" in why


class TestTheWiring:
    def _src(self) -> str:
        return (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()

    def test_the_node_uses_the_widened_decision(self):
        src = self._src()
        assert "spec_outranks_the_code(" in src
        assert "proof_tests=proof_tests" in src

    def test_the_node_feeds_it_the_judge_diagnosis(self):
        src = self._src()
        assert "judge_diagnosis=_judge_dx" in src

    def test_the_judge_diagnosis_is_computed_once(self):
        """It was computed inline at the prompt call; the anchor decision
        needs it earlier, and computing it twice invites the two call sites
        to drift apart."""
        src = self._src()
        assert src.count("_judge_diagnosis(state, rel)") == 1

    def test_the_log_names_the_evidence_that_moved_it(self):
        src = self._src()
        assert "the SPECIFICATION leads this rewrite" in src
        assert "rel, _spec_why" in src

    def test_a_flip_with_no_spec_in_context_is_logged_not_silent(self):
        """spec_authoritative requires BOTH the flip and a spec to anchor on.
        When the evidence says flip but no spec is loaded, the rebuild falls
        back to the code -- which is the right action and the wrong thing to
        do quietly."""
        src = self._src()
        assert "no specification is in " in src
        assert "_spec_flip and not spec_tiebreaker.strip()" in src
