"""A unit test that contradicts the SRS must be rebuilt from the SRS.

Unit-test regeneration anchors on the CODE by design -- the 1:1 ``# @tests:``
traceability model -- and the prompt labelled the spec
``TIEBREAKER ONLY — do not cite in the test``, last and truncated to 4k
while the code led with 12k.

That is right for a test disagreeing with its module, and exactly wrong for
a test disagreeing with the SRS: the production code was written to satisfy
the very test being replaced, so rebuilding from that code reproduces the
falsehood with more confidence, the code now being the stated source of
truth.

Four of the six false tests seen across lumina runs 5-9 were unit tests
contradicting the SRS, every one rebuilt from code. The two ACCEPTANCE cases
were rebuilt from their criteria and came back correct -- run 8's envelope
test is the worked example:

    assert body["error"].get("message"), body
    assert body["error"].get("error_ref_id"), body
"""

from __future__ import annotations

from harness.test_regeneration import (
    build_regeneration_messages,
    reason_cites_spec_conflict,
)

#: Verbatim-shaped declarations from the runs.
SPEC_CONFLICTS = {
    "asm-009": (
        "the test asserts an unconditional 400 TIMEZONE_REQUIRED but ASM-009 "
        "in the SRS resolves a stored device-profile timezone first"
    ),
    "wa.me": (
        "the test asserts `wa.me/+15551234567` with a leading `+`, but "
        "STORY-003's acceptance criterion in the spec says the message URI "
        "is `wa.me/15551234567`"
    ),
    "envelope": (
        "the test asserts body['error'] is a string, but the SRS mandates a "
        "JSON object with code/message/error_ref_id"
    ),
}

NOT_SPEC_CONFLICTS = {
    "undefined-name": "NameError: name 'resolve_caller_timezone' is not defined",
    "domain-arithmetic": (
        "the test asserts len(result) == 2 but only one seeded birthday falls "
        "in the 30-day window"
    ),
    "internal-rule": (
        "`_relative_label` must return 'This Saturday' for +3d and 'In 2 "
        "days' for +2d — no single rule satisfies both"
    ),
    "empty": "",
}


def _build(reason: str, authoritative: bool):
    return build_regeneration_messages(
        test_rel_path="server/tests/test_api_birthdays.py",
        test_source="OLD_TEST_BODY",
        code_module_path="server/app/api/birthdays.py",
        code_module_source="CODE_BODY",
        module_symbols=["list_birthdays"],
        unsat_reason=reason,
        failing_output="E assert 400 == 200",
        spec_tiebreaker="SRS_BODY",
        spec_authoritative=authoritative,
    )[-1]["content"]


class TestThePredicate:
    def test_every_real_spec_conflict_is_detected(self):
        for name, reason in SPEC_CONFLICTS.items():
            assert reason_cites_spec_conflict(reason), name

    def test_a_plain_code_gap_is_not(self):
        """A false positive would promote the spec over the code contract for
        the COMMON case, which is the wrong anchor for an ordinary repair."""
        for name, reason in NOT_SPEC_CONFLICTS.items():
            assert not reason_cites_spec_conflict(reason), name

    def test_it_is_case_insensitive(self):
        assert reason_cites_spec_conflict("The SRS Mandates an object")
        assert reason_cites_spec_conflict("contradicts the srs")

    def test_it_tolerates_none(self):
        assert not reason_cites_spec_conflict(None)  # type: ignore[arg-type]


class TestTheSpecLeadsWhenAuthoritative:
    def test_the_spec_comes_before_the_code(self):
        body = _build(SPEC_CONFLICTS["asm-009"], authoritative=True)
        assert body.index("SRS_BODY") < body.index("CODE_BODY")

    def test_precedence_is_stated_outright(self):
        body = _build(SPEC_CONFLICTS["asm-009"], authoritative=True)
        assert "specification wins" in body
        assert "AUTHORITATIVE" in body

    def test_the_model_is_told_not_to_trust_the_code(self):
        """The load-bearing instruction: the code was written to satisfy the
        defective test, so deriving expected values from it repeats the bug."""
        body = _build(SPEC_CONFLICTS["envelope"], authoritative=True)
        assert "Do not derive the expected values from the code" in body
        assert "written to satisfy the defective test" in body

    def test_the_tiebreaker_label_is_gone(self):
        """'TIEBREAKER ONLY — do not cite in the test' actively tells the
        model to let the code decide. That is the bug."""
        body = _build(SPEC_CONFLICTS["wa.me"], authoritative=True)
        assert "TIEBREAKER ONLY" not in body

    def test_the_spec_is_not_duplicated(self):
        body = _build(SPEC_CONFLICTS["wa.me"], authoritative=True)
        assert body.count("SRS_BODY") == 1

    def test_the_spec_gets_the_larger_budget(self):
        """Buried past 4k a spec rule is invisible; authoritative mode gives
        it the same 12k the code gets."""
        long_spec = "A" * 9000 + "RULE_AT_9K"
        body = build_regeneration_messages(
            test_rel_path="t.py", test_source="x",
            code_module_path="m.py", code_module_source="c",
            module_symbols=[], unsat_reason=SPEC_CONFLICTS["asm-009"],
            failing_output="", spec_tiebreaker=long_spec,
            spec_authoritative=True,
        )[-1]["content"]
        assert "RULE_AT_9K" in body

    def test_the_code_is_still_present(self):
        """Demoted, not removed — the test still has to compile against the
        module's real symbols."""
        body = _build(SPEC_CONFLICTS["asm-009"], authoritative=True)
        assert "CODE_BODY" in body
        assert "list_birthdays" in body


class TestTheDefaultIsUnchanged:
    def test_code_still_leads_for_an_ordinary_repair(self):
        body = _build(NOT_SPEC_CONFLICTS["undefined-name"], authoritative=False)
        assert body.index("CODE_BODY") < body.index("SRS_BODY")

    def test_the_spec_is_still_only_a_tiebreaker(self):
        body = _build(NOT_SPEC_CONFLICTS["undefined-name"], authoritative=False)
        assert "TIEBREAKER ONLY" in body
        assert "AUTHORITATIVE" not in body

    def test_authoritative_defaults_to_off(self):
        """The 1:1 traceability model must survive: only the
        repair-of-a-false-test path flips."""
        body = build_regeneration_messages(
            test_rel_path="t.py", test_source="x",
            code_module_path="m.py", code_module_source="CODE_BODY",
            module_symbols=[], unsat_reason="whatever",
            failing_output="", spec_tiebreaker="SRS_BODY",
        )[-1]["content"]
        assert body.index("CODE_BODY") < body.index("SRS_BODY")

    def test_no_spec_available_means_no_authoritative_section(self):
        body = build_regeneration_messages(
            test_rel_path="t.py", test_source="x",
            code_module_path="m.py", code_module_source="CODE_BODY",
            module_symbols=[], unsat_reason=SPEC_CONFLICTS["asm-009"],
            failing_output="", spec_tiebreaker="",
            spec_authoritative=True,
        )[-1]["content"]
        assert "AUTHORITATIVE" not in body
        assert "CODE_BODY" in body


class TestTheWiring:
    def test_the_node_computes_the_flag_from_the_reason(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        assert "reason_cites_spec_conflict(reason)" in src
        assert "spec_authoritative=_spec_authoritative" in src

    def test_the_flip_is_logged(self):
        """An anchor change this consequential must be visible in a run log,
        or the next post-mortem cannot tell which anchor was used."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        assert "the SPECIFICATION leads this rewrite" in src

    def test_the_acceptance_path_already_anchors_on_criteria(self):
        """Why this fix is unit-only: the acceptance suite is regenerated
        from gather_story_acceptance_context, i.e. the criteria themselves,
        and run 8 showed that path returning the SRS-correct envelope."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        at = src.index("async def _regenerate_acceptance_suite")
        body = src[at:]
        assert "gather_story_acceptance_context" in body, (
            "the acceptance path must rebuild from the CRITERIA; if that "
            "changes, unit-style code anchoring would reintroduce the bug "
            "this fix exists for"
        )
