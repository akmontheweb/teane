"""Anti-repetition wiring for repair_node's reflection judge.

The reflection judge previously saw no memory of its own past verdicts,
so a stream of narratively-identical DISTRACTION / REGRESSION verdicts
could recur round after round with no signal to break the loop.
Finsearch STORY-042 hit this: 10+ rounds of "session tokens
deterministic" variants before the repair LLM ever converged. The fix
feeds the last round's verdict into the next reflection prompt so the
judge can textually detect the repeat and pivot its recommendation.
"""

from __future__ import annotations

from harness.graph import _build_repair_reflection_prompt


def _base_kwargs() -> dict:
    """Minimal well-formed input to the prompt builder — reused across
    tests so each case can toggle one field."""
    return {
        "prior_diagnostics_count": 3,
        "current_diagnostics_count": 3,
        "resolved_fingerprints": [],
        "persisted_fingerprints": ["err::a", "err::b", "err::c"],
        "new_fingerprints": [],
        "top_persisted_diagnostics": [
            {"error_code": "AssertionError",
             "file": "app/services/rate_limit.py", "line": 130,
             "message": "session tokens deterministic"},
        ],
    }


class TestPriorVerdictBlockRendering:
    """The block only renders when there IS a prior verdict with
    substantive content — first repair round sees no block, and a
    prior PROGRESS verdict with empty real_blocker sees no block."""

    def test_no_prior_verdict_no_block(self) -> None:
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "YOUR PREVIOUS-ROUND VERDICT" not in prompt

    def test_prior_verdict_none_no_block(self) -> None:
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(), prior_reflection_verdict=None,
        )
        assert "YOUR PREVIOUS-ROUND VERDICT" not in prompt

    def test_prior_verdict_empty_dict_no_block(self) -> None:
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(), prior_reflection_verdict={},
        )
        assert "YOUR PREVIOUS-ROUND VERDICT" not in prompt

    def test_prior_verdict_missing_blocker_no_block(self) -> None:
        # PROGRESS verdict has no real_blocker — nothing to feed back.
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "PROGRESS",
                "real_blocker": "",
                "recommendation": "keep going",
            },
        )
        assert "YOUR PREVIOUS-ROUND VERDICT" not in prompt

    def test_prior_verdict_with_content_renders_block(self) -> None:
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "DISTRACTION",
                "real_blocker": "session tokens are deterministic",
                "recommendation": "add uuid to session id",
            },
        )
        assert "YOUR PREVIOUS-ROUND VERDICT" in prompt
        assert "DISTRACTION" in prompt
        assert "session tokens are deterministic" in prompt
        assert "add uuid to session id" in prompt


class TestPriorVerdictBlockContent:
    """The block must give the judge a clear instruction: either pivot
    the recommendation, or return a different real_blocker."""

    def test_prompts_pivot_when_same_blocker_recurs(self) -> None:
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "DISTRACTION",
                "real_blocker": "x",
                "recommendation": "y",
            },
        )
        # The block MUST tell the judge what to do when the same
        # blocker would recur.
        assert "PIVOT" in prompt or "structurally different" in prompt.lower()
        assert "NEVER verbatim-repeat" in prompt or "verbatim" in prompt.lower()

    def test_field_lengths_capped(self) -> None:
        # A truly runaway real_blocker (rare but possible under
        # WORKING_HYPOTHESIS with lots of grounding text) must be
        # capped to keep the prompt bounded.
        long_text = "boom " * 200  # 1000 chars
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "DISTRACTION",
                "real_blocker": long_text,
                "recommendation": long_text,
            },
        )
        # Cap is 300 per field; the whole rendered block should be
        # well under 1500 chars (headers + two capped fields).
        block_start = prompt.find("YOUR PREVIOUS-ROUND VERDICT")
        block_end = prompt.find("\n\n", block_start + 200)
        block = prompt[block_start:block_end]
        assert len(block) < 1500

    def test_block_position_after_diagnostics_before_hints(self) -> None:
        # Cache-friendly layout: the task/definitions (stable) lead the
        # prompt, then the EVIDENCE section carries the diagnostics and, after
        # them, the anti-repetition prior-verdict context. Prior verdict must
        # still come AFTER the fresh diagnostics so it doesn't bias the read.
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "DISTRACTION",
                "real_blocker": "prior blocker text",
                "recommendation": "prior recommendation",
            },
        )
        answer_pos = prompt.find("Answer ONE structured question")
        top_errors_pos = prompt.find("Top persistent errors")
        prior_verdict_pos = prompt.find("YOUR PREVIOUS-ROUND VERDICT")
        assert answer_pos > 0
        assert top_errors_pos > 0
        # Stable task/definitions lead; diagnostics then prior-verdict trail.
        assert answer_pos < top_errors_pos < prior_verdict_pos


class TestBackwardsCompat:
    """The new kwarg is optional; every existing caller that omits it
    must continue to work byte-identically to before."""

    def test_omitting_kwarg_leaves_prompt_unchanged_from_none(self) -> None:
        omit = _build_repair_reflection_prompt(**_base_kwargs())
        explicit_none = _build_repair_reflection_prompt(
            **_base_kwargs(), prior_reflection_verdict=None,
        )
        assert omit == explicit_none

    def test_invalid_shape_no_block(self) -> None:
        # Robustness: a stashed-but-corrupted prior verdict (list, str,
        # None-values only) must not crash and must not render the
        # block.
        for bogus in (None, "", 0, [], {"verdict": "", "real_blocker": ""}):
            prompt = _build_repair_reflection_prompt(
                **_base_kwargs(), prior_reflection_verdict=bogus,  # type: ignore[arg-type]
            )
            assert "YOUR PREVIOUS-ROUND VERDICT" not in prompt


class TestTestAssertionGuardrails:
    """When the top persisted diagnostic is an assertion failure inside a
    test file, the TEST-ASSERTION HINT must (a) forbid concluding the test
    is 'reversed/wrong' from a mere order/value difference, and (b) forbid
    recommending a test edit (the repair loop can't modify test files).
    Regression for lumina 019fa046: the judge misread [3,2,1] (last_name
    order) as a reversed test and recommended editing the assertion.
    """

    def _test_assertion_kwargs(self) -> dict:
        kw = _base_kwargs()
        kw["top_persisted_diagnostics"] = [
            {"error_code": "AssertionError",
             "file": "server/tests/test_repository.py", "line": 32,
             "message": "assert [c.id for c in results] == [3, 2, 1]"},
        ]
        return kw

    def test_order_difference_not_evidence_test_wrong(self) -> None:
        prompt = _build_repair_reflection_prompt(**self._test_assertion_kwargs())
        assert "A DIFFERENCE IN LIST ORDER OR RETURNED VALUES IS NOT" in prompt
        assert "[1,2,3] vs [3,2,1]" in prompt

    def test_never_recommend_editing_a_test(self) -> None:
        prompt = _build_repair_reflection_prompt(**self._test_assertion_kwargs())
        assert "repair loop CANNOT modify test files" in prompt
        assert "do not direct a test edit" in prompt

    def test_guardrails_absent_for_source_assertion(self) -> None:
        # Top error in a non-test file → hint block (and its guardrails)
        # must not render.
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "A DIFFERENCE IN LIST ORDER OR RETURNED VALUES IS NOT" not in prompt


class TestJudgeSourceEvidence:
    """Fix #7: on a test-assertion round the judge is handed the failing
    test's source (+ the impl it targets via the @tests marker), so it can
    read the documented intent instead of guessing from object-repr
    fingerprints (lumina 019fa046)."""

    def _kw(self, tmp_path) -> dict:
        (tmp_path / "server" / "app" / "repositories").mkdir(parents=True)
        (tmp_path / "server" / "app" / "repositories" / "contacts_repository.py").write_text(
            "class ContactsRepository:\n"
            "    def list_all(self):\n"
            "        return self._db.query(Contact).all()\n"
        )
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / "server" / "tests" / "test_repository.py").write_text(
            "# @tests: server/app/repositories/contacts_repository.py\n"
            "def test_list_all_sorted():\n"
            "    # Order: first by last_name (nulls as empty), then first_name\n"
            "    assert [c.id for c in repo.list_all()] == [c3.id, c2.id, c1.id]\n"
        )
        kw = _base_kwargs()
        kw["top_persisted_diagnostics"] = [{
            "error_code": "AssertionError",
            "file": "server/tests/test_repository.py", "line": 4,
            "message": "assert [<Contact at 0x1>, <Contact at 0x2>] == [<Contact at 0x2>, <Contact at 0x1>]",
        }]
        kw["workspace_path"] = str(tmp_path)
        return kw

    def test_injects_test_and_impl_source(self, tmp_path) -> None:
        prompt = _build_repair_reflection_prompt(**self._kw(tmp_path))
        assert "FAILING TEST SOURCE" in prompt
        # The documented intent — the exact thing that was missing when the
        # judge misread [3,2,1] as a reversed test.
        assert "Order: first by last_name" in prompt
        # The impl resolved via the @tests marker is included too.
        assert "IMPLEMENTATION UNDER TEST" in prompt
        assert "def list_all" in prompt

    def test_no_source_without_workspace(self, tmp_path) -> None:
        kw = self._kw(tmp_path)
        kw["workspace_path"] = ""
        prompt = _build_repair_reflection_prompt(**kw)
        assert "FAILING TEST SOURCE" not in prompt

    def test_no_source_for_non_assertion(self) -> None:
        # Compile error in a source file → no test-source injection.
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "FAILING TEST SOURCE" not in prompt


class TestPromptCacheLayout:
    """Cache-prefix cleanup: the stable instruction preamble must lead the
    prompt and the volatile per-round data must trail it, so consecutive
    judge calls share the cacheable prefix (all 6 calls were cached=0 before
    — lumina 019fa046)."""

    def test_stable_preamble_precedes_volatile_evidence(self) -> None:
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        intro = prompt.index("You are auditing")
        grounding = prompt.index("GROUNDING RULES")
        evidence = prompt.index("=== EVIDENCE (this round) ===")
        counts = prompt.index("Diagnostics before this round")
        # Stable instruction blocks come first; volatile counts live inside
        # the trailing EVIDENCE section.
        assert intro < grounding < evidence < counts
