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
        # The judge should see the diagnostics FIRST (fresh signal),
        # then the anti-repetition context. Prior verdict text
        # appearing before the diagnostics would bias the read.
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(),
            prior_reflection_verdict={
                "verdict": "DISTRACTION",
                "real_blocker": "prior blocker text",
                "recommendation": "prior recommendation",
            },
        )
        top_errors_pos = prompt.find("Top persistent errors")
        prior_verdict_pos = prompt.find("YOUR PREVIOUS-ROUND VERDICT")
        answer_pos = prompt.find("Answer ONE structured question")
        assert top_errors_pos > 0
        assert prior_verdict_pos > top_errors_pos
        assert answer_pos > prior_verdict_pos


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
