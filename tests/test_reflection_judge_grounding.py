"""The reflection judge must be fed — and must answer within — the same
ground truth the repair LLM gets.

lumina session 01a079dc escalated to HITL on ``low_signal_verdict_loop:5``
after 8 repair rounds against 3 stable pytest failures. Two structural
defects in the judge path produced the loop:

Fix H — asymmetric evidence. The repair prompt received each diagnostic's
``semantic_context`` (workspace call chain, pytest assertion-rewrite lines
naming the resolved values, failure-frame locals, runtime object detail)
while the judge received only ``[AssertionError] test_middleware.py:118 ::
assert 2 == 1``. The judge therefore returned the "insufficient data"
sentinel five rounds running — correctly, given what it was shown — and it
is the judge's verdict, not the repair LLM's, that trips the HITL cap. The
build-output slice compounded it: a blind ``[-2500:]`` cut began mid-line at
"equest FAILED [ 98%]" and dropped the entire ``test_main.py`` RuntimeError
traceback body into the discarded middle.

Fix I — unactionable recommendations. The judge answered "Run pytest with
-vv and --log-cli-level=DEBUG to capture the full exception traceback" every
round. The repair LLM's whole action space is emitting patch blocks; it has
no shell and cannot re-run anything. The text was still injected downstream
as a REQUIRED ACTION, so each round was spent on guidance that could not be
followed.
"""

from __future__ import annotations

from harness.graph import (
    _build_repair_reflection_prompt,
    _parse_repair_reflection_verdict,
    _recommendation_is_unactionable,
)

# Trimmed from the real lumina round-8 diagnostic.
LUMINA_CONTEXT = (
    "failing source: assert len(caplog.records) == 1\n"
    "assertion-rewrite:\n"
    "  assert 2 == 1\n"
    "  +  where 2 = len([<LogRecord: audit, 20, "
    "server/app/middleware/audit.py, 34, \"{...}\">, "
    "<LogRecord: httpx, 20, httpx/_client.py, 1025, "
    "\"HTTP Request: %s %s\">])\n"
    "locals at failure frame:\n"
    "  caplog = <_pytest.logging.LogCaptureFixture object>\n"
)


def _base_kwargs() -> dict:
    return {
        "prior_diagnostics_count": 3,
        "current_diagnostics_count": 3,
        "resolved_fingerprints": [],
        "persisted_fingerprints": ["err::a", "err::b", "err::c"],
        "new_fingerprints": [],
        "top_persisted_diagnostics": [
            {
                "error_code": "AssertionError",
                "file": "server/tests/test_middleware.py",
                "line": 118,
                "message": "AssertionError: assert 2 == 1",
                "semantic_context": LUMINA_CONTEXT,
            },
        ],
    }


class TestSemanticContextReachesTheJudge:
    def test_context_is_rendered(self) -> None:
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "context:" in prompt
        # The line that actually localizes the bug: the second LogRecord is
        # httpx's, not the app's — i.e. the test's assertion is wrong.
        assert "httpx/_client.py" in prompt
        assert "assertion-rewrite" in prompt

    def test_diagnostic_without_context_still_renders(self) -> None:
        kwargs = _base_kwargs()
        kwargs["top_persisted_diagnostics"][0].pop("semantic_context")
        prompt = _build_repair_reflection_prompt(**kwargs)
        assert "server/tests/test_middleware.py:118" in prompt
        assert "context:" not in prompt

    def test_context_is_budget_capped(self) -> None:
        kwargs = _base_kwargs()
        kwargs["top_persisted_diagnostics"] = [
            {
                "error_code": "AssertionError",
                "file": f"t{i}.py",
                "line": i,
                "message": "AssertionError: assert 2 == 1",
                "semantic_context": "X" * 5000,
            }
            for i in range(3)
        ]
        prompt = _build_repair_reflection_prompt(**kwargs)
        # Total context budget is 3000 chars across all three diagnostics,
        # so the judge prompt cannot be swamped by one huge traceback.
        # Measure against a context-free baseline: the static preamble
        # contains the word "EXCEPTION", which carries an X of its own.
        baseline = _build_repair_reflection_prompt(
            **{**kwargs, "top_persisted_diagnostics": [
                {k: v for k, v in d.items() if k != "semantic_context"}
                for d in kwargs["top_persisted_diagnostics"]
            ]},
        )
        assert prompt.count("X") - baseline.count("X") == 3000
        assert "context truncated" in prompt


class TestBuildOutputSlicing:
    def test_head_and_tail_both_survive(self) -> None:
        raw = (
            "HEAD_MARKER first failure at collection\n"
            + ("filler line to push past the slice window\n" * 200)
            + "TAIL_MARKER 3 failed, 74 passed\n"
        )
        prompt = _build_repair_reflection_prompt(
            **_base_kwargs(), build_output_tail=raw,
        )
        # A blind last-2500-bytes cut kept only TAIL_MARKER. The repair
        # slicer keeps the root cause at the top too.
        assert "HEAD_MARKER" in prompt
        assert "TAIL_MARKER" in prompt

    def test_bare_message_below_top_still_attaches_the_log(self) -> None:
        """The gate scans the top 3, not just position 0 — ordering must not
        decide whether the judge sees a traceback."""
        kwargs = _base_kwargs()
        kwargs["top_persisted_diagnostics"] = [
            {
                "error_code": "AssertionError",
                "file": "a.py", "line": 1,
                "message": "AssertionError: expected the upcoming list to be "
                           "ordered by next-occurrence date, got insertion order",
            },
            {
                "error_code": "RuntimeError",
                "file": "b.py", "line": 22,
                "message": "boom",
            },
        ]
        prompt = _build_repair_reflection_prompt(
            **kwargs, build_output_tail="TAIL_MARKER 3 failed\n",
        )
        assert "TAIL_MARKER" in prompt

    def test_no_bare_message_omits_the_log(self) -> None:
        kwargs = _base_kwargs()
        kwargs["top_persisted_diagnostics"] = [
            {
                "error_code": "AssertionError",
                "file": "a.py", "line": 1,
                "message": "AssertionError: expected the upcoming list to be "
                           "ordered by next-occurrence date, got insertion order",
            },
        ]
        prompt = _build_repair_reflection_prompt(
            **kwargs, build_output_tail="TAIL_MARKER 3 failed\n",
        )
        assert "TAIL_MARKER" not in prompt


class TestActionabilityContractInPrompt:
    def test_contract_is_present(self) -> None:
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "ACTIONABILITY CONTRACT" in prompt
        assert "It does not have a shell" in prompt

    def test_empty_recommendation_is_offered_as_the_honest_answer(self) -> None:
        prompt = _build_repair_reflection_prompt(**_base_kwargs())
        assert "leave ``recommendation`` EMPTY" in prompt


class TestRecommendationActionability:
    def test_the_lumina_recommendation_is_rejected(self) -> None:
        assert _recommendation_is_unactionable(
            "Run pytest with -vv and --log-cli-level=DEBUG to capture the "
            "full exception traceback and identify whether the RuntimeError "
            "originates from the test fixture, the route handler, or "
            "middleware."
        ) is True

    def test_observation_verbs_are_rejected(self) -> None:
        for text in (
            "Investigate the data flow into the assertion.",
            "Check whether the middleware ordering is correct.",
            "Re-run the suite with --tb=long.",
            "Debug the audit middleware to see which logger emits.",
            "Add logging to the dispatch method to trace the exception.",
            "Please verify the response body shape first.",
        ):
            assert _recommendation_is_unactionable(text) is True, text

    def test_edit_recommendations_are_kept(self) -> None:
        for text in (
            "Edit server/app/middleware/audit.py:21 to await call_next "
            "inside a try/finally so duration is recorded on the error path.",
            "Add pytest-asyncio to server/requirements-dev.txt.",
            "Set asyncio_mode = auto in pytest.ini.",
            "Change the return type of get_server_date to date in "
            "server/app/utils/dates.py.",
            "Remove the propagate=False line from audit.py so records reach "
            "the root handler — then run the suite to confirm.",
        ):
            assert _recommendation_is_unactionable(text) is False, text

    def test_midsentence_check_is_not_a_rejection(self) -> None:
        # "check" as a qualifier on a real edit, not as the imperative.
        assert _recommendation_is_unactionable(
            "Edit foo.py:20 to check for None before indexing."
        ) is False

    def test_empty_is_not_flagged(self) -> None:
        assert _recommendation_is_unactionable("") is False


class TestParserDropsUnactionableRecommendation:
    def test_dropped_but_verdict_survives(self) -> None:
        raw = (
            '{"verdict": "DISTRACTION", '
            '"real_blocker": "insufficient data — investigate '
            'server/tests/test_main.py\'s data flow into the assertion", '
            '"recommendation": "Run pytest with -vv and '
            '--log-cli-level=DEBUG to capture the full traceback."}'
        )
        parsed = _parse_repair_reflection_verdict(raw)
        assert parsed is not None
        assert parsed["verdict"] == "DISTRACTION"
        assert parsed["real_blocker"].startswith("insufficient data")
        assert parsed["recommendation"] == ""

    def test_actionable_recommendation_is_preserved(self) -> None:
        raw = (
            '{"verdict": "DISTRACTION", '
            '"real_blocker": "server/app/middleware/audit.py:12 sets '
            'propagate=False so audit records never reach the root handler", '
            '"recommendation": "Delete the logger.propagate = False line at '
            'server/app/middleware/audit.py:12."}'
        )
        parsed = _parse_repair_reflection_verdict(raw)
        assert parsed is not None
        assert parsed["recommendation"].startswith("Delete the logger")

    def test_fenced_json_still_parses(self) -> None:
        raw = (
            '```json\n{"verdict": "PROGRESS", "real_blocker": "", '
            '"recommendation": "Investigate the middleware chain."}\n```'
        )
        parsed = _parse_repair_reflection_verdict(raw)
        assert parsed is not None
        assert parsed["verdict"] == "PROGRESS"
        assert parsed["recommendation"] == ""
