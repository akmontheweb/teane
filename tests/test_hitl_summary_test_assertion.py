"""Verify the HITL escalation summariser steers the operator toward the
implementation module — not the test — when the top persisted error is a
pytest assertion failure inside a test file.

Mirrors the Fix G guardrail already present in the repair reflection
prompt (see ``_build_repair_reflection_prompt``). Without this extension
the summariser would satisfy its "cite filenames from the evidence"
instruction by recommending edits to the test's assertion, which masks
the regression the repair loop just failed to fix. Motivating incident:
FinancialResearch 2026-07-07 run recommended flipping
``'2024FY'`` → ``'Latest FY'`` in ``test_report_generator.py:64`` when
the real bug was in ``report_generator.py``'s ``_build_metrics_html``
label.
"""

from harness.graph import _build_hitl_escalation_summary_prompt


_HINT_MARKER = "TEST-ASSERTION NOTE"


def _state(errors, **overrides):
    """Minimal AgentState-shaped dict for the summariser."""
    base = {
        "compiler_errors": errors,
        "node_state": {},
        "loop_counter": {"total_repairs": 12},
        "modified_files": [],
        "exit_code": 1,
    }
    base.update(overrides)
    return base


def _assertion_err(file_="tests/unit/backend/test_report_generator.py"):
    return {
        "file": file_,
        "line": 64,
        "error_code": "AssertionError",
        "message": "AssertionError: assert '2024FY' in '<table>...</table>'",
    }


class TestHitlSummaryTestAssertionHint:
    def test_hint_present_for_assertion_in_test_file(self):
        state = _state([_assertion_err()])
        prompt = _build_hitl_escalation_summary_prompt(state, "repair_loop_limit")
        assert _HINT_MARKER in prompt
        assert "implementation module the test exercises" in prompt

    def test_hint_absent_for_syntax_error_in_source(self):
        errors = [{
            "file": "backend/services/report_generator.py",
            "line": 348,
            "error_code": "SyntaxError",
            "message": "invalid syntax",
        }]
        prompt = _build_hitl_escalation_summary_prompt(_state(errors), "repair_loop_limit")
        assert _HINT_MARKER not in prompt

    def test_hint_absent_for_import_error_in_test_file(self):
        errors = [{
            "file": "tests/unit/backend/test_report_generator.py",
            "line": 6,
            "error_code": "ImportError",
            "message": "ImportError: cannot import name '_build_metrics_html'",
        }]
        prompt = _build_hitl_escalation_summary_prompt(_state(errors), "repair_loop_limit")
        assert _HINT_MARKER not in prompt

    def test_hint_absent_for_assertion_in_non_test_file(self):
        errors = [{
            "file": "backend/services/report_generator.py",
            "line": 200,
            "error_code": "AssertionError",
            "message": "AssertionError: expected condition failed",
        }]
        prompt = _build_hitl_escalation_summary_prompt(_state(errors), "repair_loop_limit")
        assert _HINT_MARKER not in prompt

    def test_hint_absent_when_no_compiler_errors(self):
        prompt = _build_hitl_escalation_summary_prompt(_state([]), "repair_loop_limit")
        assert _HINT_MARKER not in prompt

    def test_hint_fires_when_test_file_named_foo_test_py(self):
        # The Go/Rust-style ``foo_test.py`` convention should also trigger.
        err = _assertion_err(file_="tests/backend/report_generator_test.py")
        prompt = _build_hitl_escalation_summary_prompt(_state([err]), "repair_loop_limit")
        assert _HINT_MARKER in prompt

    def test_hint_precedes_evidence_in_prompt(self):
        # The steering guidance must appear BEFORE the compiler-errors
        # block, otherwise the LLM has already anchored on the test file
        # by the time it reads the caveat.
        state = _state([_assertion_err()])
        prompt = _build_hitl_escalation_summary_prompt(state, "repair_loop_limit")
        assert prompt.index(_HINT_MARKER) < prompt.index("Recent compiler errors:")

    def test_hint_absent_when_mixed_top_error_is_compile(self):
        # _top_error_is_test_assertion inspects only the top diagnostic.
        # A SyntaxError ranked first (with an assertion below) is
        # compile-class and must not trigger the hint — compile errors
        # take priority.
        errors = [
            {
                "file": "backend/services/report_generator.py",
                "line": 10,
                "error_code": "SyntaxError",
                "message": "invalid syntax",
            },
            _assertion_err(),
        ]
        prompt = _build_hitl_escalation_summary_prompt(_state(errors), "repair_loop_limit")
        assert _HINT_MARKER not in prompt
