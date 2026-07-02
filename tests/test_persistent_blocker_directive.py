"""Verify the persistent-blocker directive (Fixes #2/#3) — when the judge
names the same ``(file, line)`` two rounds running, the repair prompt
injects a hard directive requiring the LLM to alter that exact line."""

from harness.graph import _verdict_named_file_lines


def _errs(*pairs):
    return [
        {"file": f, "line": ln, "error_code": "SyntaxError", "message": "x"}
        for f, ln in pairs
    ]


class TestVerdictNamedFileLines:
    def test_colon_form_matches_compiler_error(self):
        verdict = {
            "real_blocker": "IndentationError at tests/foo/test_bar.py:126",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("tests/foo/test_bar.py", 126)))
        assert out == [("tests/foo/test_bar.py", 126)]

    def test_line_of_form(self):
        verdict = {
            "real_blocker": "unexpected indent at line 42 of test_bar.py",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("tests/test_bar.py", 42)))
        assert out == [("tests/test_bar.py", 42)]

    def test_line_in_form(self):
        verdict = {
            "real_blocker": "SyntaxError at line 9 in server/main.py",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("server/main.py", 9)))
        assert out == [("server/main.py", 9)]

    def test_file_line_form(self):
        verdict = {
            "real_blocker": "IndentationError in test_bar.py line 200",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("tests/test_bar.py", 200)))
        assert out == [("tests/test_bar.py", 200)]

    def test_falls_back_to_recommendation(self):
        verdict = {
            "real_blocker": "insufficient data — no diagnostic locations available",
            "recommendation": "Fix the indent at line 42 of tests/test_bar.py.",
        }
        out = _verdict_named_file_lines(verdict, _errs(("tests/test_bar.py", 42)))
        assert out == [("tests/test_bar.py", 42)]

    def test_not_grounded_in_compiler_errors_returns_empty(self):
        # Judge mentions a file:line that isn't in the current failing set —
        # never promote to a hard directive; that's a stale reference.
        verdict = {
            "real_blocker": "IndentationError at test_stale.py:99",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("test_other.py", 5)))
        assert out == []

    def test_line_zero_is_filtered(self):
        # ``_parse_pytest_summary`` produces line=0 diagnostics for
        # collection errors. Those should not become hard directives.
        verdict = {
            "real_blocker": "ImportError at test_bar.py:0",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(verdict, _errs(("test_bar.py", 0)))
        assert out == []

    def test_multiple_matches_preserved(self):
        verdict = {
            "real_blocker": (
                "IndentationError at test_bar.py:126; "
                "also SyntaxError at server/main.py:9"
            ),
            "recommendation": "",
        }
        out = _verdict_named_file_lines(
            verdict, _errs(("test_bar.py", 126), ("server/main.py", 9)),
        )
        assert set(out) == {("test_bar.py", 126), ("server/main.py", 9)}

    def test_basename_suffix_match(self):
        # Judge writes bare basename; compiler emits full relative path —
        # they must still ground each other.
        verdict = {
            "real_blocker": "unexpected indent at test_bar.py:126",
            "recommendation": "",
        }
        out = _verdict_named_file_lines(
            verdict, _errs(("tests/backend/test_bar.py", 126)),
        )
        assert out == [("tests/backend/test_bar.py", 126)]

    def test_empty_verdict_returns_empty(self):
        assert _verdict_named_file_lines({}, _errs(("a.py", 1))) == []
        assert _verdict_named_file_lines(
            {"real_blocker": "", "recommendation": ""},
            _errs(("a.py", 1)),
        ) == []
