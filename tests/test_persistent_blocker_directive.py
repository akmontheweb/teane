"""Verify the persistent-blocker directive (Fixes #2/#3) — when the judge
names the same ``(file, line)`` two rounds running, the repair prompt
injects a hard directive requiring the LLM to alter that exact line."""

import os

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


class TestVerdictReferencedFiles:
    """The loose extractor used by the persistent-blocker save site — grounds
    on workspace existence instead of compiler_errors membership, so a
    judge that names a source file whose failure surfaces in a DIFFERENT
    file (session b8vbfdxxa: judge blames ``parser.py``, the ImportError
    location is ``tests/test_parser.py``) still gets remembered."""

    def test_finds_file_when_workspace_has_it(self, tmp_path):
        from harness.graph import _verdict_referenced_files
        (tmp_path / "backend" / "services").mkdir(parents=True)
        (tmp_path / "backend" / "services" / "parser.py").write_text("x = 1")
        verdict = {
            "real_blocker": (
                "backend/services/parser.py does not export HAS_BS4"
            ),
            "recommendation": "",
        }
        out = _verdict_referenced_files(verdict, str(tmp_path))
        assert out == [os.path.join("backend", "services", "parser.py")]

    def test_ignores_files_that_dont_exist(self, tmp_path):
        from harness.graph import _verdict_referenced_files
        verdict = {
            "real_blocker": "phantom/does_not_exist.py is broken",
            "recommendation": "",
        }
        assert _verdict_referenced_files(verdict, str(tmp_path)) == []

    def test_blocks_path_traversal(self, tmp_path):
        # A judge that emits ../etc/passwd should NOT return a match
        # even if the traversal target happens to exist on the host.
        from harness.graph import _verdict_referenced_files
        verdict = {
            "real_blocker": "../outside/file.py is the issue",
            "recommendation": "",
        }
        assert _verdict_referenced_files(verdict, str(tmp_path)) == []

    def test_only_accepts_source_like_extensions(self, tmp_path):
        # Random-extension matches (``.txt``, ``.log``, arbitrary
        # regex-y strings) don't count — narrows the surface.
        from harness.graph import _verdict_referenced_files
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "config.json").write_text("{}")
        verdict = {
            "real_blocker": "see notes.txt and config.json",
            "recommendation": "",
        }
        out = _verdict_referenced_files(verdict, str(tmp_path))
        assert out == ["config.json"]  # .txt ignored, .json accepted

    def test_dedupes_within_and_across_fields(self, tmp_path):
        from harness.graph import _verdict_referenced_files
        (tmp_path / "a.py").write_text("x")
        verdict = {
            "real_blocker": "a.py has an issue at a.py",
            "recommendation": "fix a.py",
        }
        out = _verdict_referenced_files(verdict, str(tmp_path))
        assert out == ["a.py"]

    def test_empty_verdict_returns_empty(self, tmp_path):
        from harness.graph import _verdict_referenced_files
        assert _verdict_referenced_files({}, str(tmp_path)) == []
        assert _verdict_referenced_files(
            {"real_blocker": "", "recommendation": ""},
            str(tmp_path),
        ) == []


class TestVerdictNamedFilesFileOnly:
    """Independent tests for _verdict_named_files — the file-only detector
    that backs the "PERSISTENT BLOCKER (file scope)" banner. Extends the
    existing file:line detector for diagnostics that lack a line number
    (missing-symbol import errors, session bz4xcajwa)."""

    def test_matches_file_named_without_line(self):
        from harness.graph import _verdict_named_files
        verdict = {
            "real_blocker": (
                "backend/services/parser.py does not export the name "
                "'FilingParser', causing ImportError."
            ),
            "recommendation": "",
        }
        errs = [
            {
                "file": "backend/services/parser.py",
                "line": 0,
                "error_code": "ImportError",
                "message": "cannot import name 'FilingParser'",
            }
        ]
        out = _verdict_named_files(verdict, errs)
        assert out == ["backend/services/parser.py"]

    def test_returns_empty_when_file_not_in_compiler_errors(self):
        from harness.graph import _verdict_named_files
        verdict = {
            "real_blocker": "utils.py is missing helper_x",
            "recommendation": "",
        }
        errs = [
            {
                "file": "backend/api/other.py", "line": 5,
                "error_code": "NameError", "message": "helper_x undefined",
            }
        ]
        assert _verdict_named_files(verdict, errs) == []
