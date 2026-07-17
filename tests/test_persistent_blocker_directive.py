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


class TestFixGTestAssertionMode:
    """Fix G — for pytest assertion failures the judge is asked to name the
    impl file the test exercises; the seeder swaps that in for the compiler-
    errors-derived test file so the ``MUST MODIFY`` banner points at the
    impl instead of the test. Session 52c16e16-* burned 3 rounds patching
    ``tests/unit/backend/test_edgar.py`` when the fix belonged in
    ``backend/services/edgar.py``."""

    def test_detector_matches_assertion_in_test_file(self):
        from harness.graph import _top_error_is_test_assertion
        diags = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 49,
            "error_code": "AssertionError",
            "message": "assert len(result) >= 1",
        }]
        assert _top_error_is_test_assertion(diags) is True

    def test_detector_test_filename_pattern_outside_tests_dir(self):
        from harness.graph import _top_error_is_test_assertion
        diags = [{
            "file": "backend/test_edgar.py",
            "line": 10,
            "error_code": "AssertionError",
            "message": "assertion failed",
        }]
        assert _top_error_is_test_assertion(diags) is True

    def test_detector_rejects_non_test_file(self):
        from harness.graph import _top_error_is_test_assertion
        diags = [{
            "file": "backend/services/edgar.py",
            "line": 100,
            "error_code": "AssertionError",
            "message": "assert x is None",
        }]
        assert _top_error_is_test_assertion(diags) is False

    def test_detector_rejects_compile_error_in_test_file(self):
        from harness.graph import _top_error_is_test_assertion
        diags = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 5,
            "error_code": "ImportError",
            "message": "cannot import name X",
        }]
        assert _top_error_is_test_assertion(diags) is False

    def test_detector_rejects_empty(self):
        from harness.graph import _top_error_is_test_assertion
        assert _top_error_is_test_assertion([]) is False

    def test_parser_extracts_impl_file(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = (
            '{"verdict": "DISTRACTION",'
            '"real_blocker": "AssertionError at tests/unit/backend/test_edgar.py:49",'
            '"recommendation": "Fix search impl.",'
            '"impl_file": "backend/services/edgar.py"}'
        )
        out = _parse_repair_reflection_verdict(raw)
        assert out is not None
        assert out.get("impl_file") == "backend/services/edgar.py"

    def test_parser_absent_impl_file_is_ok(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = (
            '{"verdict": "DISTRACTION",'
            '"real_blocker": "AssertionError at foo.py:1",'
            '"recommendation": "Fix it."}'
        )
        out = _parse_repair_reflection_verdict(raw)
        assert out is not None
        assert "impl_file" not in out

    def test_parser_rejects_placeholder_impl_file(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = (
            '{"verdict": "DISTRACTION",'
            '"real_blocker": "AssertionError at foo.py:1",'
            '"recommendation": "Fix it.",'
            '"impl_file": "<file>"}'
        )
        out = _parse_repair_reflection_verdict(raw)
        assert out is not None
        assert "impl_file" not in out

    def test_seeder_swaps_impl_for_test_when_applicable(self, tmp_path):
        from harness.graph import _effective_judge_named_files
        (tmp_path / "backend" / "services").mkdir(parents=True)
        impl = tmp_path / "backend" / "services" / "edgar.py"
        impl.write_text("# impl\n")
        verdict = {
            "real_blocker": "AssertionError at tests/unit/backend/test_edgar.py:49",
            "recommendation": "Modify the typeahead search implementation.",
            "impl_file": "backend/services/edgar.py",
        }
        errs = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 49,
            "error_code": "AssertionError",
            "message": "assert len(result) >= 1",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert files == ["backend/services/edgar.py"]
        assert promoted == "backend/services/edgar.py"
        assert guarded == []

    def test_seeder_falls_back_when_impl_missing_on_disk(self, tmp_path):
        from harness.graph import _effective_judge_named_files
        verdict = {
            "real_blocker": "AssertionError at tests/unit/backend/test_edgar.py:49",
            "recommendation": "Modify the typeahead search implementation.",
            "impl_file": "backend/services/edgar.py",  # not created
        }
        errs = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 49,
            "error_code": "AssertionError",
            "message": "assert len(result) >= 1",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        # No swap, and the test file is tamper-guarded (an assertion
        # failure, not a parse error) — it must move to ``guarded``
        # rather than become a MUST-MODIFY mandate the repair guard
        # would veto (lumina 019f7109).
        assert files == []
        assert promoted is None
        assert guarded == ["tests/unit/backend/test_edgar.py"]

    def test_seeder_refuses_to_promote_test_path_as_impl(self, tmp_path):
        from harness.graph import _effective_judge_named_files
        (tmp_path / "tests" / "unit").mkdir(parents=True)
        bogus = tmp_path / "tests" / "unit" / "test_other.py"
        bogus.write_text("# also a test\n")
        verdict = {
            "real_blocker": "AssertionError at tests/unit/backend/test_edgar.py:49",
            "recommendation": "Rename impl.",
            "impl_file": "tests/unit/test_other.py",
        }
        errs = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 49,
            "error_code": "AssertionError",
            "message": "assert len(result) >= 1",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert promoted is None
        assert files == []
        assert guarded == ["tests/unit/backend/test_edgar.py"]

    def test_seeder_unchanged_for_compile_errors(self, tmp_path):
        """Non-test-assertion rounds must be byte-identical to the old
        _verdict_named_files behavior — the whole point of Fix G being
        isolated."""
        from harness.graph import _effective_judge_named_files
        (tmp_path / "backend" / "services").mkdir(parents=True)
        (tmp_path / "backend" / "services" / "parser.py").write_text("# p\n")
        verdict = {
            "real_blocker": "ImportError from backend/services/parser.py",
            "recommendation": "Fix the import.",
            "impl_file": "backend/services/parser.py",  # present but ignored
        }
        errs = [{
            "file": "backend/services/parser.py",
            "line": 3,
            "error_code": "ImportError",
            "message": "cannot import name X",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert promoted is None
        assert files == ["backend/services/parser.py"]
        assert guarded == []

    def test_seeder_rejects_absolute_and_traversal_paths(self, tmp_path):
        from harness.graph import _effective_judge_named_files
        errs = [{
            "file": "tests/test_x.py", "line": 1,
            "error_code": "AssertionError", "message": "assert False",
        }]
        for bad in ("/etc/passwd", "../secrets.py", "backend/../../etc"):
            verdict = {
                "real_blocker": "AssertionError at tests/test_x.py:1",
                "recommendation": "Fix.",
                "impl_file": bad,
            }
            files, promoted, _guarded = _effective_judge_named_files(
                verdict, errs, errs, str(tmp_path),
            )
            assert promoted is None, f"Should reject {bad}"

    def test_prompt_test_assertion_hint_renders_when_detected(self):
        from harness.graph import _build_repair_reflection_prompt
        diags = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 49,
            "error_code": "AssertionError",
            "message": "assert len(result) >= 1",
        }]
        prompt = _build_repair_reflection_prompt(
            prior_diagnostics_count=2,
            current_diagnostics_count=1,
            resolved_fingerprints=[],
            persisted_fingerprints=["AssertionError::x"],
            new_fingerprints=[],
            top_persisted_diagnostics=diags,
        )
        assert "TEST-ASSERTION HINT" in prompt
        assert "impl_file" in prompt

    def test_prompt_test_assertion_hint_absent_otherwise(self):
        from harness.graph import _build_repair_reflection_prompt
        diags = [{
            "file": "backend/services/edgar.py",
            "line": 10,
            "error_code": "SyntaxError",
            "message": "invalid syntax",
        }]
        prompt = _build_repair_reflection_prompt(
            prior_diagnostics_count=2,
            current_diagnostics_count=1,
            resolved_fingerprints=[],
            persisted_fingerprints=["SyntaxError::x"],
            new_fingerprints=[],
            top_persisted_diagnostics=diags,
        )
        assert "TEST-ASSERTION HINT" not in prompt
        assert "impl_file" not in prompt


class TestImplFileResolution:
    """_resolve_impl_file_in_workspace — lumina 019f7109: the reflection
    judge guesses basenames ("db.py") because it sees neither the build
    imports nor a workspace file list; Fix G must meet the guess halfway
    instead of silently dropping it."""

    def test_unique_basename_match_resolves(self, tmp_path):
        from harness.graph import _resolve_impl_file_in_workspace
        (tmp_path / "server" / "app").mkdir(parents=True)
        (tmp_path / "server" / "app" / "db.py").write_text("# impl\n")
        out = _resolve_impl_file_in_workspace("db.py", str(tmp_path))
        assert out == "server/app/db.py"

    def test_unique_suffix_match_resolves(self, tmp_path):
        from harness.graph import _resolve_impl_file_in_workspace
        (tmp_path / "server" / "app").mkdir(parents=True)
        (tmp_path / "server" / "app" / "db.py").write_text("# impl\n")
        out = _resolve_impl_file_in_workspace("app/db.py", str(tmp_path))
        assert out == "server/app/db.py"

    def test_ambiguous_basename_refuses(self, tmp_path):
        from harness.graph import _resolve_impl_file_in_workspace
        (tmp_path / "server").mkdir()
        (tmp_path / "client").mkdir()
        (tmp_path / "server" / "db.py").write_text("# a\n")
        (tmp_path / "client" / "db.py").write_text("# b\n")
        assert _resolve_impl_file_in_workspace("db.py", str(tmp_path)) is None

    def test_no_match_returns_none(self, tmp_path):
        from harness.graph import _resolve_impl_file_in_workspace
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "other.py").write_text("# x\n")
        assert _resolve_impl_file_in_workspace("db.py", str(tmp_path)) is None

    def test_test_dirs_pruned_from_walk(self, tmp_path):
        # A basename that only exists under tests/ must NOT resolve —
        # promoting a test path would recreate the deadlock the
        # resolution exists to prevent.
        from harness.graph import _resolve_impl_file_in_workspace
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "db.py").write_text("# test helper\n")
        assert _resolve_impl_file_in_workspace("db.py", str(tmp_path)) is None

    def test_seeder_promotes_via_basename_resolution(self, tmp_path):
        # End-to-end through _effective_judge_named_files: the exact
        # lumina 019f7109 shape — judge answers impl_file "db.py", real
        # module is server/app/db.py, failing anchor is the test file.
        from harness.graph import _effective_judge_named_files
        (tmp_path / "server" / "app").mkdir(parents=True)
        (tmp_path / "server" / "app" / "db.py").write_text("# impl\n")
        verdict = {
            "real_blocker": (
                "Assertion at tests/test_db.py:46 expects WAL journal "
                "mode but gets 'memory'."
            ),
            "recommendation": "Set journal_mode=WAL in db.py.",
            "impl_file": "db.py",
        }
        errs = [{
            "file": "tests/test_db.py",
            "line": 46,
            "error_code": "AssertionError",
            "message": "assert 'memory' == 'wal'",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert files == ["server/app/db.py"]
        assert promoted == "server/app/db.py"
        assert guarded == []


class TestGuardAwareSeeder:
    """Guard-parity filtering in _effective_judge_named_files — the
    MUST-MODIFY list must never mandate a file the repair tamper guard
    will refuse to let the LLM patch (lumina 019f7109 rounds 12-13)."""

    def test_parse_broken_test_stays_mandatable(self, tmp_path):
        # The carve-out case (lumina 019f7054): a test file whose current
        # diagnostic is a SyntaxError MAY be repaired, so it must stay in
        # the mandate list.
        from harness.graph import _effective_judge_named_files
        verdict = {
            "real_blocker": "SyntaxError at tests/test_db.py:3",
            "recommendation": "Fix the comment syntax.",
        }
        errs = [{
            "file": "tests/test_db.py",
            "line": 3,
            "error_code": "SyntaxError",
            "message": "invalid syntax",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert files == ["tests/test_db.py"]
        assert promoted is None
        assert guarded == []

    def test_non_assertion_round_also_filters_guarded_tests(self, tmp_path):
        # An ImportError anchored in a healthy test file: the guard would
        # veto repair edits to it in ANY round type, so the filter must
        # apply outside test-assertion mode too.
        from harness.graph import _effective_judge_named_files
        verdict = {
            "real_blocker": "ImportError at tests/test_api.py:2",
            "recommendation": "Export the symbol from the api module.",
        }
        errs = [{
            "file": "tests/test_api.py",
            "line": 2,
            "error_code": "ImportError",
            "message": "cannot import name 'create_app'",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert files == []
        assert promoted is None
        assert guarded == ["tests/test_api.py"]

    def test_mixed_files_keep_production_drop_guarded_test(self, tmp_path):
        # Judge names a production file in the recommendation AND the
        # test anchor: production stays mandatable, test moves to guarded.
        from harness.graph import _effective_judge_named_files
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "main.py").write_text("# app\n")
        verdict = {
            "real_blocker": "AssertionError at tests/test_main.py:9",
            "recommendation": (
                "Edit server/main.py to include the search router."
            ),
        }
        errs = [{
            "file": "tests/test_main.py",
            "line": 9,
            "error_code": "AssertionError",
            "message": "assert 404 == 200",
        }]
        files, promoted, guarded = _effective_judge_named_files(
            verdict, errs, errs, str(tmp_path),
        )
        assert files == ["server/main.py"]
        assert guarded == ["tests/test_main.py"]


class TestUnsatisfiableTestDeclaration:
    """_parse_unsatisfiable_test_declaration — the declared-dead-end
    marker offered by the defective-test banner directive."""

    def test_em_dash_form(self):
        from harness.graph import _parse_unsatisfiable_test_declaration
        out = _parse_unsatisfiable_test_declaration(
            "UNSATISFIABLE_TEST: tests/test_db.py — SQLite cannot use WAL "
            "on an in-memory connection"
        )
        assert out == (
            "tests/test_db.py",
            "SQLite cannot use WAL on an in-memory connection",
        )

    def test_hyphen_and_colon_forms(self):
        from harness.graph import _parse_unsatisfiable_test_declaration
        assert _parse_unsatisfiable_test_declaration(
            "UNSATISFIABLE_TEST: tests/a.py - impossible expectation"
        ) == ("tests/a.py", "impossible expectation")
        assert _parse_unsatisfiable_test_declaration(
            "UNSATISFIABLE_TEST: tests/a.py: impossible expectation"
        ) == ("tests/a.py", "impossible expectation")

    def test_reason_optional(self):
        from harness.graph import _parse_unsatisfiable_test_declaration
        assert _parse_unsatisfiable_test_declaration(
            "UNSATISFIABLE_TEST: tests/a.py"
        ) == ("tests/a.py", "")

    def test_embedded_in_prose_and_first_wins(self):
        from harness.graph import _parse_unsatisfiable_test_declaration
        text = (
            "Some analysis first.\n"
            "UNSATISFIABLE_TEST: tests/first.py — reason one\n"
            "UNSATISFIABLE_TEST: tests/second.py — reason two\n"
        )
        assert _parse_unsatisfiable_test_declaration(text) == (
            "tests/first.py", "reason one",
        )

    def test_absent_returns_none(self):
        from harness.graph import _parse_unsatisfiable_test_declaration
        assert _parse_unsatisfiable_test_declaration("") is None
        assert _parse_unsatisfiable_test_declaration("no marker here") is None
        assert _parse_unsatisfiable_test_declaration(
            "UNSATISFIABLE_TEST:"
        ) is None
