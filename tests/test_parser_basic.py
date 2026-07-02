"""Tests for harness/parser_registry.py — error parsing utilities."""


from harness.parser_registry import (
    _strip_ansi,
    GenericParser,
    JavaParser,
    PythonParser,
    TypeScriptParser,
    detect_and_parse,
    get_parser,
)


class TestStripAnsi:
    """Test ANSI escape code stripping."""

    def test_removes_color_code(self):
        """Should remove ANSI color codes."""
        text = "\x1b[31merror\x1b[0m"  # red
        result = _strip_ansi(text)
        assert result == "error"

    def test_removes_bold(self):
        """Should remove bold codes."""
        text = "\x1b[1mbold\x1b[0m"
        result = _strip_ansi(text)
        assert result == "bold"

    def test_plain_text_unchanged(self):
        """Plain text should be unchanged."""
        text = "hello world"
        result = _strip_ansi(text)
        assert result == "hello world"

    def test_multiple_codes(self):
        """Should remove multiple ANSI sequences."""
        text = "\x1b[1m\x1b[31m\x1b[1mRED BOLD\x1b[0m"
        result = _strip_ansi(text)
        assert result == "RED BOLD"
        assert "\x1b" not in result

    def test_empty_string(self):
        """Empty string should return empty."""
        assert _strip_ansi("") == ""


class TestParserDiagnostics:
    """Test parser diagnostics methods."""

    def test_generic_parser_parse_diagnostics(self):
        """GenericParser should have parse_diagnostics static method."""
        output = "error: something failed"
        diagnostics = GenericParser.parse_diagnostics(output)
        assert isinstance(diagnostics, list)


class TestJavaParser:
    """Cover Maven [ERROR] /path:[L,C] and javac/Gradle short forms."""

    def test_maven_error_extracted(self):
        output = (
            "[INFO] Scanning for projects...\n"
            "[ERROR] /repo/src/main/java/UserService.java:[42,17] cannot find symbol\n"
            "[ERROR]   symbol:   variable userRepo\n"
            "[ERROR]   location: class UserService\n"
        )
        diags = JavaParser.parse_diagnostics(output)
        assert len(diags) == 1
        d = diags[0]
        assert d.file.endswith("UserService.java")
        assert d.line == 42
        assert d.column == 17
        assert d.severity == "error"
        assert "cannot find symbol" in d.message

    def test_maven_warning_severity(self):
        output = "[WARNING] /repo/Foo.java:[5,1] deprecated API usage\n"
        diags = JavaParser.parse_diagnostics(output)
        assert len(diags) == 1
        assert diags[0].severity == "warning"

    def test_javac_short_form_extracted(self):
        output = (
            "src/main/java/Foo.java:10: error: ';' expected\n"
            "        int x = 1\n"
            "                 ^\n"
        )
        diags = JavaParser.parse_diagnostics(output)
        assert len(diags) == 1
        assert diags[0].file.endswith("Foo.java")
        assert diags[0].line == 10
        assert diags[0].severity == "error"

    def test_no_match_returns_empty(self):
        diags = JavaParser.parse_diagnostics("BUILD SUCCESSFUL in 4s\n")
        assert diags == []


class TestTypeScriptParser:
    """Cover the tsc parens form path.ts(L,C): error TSXXXX: msg."""

    def test_tsc_error_extracted(self):
        output = "src/services/user.ts(42,17): error TS2304: Cannot find name 'bar'.\n"
        diags = TypeScriptParser.parse_diagnostics(output)
        assert len(diags) == 1
        d = diags[0]
        assert d.file == "src/services/user.ts"
        assert d.line == 42
        assert d.column == 17
        assert d.error_code == "TS2304"
        assert d.severity == "error"
        assert "Cannot find name" in d.message

    def test_tsx_extension_supported(self):
        output = "components/Button.tsx(5,9): error TS2554: Expected 1 arguments, but got 0.\n"
        diags = TypeScriptParser.parse_diagnostics(output)
        assert len(diags) == 1
        assert diags[0].file.endswith("Button.tsx")

    def test_warning_severity(self):
        output = "lib/foo.ts(1,1): warning TS6133: 'x' is declared but never used.\n"
        diags = TypeScriptParser.parse_diagnostics(output)
        assert len(diags) == 1
        assert diags[0].severity == "warning"

    def test_no_match_returns_empty(self):
        diags = TypeScriptParser.parse_diagnostics("Found 0 errors. Watching for file changes.\n")
        assert diags == []


class TestParserDispatch:
    """detect_and_parse should pick the right parser from build_command."""

    def test_mvn_routes_to_java_parser(self):
        assert get_parser("mvn") is JavaParser
        assert get_parser("gradle") is JavaParser
        assert get_parser("javac") is JavaParser

    def test_tsc_routes_to_typescript_parser(self):
        assert get_parser("tsc") is TypeScriptParser
        assert get_parser("vite") is TypeScriptParser

    def test_detect_and_parse_uses_java_on_mvn_command(self):
        output = "[ERROR] /repo/Foo.java:[5,1] cannot find symbol\n"
        diags = detect_and_parse(output, build_command="mvn compile")
        assert len(diags) == 1
        assert diags[0].file.endswith("Foo.java")
        assert diags[0].line == 5

    def test_detect_and_parse_uses_typescript_on_tsc_command(self):
        output = "src/x.ts(3,2): error TS1005: ',' expected.\n"
        diags = detect_and_parse(output, build_command="tsc --noEmit")
        assert len(diags) == 1
        assert diags[0].error_code == "TS1005"

    def test_output_signature_sniff_finds_tsc_under_npm_wrapper(self):
        """Regression for the ciod build: ``npm install && npm run build``
        wraps a ``tsc --noEmit`` inside, but the build_command string
        has no entry in ``_PARSER_REGISTRY``. Before output-signature
        sniffing, detection fell through to ``GenericParser`` which
        doesn't recognise tsc's ``path(line,col): error TSXXXX:`` form
        → ``diagnostics=0`` and the repair LLM saw nothing to fix.
        """
        output = (
            "> server@0.0.0 build\n"
            "> tsc --noEmit\n\n"
            "src/db/seed.ts(10,1): error TS1109: Expression expected.\n"
            "src/db/seed.ts(11,5): error TS1005: ';' expected.\n"
            "src/db/seed.ts(69,1): error TS1005: '}' expected.\n"
            "npm error Lifecycle script `build` failed with error:\n"
            "npm error code 2\n"
        )
        diags = detect_and_parse(
            output, build_command="npm install && npm run build",
        )
        assert len(diags) == 3
        assert all(d.file == "src/db/seed.ts" for d in diags)
        assert {d.error_code for d in diags} == {"TS1109", "TS1005"}

class TestPythonParserAssertionBody:
    """Verify pytest plain-`assert` failures keep their message body so the
    repair LLM sees what the test actually expected (session 2d0164f0).
    """

    def test_terminal_fail_line_inherits_e_line_message(self):
        # Pytest short-traceback layout for a plain assert failure:
        # the E-line carries the actual assertion text, and the terminal
        # `file:line: ErrorType` line lacks a message body. The parser
        # must paint the E-line message onto the terminal-line diagnostic
        # so the repair prompt's `## Compiler Diagnostics` block contains
        # the assertion expression — not just "AssertionError".
        output = (
            "______________ TestModel.test_type_empty_string_raises ______________\n"
            "    def test_type_empty_string_raises(self):\n"
            "        with pytest.raises(ValidationError) as exc_info:\n"
            "            Job(type='', payload={})\n"
            "        errors = exc_info.value.errors()\n"
            ">       assert any(\"type must be a non-empty string\" in e[\"msg\"] for e in errors)\n"
            "E       AssertionError: assert False\n"
            "E        +  where False = any(<generator>)\n"
            "\n"
            "tests/task_dispatcher/test_models.py:59: AssertionError\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        assert diags, "expected at least one diagnostic"
        # Find the per-failure-block diag (line=59) not any summary aggregate.
        precise = [d for d in diags if d.line == 59]
        assert precise, f"expected diagnostic on line 59, got {[(d.file, d.line, d.message) for d in diags]}"
        d = precise[0]
        assert d.file == "tests/task_dispatcher/test_models.py"
        assert d.error_code == "AssertionError"
        # Critical: the message must include the E-line body, NOT just
        # the bare error type.
        assert d.message != "AssertionError", (
            "regression: terminal fail-line dropped the E-line message body"
        )
        assert "assert False" in d.message
        # The failing source from the `>` marker should land in
        # semantic_context so the repair prompt can show the LLM exactly
        # which expression failed.
        assert "type must be a non-empty string" in d.semantic_context

    def test_bare_e_lines_carry_resolved_values(self):
        # Pytest's default assertion-rewrite output for a plain ``assert``
        # with no custom message emits BARE E-lines (no ``ErrorType:``
        # prefix) that carry the actual resolved values:
        #
        #     E       assert True is False
        #     E        +  where True = <Session>.is_active
        #
        # Before this fix these lines were dropped by the parser, leaving
        # the diagnostic message as just ``AssertionError`` — so the
        # judge/repair prompts had no way to know that ``session.is_active``
        # was ``True`` when the test expected ``False``. Observed in
        # session 116667f5 where six repair rounds failed to converge
        # because the LLM kept guessing at the value. See docs in
        # parser_registry.PythonParser._PYTEST_E_BARE_PATTERN.
        output = (
            "____________ test_get_db_yields_session ____________\n"
            "    def test_get_db_yields_session():\n"
            "        # After close, session should be closed\n"
            ">       assert session.is_active is False\n"
            "E       assert True is False\n"
            "E        +  where True = <Session at 0x7f>.is_active\n"
            "\n"
            "tests/test_db_base.py:38: AssertionError\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        precise = [d for d in diags if d.line == 38]
        assert precise, (
            f"expected diagnostic on line 38, got "
            f"{[(d.file, d.line, d.message) for d in diags]}"
        )
        d = precise[0]
        assert d.error_code == "AssertionError"
        # The bare E-line body must be promoted into the message so the
        # ranked diagnostic list in the judge/repair prompts shows the
        # resolved expression instead of just the exception type.
        assert d.message != "AssertionError", (
            "regression: bare E-lines were dropped and message collapsed "
            "to the exception type"
        )
        assert "assert True is False" in d.message
        # The `+  where True = ...` explanation is the highest-signal
        # context — it names the object AND the attribute that produced
        # the unexpected value. It must land in semantic_context.
        assert "is_active" in d.semantic_context
        assert "where True" in d.semantic_context

    def test_syntax_error_recovers_user_file_from_e_bare(self):
        # Pytest short-traceback layout for a SyntaxError raised during
        # collection: the traceback frame points at Python's ast.py (the
        # stdlib compile() call), and the user file+line live in the
        # bare-E ``File "<path>", line N`` block that pytest inlines.
        # Without recovery, the diagnostic anchors to /usr/lib/python3.11/
        # ast.py — the reflection judge marks it "insufficient data — no
        # diagnostic locations available", the repair LLM guesses, and
        # the loop fails to converge (session 70877929).
        output = (
            "___ ERROR collecting tests/backend/test_rate_limiter.py ___\n"
            "/usr/lib/python3.11/ast.py:50: in parse\n"
            "    return compile(source, filename, mode, flags | PyCF_ONLY_AST)\n"
            'E     File "tests/backend/test_rate_limiter.py", line 129\n'
            "E       assert exc_info.value.response.status_code == 500tries exhausted...\n"
            "E                                                          ^\n"
            "E   SyntaxError: invalid decimal literal\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        precise = [d for d in diags if d.error_code == "SyntaxError"]
        assert precise, (
            f"expected SyntaxError diagnostic, got "
            f"{[(d.file, d.line, d.error_code) for d in diags]}"
        )
        d = precise[0]
        assert d.file == "tests/backend/test_rate_limiter.py", (
            f"regression: SyntaxError anchored to stdlib instead of user "
            f"file (got {d.file!r})"
        )
        assert d.line == 129, f"expected line 129, got {d.line}"

    def test_bare_e_line_dedupes_against_failing_source(self):
        # When the assertion has no rewrite explanation (e.g. ``assert
        # 1 == 2`` where pytest just echoes the same expression back on
        # the E-line), the parser should NOT double-attach the same text
        # in both ``failing source:`` and ``assertion-rewrite:`` slots.
        output = (
            "____________ test_simple ____________\n"
            "    def test_simple():\n"
            ">       assert 1 == 2\n"
            "E       assert 1 == 2\n"
            "\n"
            "tests/test_simple.py:2: AssertionError\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        precise = [d for d in diags if d.line == 2]
        assert precise
        d = precise[0]
        # The failing source should still be present; the redundant E-line
        # should be suppressed so we don't ship duplicate context.
        assert "failing source: assert 1 == 2" in d.semantic_context
        assert d.semantic_context.count("assert 1 == 2") == 1, (
            f"expected the failing source to appear once, "
            f"got context {d.semantic_context!r}"
        )
