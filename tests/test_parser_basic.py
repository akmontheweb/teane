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

class TestPythonParserWorkspaceFrames:
    """The diagnostic must anchor on workspace code, not the library frame
    where the exception happened to be raised (session 22471c0c: five
    repair rounds re-edited test fixtures because "no such table" surfaced
    only as site-packages/sqlalchemy/engine/default.py:952 — every
    app-side frame had been discarded, so file auto-injection never
    showed the repair LLM the files where the fix lived)."""

    _SQLALCHEMY_STYLE_OUTPUT = (
        "____________ TestCompanyLookup.test_unknown_cik ____________\n"
        "\n"
        "tests/api/test_financials.py:95: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "server/app/api/companies.py:41: in get_company\n"
        "    company = repo.find_by_cik(cik)\n"
        "server/app/repositories/company_repository.py:52: in find_by_cik\n"
        "    return self.db.query(Company).filter(Company.cik == cik).first()\n"
        "/tmp/teane-venv/lib/python3.11/site-packages/sqlalchemy/orm/query.py:2728: in first\n"
        "    return self.limit(1)._iter().first()\n"
        "/tmp/teane-venv/lib/python3.11/site-packages/sqlalchemy/engine/default.py:952: in do_execute\n"
        "    cursor.execute(statement, parameters)\n"
        "E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: companies\n"
        "/tmp/teane-venv/lib/python3.11/site-packages/sqlalchemy/engine/default.py:952: OperationalError\n"
    )

    def test_library_terminal_reanchors_on_workspace_frame(self):
        diags = PythonParser.parse_diagnostics(self._SQLALCHEMY_STYLE_OUTPUT)
        hits = [d for d in diags if "no such table" in d.message]
        assert hits, (
            f"expected a 'no such table' diagnostic, got "
            f"{[(d.file, d.line, d.message) for d in diags]}"
        )
        d = hits[0]
        # The anchor must be the innermost WORKSPACE frame — this drives
        # file auto-injection into the repair prompt and the judge's
        # grounding vocabulary.
        assert d.file == "server/app/repositories/company_repository.py", (
            f"diagnostic anchored on {d.file!r}; a site-packages anchor "
            f"is unpatchable and hides the app-side call chain"
        )
        assert d.line == 52

    def test_library_location_preserved_in_context(self):
        diags = PythonParser.parse_diagnostics(self._SQLALCHEMY_STYLE_OUTPUT)
        d = [x for x in diags if "no such table" in x.message][0]
        assert "raised in library frame:" in d.semantic_context
        assert "sqlalchemy/engine/default.py:952" in d.semantic_context

    def test_workspace_call_chain_rendered_in_context(self):
        diags = PythonParser.parse_diagnostics(self._SQLALCHEMY_STYLE_OUTPUT)
        d = [x for x in diags if "no such table" in x.message][0]
        assert "workspace call chain" in d.semantic_context
        assert "server/app/api/companies.py:41 in get_company" in (
            d.semantic_context
        )
        assert (
            "server/app/repositories/company_repository.py:52 in find_by_cik"
            in d.semantic_context
        )

    # Condensed from the REAL --tb=long output captured from finsearch
    # (scratchpad/full_pytest_output.txt): frames render as BARE
    # "path:line: " boundary lines — no "in <func>" suffix — and the
    # chained cause (sqlite3.OperationalError) renders its own
    # library-frames-only traceback ending in its own terminal line
    # BEFORE the outer chain that contains the workspace frames.
    _LONG_TB_CHAINED_OUTPUT = (
        "____________________ test_search_by_ticker ____________________\n"
        "\n"
        "self = <SQLiteDialect_pysqlite object at 0x1>\n"
        "\n"
        "    def do_execute(self, cursor, statement, parameters, context=None):\n"
        ">       cursor.execute(statement, parameters)\n"
        "E       sqlite3.OperationalError: no such table: companies\n"
        "\n"
        "/tmp/venv/lib/python3.11/site-packages/sqlalchemy/engine/default.py:952: OperationalError\n"
        "\n"
        "The above exception was the direct cause of the following exception:\n"
        "\n"
        "client = <starlette.testclient.TestClient object at 0x2>\n"
        "\n"
        "    def test_search_by_ticker(client):\n"
        '        """Search by exact ticker returns that company."""\n'
        ">       response = client.get(\"/api/companies/search?q=AAPL\")\n"
        "\n"
        "server/tests/test_api.py:62: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "\n"
        "    def get(self, url, **kwargs):\n"
        "        return super().request(...)\n"
        "\n"
        "/tmp/venv/lib/python3.11/site-packages/starlette/testclient.py:482: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "\n"
        "    async def search_companies(q: str, db: Session = Depends(get_db)):\n"
        "        return service.search_companies(q)\n"
        "\n"
        "server/app/api/companies.py:28: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "\n"
        "    def search_by_ticker(self, ticker):\n"
        "        return self.db.query(Company).filter(...).first()\n"
        "\n"
        "server/app/repositories/company_repository.py:22: \n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "\n"
        "    def do_execute(self, cursor, statement, parameters, context=None):\n"
        ">       cursor.execute(statement, parameters)\n"
        "E       sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: companies\n"
        "\n"
        "/tmp/venv/lib/python3.11/site-packages/sqlalchemy/engine/default.py:952: OperationalError\n"
    )

    def test_long_tb_bare_frame_boundaries_are_captured(self):
        diags = PythonParser.parse_diagnostics(self._LONG_TB_CHAINED_OUTPUT)
        hits = [
            d for d in diags
            if d.error_code == "OperationalError"
            and "no such table" in d.message
        ]
        assert hits, (
            f"expected an OperationalError diagnostic, got "
            f"{[(d.file, d.line, d.message) for d in diags]}"
        )
        d = hits[0]
        assert d.file == "server/app/repositories/company_repository.py"
        assert d.line == 22
        assert "server/tests/test_api.py:62 in test_search_by_ticker" in (
            d.semantic_context
        )
        assert "server/app/api/companies.py:28 in search_companies" in (
            d.semantic_context
        )
        assert "raised in library frame:" in d.semantic_context

    # Regression (2026-07-16 review): the long-tb boundary line for frame k
    # prints BEFORE frame k+1's source and E-lines, so when the boundary
    # capture landed (817babc) the typed-E branch started emitting against
    # the OUTER frame — anchoring on the calling test file, stealing the
    # summary row's nodeid in _dedup, and leaving a context-less duplicate
    # from the terminal branch. 817babc's own tests only used dotted
    # exception names (which the \w+ E-pattern can't match), so the
    # early-emit path was never exercised.
    _LONG_TB_NESTED_UNDOTTED = (
        "=================================== FAILURES "
        "===================================\n"
        "_______________________________ test_total "
        "_____________________________________\n"
        "\n"
        "    def test_total(self):\n"
        ">       assert compute_total([1, 2]) == 3\n"
        "tests/test_c.py:4:\n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n"
        "\n"
        "    def compute_total(items):\n"
        ">       raise ValueError(\"boom\")\n"
        "E       ValueError: boom\n"
        "app/util.py:2: ValueError\n"
        "=========================== short test summary info "
        "============================\n"
        "FAILED tests/test_c.py::test_total - ValueError: boom\n"
    )

    def test_long_tb_typed_e_anchors_on_raising_frame_not_caller(self):
        diags = PythonParser.parse_diagnostics(self._LONG_TB_NESTED_UNDOTTED)
        block = [d for d in diags if d.line > 0]
        assert len(block) == 1, (
            f"expected exactly one anchored diagnostic, got "
            f"{[(d.file, d.line, d.message) for d in block]}"
        )
        d = block[0]
        # The RAISING frame (app code), not the calling test frame.
        assert d.file == "app/util.py"
        assert d.line == 2
        # Full message from the E-line, not a bare type.
        assert d.message == "ValueError: boom"

    def test_long_tb_typed_e_no_junk_duplicate_and_nodeid_survives(self):
        diags = PythonParser.parse_diagnostics(self._LONG_TB_NESTED_UNDOTTED)
        # No block-level diagnostic may sit on the test file — that anchor
        # sends file auto-injection at the test instead of the app code.
        assert not [
            d for d in diags if d.file == "tests/test_c.py" and d.line > 0
        ]
        # The summary row keeps its re-runnable nodeid.
        assert [
            d for d in diags
            if d.pytest_nodeid == "tests/test_c.py::test_total"
        ]

    def test_chained_cause_duplicate_is_collapsed(self):
        # The cause chain's own terminal (library-frames-only) must NOT
        # survive as a second venv-anchored OperationalError — it would
        # win downstream dedupe and hide the workspace anchor again.
        diags = PythonParser.parse_diagnostics(self._LONG_TB_CHAINED_OUTPUT)
        op_errors = [
            d for d in diags
            if d.error_code == "OperationalError"
            and "no such table" in d.message
        ]
        assert len(op_errors) == 1, (
            f"chained duplicate not collapsed: "
            f"{[(d.file, d.line) for d in op_errors]}"
        )
        assert op_errors[0].file == (
            "server/app/repositories/company_repository.py"
        )

    def test_workspace_terminal_keeps_its_anchor(self):
        # A plain assert failure whose terminal line already points at
        # workspace code must be untouched by the re-anchoring.
        output = (
            "____________ test_revenue ____________\n"
            "    def test_revenue():\n"
            ">       assert data == [1000.0]\n"
            "E       assert [] == [1000.0]\n"
            "\n"
            "tests/api/test_financials.py:95: AssertionError\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        precise = [d for d in diags if d.line == 95]
        assert precise
        d = precise[0]
        assert d.file == "tests/api/test_financials.py"
        assert "raised in library frame" not in (d.semantic_context or "")

    def test_no_chain_section_for_single_frame(self):
        # One workspace frame is the diagnostic's own location — a
        # one-entry "call chain" would be noise.
        output = (
            "____________ test_single ____________\n"
            "tests/test_single.py:10: in test_single\n"
            "    do_thing()\n"
            "E   ValueError: boom\n"
            "tests/test_single.py:10: ValueError\n"
        )
        diags = PythonParser.parse_diagnostics(output)
        hits = [d for d in diags if d.error_code == "ValueError"]
        assert hits
        assert "workspace call chain" not in (hits[0].semantic_context or "")


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
