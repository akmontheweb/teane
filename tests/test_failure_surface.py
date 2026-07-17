"""Tests for _extract_failure_surface — the RAW_BUILD_STDERR excerpt.

Regression for finsearch session b674f3ca: the sandbox streamers return
stdout and stderr CONCATENATED (all stdout, then all stderr), so the old
"last 40 lines of raw_log" was really the tail of stderr. The build
command installed deps with uv (whose progress goes to stderr) before
running pytest (whose report goes to stdout) — the synthesized
diagnostic carried nothing but "+ package==version" lines and the
repair loop burned 12 blind rounds before HITL.
"""

from __future__ import annotations

import asyncio

from harness.graph import (
    _FAILURE_SURFACE_MAX_BYTES,
    _extract_failure_surface,
)
from harness.sandbox import SandboxExecutor

_UV_STDERR = "\n".join(
    f" + package-{i}=={i}.0.0" for i in range(41)
)

_PYTEST_STDOUT = """\
============================= test session starts ==============================
collected 12 items

tests/test_edgar_xbrl_service.py::TestPeriodEndToFiscalStr::test_q1_period FAILED
=================================== FAILURES ===================================
_____________ TestPeriodEndToFiscalStr.test_q1_period ______________
    def test_q1_period(self):
>       assert period_end_to_fiscal_str("2026-03-31") == "Q1 2026"
E       AssertionError: assert 'Q2 2026' == 'Q1 2026'
tests/test_edgar_xbrl_service.py:44: AssertionError
=========================== short test summary info ============================
FAILED tests/test_edgar_xbrl_service.py::TestPeriodEndToFiscalStr::test_q1_period
========================= 1 failed, 11 passed in 2.31s =========================
"""


class TestFinsearchRegression:
    def test_pytest_report_found_under_stderr_noise(self):
        # stdout (pytest report) + stderr (uv install listing) in
        # streamer concatenation order — the exact finsearch shape.
        raw = _PYTEST_STDOUT + "\n" + _UV_STDERR
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "pytest_report"
        assert "AssertionError: assert 'Q2 2026' == 'Q1 2026'" in surface
        assert "test_q1_period" in surface
        # Installer noise must not dominate the excerpt.
        assert "package-40" not in surface

    def test_report_ends_at_summary_line_not_log_end(self):
        raw = _PYTEST_STDOUT + "\n" + _UV_STDERR
        _, surface = _extract_failure_surface(raw)
        assert surface.rstrip().endswith(
            "========================= 1 failed, 11 passed in 2.31s "
            "========================="
        )


class TestStrategies:
    def test_last_report_wins_on_rerun(self):
        # Two pytest reports (e.g. install → test → reinstall → retest):
        # the LAST one is the current state.
        first = _PYTEST_STDOUT.replace("test_q1_period", "test_old_name")
        raw = first + "\n" + _PYTEST_STDOUT
        _, surface = _extract_failure_surface(raw)
        assert "test_q1_period" in surface

    def test_traceback_block_extracted(self):
        raw = _UV_STDERR + """
Traceback (most recent call last):
  File "server/app/main.py", line 3, in <module>
    from app.missing import thing
ModuleNotFoundError: No module named 'app.missing'
""" + _UV_STDERR
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "traceback"
        assert "ModuleNotFoundError: No module named 'app.missing'" in surface

    def test_error_window_when_no_report_or_traceback(self):
        raw = "\n".join(f"build step {i} ok" for i in range(50))
        raw += "\nmake: *** [target] Error 2\nsome trailing line"
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "error_window"
        assert "Error 2" in surface

    def test_tail_fallback_when_no_signal(self):
        raw = "\n".join(f"line {i}" for i in range(100))
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "tail"
        assert "line 99" in surface
        assert "line 0" not in surface

    def test_empty_log(self):
        strategy, surface = _extract_failure_surface("")
        assert strategy == "empty"
        assert "empty" in surface

    def test_jest_summary_extracted(self):
        raw = (
            "npm noise\n"
            "  ● Button renders › shows label\n"
            "    expect(received).toBe(expected)\n"
            "Tests:       1 failed, 4 passed, 5 total\n"
            + _UV_STDERR
        )
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "jest_report"
        assert "shows label" in surface
        assert surface.rstrip().endswith("Tests:       1 failed, 4 passed, 5 total")


_FIXTURE_ERROR_STDOUT = """\
============================= test session starts ==============================
collected 247 items

tests/test_edgar_client.py::TestEdgarClient::test_user_agent PASSED
==================================== ERRORS ====================================
__________ ERROR at setup of TestSearchEndpoint.test_search_by_ticker __________
file /ws/tests/test_companies_api.py, line 11
      async def test_search_by_ticker(self, client: AsyncClient, seed_db):
E       fixture 'client' not found
>       available fixtures: api_client, db_session, seed_companies, tmp_path
>       use 'pytest --fixtures [testpath]' for help on them.

/ws/tests/test_companies_api.py:11
=========================== short test summary info ============================
ERROR tests/test_companies_api.py::TestSearchEndpoint::test_search_by_ticker
================= 198 passed, 14 warnings, 49 errors in 1.30s ==================
"""


class TestFixtureErrorShape:
    """Regression for the second finsearch b674f3ca blindness (2026-07-16).

    Pytest fixture-not-found errors produce a report with NO line matching
    any _CRITICAL_ERROR_PATTERNS entry: ``ERROR at setup of`` and ``ERROR
    tests/...`` have no colon (``\\bERROR:\\s`` misses them), there is no
    FAILED, no Traceback, no exception class name. filter_critical_errors
    then falls back to "last 50 lines" — the stderr installer tail — and
    the whole-log scan of commit ed161ff went blind because it received
    that filtered husk instead of the true capture.
    """

    def test_errors_section_extracted(self):
        raw = _FIXTURE_ERROR_STDOUT + "\n" + _UV_STDERR
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "pytest_report"
        assert "fixture 'client' not found" in surface
        assert surface.rstrip().endswith(
            "================= 198 passed, 14 warnings, 49 errors in 1.30s "
            "=================="
        )
        assert "package-40" not in surface


class _StaticBackend:
    """Minimal SandboxBackend stand-in returning a canned capture."""

    name = "static-test"

    def __init__(self, exit_code: int, output: str) -> None:
        self._exit_code = exit_code
        self._output = output

    async def run(self, **_kwargs):
        return self._exit_code, self._output, False, False


class _NoopValidator:
    def validate_or_raise(self, _command: str) -> None:
        return None


class TestExecutorFullOutput:
    """SandboxExecutor must hand compiler_node the UNFILTERED capture via
    BuildResult.full_output — raw_output alone is the filtered view on
    failure and may have been reduced to the stderr tail."""

    def _run(self, exit_code: int, output: str):
        executor = SandboxExecutor(
            workspace_path="/tmp",
            backend=_StaticBackend(exit_code, output),
            command_validator=_NoopValidator(),
        )
        return asyncio.run(executor.run("python3 -m pytest"))

    def test_full_output_carries_complete_unfiltered_capture(self):
        # Enough trailing stderr noise that the filter's "last 50 lines"
        # fallback (were it to fire) would hold only installer chatter
        # (real finsearch logs ran to thousands of lines).
        long_stderr = "\n".join(
            f" + package-{i}=={i}.0.0" for i in range(80)
        )
        combined = _FIXTURE_ERROR_STDOUT + "\n" + long_stderr
        result = self._run(1, combined)
        # The unfiltered capture is intact — full_output is load-bearing
        # for every failure shape the filter patterns don't recognise.
        assert result.full_output == combined
        # ...and feeding it to the scanner recovers the real failure.
        strategy, surface = _extract_failure_surface(result.full_output)
        assert strategy == "pytest_report"
        assert "fixture 'client' not found" in surface

    def test_full_output_equals_raw_output_on_success(self):
        result = self._run(0, "all good\n")
        assert result.full_output == result.raw_output == "all good\n"


class TestFixtureErrorFilter:
    """filter_critical_errors must recognise pytest's ERROR-section shapes.

    Before the pattern additions, a report whose only failures were
    fixture-lookup errors matched NOTHING critical: "ERROR at setup of"
    has no colon (``\\bERROR:\\s`` misses), summary rows are "ERROR
    path.py::node" (no colon, no dash tail), and the cause line names no
    exception class. The no-match fallback then reduced the whole log to
    its last 50 lines — the stderr installer tail (finsearch b674f3ca).
    """

    def test_fixture_error_report_survives_filtering(self):
        from harness.sandbox import filter_critical_errors
        long_stderr = "\n".join(
            f" + package-{i}=={i}.0.0" for i in range(80)
        )
        combined = _FIXTURE_ERROR_STDOUT + "\n" + long_stderr
        filtered = filter_critical_errors(combined)
        assert "fixture 'client' not found" in filtered
        assert "ERROR at setup of TestSearchEndpoint.test_search_by_ticker" in filtered
        # Summary row (± context) — the nodeid the isolation re-run needs.
        assert "ERROR tests/test_companies_api.py::TestSearchEndpoint" in filtered
        # The critical-match path fired (context blocks), not the
        # last-50-lines fallback: deep installer noise must be gone.
        assert "package-40" not in filtered

    def test_lowercase_prose_does_not_match_error_shapes(self):
        # App logging like "error at setup of connection pool" must not
        # drag context blocks in — the new ERROR patterns are
        # case-sensitive because pytest always prints uppercase.
        from harness.sandbox import _is_critical_line
        assert not _is_critical_line("retrying error at setup of pool")
        assert _is_critical_line("ERROR at setup of TestX.test_y")
        assert _is_critical_line("ERROR tests/test_x.py::TestX::test_y")
        assert _is_critical_line("ERROR collecting tests/test_x.py")
        assert _is_critical_line(
            "==================================== ERRORS "
            "===================================="
        )
        assert _is_critical_line("E       fixture 'client' not found")


class TestFixtureErrorParser:
    """PythonParser must emit structured diagnostics for fixture-lookup
    errors — before this, the block has no typed E-line and no terminal
    ``file:line: ErrorType`` row, and the summary rows have no `` - ``
    tail, so the whole shape parsed to zero diagnostics and the harness
    logged "extracted 0 fresh diagnostics" against a 49-error run."""

    def _parse(self):
        from harness.parser_registry import PythonParser
        return PythonParser.parse_diagnostics(_FIXTURE_ERROR_STDOUT)

    def test_block_yields_anchored_fixture_diagnostic(self):
        diags = self._parse()
        fixture_diags = [
            d for d in diags if d.error_code == "FixtureLookupError"
        ]
        assert len(fixture_diags) == 1
        d = fixture_diags[0]
        assert d.file == "/ws/tests/test_companies_api.py"
        assert d.line == 11
        assert "fixture 'client' not found" in d.message

    def test_summary_row_yields_nodeid_diagnostic(self):
        diags = self._parse()
        with_nodeid = [d for d in diags if d.pytest_nodeid]
        assert len(with_nodeid) == 1
        assert with_nodeid[0].pytest_nodeid == (
            "tests/test_companies_api.py::TestSearchEndpoint"
            "::test_search_by_ticker"
        )

    def test_dashless_summary_row_still_parses_with_reason(self):
        # The classic dash-tail form must keep working after making the
        # tail optional.
        from harness.parser_registry import PythonParser
        diags = PythonParser.parse_diagnostics(
            "=========================== short test summary info "
            "============================\n"
            "FAILED tests/test_x.py::test_one - AssertionError: assert 1 == 2\n"
        )
        assert len(diags) == 1
        assert diags[0].error_code == "AssertionError"
        assert diags[0].message == "assert 1 == 2"
        assert diags[0].pytest_nodeid == "tests/test_x.py::test_one"


class TestEndAnchoring:
    def test_embedded_summary_in_captured_output_does_not_end_section(self):
        # A test that shells out to pytest echoes a pytest-style summary
        # line inside its own captured output — the section must end at
        # the REAL (last) summary line, not the embedded one.
        raw = (
            "=================================== FAILURES "
            "===================================\n"
            "____________ test_runs_child_pytest ____________\n"
            "    captured stdout:\n"
            "========================= 1 failed, 2 passed in 0.10s "
            "=========================\n"
            ">       assert child.returncode == 0\n"
            "E       AssertionError: child pytest failed\n"
            "tests/test_meta.py:9: AssertionError\n"
            "=========================== short test summary info "
            "============================\n"
            "FAILED tests/test_meta.py::test_runs_child_pytest\n"
            "========================= 1 failed, 5 passed in 1.00s "
            "=========================\n"
            + _UV_STDERR
        )
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "pytest_report"
        # The real summary (and everything before it) survives.
        assert "AssertionError: child pytest failed" in surface
        assert "1 failed, 5 passed in 1.00s" in surface
        assert "package-40" not in surface

    def test_missing_summary_keeps_report_front_not_stderr_tail(self):
        # Killed run: FAILURES section starts but no closing summary line
        # exists. The section runs to EOF — the clip must keep the FRONT
        # (the failures) rather than head+tail halves whose tail is the
        # stderr installer noise.
        body = "\n".join(
            f"tests/test_x.py::test_{i} FAILED  - boom" for i in range(200)
        )
        raw = (
            "=================================== FAILURES "
            "===================================\n"
            + body + "\n"
            + _UV_STDERR
        )
        strategy, surface = _extract_failure_surface(raw)
        assert strategy == "pytest_report"
        assert "test_0 FAILED" in surface
        assert "package-40" not in surface


class TestBounds:
    def test_huge_report_is_clipped_keeping_the_end(self):
        body = "\n".join(
            f"tests/test_x.py::test_{i} FAILED  - boom" for i in range(2000)
        )
        raw = (
            "=================================== FAILURES "
            "===================================\n"
            + body
            + "\n========================= 2000 failed in 60.00s "
            "=========================\n"
        )
        _, surface = _extract_failure_surface(raw)
        assert len(surface) <= _FAILURE_SURFACE_MAX_BYTES + 100
        # The end (summary line) survives clipping.
        assert "2000 failed in 60.00s" in surface
