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

from harness.graph import (
    _FAILURE_SURFACE_MAX_BYTES,
    _extract_failure_surface,
)

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
