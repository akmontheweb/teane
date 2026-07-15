"""Diagnostics-visibility fixes from session 22471c0c's second stall.

The repair prompt buried 14 real pytest failures below 456 tsc
type-environment diagnostics, taught the LLM absolute paths the patcher
rejects, and had no way to reveal that the failing test files passed in
isolation (cross-module import-time ``app.dependency_overrides``
pollution). Three fixes, one test class each:

  1. Ranking: test-environment noise (missing jest/@testing-library
     types) never outranks runtime test failures.
  2. The diagnostics gate relativizes pyright/tsc absolute paths at the
     boundary.
  3. Multi-failure isolation reruns: when failures cluster into ≤3 test
     files, each file is re-run alone and files that pass get a
     suite-order-pollution hint on every diagnostic.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.graph import (
    _format_diagnostics_for_repair,
    _maybe_pytest_isolation_rerun_multi,
)


def _diag(code: str, message: str, files: list[str]) -> list[dict]:
    return [
        {
            "file": f,
            "line": 1,
            "column": 0,
            "error_code": code,
            "message": message,
            "severity": "error",
            "semantic_context": "",
        }
        for f in files
    ]


class TestEnvNoiseDemotion:
    """Replicates the 22471c0c shape: huge tsc type-noise groups vs a
    handful of runtime pytest failures. The runtime failures must own
    the top-N full-context slots; the noise goes to the deferred list."""

    @staticmethod
    def _errors() -> list[dict]:
        errors: list[dict] = []
        # 4 distinct env-noise shapes with big counts (test files).
        errors += _diag(
            "TS2307",
            "Cannot find module '@testing-library/react' or its "
            "corresponding type declarations.",
            [f"client/src/__tests__/Panel{i}.test.tsx" for i in range(8)],
        )
        errors += _diag(
            "TS2304", "Cannot find name 'expect'.",
            [f"client/src/__tests__/Panel{i}.test.tsx" for i in range(20)],
        )
        errors += _diag(
            "TS2593",
            "Cannot find name 'describe'. Do you need to install type "
            "definitions for a test runner?",
            [f"client/src/__tests__/Panel{i}.test.tsx" for i in range(9)],
        )
        errors += _diag(
            "TS7026",
            "JSX element implicitly has type 'any' because no interface "
            "'JSX.IntrinsicElements' exists.",
            [f"client/src/Panel{i}.tsx" for i in range(15)],
        )
        # 2 runtime pytest failures + 1 genuine static finding.
        errors += _diag(
            "AssertionError", "AssertionError: assert 404 == 400",
            ["server/tests/test_company_search_api.py"],
        )
        errors += _diag(
            "OperationalError",
            "OperationalError: sqlalchemy.exc.OperationalError: "
            "(sqlite3.OperationalError) no such table: companies",
            ["server/tests/test_health_score_api.py"],
        )
        errors += _diag(
            "F821", "Undefined name 'compute_score'",
            ["server/app/services/health_score_service.py"],
        )
        return errors

    def test_runtime_failures_own_the_top_slots(self):
        rendered = _format_diagnostics_for_repair(self._errors())
        shown, _, _ = rendered.partition("### Suppressed")
        assert "assert 404 == 400" in shown
        assert "no such table: companies" in shown

    def test_noise_compressed_into_one_classified_section(self):
        rendered = _format_diagnostics_for_repair(self._errors())
        _, sep, suppressed = rendered.partition(
            "### Suppressed: test-environment type-config noise"
        )
        assert sep, "expected the suppressed-noise section"
        # Shapes are listed as code×count only — the full messages are
        # compressed out of the prompt entirely.
        assert "`TS2304`×20" in suppressed
        assert "`TS2307`×8" in suppressed
        assert "Cannot find name 'expect'" not in rendered
        assert "@testing-library/react" not in rendered
        # The class explanation tells the LLM what NOT to do.
        assert "not code defects" in suppressed
        assert "Do NOT patch" in suppressed

    def test_noise_excluded_from_structured_payload(self):
        rendered = _format_diagnostics_for_repair(self._errors())
        _, sep, payload = rendered.partition("### Structured payload")
        assert sep, "expected the structured payload section"
        assert "TS2304" not in payload
        assert "TS2307" not in payload
        # Real failures remain machine-readable.
        assert "AssertionError" in payload

    def test_genuine_static_finding_not_demoted(self):
        # F821 on a production file is not env noise — it stays in the
        # ranked sections, ahead of the suppressed summary.
        rendered = _format_diagnostics_for_repair(self._errors())
        assert rendered.find("compute_score") < rendered.find("### Suppressed")

    def test_env_codes_on_production_files_with_env_message_compressed(self):
        # TS7026 hits production .tsx files, but the message is the
        # missing-JSX-types signature — env noise regardless of path.
        rendered = _format_diagnostics_for_repair(self._errors())
        assert "JSX element implicitly" not in rendered
        _, _, suppressed = rendered.partition("### Suppressed")
        assert "`TS7026`×15" in suppressed

    def test_all_noise_run_is_not_compressed(self):
        # When env noise is ALL there is, it is the diagnosis — the
        # prompt must show it rather than compress itself to nothing.
        noise_only = _diag(
            "TS2304", "Cannot find name 'expect'.",
            [f"client/src/__tests__/P{i}.test.tsx" for i in range(6)],
        ) + _diag(
            "TS2307",
            "Cannot find module '@testing-library/react' or its "
            "corresponding type declarations.",
            [f"client/src/__tests__/P{i}.test.tsx" for i in range(6)],
        )
        rendered = _format_diagnostics_for_repair(noise_only)
        assert "### Suppressed" not in rendered
        assert "Cannot find name 'expect'" in rendered

    def test_promoted_noise_code_escapes_compression(self):
        rendered = _format_diagnostics_for_repair(
            self._errors(), promoted_codes={"TS2307"},
        )
        # The promoted shape gets full context back...
        assert "@testing-library/react" in rendered
        # ...while its unpromoted siblings stay compressed.
        assert "Cannot find name 'expect'" not in rendered


class TestGateRelativizesPaths:
    def test_run_checker_relativizes_absolute_paths(self, monkeypatch, tmp_path):
        from harness import diagnostics_gate as dg
        from harness.parser_registry import DiagnosticObject

        abs_file = str(tmp_path / "server" / "app" / "repo.py")

        class _StubParser:
            @staticmethod
            def parse_diagnostics(output: str):
                return [DiagnosticObject(
                    file=abs_file, line=7, column=0, severity="error",
                    error_code="reportMissingImports",
                    message='Import "sqlalchemy.exc" could not be resolved',
                    semantic_context="",
                )]

        class _Spec:
            tool = "pyright"
            argv = ["pyright"]
            parser = _StubParser
            filter_output = False
            scoped_files: list[str] = []

        async def _fake_run(argv, timeout, cwd):
            return 1, b"output", b"", False

        monkeypatch.setattr(dg, "run_subprocess_kill_on_timeout", _fake_run)
        diags, timed_out = asyncio.run(
            dg.run_checker(_Spec(), str(tmp_path), timeout=5.0)
        )
        assert not timed_out
        assert diags[0].file == "server/app/repo.py", (
            "absolute checker paths must be relativized at the gate "
            "boundary — the patcher rejects '/'-prefixed patch targets"
        )


class _StubExecutor:
    """Records commands; returns a canned exit code per selector."""

    def __init__(self, exit_codes: dict[str, int]):
        self.exit_codes = exit_codes
        self.commands: list[str] = []

    async def run(self, cmd: str):
        self.commands.append(cmd)
        exit_code = 1
        for selector, code in self.exit_codes.items():
            if cmd.rstrip().endswith(selector):
                exit_code = code
                break

        class _R:
            pass

        r = _R()
        r.exit_code = exit_code
        return r


class TestMultiFileIsolation:
    @staticmethod
    def _failures(files_and_tests: list[tuple[str, str]]) -> list[dict]:
        return [
            {
                "file": f, "line": 1, "severity": "error",
                "error_code": "AssertionError", "message": "assert False",
                "semantic_context": "",
                "pytest_nodeid": f"{f}::{t}",
            }
            for f, t in files_and_tests
        ]

    def test_passing_file_gets_pollution_hint_on_every_failure(self):
        failures = self._failures([
            ("server/tests/test_company_search_api.py", "test_a"),
            ("server/tests/test_company_search_api.py", "test_b"),
            ("server/tests/test_health_score_api.py", "test_c"),
        ])
        executor = _StubExecutor({
            "server/tests/test_company_search_api.py": 0,  # passes alone
            "server/tests/test_health_score_api.py": 1,    # real bug
        })
        lc: dict = {}
        asyncio.run(_maybe_pytest_isolation_rerun_multi(
            executor, failures, lc, "python3 -m pytest",
        ))
        company = [f for f in failures if "company" in f["file"]]
        health = [f for f in failures if "health" in f["file"]]
        for f in company:
            assert "ISOLATION SIGNAL" in f["semantic_context"]
            assert "dependency_overrides" in f["semantic_context"]
        for f in health:
            assert "ISOLATION SIGNAL" not in f["semantic_context"]

    def test_results_cached_per_file(self):
        failures = self._failures([
            ("server/tests/test_a.py", "test_1"),
            ("server/tests/test_a.py", "test_2"),
        ])
        executor = _StubExecutor({"server/tests/test_a.py": 0})
        lc: dict = {}
        asyncio.run(_maybe_pytest_isolation_rerun_multi(
            executor, failures, lc, "python3 -m pytest",
        ))
        asyncio.run(_maybe_pytest_isolation_rerun_multi(
            executor, self._failures([("server/tests/test_a.py", "test_1")] * 2),
            lc, "python3 -m pytest",
        ))
        # Only one sandbox invocation despite two rounds.
        assert len(executor.commands) == 1
        assert lc["_isolation_rerun_cache"] == {
            "file::server/tests/test_a.py": True,
        }

    def test_skips_when_failures_span_too_many_files(self):
        failures = self._failures([
            (f"server/tests/test_{i}.py", "test_x") for i in range(4)
        ])
        executor = _StubExecutor({})
        asyncio.run(_maybe_pytest_isolation_rerun_multi(
            executor, failures, {}, "python3 -m pytest",
        ))
        assert executor.commands == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


class TestTargetedTestsFirst:
    """Fail-to-pass fast path: re-run the previous round's failing pytest
    selectors before paying for the full suite (SWE-bench methodology)."""

    @staticmethod
    def _state(errors, cfg=None):
        s = {"compiler_errors": errors}
        if cfg is not None:
            s["compiler_config"] = cfg
        return s

    @staticmethod
    def _fail(nodeid):
        return {
            "file": nodeid.split("::")[0], "line": 1, "severity": "error",
            "error_code": "AssertionError", "message": "assert False",
            "pytest_nodeid": nodeid,
        }

    def test_eligible_when_all_failures_have_nodeids(self):
        from harness.graph import _targeted_tests_first_selectors
        state = self._state([
            self._fail("tests/test_a.py::test_1"),
            self._fail("tests/test_b.py::test_2"),
        ])
        assert _targeted_tests_first_selectors(
            state, "python3 -m pytest -q", is_pre_exit_verify=False,
        ) == ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]

    def test_ineligible_when_any_failure_lacks_nodeid(self):
        # A non-pytest diagnostic in the prior set means a targeted run
        # would only cover a subset and confuse progress fingerprints.
        from harness.graph import _targeted_tests_first_selectors
        state = self._state([
            self._fail("tests/test_a.py::test_1"),
            {"file": "x.py", "line": 1, "severity": "error",
             "error_code": "F821", "message": "undefined name"},
        ])
        assert _targeted_tests_first_selectors(
            state, "python3 -m pytest -q", is_pre_exit_verify=False,
        ) == []

    def test_ineligible_over_nodeid_cap(self):
        from harness.graph import _targeted_tests_first_selectors
        state = self._state([
            self._fail(f"tests/test_a.py::test_{i}") for i in range(6)
        ])
        assert _targeted_tests_first_selectors(
            state, "python3 -m pytest -q", is_pre_exit_verify=False,
        ) == []

    def test_ineligible_for_non_pytest_build(self):
        from harness.graph import _targeted_tests_first_selectors
        state = self._state([self._fail("tests/test_a.py::test_1")])
        assert _targeted_tests_first_selectors(
            state, "npx jest --silent", is_pre_exit_verify=False,
        ) == []

    def test_ineligible_on_pre_exit_verify(self):
        from harness.graph import _targeted_tests_first_selectors
        state = self._state([self._fail("tests/test_a.py::test_1")])
        assert _targeted_tests_first_selectors(
            state, "python3 -m pytest -q", is_pre_exit_verify=True,
        ) == []

    def test_config_opt_out(self):
        from harness.graph import _targeted_tests_first_selectors
        state = self._state(
            [self._fail("tests/test_a.py::test_1")],
            cfg={"targeted_tests_first": False},
        )
        assert _targeted_tests_first_selectors(
            state, "python3 -m pytest -q", is_pre_exit_verify=False,
        ) == []

    def test_no_prior_errors_is_ineligible(self):
        from harness.graph import _targeted_tests_first_selectors
        assert _targeted_tests_first_selectors(
            self._state([]), "python3 -m pytest -q",
            is_pre_exit_verify=False,
        ) == []

    def test_config_key_wired_into_validator_and_template(self):
        import json as _json
        from pathlib import Path
        from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA
        assert "targeted_tests_first" in _KNOWN_NESTED_KEYS["compiler"]
        assert _TYPE_SCHEMA["compiler.targeted_tests_first"] == (bool,)
        repo_root = Path(__file__).resolve().parents[1]
        cfg = _json.loads(
            (repo_root / "config" / "config.json").read_text(encoding="utf-8")
        )
        assert cfg["compiler"]["targeted_tests_first"] is True
