"""A failure the APPLICATION caused must never be declared an unsatisfiable test.

lumina 01a1de20 (run 16, 2026-11-15) ended 0/16 on ``unsatisfiable_test`` 6/6
against ``tests/acceptance/test_story_001_acceptance.py`` — a test that was
right. Two production defects were behind it:

* ``birthdays.py`` raised ``HTTPException(detail=<error envelope>)``, so the
  framework wrapped the envelope: ``{"detail":{"error":{...}}}``. The test read
  ``body["error"]`` per the contract and got ``KeyError: 'error'``.
* the route's ``except Exception`` turned a real crash into a generic
  ``{"code":"APP_ERROR","message":"Internal server error"}`` 500 and threw the
  traceback away, so repair had nothing to work from.

Repair declined the UNSATISFIABLE_TEST offer four times — correctly — and the
declined-offer floor declared the test unsatisfiable on its behalf anyway; the
judge's contradiction ruling did the same once. These tests pin both halves of
the fix: the evidence gate on every test-blaming path, and the plugin layer
that recovers the swallowed traceback.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from harness import graph as g
from harness import sandbox as sb
from harness.parser_registry import PythonParser

T001 = "tests/acceptance/test_story_001_acceptance.py"

# Verbatim shapes from the run-16 repair dump (01a1de20_0219).
ENVELOPE_DIAG = {
    "file": T001, "line": 57, "severity": "error", "error_code": "KeyError",
    "message": "KeyError: 'error'",
    "semantic_context": (
        "body = {'detail': {'error': {'code': 'INVALID_DAYS', "
        "'message': 'days must be a positive integer'}}}\n"
        "runtime object detail (harness enrichment):\n  body:    "
        "'{\"detail\":{\"error\":{\"code\":\"INVALID_DAYS\"}}}'"
    ),
}
CRASH_DIAG = {
    "file": T001, "line": 21, "severity": "error", "error_code": "AssertionError",
    "message": (
        'AssertionError: {"detail":{"error":{"code":"APP_ERROR",'
        '"message":"Internal server error","error_ref_id":"err_x"}}}'
    ),
}
# A plain assertion with nothing the application can be blamed for.
PLAIN_DIAG = {
    "file": T001, "line": 30, "severity": "error", "error_code": "AssertionError",
    "message": "AssertionError: assert 3 == 4",
}


def _state(*diags, ws="/ws"):
    return {"workspace_path": ws, "compiler_errors": list(diags)}


class TestTheEvidence:
    def test_a_double_wrapped_error_envelope_is_production(self):
        why = g._code_gap_evidence(_state(ENVELOPE_DIAG), T001)
        assert "detail" in why and "envelope" in why

    def test_run16_crash_body_is_production(self):
        # Verbatim run-16 body: double-wrapped AND a crash; either suffices.
        assert g._code_gap_evidence(_state(CRASH_DIAG), T001)

    def test_a_generic_internal_server_error_body_is_production(self):
        d = dict(CRASH_DIAG, message=(
            'AssertionError: {"error":{"code":"APP_ERROR",'
            '"message":"Internal server error"}}'
        ))
        assert "crashed" in g._code_gap_evidence(_state(d), T001)

    def test_a_500_where_the_test_expected_success_is_production(self):
        d = dict(PLAIN_DIAG, message="AssertionError: assert 500 == 200")
        assert g._code_gap_evidence(_state(d), T001)

    def test_the_plugins_swallowed_exception_line_is_production(self):
        d = dict(PLAIN_DIAG, semantic_context=(
            "runtime object detail (harness enrichment):\n  exception "
            "swallowed by application code: TypeError: x — caught by broad "
            "`except Exception` at server/app/api/birthdays.py:92"
        ))
        assert "broad" in g._code_gap_evidence(_state(d), T001)

    def test_a_plain_assertion_is_not_evidence(self):
        assert g._code_gap_evidence(_state(PLAIN_DIAG), T001) == ""

    def test_the_top_level_envelope_the_contract_asks_for_is_not_evidence(self):
        d = dict(PLAIN_DIAG, message=(
            'AssertionError: {"error":{"code":"NOT_FOUND","message":"x"}}'
        ))
        assert g._code_gap_evidence(_state(d), T001) == ""

    def test_evidence_is_per_file(self):
        assert g._production_fault_evidence(
            _state(CRASH_DIAG), {"tests/acceptance/test_story_002_acceptance.py"},
        ) == ("", "")
        f, why = g._production_fault_evidence(_state(CRASH_DIAG), {T001})
        assert f == T001 and why


class TestTheOfferIsWithheld:
    @pytest.mark.parametrize("kwargs", [
        {"has_mandatable_target": False, "distraction_streak": 0},
        {"has_mandatable_target": True, "distraction_streak": 5},
        {"has_mandatable_target": True, "distraction_streak": 0,
         "low_signal_streak": 5},
        {"has_mandatable_target": True, "distraction_streak": 0,
         "triage_test_bug": True},
    ])
    def test_evidence_vetoes_every_trigger(self, kwargs):
        assert g._should_offer_unsatisfiable_escape(
            has_guarded_blocker=True, **kwargs,
        ) is True
        assert g._should_offer_unsatisfiable_escape(
            has_guarded_blocker=True, production_fault_evidence="crashed",
            **kwargs,
        ) is False

    def test_the_banner_tells_repair_where_the_fault_is(self):
        text = g._production_fault_banner(T001, "the application crashed")
        assert T001 in text and "production code" in text
        assert "NOT" in text and "UNSATISFIABLE_TEST" in text


class TestTheFloorNeverForcesAgainstEvidence:
    def _floor(self, declined, **kw):
        return g._apply_unsat_offer_floor(
            declined, {T001}, any_real_patch=False, already_declared=False,
            floor=3, **kw,
        )

    def test_run16_four_declines_no_longer_declare_the_test(self):
        declined: dict[str, int] = {}
        for _ in range(4):
            assert self._floor(
                declined, production_fault_evidence="envelope double-wrapped",
            ) is None
        # Cleared, not held at the floor: a later round without evidence
        # must earn the floor from scratch, not fire a spent count.
        assert declined == {}

    def test_without_evidence_the_floor_still_fires(self):
        declined: dict[str, int] = {}
        assert self._floor(declined) is None
        assert self._floor(declined) is None
        path, reason = self._floor(declined)
        assert path == T001
        assert "no round making progress" in reason


class TestTheJudgeCannotOverruleEvidence:
    def _proof(self, ws):
        return {"oscillation_contradiction": {
            "sides": [{"tests": [f"{T001}::test_x"]}, {"tests": ["tests/b.py::t"]}],
        }}

    def test_a_ruling_against_a_production_fault_is_ignored(self, tmp_path):
        ws = str(tmp_path)
        verdict = {"verdict": "DISTRACTION", "real_blocker": "x",
                   "recommendation": "", "wrong_test": T001}
        assert g._adjudicated_wrong_test(
            verdict, self._proof(ws), _state(ENVELOPE_DIAG, ws=ws),
        ) is None

    def test_a_ruling_without_evidence_still_stands(self, tmp_path):
        ws = str(tmp_path)
        verdict = {"verdict": "DISTRACTION", "real_blocker": "x",
                   "recommendation": "", "wrong_test": T001}
        got = g._adjudicated_wrong_test(
            verdict, self._proof(ws), _state(PLAIN_DIAG, ws=ws),
        )
        assert got is not None and got[0] == T001


# --------------------------------------------------------------------------
# Plugin layer — the swallowed traceback reaches the diagnostic.
# --------------------------------------------------------------------------

APP = '''
class AppError(Exception):
    pass


def load(key):
    return {"a": 1}[key]


def route(key):
    try:
        return load(key)
    except ValueError:
        raise AppError("bad value")
    except Exception as exc:
        raise AppError("Internal server error") from exc


def narrow(text):
    try:
        return int(text)
    except ValueError:
        return -1


def tolerant(key):
    try:
        return load(key)
    except Exception:
        return None
'''

TESTS = '''
from app.svc import narrow, route, tolerant


def test_route():
    assert route("zzz") == 1


def test_narrow():
    assert narrow("q") == 5


def test_tolerant_passes():
    assert tolerant("zzz") is None
'''

PLUGIN_DIR = os.path.join(os.path.dirname(g.__file__), "pytest_plugins")

needs_monitoring = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason="sys.monitoring needs Python 3.12+",
)


def _run_toy(tmp_path, **env):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "svc.py").write_text(textwrap.dedent(APP))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(textwrap.dedent(TESTS))
    full_env = {
        **os.environ,
        "PYTHONPATH": f"{PLUGIN_DIR}{os.pathsep}{tmp_path}",
        "PYTEST_ADDOPTS": "",
        **env,
    }
    full_env.pop("TEANE_DIAGNOSTICS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "teane_diagnostics",
         "-p", "no:cacheprovider", "-q", "--tb=long", "tests"],
        cwd=tmp_path, env=full_env, capture_output=True, text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


@needs_monitoring
class TestThePluginRecoversTheSwallowedTraceback:
    def test_a_broad_catch_reports_the_real_exception(self, tmp_path):
        out = _run_toy(tmp_path)
        assert "exception swallowed by application code: KeyError: 'zzz'" in out
        assert "caught by broad `except Exception` at app/svc.py:" in out
        # The real raise site, which the response body never carried.
        assert 'File "app/svc.py", line 7, in load' in out

    def test_a_narrow_handler_is_not_reported(self, tmp_path):
        out = _run_toy(tmp_path)
        assert "ValueError" not in out.split("test_narrow", 1)[-1].split(
            "short test summary", 1)[0].replace("except ValueError", "")

    def test_a_passing_test_reports_nothing(self, tmp_path):
        out = _run_toy(tmp_path)
        assert out.count("exception swallowed by application code") == 1
        assert "2 failed, 1 passed" in out

    def test_zero_disables_it(self, tmp_path):
        out = _run_toy(tmp_path, TEANE_SWALLOWED_MAX="0")
        assert "exception swallowed by application code" not in out
        assert "2 failed, 1 passed" in out

    def test_the_traceback_cap_applies(self, tmp_path):
        out = _run_toy(tmp_path, TEANE_SWALLOWED_TB_LINES="3")
        assert "line(s) elided" in out

    def test_the_line_survives_the_parser_and_is_evidence(self, tmp_path):
        out = _run_toy(tmp_path)
        diags = [d.to_dict() for d in PythonParser.parse_diagnostics(out)]
        route = [d for d in diags if "exception swallowed" in str(
            d.get("semantic_context") or "")]
        assert route, diags
        f = route[0]["file"]
        state = {"workspace_path": str(tmp_path), "compiler_errors": diags}
        assert "broad" in g._code_gap_evidence(state, f)


class TestTheCapsComeFromConfig:
    def test_sandbox_config_reaches_the_plugin_env(self, tmp_path):
        ex = sb.SandboxExecutor(
            str(tmp_path), backend=sb.BareBackend(),
            sandbox_config={
                "swallowed_exception_report_max": 4,
                "swallowed_exception_traceback_lines": 12,
            },
        )
        assert ex.extra_env["TEANE_SWALLOWED_MAX"] == "4"
        assert ex.extra_env["TEANE_SWALLOWED_TB_LINES"] == "12"

    def test_an_explicit_env_wins(self, tmp_path):
        ex = sb.SandboxExecutor(
            str(tmp_path), backend=sb.BareBackend(),
            extra_env={"TEANE_SWALLOWED_MAX": "1"},
            sandbox_config={"swallowed_exception_report_max": 4},
        )
        assert ex.extra_env["TEANE_SWALLOWED_MAX"] == "1"

    def test_the_keys_are_known_and_typed(self):
        from harness import cli
        assert "swallowed_exception_report_max" in cli._KNOWN_NESTED_KEYS["sandbox"]
        assert "swallowed_exception_traceback_lines" in cli._KNOWN_NESTED_KEYS["sandbox"]
