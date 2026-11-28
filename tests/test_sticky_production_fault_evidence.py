"""Proof that the APPLICATION is at fault must not evaporate for one round.

lumina run 01a21db1 (2026-11-27) forced ``UNSATISFIABLE_TEST`` onto three
correct acceptance tests. ``_code_gap_evidence`` matches fingerprints against
one round's diagnostic text, and ``_production_fault_evidence_map`` rebuilds
from scratch at every offer site AND at force time, so the per-file veto
flickers -- and the declined-offer floor can force the escape inside the
flicker. All three of this file's declines fell inside one::

    21:48:25  vetoed   {"detail":"Not Found"} -- the route does not exist
    21:53:35  offered  (bare AssertionError; no fingerprint matched)
    21:56:16  offered
    21:57:41  offered  -> declined count reaches the floor of 3
    21:58:19  FORCED   "taking the escape on the model's behalf"
    22:00:29  vetoed   assert 2xx == 4xx -- success where the criterion
                       requires rejection

One test, one root cause, proof on both sides of the gap, and the file never
left the failing set in between. Remembering the proof while the file keeps
failing closes exactly that window.

REJECTED HYPOTHESIS, recorded so it is not retried. The same run logs
``[repair_node:triage] round N: test_bug=0 code_gap=6`` in both of the rounds
whose offers reached the floor, which looks like the harness contradicting
itself -- ADR-0005's classifier gives its POSITIVE verdict a routing
consequence (a test-authoring fingerprint unlocks the escape ahead of every
streak) and its negative verdict none. Vetoing the escape on
``test_bug=0 code_gap=N`` was built and then reverted: ``CODE_GAP`` is the
classifier's RESIDUAL bucket, not a finding ("no test-authoring fingerprint
matched -- default to repair", confidence "n/a"), so that shape holds on
almost every round. It also re-blocks run 20's genuinely unsatisfiable
``test_birthday_contract.py``, which classifies as CODE_GAP too. The
asymmetry is deliberate: the classifier speaks only when it has positive
evidence, so it can never mask a real code defect.
"""
from __future__ import annotations

import json
import pathlib
import types

import pytest

from harness import graph as g

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

T001 = "tests/acceptance/test_story_001_acceptance.py"
T002 = "tests/acceptance/test_story_002_acceptance.py"

# Verbatim run-01a21db1 shapes.
#
# 21:48:25 -- Starlette's own 404 body. The route genuinely did not exist.
ROUTE_MISSING_DIAG = {
    "file": T001, "line": 40, "severity": "error",
    "error_code": "AssertionError",
    "message": 'AssertionError: {"detail":"Not Found"}',
}
# 21:53-21:57 -- the rate-limit assertion, which carries a test innermost
# frame and no fingerprint at all. This is the gap.
RATE_LIMIT_DIAG = {
    "file": T001, "line": 115, "severity": "error",
    "error_code": "AssertionError",
    "message": "AssertionError",
    "semantic_context": "test_rate_limit_101_requests_returns_429",
}
# 22:00:29 -- the same defect, finally legible.
SUCCESS_WHERE_REJECTION_DIAG = {
    "file": T001, "line": 115, "severity": "error",
    "error_code": "AssertionError",
    "message": "AssertionError: assert 200 == 429",
}


def _state(*diags, ws="/ws"):
    return {"workspace_path": ws, "compiler_errors": list(diags)}


def _gw(sticky=True):
    """A stand-in gateway exposing only the flag under test."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            repair_sticky_production_fault_evidence=sticky,
        ),
    )


class TestEvidenceSurvivesAnIllegibleRound:
    def test_the_gap_round_keeps_the_earlier_proof(self):
        sticky: dict[str, str] = {}
        # 21:48:25 -- proof observed and recorded.
        first = g._production_fault_evidence_map(
            _state(ROUTE_MISSING_DIAG), {T001}, sticky=sticky,
        )
        assert "route does not exist" in first[T001]
        assert T001 in sticky
        # 21:53-21:57 -- same file, still failing, nothing matches.
        gap = g._production_fault_evidence_map(
            _state(RATE_LIMIT_DIAG), {T001}, sticky=sticky,
        )
        assert T001 in gap, "the proof must not evaporate"
        assert "route does not exist" in gap[T001]

    def test_without_sticky_the_gap_round_loses_the_proof(self):
        # The pre-fix behaviour, kept explicit so the test above cannot
        # pass by accident.
        g._production_fault_evidence_map(_state(ROUTE_MISSING_DIAG), {T001})
        assert g._production_fault_evidence_map(
            _state(RATE_LIMIT_DIAG), {T001},
        ) == {}

    def test_fresh_evidence_overwrites_the_remembered_reason(self):
        # 22:00:29 -- the defect became legible again, differently. The
        # banner should quote what this round actually shows.
        sticky: dict[str, str] = {}
        g._production_fault_evidence_map(
            _state(ROUTE_MISSING_DIAG), {T001}, sticky=sticky,
        )
        now = g._production_fault_evidence_map(
            _state(SUCCESS_WHERE_REJECTION_DIAG), {T001}, sticky=sticky,
        )
        assert "requires rejection" in now[T001]
        assert "requires rejection" in sticky[T001]

    def test_evidence_retires_when_the_file_stops_failing(self):
        sticky: dict[str, str] = {}
        g._production_fault_evidence_map(
            _state(ROUTE_MISSING_DIAG), {T001}, sticky=sticky,
        )
        # T001 passes; only T002 fails now.
        other = dict(ROUTE_MISSING_DIAG, file=T002)
        out = g._production_fault_evidence_map(
            _state(other), {T001}, sticky=sticky,
        )
        assert out == {}, "evidence about a passing file is moot"
        assert T001 not in sticky

    def test_a_batch_transition_retires_without_a_reset_hook(self):
        # Every file from the old batch leaves the failing set at once.
        sticky: dict[str, str] = {}
        g._production_fault_evidence_map(
            _state(ROUTE_MISSING_DIAG, dict(ROUTE_MISSING_DIAG, file=T002)),
            {T001, T002}, sticky=sticky,
        )
        assert set(sticky) == {T001, T002}
        g._production_fault_evidence_map(
            _state(dict(ROUTE_MISSING_DIAG, file="tests/acceptance/test_x.py")),
            {"tests/acceptance/test_x.py"}, sticky=sticky,
        )
        assert T001 not in sticky and T002 not in sticky

    def test_the_proof_is_per_file(self):
        # Run 20's correction must survive: one evidence-bearing file does
        # not shield a genuinely unsatisfiable sibling.
        sticky: dict[str, str] = {}
        out = g._production_fault_evidence_map(
            _state(ROUTE_MISSING_DIAG, dict(RATE_LIMIT_DIAG, file=T002)),
            {T001, T002}, sticky=sticky,
        )
        assert T001 in out and T002 not in out

    def test_the_single_file_wrapper_threads_sticky(self):
        sticky: dict[str, str] = {}
        g._production_fault_evidence(
            _state(ROUTE_MISSING_DIAG), {T001}, sticky=sticky,
        )
        f, why = g._production_fault_evidence(
            _state(RATE_LIMIT_DIAG), {T001}, sticky=sticky,
        )
        assert f == T001 and "route does not exist" in why


class TestTheStickyStoreIsSafeAndGated:
    def test_the_store_lives_in_loop_counter_and_is_the_live_object(self):
        lc: dict[str, object] = {}
        store = g._sticky_pf_evidence(lc)
        assert store is not None
        store["x"] = "y"
        assert lc["production_fault_evidence_sticky"] == {"x": "y"}
        assert g._sticky_pf_evidence(lc) is store

    def test_a_checkpoint_round_trip_cannot_poison_it(self):
        lc: dict[str, object] = {
            "production_fault_evidence_sticky": {
                T001: "real", 7: "bad key", T002: None, "empty": "",
            },
        }
        store = g._sticky_pf_evidence(lc)
        assert store == {T001: "real"}
        # Same object, so later mutations still reach loop_counter.
        assert store is lc["production_fault_evidence_sticky"]

    def test_a_non_dict_value_is_replaced(self):
        lc: dict[str, object] = {"production_fault_evidence_sticky": "junk"}
        assert g._sticky_pf_evidence(lc) == {}

    def test_the_flag_can_switch_it_off(self, monkeypatch):
        monkeypatch.setattr(g, "get_gateway", lambda: _gw(sticky=False))
        assert g._sticky_pf_evidence({}) is None
        monkeypatch.setattr(g, "get_gateway", lambda: _gw(sticky=True))
        assert g._sticky_pf_evidence({}) == {}

    def test_a_gateway_fault_defaults_to_remembering(self, monkeypatch):
        def _boom():
            raise RuntimeError("no gateway")
        monkeypatch.setattr(g, "get_gateway", _boom)
        assert g._sticky_pf_evidence({}) == {}

    def test_every_evidence_site_threads_the_store(self):
        src = (REPO_ROOT / "harness" / "graph.py").read_text()
        # Two offer sites, the forcing floor, and the judge's contradiction
        # ruling. A site left unwired reopens the flicker.
        assert src.count("sticky=_sticky_pf_evidence(") == 4

    def test_the_abandon_rewind_clears_remembered_evidence(self):
        # An operator's manual fix invalidates a claim about code they may
        # have just rewritten, and normal retirement cannot catch it: the
        # file may still be failing, for a different reason.
        src = (REPO_ROOT / "harness" / "graph.py").read_text()
        assert 'new_lc["production_fault_evidence_sticky"] = {}' in src


# ---------------------------------------------------------------------------
# Config promotion -- no behaviour switch may be hardcoded
# ---------------------------------------------------------------------------


class TestTheFlagIsPromoted:
    FLAGS = ("sticky_production_fault_evidence",)

    @pytest.mark.parametrize("flag", FLAGS)
    def test_gateway_config_exposes_it(self, flag):
        from harness.gateway import GatewayConfig

        assert hasattr(GatewayConfig, f"repair_{flag}")

    @pytest.mark.parametrize("flag", FLAGS)
    def test_config_json_carries_it_under_repair(self, flag):
        cfg = json.loads((REPO_ROOT / "config" / "config.json").read_text())
        assert isinstance(cfg["repair"][flag], bool)

    @pytest.mark.parametrize("flag", FLAGS)
    def test_cli_accepts_and_type_checks_it(self, flag):
        from harness import cli

        assert flag in cli._KNOWN_NESTED_KEYS["repair"]
        assert cli._TYPE_SCHEMA[f"repair.{flag}"] == (bool,)

    @pytest.mark.parametrize("flag", FLAGS)
    def test_config_help_documents_it(self, flag):
        # Standing rule: every config change updates config_help.md.
        assert flag in (REPO_ROOT / "config" / "config_help.md").read_text()

    @pytest.mark.parametrize("flag", FLAGS)
    def test_both_default_on_and_round_trip(self, flag):
        from harness.gateway import create_gateway_from_config

        assert getattr(
            create_gateway_from_config({}).config, f"repair_{flag}",
        ) is True
        assert getattr(
            create_gateway_from_config(
                {"repair": {flag: False}},
            ).config,
            f"repair_{flag}",
        ) is False
