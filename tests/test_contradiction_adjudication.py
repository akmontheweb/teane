"""A proved contradiction is adjudicated and acted on, not just reported.

lumina-run24-20260929-2327: repair flipped ``rate_limiter.py`` between a
limit of 30 and 250. ``_prove_oscillation_contradiction`` proved it — each
state left failures the other cleared — but the judge saw the two sides
only as masked fingerprints (``AssertionError::{'*':'*'}``), never the tests
or criteria behind them, and alternated between them. Even a correct ruling
had nowhere to go: the wrong test was a generated acceptance suite, which
the regeneration node refused to HITL.

Now each side is named by its tests and the criteria they verify, in the
specification's words; the judge may name the wrong test; repair declares
it unsatisfiable with that proof; and the regeneration node rewrites a
generated acceptance scenario through the scenario generator, recording why
so later generations of that criterion do not repeat it.

Also covered: the not-runtime classification (run 24's "CI runs pip-audit"
criterion became a test that read CI files from disk).
"""

from __future__ import annotations

import json
import os

import pytest

from harness import acceptance_gen as ag
from harness import graph as g

STORY_003 = '''"""Acceptance tests for STORY-003 (ADR-0006 integration altitude).

AUTO-GENERATED. Each test asserts one acceptance criterion in-process via
the `client` fixture (real collaborators, seeded DB). The `@verifies:`
marker links the test to its criterion for traceability.
"""

import pytest

# @verifies: STORY-003.AC-1
def test_directory_sorted(client):
    resp = client.get('/api/birthdays')
    assert resp.status_code == 200, resp.text


# @verifies: STORY-003.AC-2
def test_large_directory_paginates_100_per_page(client):
    for i in range(250):
        resp = client.post('/api/birthdays', json={'first_name': f'F{i}'})
        assert resp.status_code == 201, resp.text
    assert client.get('/api/birthdays').json()['total'] == 250
'''

NFR_004 = '''"""Acceptance tests for STORY-NFR-004 (ADR-0006 integration altitude).

AUTO-GENERATED. Each test asserts one acceptance criterion in-process via
the `client` fixture (real collaborators, seeded DB). The `@verifies:`
marker links the test to its criterion for traceability.
"""

import pytest

# @verifies: STORY-NFR-004.AC-1
def test_rate_limit_blocks_mutation_endpoints(client):
    for i in range(30):
        assert client.post('/api/birthdays', json={}).status_code == 201
    throttled = client.post('/api/birthdays', json={})
    assert throttled.status_code == 429, throttled.text
'''

SPEC = {
    "STORY-003.AC-2": "**Scenario: Large directory is paginated**\n"
                      "- Given the database contains 250 employee records",
    "STORY-NFR-004.AC-1": "**Scenario: Rate limiting protects mutation endpoints**\n"
                          "- Given I issue more than 30 mutation requests from one IP"
                          " within one minute\n- Then the API returns 429",
}
S003 = "tests/acceptance/test_story_003_acceptance.py"
SN04 = "tests/acceptance/test_story_nfr_004_acceptance.py"
FP_A = "AssertionError::AssertionError: {'*':'*'}"
FP_B = "AssertionError::AssertionError: {'*':31,'*':'*'}"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    d = tmp_path / "tests" / "acceptance"
    d.mkdir(parents=True)
    (d / "test_story_003_acceptance.py").write_text(STORY_003)
    (d / "test_story_nfr_004_acceptance.py").write_text(NFR_004)
    monkeypatch.setattr(ag, "criterion_as_specified",
                        lambda _ws, key, **_k: SPEC.get(key, ""))
    return str(tmp_path)


def _errors():
    return [
        {"file": S003, "line": 20, "severity": "error", "error_code": "AssertionError",
         "message": 'AssertionError: {"detail":"Rate limit exceeded"}'},
        {"file": SN04, "line": 14, "severity": "error", "error_code": "AssertionError",
         "message": "AssertionError: {'rows':31,'x':'y'}"},
    ]


def _proved(ws):
    lc: dict = {}
    errs = _errors()
    g._record_fingerprint_locations(lc, errs, ws)
    fp_a, fp_b = (g._diagnostic_fingerprint(e) for e in errs)
    con = {"file": "server/app/services/rate_limiter.py",
           "only_a": [fp_a], "only_b": [fp_b]}
    con["sides"] = g._contradiction_sides(con, lc, ws)
    lc["oscillation_contradiction"] = con
    return lc, con


class TestTheSidesAreNamed:
    def test_each_side_names_its_test_and_criterion_in_spec_words(self, ws):
        _, con = _proved(ws)
        a, b = con["sides"]
        assert a["tests"] == [f"{S003}::test_large_directory_paginates_100_per_page"]
        assert a["criteria"] == {"STORY-003.AC-2": SPEC["STORY-003.AC-2"]}
        assert b["tests"] == [f"{SN04}::test_rate_limit_blocks_mutation_endpoints"]
        assert "returns 429" in b["criteria"]["STORY-NFR-004.AC-1"]

    def test_the_judge_sees_them_and_may_name_the_wrong_test(self, ws):
        _, con = _proved(ws)
        prompt = g._build_repair_reflection_prompt(
            prior_diagnostics_count=2, current_diagnostics_count=2,
            resolved_fingerprints=[], persisted_fingerprints=[FP_A],
            new_fingerprints=[], top_persisted_diagnostics=[],
            oscillation_contradiction=con,
        )
        assert "Given the database contains 250 employee records" in prompt
        assert "more than 30 mutation requests" in prompt
        assert "test_large_directory_paginates_100_per_page" in prompt
        assert '"wrong_test"' in prompt

    def test_wrong_test_is_only_asked_for_with_a_proof(self):
        prompt = g._build_repair_reflection_prompt(
            prior_diagnostics_count=1, current_diagnostics_count=1,
            resolved_fingerprints=[], persisted_fingerprints=[FP_A],
            new_fingerprints=[], top_persisted_diagnostics=[],
        )
        assert '"wrong_test"' not in prompt

    def test_the_parser_keeps_the_file_part(self):
        v = g._parse_repair_reflection_verdict(json.dumps({
            "verdict": "DISTRACTION", "real_blocker": "STORY-003 seeds via the API",
            "recommendation": "", "wrong_test": f"{S003}::test_large",
        }))
        assert v["wrong_test"] == S003


class TestTheRulingIsActedOn:
    def _state(self, ws):
        return {"workspace_path": ws, "compiler_errors": _errors()}

    def test_a_ruling_on_a_proved_side_is_declared(self, ws):
        lc, _ = _proved(ws)
        verdict = {"verdict": "DISTRACTION", "real_blocker": "seeds a Given via POST",
                   "recommendation": "", "wrong_test": S003}
        path, reason = g._adjudicated_wrong_test(verdict, lc, self._state(ws))
        assert path == S003 and "seeds a Given via POST" in reason

    def test_a_file_outside_the_proof_is_ignored(self, ws):
        lc, _ = _proved(ws)
        verdict = {"verdict": "DISTRACTION", "real_blocker": "x",
                   "recommendation": "", "wrong_test": "tests/acceptance/other.py"}
        assert g._adjudicated_wrong_test(verdict, lc, self._state(ws)) is None

    def test_no_proof_no_ruling(self, ws):
        verdict = {"verdict": "DISTRACTION", "real_blocker": "x",
                   "recommendation": "", "wrong_test": S003}
        assert g._adjudicated_wrong_test(verdict, {}, self._state(ws)) is None

    def test_the_router_treats_the_proof_as_machine_proof(self, ws):
        state = {
            "workspace_path": ws, "compiler_errors": _errors(),
            "loop_counter": {},
            "node_state": {"unsatisfiable_test": S003,
                           "unsatisfiable_test_proof": "oscillation_contradiction"},
            "test_regeneration_config": {"enabled": True, "tier_b_auto": False,
                                         "max_attempts_per_test": 2},
        }
        assert g.route_after_unsatisfiable(state, S003) == "test_regeneration_node"
        state["node_state"].pop("unsatisfiable_test_proof")
        assert g.route_after_unsatisfiable(state, S003) == "human_intervention_node"


class TestSplice:
    def test_only_the_named_criterion_is_replaced(self):
        new = ag.AcceptanceScenario(
            "STORY-003.AC-2", "test_large_directory_paginates_100_per_page",
            ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND,
            "client.seed('employees', [{'first_name': 'F'}])\n"
            "assert client.get('/api/birthdays').status_code == 200")
        out = ag.splice_integration_scenarios(STORY_003, {"STORY-003.AC-2": new})
        assert "client.seed('employees'" in out
        assert "range(250)" not in out
        head = STORY_003.split("# @verifies: STORY-003.AC-2")[0]
        assert out.startswith(head), "AC-1's test must be byte-for-byte untouched"
        compile(out, "x", "exec")

    def test_none_removes_the_test(self):
        out = ag.splice_integration_scenarios(STORY_003, {"STORY-003.AC-2": None})
        assert "STORY-003.AC-2" not in out and "test_directory_sorted" in out
        compile(out, "x", "exec")


class TestFeedback:
    def test_notes_persist_and_reach_the_prompt(self, tmp_path):
        d = str(tmp_path)
        ag.record_scenario_feedback(d, "STORY-003.AC-2", "seeded a Given via POST")
        ag.record_scenario_feedback(d, "STORY-003.AC-2", "seeded a Given via POST")
        assert ag.load_scenario_feedback(d) == {
            "STORY-003.AC-2": ["seeded a Given via POST"]}
        ctx = ag.StoryAcceptanceContext(
            story_key="STORY-003", title="t", description="",
            acceptance_criteria=[{"ac_key": "STORY-003.AC-2", "text": "Large"}],
            scenario_feedback=ag.load_scenario_feedback(d),
        )
        prompt = ag.build_user_prompt(ctx, max_scenarios=2)
        assert "ruled WRONG" in prompt and "seeded a Given via POST" in prompt


class TestRegenerationNode:
    @pytest.mark.asyncio
    async def test_the_wrong_acceptance_scenario_is_rewritten(self, ws, monkeypatch):
        from harness import test_regeneration as tr

        seen = {}

        def fake_ctx(workspace, story, **kw):
            seen["suite_dir"] = kw.get("suite_dir")
            return ag.StoryAcceptanceContext(
                story_key=story, title="t", description="",
                acceptance_criteria=[
                    {"ac_key": "STORY-003.AC-1", "text": "sorted"},
                    {"ac_key": "STORY-003.AC-2", "text": "Large"}],
                scenario_feedback=ag.load_scenario_feedback(kw["suite_dir"]),
            )

        async def fake_gen(ctx, **kw):
            seen["asked"] = sorted(ctx.ac_keys())
            seen["feedback"] = ctx.scenario_feedback
            return ag.AcceptanceGenResult(story_key=ctx.story_key, scenarios=[
                ag.AcceptanceScenario(
                    "STORY-003.AC-2", "test_large_directory_paginates_100_per_page",
                    ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND,
                    "ids = client.seed('employees', [{'first_name': 'F'}])\n"
                    "assert client.get('/api/birthdays').status_code == 200"),
            ], budget_remaining_usd=4.5)

        monkeypatch.setattr(ag, "gather_story_acceptance_context", fake_ctx)
        monkeypatch.setattr(ag, "generate_acceptance_scenarios", fake_gen)
        monkeypatch.setattr(g, "get_gateway", lambda: object())

        state = {
            "workspace_path": ws,
            "node_state": {"unsatisfiable_test": S003,
                           "unsatisfiable_test_reason": "seeds a Given through a throttled POST"},
            "loop_counter": {},
            "test_regeneration_config": {"enabled": True, "require_code_linkage": True},
            "acceptance_config": {"scenario_source": "llm"},
            "messages": [], "budget_remaining_usd": 5.0,
            "compiler_errors": _errors(), "modified_files": [],
        }
        out = await tr.test_regeneration_node(state)

        assert out["node_state"]["test_regeneration"]["status"] == "acceptance_regenerated"
        assert seen["asked"] == ["STORY-003.AC-2"], "only the failing criterion is regenerated"
        assert "throttled POST" in seen["feedback"]["STORY-003.AC-2"][0]
        body = open(os.path.join(ws, S003)).read()
        assert "client.seed('employees'" in body and "range(250)" not in body
        assert "def test_directory_sorted" in body
        assert out["budget_remaining_usd"] == 4.5
        assert S003 in out["modified_files"]


class TestNotRuntime:
    _RAW = {"scenarios": [{
        "verifies": "STORY-NFR-004.AC-5", "classification": "not-runtime",
        "rationale": "verified by the CI pipeline configuration",
        "integration": {"name": "test_ci", "body": "assert open('ci.yml')"},
        "e2e": None,
    }]}

    def _ctx(self):
        return ag.StoryAcceptanceContext(
            story_key="STORY-NFR-004", title="t", description="",
            acceptance_criteria=[{"ac_key": "STORY-NFR-004.AC-5", "text": "CI"}])

    def test_the_prompt_offers_it(self):
        assert '"not-runtime"' in ag.build_system_prompt(db_isolated=True)

    def test_its_criteria_are_kept_and_its_scenarios_dropped(self):
        assert ag._not_runtime_acs(self._RAW, self._ctx()) == {
            "STORY-NFR-004.AC-5": "verified by the CI pipeline configuration"}
        cands = ag._parse_candidates(self._RAW, frozenset({ag.ALTITUDE_INTEGRATION}))
        kept, dropped = ag.validate_scenarios(cands, self._ctx())
        assert kept == [] and "not-runtime" in dropped[0]["reason"]

    def test_it_is_a_deferral(self):
        from harness import acceptance_run as ar
        assert ar.ACOutcome("X", ar.STATUS_DEFERRED_NOT_RUNTIME).is_deferred


class TestAcceptanceGapLocation:
    def test_each_gap_points_at_its_own_test(self, ws):
        # Was written[0], line 0, for every criterion: on run 24 the NFR-004
        # gap was attributed to test_story_nfr_001_acceptance.py.
        from harness import acceptance_node as an
        from harness import acceptance_run as ar
        written = [os.path.join(ws, S003), os.path.join(ws, SN04)]
        [diag] = an._acceptance_diagnostics(
            [ar.ACOutcome("STORY-NFR-004.AC-1", ar.STATUS_ATTRIBUTABLE)], written)
        assert diag["file"] == os.path.join(ws, SN04)
        src = NFR_004.splitlines()
        assert src[diag["line"] - 1].startswith("def test_rate_limit_blocks_mutation_endpoints")
