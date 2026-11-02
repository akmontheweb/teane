"""A test that cannot RUN is not repair's problem, and must not kill the run.

lumina run 12 ended **0/16** — worse than run 11's 4/16 — holding 13 of 17
criteria verified with nothing attributable left. Acceptance regeneration
emitted:

    with patch('datetime.date.today', return_value=fixed_dt.date()), ...

``datetime.date`` is an immutable C type, so CPython raises ``TypeError:
cannot set 'today' attribute of immutable type`` **at patch setup, before any
application frame executes**. The model's declaration was correct six times
over. The router's floor, seeing the file's regeneration attempts spent, handed
each round to repair *"which can still fix the production side of the
failure"*. There was no production side. Six HITL charges in nine minutes
exhausted the ``unsatisfiable_test`` class cap and terminated the run.

Five fixes, all covered here:

1. the clock rule offers ``not-runtime`` instead of a dead end, so a leap-day
   criterion with no injectable clock has a legal answer;
2. ``immutable_patch_reason`` proves the shape from the AST, and
   ``validate_scenarios`` rejects it before it is ever written;
3. the router's floor quarantines instead of handing it to repair;
4. so the six-charge loop cannot form;
5. a regeneration that introduced its own blocker gets one grace attempt.

:class:`TestTheProver` is the foundation — everything else keys off it, and
its false-positive case (a module-level ``date`` reference IS patchable) is
the one that must never regress.
"""

from __future__ import annotations

import pytest

from harness import acceptance_gen as ag
from harness.test_contradiction import (
    immutable_patch_reason,
    machine_unsatisfiable_reason,
)

#: Run 12's actual test body, trimmed.
RUN12_BODY = '''fixed_dt = datetime.datetime(2028, 2, 29, 12, 0, 0)
with patch('datetime.date.today', return_value=fixed_dt.date()):
    upcoming = client.get('/api/v1/birthdays/upcoming', params={'days': 1})
assert upcoming.status_code == 200, upcoming.text
'''

RUN12_FILE = f'''import datetime
from unittest.mock import patch


# @verifies: STORY-002.AC-1
def test_feb_29_policy(client):
    {RUN12_BODY.replace(chr(10), chr(10) + "    ").rstrip()}
'''

#: The legitimate form: the module under test did `from datetime import date`,
#: so ITS `date` name is an ordinary module attribute and patchable.
LEGITIMATE = '''from unittest.mock import patch


def test_feb_29_policy(client):
    with patch('server.app.services.birthday_service.date') as d:
        d.today.return_value = None
        assert client.get('/api/v1/birthdays/upcoming').status_code == 200
'''


class TestTheProver:
    def test_run_12s_shape_is_proven_unrunnable(self):
        why = immutable_patch_reason(RUN12_FILE, filename="t.py")
        assert why
        assert "datetime.date" in why
        assert "immutable" in why

    def test_it_names_the_sanctioned_alternatives(self):
        """A rejection that does not say what to do instead just produces the
        same output again."""
        why = immutable_patch_reason(RUN12_FILE, filename="t.py")
        assert "freezegun" in why or "module-level reference" in why

    def test_a_module_level_reference_is_NOT_flagged(self):
        """The case that must never regress: this is the correct fix, and
        flagging it would make the right answer unreachable."""
        assert immutable_patch_reason(LEGITIMATE, filename="t.py") is None

    @pytest.mark.parametrize("target", [
        "datetime.datetime.now", "datetime.time.min", "str.upper",
        "int.bit_length", "dict.get", "builtins.str.upper",
    ])
    def test_the_whole_immutable_family_is_covered(self, target):
        src = f"from unittest.mock import patch\ndef test_x():\n    with patch('{target}'):\n        pass\n"
        assert immutable_patch_reason(src, filename="t.py")

    def test_patch_object_against_a_c_type_is_caught(self):
        src = (
            "import datetime\nfrom unittest.mock import patch\n"
            "def test_x():\n"
            "    with patch.object(datetime.date, 'today'):\n        pass\n"
        )
        assert immutable_patch_reason(src, filename="t.py")

    def test_bare_setattr_is_caught(self):
        src = (
            "import datetime\ndef test_x():\n"
            "    setattr(datetime.date, 'today', lambda: None)\n"
        )
        assert immutable_patch_reason(src, filename="t.py")

    def test_an_ordinary_patch_is_untouched(self):
        src = (
            "from unittest.mock import patch\ndef test_x():\n"
            "    with patch('server.app.api.birthdays.get_db'):\n        pass\n"
        )
        assert immutable_patch_reason(src, filename="t.py") is None

    def test_an_unparseable_file_defers_to_its_own_prover(self):
        assert immutable_patch_reason("def (", filename="t.py") is None

    def test_a_non_python_file_is_declined(self):
        assert immutable_patch_reason(RUN12_FILE, filename="a.test.ts") is None

    def test_the_router_can_now_prove_it_without_a_declaration(self):
        """Previously only the model's prose said so, which is why the
        decision depended on a verdict rather than a proof."""
        why = machine_unsatisfiable_reason(RUN12_FILE, filename="t.py")
        assert why and "immutable" in why

    def test_cannot_run_is_checked_before_cannot_pass(self):
        """A test that dies at patch setup never reaches an assertion, so the
        contradiction detector has nothing to read."""
        import inspect

        import harness.test_contradiction as tc

        src = inspect.getsource(tc.machine_unsatisfiable_reason)
        assert src.index("immutable_patch_reason") < src.index(
            "find_contradictions("
        )


class TestItIsRejectedBeforeItIsWritten:
    """Fix 2's consumer: cheapest possible place to stop it."""

    def _scen(self, body: str) -> ag.AcceptanceScenario:
        return ag.AcceptanceScenario(
            "STORY-002.AC-1", "test_feb_29_policy",
            ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND, body,
        )

    def _ctx(self):
        return ag.StoryAcceptanceContext(
            story_key="STORY-002", title="t", description="",
            acceptance_criteria=[{"ac_key": "STORY-002.AC-1", "text": "leap"}],
        )

    def test_the_unrunnable_scenario_is_dropped(self):
        kept, dropped = ag.validate_scenarios(
            [self._scen(RUN12_BODY)], self._ctx(),
        )
        assert kept == []
        assert len(dropped) == 1
        assert "immutable" in dropped[0]["reason"]

    def test_a_legitimate_scenario_survives(self):
        body = (
            "with patch('server.app.services.birthday_service.date') as d:\n"
            "    d.today.return_value = None\n"
            "assert client.get('/api/v1/birthdays/upcoming').status_code == 200\n"
        )
        kept, dropped = ag.validate_scenarios(
            [self._scen(body)], self._ctx(),
        )
        assert len(kept) == 1, dropped


class TestTheClockRuleOffersAWayOut:
    """Fix 1. The generator did not misbehave — the rule forbade the only
    honest approach and offered nothing for a date-pinned criterion."""

    def _prompt(self) -> str:
        return ag.build_system_prompt(db_isolated=True, clock_available=False)

    def test_it_points_at_not_runtime(self):
        p = self._prompt()
        assert "cannot be verified in-process at all" in p
        assert '"not-runtime"' in p

    def test_it_names_the_leap_day_case_explicitly(self):
        p = self._prompt()
        assert "29 February" in p or "leap-day" in p

    def test_it_forbids_patching_a_builtin(self):
        p = self._prompt()
        assert "NEVER patch a built-in type" in p
        assert "datetime.date.today" in p

    def test_it_still_allows_the_module_level_patch(self):
        """Forbidding too much would remove the correct answer along with the
        wrong one."""
        p = self._prompt()
        assert "module-level reference" in p
        assert "from datetime import date" in p

    def test_the_clock_available_rule_is_unaffected(self):
        p = ag.build_system_prompt(db_isolated=True, clock_available=True)
        assert "NEVER patch a built-in type" not in p


class TestTheFloorQuarantinesInsteadOfLooping:
    """Fixes 3 and 4 — the containment that would have saved run 12."""

    def _state(self, tmp_path, *, body: str = RUN12_FILE):
        rel = "tests/acceptance/test_story_002_acceptance.py"
        (tmp_path / "tests" / "acceptance").mkdir(parents=True)
        (tmp_path / rel).write_text(body)
        return {
            "workspace_path": str(tmp_path),
            "compiler_errors": [{
                "file": rel, "line": 6,
                "message": "TypeError: cannot set 'today' attribute of "
                           "immutable type 'datetime.date'",
                "pytest_nodeid": f"{rel}::test_feb_29_policy",
            }],
        }, rel

    def test_the_evidence_is_found_from_the_file(self, tmp_path):
        import harness.graph as g

        state, rel = self._state(tmp_path)
        assert "immutable" in g._unrunnable_blocked_file(state, rel)

    def test_the_evidence_is_found_from_the_failure_text_alone(self, tmp_path):
        """Covers a shape the AST check has not learned: the file is fine to
        parse but the runtime says the same thing."""
        import harness.graph as g

        state, rel = self._state(tmp_path, body="def test_x():\n    pass\n")
        assert "immutable type" in g._unrunnable_blocked_file(state, rel)

    def test_an_ordinary_failure_is_not_called_unrunnable(self, tmp_path):
        """If this misfired it would quarantine real failures instead of
        repairing them — strictly worse than the bug it fixes."""
        import harness.graph as g

        state, rel = self._state(tmp_path, body=LEGITIMATE)
        state["compiler_errors"] = [{
            "file": rel, "line": 6,
            "message": "AssertionError: assert 404 == 200",
        }]
        assert g._unrunnable_blocked_file(state, rel) == ""

    def test_the_floor_branch_quarantines_before_falling_through(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "graph.py"
        ).read_text()
        at = src.index("_unrunnable = _unrunnable_blocked_file(")
        fallthrough = src.index(
            "The test-author ladder has nothing left to ", at,
        )
        window = src[at:fallthrough]
        assert "quarantine_unrunnable" in window
        assert 'return "test_regeneration_node"' in window


class TestTheNodeWritesTheQuarantine:
    """The router decides (it is a pure read); the node writes."""

    def _run(self, tmp_path, *, body=RUN12_FILE):
        import harness.test_regeneration as tr

        rel = "tests/acceptance/test_story_002_acceptance.py"
        (tmp_path / "tests" / "acceptance").mkdir(parents=True)
        (tmp_path / rel).write_text(body)
        state = {
            "workspace_path": str(tmp_path),
            "compiler_errors": [{
                "file": rel, "pytest_nodeid": f"{rel}::test_feb_29_policy",
                "message": "TypeError: cannot set 'today' attribute",
            }],
        }
        out = tr._quarantine_unrunnable_test(
            workspace=str(tmp_path), rel=rel,
            reason="patches datetime.date.today, an immutable type",
            state=state, loop_counter={},
        )
        return out, (tmp_path / rel).read_text()

    def test_the_failing_function_is_marked(self, tmp_path):
        out, on_disk = self._run(tmp_path)
        assert "xfail" in on_disk
        assert "teane-quarantine" in on_disk
        assert out["node_state"]["test_regeneration"]["status"] == (
            "quarantined_unrunnable"
        )

    def test_the_file_still_parses(self, tmp_path):
        _out, on_disk = self._run(tmp_path)
        compile(on_disk, "q.py", "exec")

    def test_the_reason_is_recorded_in_the_marker(self, tmp_path):
        _out, on_disk = self._run(tmp_path)
        assert "immutable type" in on_disk

    def test_no_attempt_is_charged(self, tmp_path):
        """The whole point: this costs nothing, so it cannot be the thing
        that exhausts a budget."""
        out, _ = self._run(tmp_path)
        assert not (out["loop_counter"].get("test_regen_attempts") or {})

    def test_the_unsatisfiable_flag_is_cleared(self, tmp_path):
        """node_state is replaced wholesale, so the recompile is judged on its
        own merits rather than re-deriving the same diversion."""
        out, _ = self._run(tmp_path)
        assert "unsatisfiable_test" not in out["node_state"]
        assert "quarantine_unrunnable" not in out["node_state"]

    def test_a_missing_file_is_survivable(self, tmp_path):
        import harness.test_regeneration as tr

        out = tr._quarantine_unrunnable_test(
            workspace=str(tmp_path), rel="tests/gone.py", reason="x",
            state={"workspace_path": str(tmp_path)}, loop_counter={},
        )
        assert out["node_state"]["test_regeneration"]["status"] == (
            "quarantine_read_failed"
        )


class TestTheGraceAttempt:
    """Fix 5: an attempt spent before a defect existed is not a fair charge
    against fixing that defect."""

    def _state(self, tmp_path, *, body: str, emission_md5: str | None):
        from harness.test_regeneration import _content_md5

        rel = "tests/t.py"
        (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
        lc: dict = {"test_regen_attempts": {rel: 2}}
        if emission_md5 is not None:
            lc["test_regen_last_emission"] = {
                rel: emission_md5 or _content_md5(body)
            }
        return {
            "workspace_path": str(tmp_path),
            "loop_counter": lc,
            "test_regeneration_config": {
                "enabled": True, "max_attempts_per_test": 2,
            },
        }, rel

    def test_its_own_output_earns_one_more_attempt(self, tmp_path):
        import harness.graph as g

        state, rel = self._state(
            tmp_path, body=RUN12_FILE, emission_md5="",
        )
        assert g._test_regen_exhausted(state, rel) is False

    def test_the_grace_is_bounded_to_once(self, tmp_path):
        """Otherwise a generator that keeps breaking its own output never
        converges."""
        import harness.graph as g

        state, rel = self._state(
            tmp_path, body=RUN12_FILE, emission_md5="",
        )
        assert g._test_regen_exhausted(state, rel) is False
        assert g._test_regen_exhausted(state, rel) is True

    def test_someone_elses_edit_earns_nothing(self, tmp_path):
        """The file on disk must BE this node's emission; a file repair
        touched is not its output."""
        import harness.graph as g

        state, rel = self._state(
            tmp_path, body=RUN12_FILE, emission_md5="deadbeef",
        )
        assert g._test_regen_exhausted(state, rel) is True

    def test_no_recorded_emission_earns_nothing(self, tmp_path):
        import harness.graph as g

        state, rel = self._state(tmp_path, body=RUN12_FILE, emission_md5=None)
        assert g._test_regen_exhausted(state, rel) is True

    def test_below_the_cap_is_unaffected(self, tmp_path):
        import harness.graph as g

        state, rel = self._state(tmp_path, body=RUN12_FILE, emission_md5="")
        state["loop_counter"]["test_regen_attempts"][rel] = 0
        assert g._test_regen_exhausted(state, rel) is False
        assert not state["loop_counter"].get("test_regen_own_output_grace"), (
            "no grace should be consumed when none was needed"
        )

    def test_disabled_regeneration_is_unaffected(self, tmp_path):
        import harness.graph as g

        state, rel = self._state(tmp_path, body=RUN12_FILE, emission_md5="")
        state["test_regeneration_config"]["enabled"] = False
        assert g._test_regen_exhausted(state, rel) is False
