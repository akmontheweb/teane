"""A regeneration attempt must count a FIX, not a wasted dispatch.

Regression cover for lumina 01a17661 (run 8), which terminated at
``unsatisfiable_test`` 6/6 with 0/16 stories because STORY-004's two
attempts (``max_attempts_per_test`` = 2) were both spent without touching
the defect:

  attempt 1  regenerated STORY-004.AC-2 -- the WRONG criterion; the model's
             declaration named ``test_empty_horizon_returns_empty_buckets``
  attempt 2  acceptance_regen_empty -- the generator returned nothing usable,
             so an attempt was charged for zero code written

The model then re-declared the same contradiction NINE times, correctly and
identically, with nothing left to act on. Separately STORY-002.AC-4 was
regenerated TWICE from a single declaration, six seconds apart, because the
router mints its own regeneration request whenever repair is blocked needing
a test edit (graph.py: ``_ns_blocked["unsatisfiable_test"] = _blocked_file``)
and charges it to the same budget.
"""

from __future__ import annotations

import pytest

from harness.acceptance_gen import verifies_by_function
from harness.test_regeneration import _REGEN_ATTEMPTS_KEY, _refund_attempt


def _lc(rel: str, attempts: int) -> dict:
    return {_REGEN_ATTEMPTS_KEY: {rel: attempts}}


def _attempts(lc: dict, rel: str) -> int:
    return int((lc.get(_REGEN_ATTEMPTS_KEY) or {}).get(rel, 0) or 0)


REL = "tests/acceptance/test_story_004_acceptance.py"


class TestTheRefund:
    def test_it_gives_the_attempt_back(self):
        lc = _lc(REL, 2)
        assert _refund_attempt(lc, REL, why="generator returned nothing")
        assert _attempts(lc, REL) == 1

    def test_it_is_bounded_to_one_reprieve_per_file(self):
        """A generator that never emits must still converge to HITL rather
        than looping on refunds forever."""
        lc = _lc(REL, 2)
        assert _refund_attempt(lc, REL, why="first") is True
        assert _refund_attempt(lc, REL, why="second") is False
        assert _attempts(lc, REL) == 1, "the second refund must not land"

    def test_each_file_gets_its_own_reprieve(self):
        other = "tests/acceptance/test_story_002_acceptance.py"
        lc = {_REGEN_ATTEMPTS_KEY: {REL: 2, other: 2}}
        assert _refund_attempt(lc, REL, why="x") is True
        assert _refund_attempt(lc, other, why="x") is True
        assert _attempts(lc, REL) == 1
        assert _attempts(lc, other) == 1

    def test_it_never_goes_below_zero(self):
        lc = _lc(REL, 0)
        _refund_attempt(lc, REL, why="x")
        assert _attempts(lc, REL) == 0

    def test_it_tolerates_a_missing_counter(self):
        lc: dict = {}
        _refund_attempt(lc, REL, why="x")  # must not raise
        assert _attempts(lc, REL) == 0

    def test_run_8_would_have_kept_an_attempt(self):
        """STORY-004 reached the cap at 2/2; with the empty generation
        refunded it sits at 1/2 and the ladder still has a real attempt for
        the contradiction the model named nine times."""
        lc = _lc(REL, 2)
        _refund_attempt(lc, REL, why="the scenario generator returned nothing")
        assert _attempts(lc, REL) < 2


#: A suite shaped like the real one: the criterion the line-mapped failure
#: points at, and the criterion the DECLARATION names, are different.
SUITE = '''
# @verifies: STORY-004.AC-2
def test_dashboard_groups_upcoming_birthdays_into_relative_buckets(client):
    resp = client.get("/api/v1/birthdays/dashboard")
    assert resp.status_code == 200


# @verifies: STORY-004.AC-3
def test_empty_horizon_returns_empty_buckets(client):
    payload = client.get("/api/v1/birthdays/dashboard").json()
    assert payload["today"]
'''

DECLARATION = (
    "`test_empty_horizon_returns_empty_buckets` seeds only a device profile "
    "(zero birthday records) yet asserts `payload[\"today\"]` is truthy; with "
    "no birthdays the `today` array is correctly empty, so no production "
    "change can satisfy it."
)


class TestTheDeclarationSelectsTheCriterion:
    """Fix 2, exercised through the real ``verifies_by_function`` mapping."""

    def _selected(self, line_mapped: dict[str, str], reason: str) -> set[str]:
        """Mirror of the node's selection step: line-mapped failures, then
        whatever the declaration names."""
        by_func = verifies_by_function(SUITE)
        failing = dict(line_mapped)
        for fn_name, (ac_key, _first, _last) in by_func.items():
            if ac_key in failing:
                continue
            if fn_name and fn_name in (reason or ""):
                failing[ac_key] = "named in the unsatisfiable declaration"
        return set(failing)

    def test_the_suite_maps_two_distinct_criteria(self):
        by_func = verifies_by_function(SUITE)
        assert by_func["test_empty_horizon_returns_empty_buckets"][0] == (
            "STORY-004.AC-3"
        )
        assert by_func[
            "test_dashboard_groups_upcoming_birthdays_into_relative_buckets"
        ][0] == "STORY-004.AC-2"

    def test_run_8s_miss_is_corrected(self):
        """Line-mapped failures pointed at AC-2; the declaration named the
        AC-3 test. Run 8 regenerated AC-2 only."""
        got = self._selected({"STORY-004.AC-2": "assert failed"}, DECLARATION)
        assert got == {"STORY-004.AC-2", "STORY-004.AC-3"}, (
            "the declared criterion must join the regeneration set"
        )

    def test_the_declared_criterion_is_picked_up_with_no_line_mapping(self):
        got = self._selected({}, DECLARATION)
        assert got == {"STORY-004.AC-3"}

    def test_a_generic_router_reason_adds_nothing(self):
        """The router mints its own reason, which names no test function —
        it must not inflate the set."""
        got = self._selected(
            {"STORY-004.AC-2": "assert failed"},
            "repair is blocked and needs a test edit",
        )
        assert got == {"STORY-004.AC-2"}

    def test_an_already_failing_criterion_is_not_duplicated(self):
        got = self._selected({"STORY-004.AC-3": "assert failed"}, DECLARATION)
        assert got == {"STORY-004.AC-3"}

    def test_an_empty_reason_is_safe(self):
        got = self._selected({"STORY-004.AC-2": "x"}, "")
        assert got == {"STORY-004.AC-2"}


class TestTheRepeatGuard:
    """Fix 3: the router re-requests regeneration for a still-red file, so a
    successful attempt could be immediately repeated at full cost."""

    def _would_repeat(self, lc: dict, rel: str, failing: set[str]) -> bool:
        sig = "|".join(sorted(failing))
        prev = lc.setdefault("test_regen_last_signature", {})
        if prev.get(rel) == sig:
            return True
        prev[rel] = sig
        return False

    def test_the_same_criteria_twice_is_refused(self):
        lc: dict = {}
        rel = "tests/acceptance/test_story_002_acceptance.py"
        assert not self._would_repeat(lc, rel, {"STORY-002.AC-4"})
        assert self._would_repeat(lc, rel, {"STORY-002.AC-4"}), (
            "run 8 regenerated STORY-002.AC-4 twice from one declaration"
        )

    def test_different_criteria_proceed(self):
        lc: dict = {}
        rel = "tests/acceptance/test_story_002_acceptance.py"
        assert not self._would_repeat(lc, rel, {"STORY-002.AC-4"})
        assert not self._would_repeat(lc, rel, {"STORY-002.AC-1"})

    def test_a_grown_criteria_set_proceeds(self):
        """Fix 2 can add the declared criterion on a later pass; that is new
        work, not a repeat."""
        lc: dict = {}
        assert not self._would_repeat(lc, REL, {"STORY-004.AC-2"})
        assert not self._would_repeat(
            lc, REL, {"STORY-004.AC-2", "STORY-004.AC-3"}
        )

    def test_files_are_tracked_independently(self):
        lc: dict = {}
        a = "tests/acceptance/test_story_002_acceptance.py"
        assert not self._would_repeat(lc, a, {"STORY-002.AC-4"})
        assert not self._would_repeat(lc, REL, {"STORY-002.AC-4"})


class TestTheWiring:
    def test_all_three_fixes_are_present_in_the_node(self):
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        # 1: empty generation refunds
        empty_at = src.index('give_up("acceptance_regen_empty"')
        assert "_refund_attempt(" in src[empty_at - 900:empty_at]
        # 2: the declaration's named function selects a criterion
        assert 'if _fn_name and _fn_name in (reason or ""):' in src
        # 3: the repeat guard
        assert 'give_up(\n            "acceptance_regen_repeat"' in src or (
            '"acceptance_regen_repeat"' in src
        )

    def test_the_repeat_guard_runs_before_the_generator_is_called(self):
        """Refusing after the dispatch would spend the money anyway."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_regeneration.py"
        ).read_text()
        guard = src.index('"acceptance_regen_repeat"')
        dispatch = src.index("await ag.generate_acceptance_scenarios(")
        assert guard < dispatch


class TestTheRealNodeExecutes:
    """Drives ``_regenerate_acceptance_suite`` itself.

    The classes above mirror the selection and repeat logic; these run the
    shipped function. Both matter: 33 tests passed against the in-turn retry
    while it raised a TypeError on every firing, because not one of them
    executed the code.
    """

    @staticmethod
    def _env(tmp_path, reason):
        """A real suite on disk plus the state the node reads."""
        suite_dir = tmp_path / "tests" / "acceptance"
        suite_dir.mkdir(parents=True)
        suite = suite_dir / "test_story_004_acceptance.py"
        suite.write_text(SUITE)
        (suite_dir / "conftest.py").write_text("# empty\n")
        rel = "tests/acceptance/test_story_004_acceptance.py"
        # The line-mapped failure points at the AC-2 test, as in run 8.
        ac2_line = SUITE.splitlines().index(
            "def test_dashboard_groups_upcoming_birthdays_into_relative"
            "_buckets(client):"
        ) + 1
        state = {
            "workspace_path": str(tmp_path),
            "compiler_errors": [
                {"file": rel, "line": ac2_line, "message": "assert failed"},
            ],
            "budget_remaining_usd": 5.0,
            "acceptance_config": {},
        }
        return state, rel, str(suite)

    @staticmethod
    def _stub_generator(monkeypatch, scenarios):
        """Make the scenario generator return `scenarios` and record which
        criteria it was ASKED for."""
        from harness import acceptance_gen as ag

        asked: list[str] = []

        class _Ctx:
            def __init__(self):
                self.acceptance_criteria = [
                    {"ac_key": "STORY-004.AC-2"},
                    {"ac_key": "STORY-004.AC-3"},
                ]

            def ac_keys(self):
                return [a["ac_key"] for a in self.acceptance_criteria]

        class _Res:
            def __init__(self):
                self.scenarios = list(scenarios)
                self.not_runtime = {}
                self.budget_remaining_usd = 5.0

        def _gather(*_a, **_k):
            return _Ctx()

        async def _gen(ctx, **_k):
            asked.extend(a["ac_key"] for a in ctx.acceptance_criteria)
            return _Res()

        monkeypatch.setattr(ag, "gather_story_acceptance_context", _gather)
        monkeypatch.setattr(ag, "generate_acceptance_scenarios", _gen)
        monkeypatch.setattr(
            ag, "record_scenario_feedback", lambda *_a, **_k: None
        )
        return asked

    @pytest.mark.asyncio
    async def test_an_empty_generation_refunds_in_the_real_node(
        self, tmp_path, monkeypatch
    ):
        """Fix 1, executed: run 8's exact failure, charged 2/2 before."""
        from harness.test_regeneration import _regenerate_acceptance_suite

        state, rel, abs_path = self._env(tmp_path, DECLARATION)
        self._stub_generator(monkeypatch, scenarios=[])   # nothing usable
        lc = _lc(rel, 2)

        out = await _regenerate_acceptance_suite(
            state, rel=rel, abs_path=abs_path, source=SUITE,
            reason=DECLARATION, gateway=object(), loop_counter=lc,
            give_up=lambda status, detail: {
                "loop_counter": lc,
                "node_state": {"test_regeneration": {"status": status}},
            },
        )
        assert out["node_state"]["test_regeneration"]["status"] == (
            "acceptance_regen_empty"
        )
        assert _attempts(lc, rel) == 1, (
            "the empty generation still spent an attempt — run 8's cap kill"
        )

    @pytest.mark.asyncio
    async def test_the_real_node_asks_for_the_declared_criterion(
        self, tmp_path, monkeypatch
    ):
        """Fix 2, executed: the generator must be asked for AC-3 (named in
        the declaration) and not AC-2 alone."""
        from harness.test_regeneration import _regenerate_acceptance_suite

        state, rel, abs_path = self._env(tmp_path, DECLARATION)
        asked = self._stub_generator(monkeypatch, scenarios=[])
        lc = _lc(rel, 1)

        await _regenerate_acceptance_suite(
            state, rel=rel, abs_path=abs_path, source=SUITE,
            reason=DECLARATION, gateway=object(), loop_counter=lc,
            give_up=lambda status, detail: {
                "loop_counter": lc, "node_state": {"x": status},
            },
        )
        assert "STORY-004.AC-3" in asked, (
            f"the declared criterion was never requested; asked={asked}"
        )

    @pytest.mark.asyncio
    async def test_the_real_node_refuses_an_identical_repeat(
        self, tmp_path, monkeypatch
    ):
        """Fix 3, executed: second call with the same criteria must refuse
        BEFORE dispatching and refund."""
        from harness.test_regeneration import _regenerate_acceptance_suite

        state, rel, abs_path = self._env(tmp_path, DECLARATION)
        asked = self._stub_generator(monkeypatch, scenarios=[])
        lc = _lc(rel, 1)
        statuses = []

        async def _call():
            return await _regenerate_acceptance_suite(
                state, rel=rel, abs_path=abs_path, source=SUITE,
                reason=DECLARATION, gateway=object(), loop_counter=lc,
                give_up=lambda status, detail: (
                    statuses.append(status) or {
                        "loop_counter": lc,
                        "node_state": {"test_regeneration": {"status": status}},
                    }
                ),
            )

        await _call()
        asked_after_first = len(asked)
        await _call()
        assert statuses[-1] == "acceptance_regen_repeat", (
            f"the duplicate was not refused; statuses={statuses}"
        )
        assert len(asked) == asked_after_first, (
            "the generator was dispatched again for identical criteria — the "
            "refusal must come before the spend"
        )
