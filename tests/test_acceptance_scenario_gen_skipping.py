"""Scenario generation must not re-buy what it already has.

``acceptance_node`` is re-entered on every green build, and
``_generate_batch_scenarios`` asked the LLM for every criterion of every story
each time. ``select_runnable`` filtered by ``already_passed`` only AFTER
generation, and ``_write_integration_suite`` writes only the runnable set — so
a passed criterion's scenario was generated, discarded, and never written.

Measured on lumina run 11, batch 94: **4 acceptance entries, 19 scenario-gen
dispatches, 13 of them over 300s — roughly 75-90 of the batch's 164 minutes,
about half its wall clock**, regenerating scenarios for criteria that had
already passed. Four stories took 2h45m; sixteen at that rate is about eleven
hours, which is the real reason no run has finished. Nothing behaved
incorrectly, which is why it stayed invisible for ten runs.

This is a cost fix, not a behaviour fix: the set of scenarios that reaches the
suite is unchanged. :class:`TestTheOutputIsUnchanged` is the part that matters
most — a throughput fix that quietly drops a criterion would be far worse than
the cost it saves.
"""

from __future__ import annotations

import pytest

from harness import acceptance_gen as ag
from harness import acceptance_node as an

STORY_A = "STORY-005"
STORY_B = "STORY-006"


def _ctx(story_key: str, ac_keys: list[str]) -> ag.StoryAcceptanceContext:
    return ag.StoryAcceptanceContext(
        story_key=story_key,
        title=f"{story_key} title",
        description="",
        acceptance_criteria=[{"ac_key": k, "text": f"{k} prose"} for k in ac_keys],
    )


class _Recorder:
    """Stands in for the scenario generator, recording what it was ASKED."""

    def __init__(self):
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def install(self, monkeypatch, *, contexts: dict[str, list[str]]):
        def _gather(workspace, story_key, **kw):
            keys = contexts.get(story_key)
            return _ctx(story_key, keys) if keys is not None else None

        async def _generate(ctx, **kw):
            self.asked.append(
                (ctx.story_key, tuple(a["ac_key"] for a in ctx.acceptance_criteria))
            )
            return ag.AcceptanceGenResult(
                story_key=ctx.story_key,
                scenarios=[
                    ag.AcceptanceScenario(
                        a["ac_key"], f"test_{a['ac_key'].replace('.', '_').lower()}",
                        ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND, "assert True",
                    )
                    for a in ctx.acceptance_criteria
                ],
                budget_remaining_usd=kw.get("budget_remaining_usd", 5.0) - 0.01,
            )

        monkeypatch.setattr(ag, "gather_story_acceptance_context", _gather)
        monkeypatch.setattr(ag, "generate_acceptance_scenarios", _generate)
        return self

    @property
    def dispatches(self) -> int:
        return len(self.asked)

    @property
    def criteria_asked(self) -> int:
        return sum(len(keys) for _sk, keys in self.asked)


CONTEXTS = {
    STORY_A: [f"{STORY_A}.AC-1", f"{STORY_A}.AC-2", f"{STORY_A}.AC-3"],
    STORY_B: [f"{STORY_B}.AC-1", f"{STORY_B}.AC-2"],
}
CFG = {"scenario_source": "llm", "integration_dir": "tests/acceptance"}


async def _run(rec, *, already_passed=None, cfg=None):
    return await an._generate_batch_scenarios(
        "/ws", [STORY_A, STORY_B], cfg if cfg is not None else CFG,
        gateway=object(), budget=5.0, already_passed=already_passed,
    )


class TestTheRegressionItself:
    @pytest.mark.asyncio
    async def test_nothing_passed_means_everything_is_generated(
        self, monkeypatch,
    ):
        """The baseline: a first pass must still ask for all five."""
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec)
        assert rec.dispatches == 2
        assert rec.criteria_asked == 5

    @pytest.mark.asyncio
    async def test_without_the_filter_a_second_pass_costs_the_same(
        self, monkeypatch,
    ):
        """What the old code did on every green build — the whole defect in
        one assertion. ``already_passed=None`` is the pre-fix call shape."""
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed=None)
        assert rec.criteria_asked == 5, (
            "passing no already_passed must reproduce the old full cost"
        )


class TestTheSaving:
    @pytest.mark.asyncio
    async def test_passed_criteria_are_not_asked_for(self, monkeypatch):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed={f"{STORY_A}.AC-1", f"{STORY_A}.AC-2"})
        asked = {k for _sk, keys in rec.asked for k in keys}
        assert f"{STORY_A}.AC-1" not in asked
        assert f"{STORY_A}.AC-2" not in asked
        assert f"{STORY_A}.AC-3" in asked

    @pytest.mark.asyncio
    async def test_a_fully_passed_story_costs_no_dispatch_at_all(
        self, monkeypatch,
    ):
        """The big saving. In run 11 each of these ran 5-6 minutes."""
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed=set(CONTEXTS[STORY_B]))
        assert rec.dispatches == 1
        assert [sk for sk, _ in rec.asked] == [STORY_A]

    @pytest.mark.asyncio
    async def test_everything_passed_costs_nothing(self, monkeypatch):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        scen, _nr, budget = await _run(
            rec, already_passed=set(CONTEXTS[STORY_A]) | set(CONTEXTS[STORY_B]),
        )
        assert rec.dispatches == 0
        assert scen == []
        assert budget == 5.0, "an untouched budget is the point"

    @pytest.mark.asyncio
    async def test_run_11s_batch_94_shape(self, monkeypatch):
        """Pass 3 of run 11 had 11 of 17 criteria already verified. Two of
        the four stories were fully done and cost two dispatches that each
        ran over five minutes."""
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed={
            f"{STORY_A}.AC-1", f"{STORY_A}.AC-2", f"{STORY_A}.AC-3",
            f"{STORY_B}.AC-1",
        })
        assert rec.dispatches == 1, "STORY-005 was fully verified"
        assert rec.criteria_asked == 1, "only STORY-006.AC-2 remained"


class TestTheOutputIsUnchanged:
    """A throughput fix that drops a criterion would be far worse than the
    cost it saves."""

    @pytest.mark.asyncio
    async def test_the_scenarios_returned_are_exactly_the_unpassed_ones(
        self, monkeypatch,
    ):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        scen, _nr, _b = await _run(rec, already_passed={f"{STORY_A}.AC-1"})
        assert {s.verifies for s in scen} == {
            f"{STORY_A}.AC-2", f"{STORY_A}.AC-3",
            f"{STORY_B}.AC-1", f"{STORY_B}.AC-2",
        }

    @pytest.mark.asyncio
    async def test_a_passed_criterion_was_never_written_anyway(
        self, monkeypatch,
    ):
        """The reason this is safe: ``select_runnable`` already excluded
        passed criteria and ``_write_integration_suite`` writes only the
        runnable set, so generating them produced bytes that went nowhere.
        This is the existing run-each-AC-once contract, made cheap — not a
        new decision about re-verification.
        """
        from harness import acceptance_run as ar

        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        scen, _nr, _b = await _run(rec)
        runnable = ar.select_runnable(
            scen, already_passed={f"{STORY_A}.AC-1"},
        )
        assert f"{STORY_A}.AC-1" not in {s.verifies for s in runnable}

    @pytest.mark.asyncio
    async def test_an_unknown_story_is_still_skipped_quietly(
        self, monkeypatch,
    ):
        rec = _Recorder().install(
            monkeypatch, contexts={STORY_A: CONTEXTS[STORY_A]},
        )
        scen, _nr, _b = await _run(rec)
        assert rec.dispatches == 1
        assert {s.verifies for s in scen} == set(CONTEXTS[STORY_A])

    @pytest.mark.asyncio
    async def test_the_fallback_path_is_narrowed_too(self, monkeypatch):
        """``scenario_source != 'llm'`` costs no tokens, but generating
        scenarios that are then filtered out is still wrong."""
        seen: list[str] = []

        def _gather(workspace, story_key, **kw):
            return _ctx(story_key, CONTEXTS[story_key])

        def _fallback(ctx):
            seen.append(ctx.story_key)
            return ag.AcceptanceGenResult(
                story_key=ctx.story_key, scenarios=[], budget_remaining_usd=0.0,
            )

        monkeypatch.setattr(ag, "gather_story_acceptance_context", _gather)
        monkeypatch.setattr(ag, "fallback_acceptance_scenarios", _fallback)
        await an._generate_batch_scenarios(
            "/ws", [STORY_A, STORY_B], {"scenario_source": "fallback"},
            gateway=None, budget=0.0,
            already_passed=set(CONTEXTS[STORY_B]),
        )
        assert seen == [STORY_A]


class TestTheEdges:
    @pytest.mark.asyncio
    async def test_an_empty_passed_set_changes_nothing(self, monkeypatch):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed=set())
        assert rec.criteria_asked == 5

    @pytest.mark.asyncio
    async def test_an_unrelated_passed_key_changes_nothing(self, monkeypatch):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        await _run(rec, already_passed={"STORY-999.AC-1"})
        assert rec.criteria_asked == 5

    @pytest.mark.asyncio
    async def test_the_budget_is_only_spent_on_what_was_asked(
        self, monkeypatch,
    ):
        rec = _Recorder().install(monkeypatch, contexts=CONTEXTS)
        _s, _nr, budget = await _run(
            rec, already_passed=set(CONTEXTS[STORY_B]),
        )
        assert budget == pytest.approx(4.99), "one dispatch, one charge"


class TestTheWiring:
    def test_the_node_passes_the_verified_set_through(self):
        import inspect

        src = inspect.getsource(an.acceptance_node)
        assert "already_passed=already_passed," in src
        at = src.index("_generate_batch_scenarios(")
        assert "already_passed=already_passed," in src[at:at + 800]

    def test_the_verified_set_is_read_before_generation(self):
        """It comes from the story DB at node entry; generating first and
        filtering after is the defect."""
        import inspect

        src = inspect.getsource(an.acceptance_node)
        assert src.index("already_passed = story_state") < src.index(
            "_generate_batch_scenarios("
        )
