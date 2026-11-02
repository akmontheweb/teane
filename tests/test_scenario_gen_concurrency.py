"""The largest cost in the harness ran one story at a time.

Measured across lumina runs 11 and 12: **96% and 95% of each run's wall clock
was time inside LLM calls**, and the ``planning`` role alone was **66% then
72% of all of it** — ~300 seconds and ~18,000 output tokens per call. That
latency is earned; the model really is generating 18k tokens. What was not
earned is that ``_generate_batch_scenarios`` awaited them **strictly in
sequence**, so a four-story batch cost 4 x 300s per acceptance pass even
though the per-story contexts share nothing and the results are only appended
to a list.

Three properties this has to keep while going concurrent, each with a test
below:

* **order** — results fold in INPUT order, so the rendered suite does not
  depend on which call returned first;
* **budget** — summed from per-call deltas, because every concurrent call is
  handed the same starting figure, so the remainder any one of them reports is
  not the remainder after all of them;
* **isolation** — one story's failure costs the batch only that story.

And what is deliberately NOT parallel: context gathering, which is local file
and sqlite reads measured in milliseconds against a 300-second dispatch.
"""

from __future__ import annotations

import asyncio

import pytest

from harness import acceptance_gen as ag
from harness import acceptance_node as an

STORIES = ["STORY-001", "STORY-002", "STORY-003", "STORY-004"]
CFG = {"scenario_source": "llm", "integration_dir": "tests/acceptance"}


def _ctx(story_key: str) -> ag.StoryAcceptanceContext:
    return ag.StoryAcceptanceContext(
        story_key=story_key, title=story_key, description="",
        acceptance_criteria=[{"ac_key": f"{story_key}.AC-1", "text": "x"}],
    )


def _scenario(story_key: str) -> ag.AcceptanceScenario:
    return ag.AcceptanceScenario(
        f"{story_key}.AC-1", f"test_{story_key.lower().replace('-', '_')}",
        ag.ALTITUDE_INTEGRATION, ag.CLASS_BACKEND, "assert True",
    )


class _Harness:
    """Records concurrency and lets each story's call be delayed or fail."""

    def __init__(self, monkeypatch, *, delays=None, fails=(), cost=0.01):
        self.inflight = 0
        self.peak = 0
        self.order: list[str] = []
        self._delays = delays or {}
        self._fails = set(fails)
        self._cost = cost

        monkeypatch.setattr(
            ag, "gather_story_acceptance_context",
            lambda ws, sk, **kw: _ctx(sk),
        )

        async def _generate(ctx, **kw):
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            try:
                await asyncio.sleep(self._delays.get(ctx.story_key, 0.01))
                if ctx.story_key in self._fails:
                    raise RuntimeError(f"provider refused {ctx.story_key}")
                self.order.append(ctx.story_key)
                return ag.AcceptanceGenResult(
                    story_key=ctx.story_key,
                    scenarios=[_scenario(ctx.story_key)],
                    not_runtime={},
                    budget_remaining_usd=(
                        kw.get("budget_remaining_usd", 10.0) - self._cost
                    ),
                )
            finally:
                self.inflight -= 1

        monkeypatch.setattr(ag, "generate_acceptance_scenarios", _generate)


async def _run(cfg=None, *, budget=10.0, stories=None):
    return await an._generate_batch_scenarios(
        "/ws", stories if stories is not None else STORIES,
        cfg if cfg is not None else CFG,
        gateway=object(), budget=budget,
    )


class TestItRunsConcurrently:
    @pytest.mark.asyncio
    async def test_all_four_stories_are_in_flight_together(self, monkeypatch):
        """The whole point. Sequentially the peak would be 1."""
        h = _Harness(monkeypatch, delays={s: 0.05 for s in STORIES})
        await _run()
        assert h.peak == 4

    @pytest.mark.asyncio
    async def test_it_is_faster_than_sequential(self, monkeypatch):
        """Four 50ms calls concurrently finish in well under 200ms."""
        h = _Harness(monkeypatch, delays={s: 0.05 for s in STORIES})
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await _run()
        assert loop.time() - t0 < 0.15, "the calls did not overlap"
        assert h.peak == 4

    @pytest.mark.asyncio
    async def test_the_concurrency_cap_is_honoured(self, monkeypatch):
        h = _Harness(monkeypatch, delays={s: 0.05 for s in STORIES})
        await _run({**CFG, "scenario_gen_concurrency": 2})
        assert h.peak == 2

    @pytest.mark.asyncio
    async def test_a_cap_of_one_restores_sequential_behaviour(
        self, monkeypatch,
    ):
        """The documented escape hatch for provider rate limits."""
        h = _Harness(monkeypatch, delays={s: 0.02 for s in STORIES})
        await _run({**CFG, "scenario_gen_concurrency": 1})
        assert h.peak == 1

    @pytest.mark.asyncio
    async def test_a_zero_or_negative_cap_is_clamped_not_deadlocked(
        self, monkeypatch,
    ):
        h = _Harness(monkeypatch)
        scen, _nr, _b = await _run({**CFG, "scenario_gen_concurrency": 0})
        assert h.peak == 1
        assert len(scen) == 4


class TestOrderIsPreserved:
    @pytest.mark.asyncio
    async def test_results_fold_in_input_order_not_completion_order(
        self, monkeypatch,
    ):
        """STORY-004 returns first; the scenario list must still read
        001, 002, 003, 004 or the rendered suite changes between runs for no
        reason."""
        h = _Harness(monkeypatch, delays={
            "STORY-001": 0.08, "STORY-002": 0.06,
            "STORY-003": 0.04, "STORY-004": 0.01,
        })
        scen, _nr, _b = await _run()
        assert h.order[0] == "STORY-004", "the fixture must invert completion"
        assert [s.verifies for s in scen] == [f"{s}.AC-1" for s in STORIES]


class TestBudgetAccounting:
    @pytest.mark.asyncio
    async def test_spend_is_summed_not_chained(self, monkeypatch):
        """Every concurrent call is handed 10.0 and each returns 9.99.
        Chaining would read the remainder as 9.99 for the whole batch; the
        real spend is four times one call."""
        _Harness(monkeypatch, cost=0.01)
        _s, _nr, budget = await _run(budget=10.0)
        assert budget == pytest.approx(9.96), (
            "four calls at $0.01 must cost $0.04, not $0.01"
        )

    @pytest.mark.asyncio
    async def test_a_failed_call_is_not_billed(self, monkeypatch):
        _Harness(monkeypatch, cost=0.01, fails=["STORY-003"])
        _s, _nr, budget = await _run(budget=10.0)
        assert budget == pytest.approx(9.97)

    @pytest.mark.asyncio
    async def test_the_budget_never_goes_negative(self, monkeypatch):
        _Harness(monkeypatch, cost=5.0)
        _s, _nr, budget = await _run(budget=1.0)
        assert budget == 0.0

    @pytest.mark.asyncio
    async def test_no_jobs_leaves_the_budget_untouched(self, monkeypatch):
        _Harness(monkeypatch)
        _s, _nr, budget = await _run(budget=7.5, stories=[])
        assert budget == 7.5


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_story_failing_does_not_lose_the_others(
        self, monkeypatch,
    ):
        """Sequentially this raised and cost the batch everything after it."""
        _Harness(monkeypatch, fails=["STORY-002"])
        scen, _nr, _b = await _run()
        assert [s.verifies for s in scen] == [
            "STORY-001.AC-1", "STORY-003.AC-1", "STORY-004.AC-1",
        ]

    @pytest.mark.asyncio
    async def test_every_story_failing_is_survivable(self, monkeypatch):
        _Harness(monkeypatch, fails=STORIES)
        scen, nr, budget = await _run(budget=10.0)
        assert scen == []
        assert nr == {}
        assert budget == 10.0


class TestWhatStaysSequential:
    @pytest.mark.asyncio
    async def test_context_gathering_is_not_parallelised(self, monkeypatch):
        """Local file + sqlite reads, milliseconds against a 300s dispatch,
        and concurrent readers on one connection are a hazard for no gain."""
        peak = {"n": 0, "cur": 0}

        def _gather(ws, sk, **kw):
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
            try:
                return _ctx(sk)
            finally:
                peak["cur"] -= 1

        monkeypatch.setattr(ag, "gather_story_acceptance_context", _gather)

        async def _generate(ctx, **kw):
            return ag.AcceptanceGenResult(
                story_key=ctx.story_key, scenarios=[],
                budget_remaining_usd=kw.get("budget_remaining_usd", 10.0),
            )

        monkeypatch.setattr(ag, "generate_acceptance_scenarios", _generate)
        await _run()
        assert peak["n"] == 1


class TestTheSkipLogicSurvives:
    """`b96bc4b` must keep working through the restructure — it is the other
    half of the same cost."""

    @pytest.mark.asyncio
    async def test_a_fully_verified_story_still_costs_no_dispatch(
        self, monkeypatch,
    ):
        h = _Harness(monkeypatch)
        await an._generate_batch_scenarios(
            "/ws", STORIES, CFG, gateway=object(), budget=10.0,
            already_passed={"STORY-002.AC-1"},
        )
        assert "STORY-002" not in h.order
        assert len(h.order) == 3

    @pytest.mark.asyncio
    async def test_the_fallback_path_still_works(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            ag, "gather_story_acceptance_context",
            lambda ws, sk, **kw: _ctx(sk),
        )
        monkeypatch.setattr(
            ag, "fallback_acceptance_scenarios",
            lambda ctx: (
                seen.append(ctx.story_key) or ag.AcceptanceGenResult(
                    story_key=ctx.story_key, scenarios=[_scenario(ctx.story_key)],
                    budget_remaining_usd=0.0,
                )
            ),
        )
        scen, _nr, _b = await an._generate_batch_scenarios(
            "/ws", STORIES, {"scenario_source": "fallback"},
            gateway=None, budget=0.0,
        )
        assert seen == STORIES
        assert len(scen) == 4


class TestTheConfigKey:
    def test_it_is_a_known_key(self):
        from harness.cli import _KNOWN_NESTED_KEYS

        assert "scenario_gen_concurrency" in _KNOWN_NESTED_KEYS["acceptance"]

    def test_it_is_typed(self):
        from harness.cli import _TYPE_SCHEMA

        assert _TYPE_SCHEMA["acceptance.scenario_gen_concurrency"] == (int,)

    def test_the_shipped_config_sets_it(self):
        import json
        import pathlib

        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent / "config" / "config.json")
            .read_text()
        )
        assert cfg["acceptance"]["scenario_gen_concurrency"] == 4

    def test_it_is_documented(self):
        """House rule: every config change updates config_help.md."""
        import pathlib

        doc = (
            pathlib.Path(__file__).parent.parent / "config" / "config_help.md"
        ).read_text()
        assert "scenario_gen_concurrency" in doc
        assert "set it to 1" in doc, "the escape hatch must be documented"
