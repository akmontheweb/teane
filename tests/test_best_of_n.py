"""Trajectory-level best-of-N orchestration (harness/best_of_n.py).

The per-variant solve is injected, so these tests exercise the parts the
module actually owns: winner selection under each strategy, N-way execution,
exception isolation (one dying variant must not sink the batch), shared-budget
stop, and the voted/judge path.
"""

from __future__ import annotations

import asyncio

from harness.best_of_n import (
    SELECT_FEWEST_CHANGES,
    SELECT_FIRST_SUCCESS,
    SELECT_VOTED,
    TrajectoryResult,
    run_best_of_n,
    select_winner,
)


def _r(vid, ok, cf=0, ll=0, cost=0.0):
    return TrajectoryResult(
        variant_id=vid, compiled_ok=ok, changed_files=cf,
        lines_changed=ll, cost_usd=cost,
    )


class TestSelectWinner:
    def test_first_success_picks_lowest_compiled_id(self):
        rs = [_r(0, False), _r(1, True, cf=5), _r(2, True, cf=2)]
        assert select_winner(rs, SELECT_FIRST_SUCCESS).variant_id == 1

    def test_fewest_changes_picks_smallest_diff(self):
        rs = [_r(1, True, cf=5, ll=50), _r(2, True, cf=2, ll=90)]
        assert select_winner(rs, SELECT_FEWEST_CHANGES).variant_id == 2

    def test_fewest_changes_breaks_ties_by_lines_then_id(self):
        rs = [_r(2, True, cf=2, ll=10), _r(1, True, cf=2, ll=10)]
        assert select_winner(rs, SELECT_FEWEST_CHANGES).variant_id == 1

    def test_none_when_nothing_compiled(self):
        assert select_winner([_r(0, False), _r(1, False)], SELECT_FIRST_SUCCESS) is None

    def test_voted_prefers_judge_winner(self):
        rs = [_r(1, True, cf=5), _r(2, True, cf=2)]
        assert select_winner(rs, SELECT_VOTED, voted_winner=rs[0]).variant_id == 1

    def test_voted_falls_back_to_fewest_changes(self):
        rs = [_r(1, True, cf=5), _r(2, True, cf=2)]
        assert select_winner(rs, SELECT_VOTED).variant_id == 2

    def test_voted_ignores_noncompiled_judge_pick(self):
        rs = [_r(1, True, cf=5), _r(2, True, cf=2)]
        bad = _r(3, False)
        # judge picked something that didn't compile → fall back
        assert select_winner(rs, SELECT_VOTED, voted_winner=bad).variant_id == 2


class TestRunBestOfN:
    def test_runs_n_and_sorts_results(self):
        async def solve(i):
            return _r(i, True, cf=i + 1, ll=(i + 1) * 10, cost=0.5)
        winner, results = asyncio.run(
            run_best_of_n(solve, 3, strategy=SELECT_FEWEST_CHANGES))
        assert [r.variant_id for r in results] == [0, 1, 2]
        assert winner.variant_id == 0  # smallest diff

    def test_exception_in_one_variant_does_not_sink_batch(self):
        async def solve(i):
            if i == 1:
                raise RuntimeError("boom")
            return _r(i, i == 2, cf=1, cost=0.1)
        winner, results = asyncio.run(
            run_best_of_n(solve, 3, strategy=SELECT_FIRST_SUCCESS))
        assert len(results) == 3
        assert results[1].compiled_ok is False and "boom" in results[1].error
        assert winner.variant_id == 2

    def test_all_failed_returns_no_winner(self):
        async def solve(i):
            return _r(i, False)
        winner, results = asyncio.run(run_best_of_n(solve, 3))
        assert winner is None and len(results) == 3

    def test_shared_budget_stops_launching(self):
        async def solve(i):
            await asyncio.sleep(0.01 * i)
            return _r(i, True, cf=1, cost=1.0)
        winner, results = asyncio.run(run_best_of_n(
            solve, 5, strategy=SELECT_FIRST_SUCCESS,
            max_concurrency=1, budget_usd=2.0))
        skipped = [r for r in results if r.error and "budget" in r.error]
        assert len(skipped) >= 1

    def test_voted_with_judge(self):
        async def judge(cands):
            return max(cands, key=lambda r: r.variant_id)
        async def solve(i):
            return _r(i, True, cf=1, cost=0.1)
        winner, _ = asyncio.run(
            run_best_of_n(solve, 3, strategy=SELECT_VOTED, judge=judge))
        assert winner.variant_id == 2

    def test_judge_failure_falls_back(self):
        async def judge(cands):
            raise RuntimeError("judge boom")
        async def solve(i):
            return _r(i, True, cf=i + 1, cost=0.1)
        winner, _ = asyncio.run(
            run_best_of_n(solve, 3, strategy=SELECT_VOTED, judge=judge))
        # fell back to fewest_changes → v0
        assert winner.variant_id == 0
