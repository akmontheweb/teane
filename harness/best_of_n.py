"""Trajectory-level best-of-N.

Runs N *independent full solve attempts* (each its own plan → patch → repair
trajectory, ideally in an isolated git worktree) and selects the best one.
This is distinct from ``harness/speculative.py``, which runs best-of-N at the
*single patch attempt* level; here each candidate is a whole trajectory, which
is where a lot of SWE-bench-style gains come from (diverse approaches, one of
which sticks).

Design: the per-variant solve is **injected** as a coroutine, so this module
owns only the parts that are provider- and graph-agnostic and therefore
unit-testable in isolation — the N-way bounded concurrency, exception
isolation (one variant dying never sinks the batch), shared-budget accounting,
and winner selection. Wiring an actual ``run_graph`` trajectory (or a
``teane build`` subprocess in a worktree) into the ``solve`` seam is the
integration step; the selection/orchestration logic here does not change.

Typical use::

    async def solve(variant_id: int) -> TrajectoryResult:
        wt = make_worktree(variant_id)
        # ...run a full solve in `wt`, diversified by model/temperature/prompt...
        return TrajectoryResult(variant_id=variant_id, label=f"v{variant_id}",
                                compiled_ok=ok, changed_files=nf,
                                lines_changed=nl, cost_usd=cost, payload=wt)

    winner, all_results = await run_best_of_n(solve, n=3,
                                              strategy=SELECT_FEWEST_CHANGES)
    if winner:
        apply(winner.payload)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("harness.best_of_n")

# Selection strategies (names mirror harness/speculative.py for consistency).
SELECT_FIRST_SUCCESS = "first_success"
SELECT_FEWEST_CHANGES = "fewest_changes"
SELECT_VOTED = "voted"
_STRATEGIES = frozenset({SELECT_FIRST_SUCCESS, SELECT_FEWEST_CHANGES, SELECT_VOTED})


@dataclass
class TrajectoryResult:
    """Outcome of one full solve trajectory."""

    variant_id: int
    label: str = ""
    compiled_ok: bool = False       # did this trajectory reach a green build?
    changed_files: int = 0          # size of the diff (for fewest-changes)
    lines_changed: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None      # set when the trajectory raised / failed
    diagnostics: list[Any] = field(default_factory=list)
    payload: Any = None              # opaque winner artifact (worktree, patch set, …)

    @property
    def diff_size(self) -> tuple[int, int]:
        return (self.changed_files, self.lines_changed)


# A solve takes a variant id and returns its TrajectoryResult. It should not
# raise for ordinary "this attempt failed" outcomes (return compiled_ok=False
# instead); the orchestrator still isolates unexpected exceptions defensively.
SolveFn = Callable[[int], Awaitable[TrajectoryResult]]

# A judge ranks successful candidates for SELECT_VOTED and returns the winner
# (or None). Injected so the adversarial-vote machinery stays out of this
# module; falls back to fewest-changes when absent.
JudgeFn = Callable[[list[TrajectoryResult]], Awaitable[Optional[TrajectoryResult]]]


def select_winner(
    results: list[TrajectoryResult],
    strategy: str,
    *,
    voted_winner: Optional[TrajectoryResult] = None,
) -> Optional[TrajectoryResult]:
    """Pick the winning trajectory from ``results`` under ``strategy``.

    Only trajectories that reached a green build (``compiled_ok``) are
    eligible. Returns None when none compiled.

    - ``first_success``  → the lowest-variant-id compiled trajectory (stable,
      deterministic; "the first one that worked").
    - ``fewest_changes`` → the compiled trajectory with the smallest diff
      (fewest files, then fewest lines) — the least invasive fix.
    - ``voted``          → ``voted_winner`` when provided (computed by a judge
      over the compiled set), else falls back to ``fewest_changes``.
    """
    successful = [r for r in results if r.compiled_ok]
    if not successful:
        return None
    if strategy == SELECT_FIRST_SUCCESS:
        return min(successful, key=lambda r: r.variant_id)
    if strategy == SELECT_FEWEST_CHANGES:
        return min(successful, key=lambda r: (r.changed_files, r.lines_changed, r.variant_id))
    if strategy == SELECT_VOTED:
        if voted_winner is not None and voted_winner.compiled_ok:
            return voted_winner
        return min(successful, key=lambda r: (r.changed_files, r.lines_changed, r.variant_id))
    # Unknown strategy → safest default.
    return min(successful, key=lambda r: r.variant_id)


async def run_best_of_n(
    solve: SolveFn,
    n: int,
    *,
    strategy: str = SELECT_FIRST_SUCCESS,
    max_concurrency: int = 3,
    budget_usd: Optional[float] = None,
    judge: Optional[JudgeFn] = None,
) -> tuple[Optional[TrajectoryResult], list[TrajectoryResult]]:
    """Run ``n`` independent solve trajectories and select a winner.

    Args:
        solve: coroutine ``solve(variant_id) -> TrajectoryResult``.
        n: number of trajectories (clamped to >= 1).
        strategy: one of ``first_success`` / ``fewest_changes`` / ``voted``.
        max_concurrency: cap on trajectories running at once (>= 1).
        budget_usd: optional shared spend ceiling. Best-effort: once the
            cumulative cost of *completed* trajectories reaches this, no
            further trajectories are launched (in-flight ones finish).
        judge: optional ranker for ``voted``; falls back to fewest-changes.

    Returns ``(winner_or_None, all_results)``. Never raises for a failing
    trajectory — a raised exception becomes a ``compiled_ok=False`` result so
    the batch and selection still complete.
    """
    n = max(1, int(n))
    if strategy not in _STRATEGIES:
        logger.warning("[best_of_n] unknown strategy %r — using %s", strategy, SELECT_FIRST_SUCCESS)
        strategy = SELECT_FIRST_SUCCESS
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    results: list[TrajectoryResult] = []
    spent = 0.0
    stop = False

    async def _one(variant_id: int) -> TrajectoryResult:
        nonlocal spent, stop
        async with sem:
            if stop:
                return TrajectoryResult(
                    variant_id=variant_id, label=f"v{variant_id}",
                    compiled_ok=False, error="skipped: shared budget exhausted",
                )
            try:
                res = await solve(variant_id)
                if not isinstance(res, TrajectoryResult):  # defensive
                    raise TypeError(f"solve returned {type(res).__name__}, expected TrajectoryResult")
            except Exception as exc:  # noqa: BLE001 — one variant must never sink the batch
                logger.warning("[best_of_n] variant %d failed: %s", variant_id, exc)
                return TrajectoryResult(
                    variant_id=variant_id, label=f"v{variant_id}",
                    compiled_ok=False, error=f"{type(exc).__name__}: {exc}",
                )
            spent += max(0.0, res.cost_usd)
            if budget_usd is not None and spent >= budget_usd:
                stop = True
            return res

    results = list(await asyncio.gather(*[_one(i) for i in range(n)]))
    results.sort(key=lambda r: r.variant_id)

    voted_winner: Optional[TrajectoryResult] = None
    if strategy == SELECT_VOTED and judge is not None:
        eligible = [r for r in results if r.compiled_ok]
        if eligible:
            try:
                voted_winner = await judge(eligible)
            except Exception as exc:  # noqa: BLE001 — judge failure → fall back
                logger.warning("[best_of_n] judge failed, falling back to fewest_changes: %s", exc)

    winner = select_winner(results, strategy, voted_winner=voted_winner)
    n_ok = sum(1 for r in results if r.compiled_ok)
    logger.info(
        "[best_of_n] %d/%d trajectories reached green; strategy=%s; winner=%s; spent=$%.4f",
        n_ok, len(results), strategy,
        (f"v{winner.variant_id}" if winner else "none"), spent,
    )
    return winner, results
