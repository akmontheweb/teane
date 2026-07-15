"""Regression guard for ``PER_BATCH_CAP_COUNTERS``.

Three sites are responsible for zeroing per-batch HITL cap counters:
  1. ``story_loop._batch_commit_node``       — batch-boundary reset
  2. ``cli._reset_iteration_counters``       — Save & Quit / [r] resume
  3. ``cli._reset_hitl_trip_counters``       — headless auto-resume

Each site historically maintained its own hand-rolled tuple of keys.
That was drift bait: a new capped counter added elsewhere (e.g.
``test_generation_zero_emit``) would silently persist across every
recovery event whose site owner forgot to update its list, so the next
batch tripped the cap on its first entry without any real iteration.
Finsearch session 156032347 batch 110 died from that exact pattern
(``test_generation=5`` carried from batch 109 through the batch-commit
boundary, tripped ``max_iterations`` immediately, HITL auto-resume cap
exhausted, run terminated).

The class fix is a single canonical tuple in
``harness.loop_counter_keys`` and this test, which seeds every key to
a sentinel non-zero value and asserts each of the three reset paths
zeros them. Any counter added to the tuple that a reset site forgets
about fails here — the failing test names the offending site.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from harness.cli import _reset_hitl_trip_counters, _reset_iteration_counters
from harness.loop_counter_keys import (
    PER_BATCH_CAP_COUNTERS,
    STALL_TRIPWIRE_KEYS,
)

SENTINEL = 9


def _seeded_loop_counter() -> dict[str, int]:
    return {key: SENTINEL for key in PER_BATCH_CAP_COUNTERS}


def test_registry_is_non_empty_and_string_only():
    assert PER_BATCH_CAP_COUNTERS, "registry cannot be empty"
    for key in PER_BATCH_CAP_COUNTERS:
        assert isinstance(key, str) and key, key


def test_reset_iteration_counters_zeros_every_registered_key():
    """``_reset_iteration_counters`` is the Save & Quit / [r] resume
    reset. If a key survives it, the resumed batch enters the graph
    with a pre-tripped cap."""
    seeded = _seeded_loop_counter()
    result = _reset_iteration_counters(seeded, total_repairs=0)
    for key in PER_BATCH_CAP_COUNTERS:
        assert result.get(key) == 0, (
            f"_reset_iteration_counters left {key!r} at "
            f"{result.get(key)!r}. Add it to the site's reset loop "
            f"(harness/cli.py) or drop it from PER_BATCH_CAP_COUNTERS "
            f"if it should NOT reset on resume."
        )


def test_reset_hitl_trip_counters_zeros_every_registered_key():
    """``_reset_hitl_trip_counters`` is the headless auto-resume path.
    A key that survives it re-trips the same HITL trigger on the very
    next router pass and the session ping-pongs to the auto-resume
    session cap (finsearch 156032347)."""
    seeded = _seeded_loop_counter()
    _reset_hitl_trip_counters(seeded)
    for key in PER_BATCH_CAP_COUNTERS:
        assert seeded.get(key) == 0, (
            f"_reset_hitl_trip_counters left {key!r} at "
            f"{seeded.get(key)!r}. Add it to the site's reset loop "
            f"(harness/cli.py)."
        )


def test_batch_commit_reset_zeros_every_registered_key():
    """``story_loop._batch_commit_node`` clears per-batch counters as
    the last step before returning control to the batch planner. A key
    that survives it means the NEXT batch's first entry into the owning
    node trips its cap with zero real work done (the finsearch 156032347
    signature).

    The reset block is a plain-python for-loop over a tuple literal;
    parsing the module and reading the exact keys is cheaper (and less
    fragile) than driving the async node against a real state.db, and
    it fails loudly the moment the block drifts back to a hand-rolled
    tuple."""
    src = Path(__file__).resolve().parents[1] / "harness" / "story_loop.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    references_registry = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.iter, ast.Name):
            continue
        if node.iter.id != "PER_BATCH_CAP_COUNTERS":
            continue
        # Body must assign 0 to loop_counter[<target>].
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Subscript)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == 0
            ):
                references_registry = True
                break
        if references_registry:
            break

    assert references_registry, (
        "harness/story_loop.py no longer contains a `for key in "
        "PER_BATCH_CAP_COUNTERS: loop_counter[key] = 0` block. The "
        "batch-boundary reset MUST iterate the canonical registry — "
        "otherwise a new cap counter added to the registry silently "
        "carries across batches and the next batch's first entry "
        "into the owning node trips its cap without any real work."
    )


@pytest.mark.parametrize("key", list(PER_BATCH_CAP_COUNTERS))
def test_every_registered_key_survives_a_round_trip(key: str):
    """Explicit per-key coverage so a failure surfaces which counter
    regressed, not just 'the tuple did'."""
    seeded = {key: SENTINEL}
    result = _reset_iteration_counters(dict(seeded), total_repairs=0)
    assert result[key] == 0
    live = dict(seeded)
    _reset_hitl_trip_counters(live)
    assert live[key] == 0


# ---------------------------------------------------------------------------
# STALL_TRIPWIRE_KEYS — the router's "N bad rounds in a row" gates
# ---------------------------------------------------------------------------
#
# Session 22471c0c: HITL fired on ``reflection_distraction_loop:3``, the
# operator pressed [r], and the SAME trigger re-fired 20 seconds later —
# route_after_compiler consults ``consecutive_distraction_rounds`` before
# repair_node can run, the counter was preserved at its cap across the
# resume, and the only resets (PROGRESS verdict / green build) were
# unreachable. The [r]/[e] resume path must step every stall tripwire
# below its trip value so the gate that fired admits exactly one more
# repair round; the headless auto-resume path zeros them outright.


@pytest.mark.parametrize("key", list(STALL_TRIPWIRE_KEYS))
def test_stall_tripwire_steps_below_trip_on_human_resume(key: str):
    """[r]/[e] resume: a counter at its cap must come back below it —
    but only by one, so streak-keyed directives (reasoning-judge
    escalation at distraction >= 2, "use a different operation") stay
    armed and the trigger re-fires after a single fruitless round."""
    result = _reset_iteration_counters({key: 3}, total_repairs=2)
    assert result[key] == 2, (
        f"_reset_iteration_counters left stall tripwire {key!r} at "
        f"{result.get(key)!r}; a value at the trip cap re-fires the "
        f"same HITL trigger off the resume compile before repair_node "
        f"ever runs (session 22471c0c dead end)."
    )


@pytest.mark.parametrize("key", list(STALL_TRIPWIRE_KEYS))
def test_stall_tripwire_zeroed_on_auto_resume(key: str):
    live = {key: 3}
    _reset_hitl_trip_counters(live)
    assert live[key] == 0


@pytest.mark.parametrize("key", list(STALL_TRIPWIRE_KEYS))
def test_stall_tripwire_reset_is_present_only(key: str):
    """A legacy checkpoint without the key must not gain it (mirrors
    the ``cheap_shots_taken`` contract in test_cheap_shots_escalation)."""
    result = _reset_iteration_counters({}, total_repairs=0)
    assert key not in result
    live: dict[str, int] = {}
    _reset_hitl_trip_counters(live)
    assert key not in live


def test_stall_tripwire_never_goes_negative():
    result = _reset_iteration_counters(
        {key: 0 for key in STALL_TRIPWIRE_KEYS}, total_repairs=0,
    )
    for key in STALL_TRIPWIRE_KEYS:
        assert result[key] == 0


def test_graph_router_uses_the_shared_registry():
    """graph.py's ``_reset_stall_tripwires_on_progress`` must iterate
    the same canonical tuple — a hand-rolled copy there is the exact
    drift this module exists to prevent. Source-level check to avoid
    importing the full graph module."""
    src = (
        Path(__file__).resolve().parents[1] / "harness" / "graph.py"
    ).read_text(encoding="utf-8")
    assert (
        "from harness.loop_counter_keys import STALL_TRIPWIRE_KEYS" in src
    ), "graph.py no longer imports STALL_TRIPWIRE_KEYS from the registry"
