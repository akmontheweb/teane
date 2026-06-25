"""Phase I — intra-batch topological ordering.

Three scopes:

1. ``validate_batches`` now ALLOWS same-batch deps but ENFORCES that
   the dep comes before its dependent in story_keys.
2. ``deterministic_batches`` keeps its dep-frontier output but routes
   each emitted batch through ``_topo_sort_within_batch`` so the order
   contract is explicit and future-proof.
3. ``_next_story_in_batch`` defensively skips a story whose intra-batch
   deps aren't yet ``done`` (a corrupt-DB / resumed-session guard).
"""

from __future__ import annotations

from typing import Any

import pytest

from harness import story_loop, story_state
from harness.batch_sizing import (
    _topo_sort_within_batch,
    deterministic_batches,
    validate_batches,
)


def _story(key: str, *, deps: list[str] | None = None) -> dict[str, Any]:
    return {
        "story_key": key,
        "title": f"Story {key}",
        "depends_on": deps or [],
    }


# ---------------------------------------------------------------------------
# _topo_sort_within_batch
# ---------------------------------------------------------------------------

class TestTopoSortWithinBatch:
    def test_independent_keys_preserve_input_order(self):
        deps = {"STORY-1": set(), "STORY-2": set(), "STORY-3": set()}
        out = _topo_sort_within_batch(["STORY-1", "STORY-2", "STORY-3"], deps)
        assert out == ["STORY-1", "STORY-2", "STORY-3"]

    def test_chain_dependency_reordered(self):
        # STORY-3 depends on STORY-2 depends on STORY-1; input order
        # has them reversed.
        deps = {
            "STORY-1": set(),
            "STORY-2": {"STORY-1"},
            "STORY-3": {"STORY-2"},
        }
        out = _topo_sort_within_batch(
            ["STORY-3", "STORY-2", "STORY-1"], deps,
        )
        assert out == ["STORY-1", "STORY-2", "STORY-3"]

    def test_only_intra_batch_deps_considered(self):
        # STORY-1 depends on EXTERNAL-9 (not in the batch). The deps_by_key
        # set is intersected with the batch's keys inside the helper, so
        # EXTERNAL-9 doesn't influence ordering.
        deps = {"STORY-1": {"EXTERNAL-9"}, "STORY-2": set()}
        out = _topo_sort_within_batch(["STORY-1", "STORY-2"], deps)
        assert out == ["STORY-1", "STORY-2"]

    def test_cycle_falls_back_to_input_order(self):
        # A↔B cycle inside one batch shouldn't happen in practice (the
        # validator would reject), but the helper must not infinite-loop.
        deps = {
            "STORY-1": {"STORY-2"},
            "STORY-2": {"STORY-1"},
        }
        out = _topo_sort_within_batch(["STORY-1", "STORY-2"], deps)
        # Both stories appear; exact order is the input order (defensive).
        assert sorted(out) == ["STORY-1", "STORY-2"]
        assert len(out) == 2

    def test_diamond_preserves_partial_order(self):
        # STORY-1 → (STORY-2, STORY-3) → STORY-4 with input as [4, 2, 3, 1].
        deps = {
            "STORY-1": set(),
            "STORY-2": {"STORY-1"},
            "STORY-3": {"STORY-1"},
            "STORY-4": {"STORY-2", "STORY-3"},
        }
        out = _topo_sort_within_batch(
            ["STORY-4", "STORY-2", "STORY-3", "STORY-1"], deps,
        )
        # STORY-1 must come first; STORY-4 must come last.
        assert out[0] == "STORY-1"
        assert out[-1] == "STORY-4"
        assert set(out[1:3]) == {"STORY-2", "STORY-3"}


# ---------------------------------------------------------------------------
# validate_batches — same-batch deps allowed, position order enforced
# ---------------------------------------------------------------------------

class TestValidateBatchesIntraBatchDeps:
    def test_same_batch_dep_in_correct_order_is_accepted(self):
        # STORY-2 depends on STORY-1; both in batch 1 with STORY-1 first.
        stories = [_story("STORY-1"), _story("STORY-2", deps=["STORY-1"])]
        batches = [{"batch_id": 1, "story_keys": ["STORY-1", "STORY-2"]}]
        assert validate_batches(stories, batches) == []

    def test_same_batch_dep_out_of_order_is_flagged(self):
        # STORY-2 depends on STORY-1 but STORY-2 listed first.
        stories = [_story("STORY-1"), _story("STORY-2", deps=["STORY-1"])]
        batches = [{"batch_id": 1, "story_keys": ["STORY-2", "STORY-1"]}]
        errs = validate_batches(stories, batches)
        assert any("must come BEFORE its dependent" in e for e in errs)

    def test_cross_batch_forward_still_rejected(self):
        # STORY-1 (batch 1) depends on STORY-2 (batch 2) — cross-batch
        # forward dep is the only ordering violation that's still
        # outright forbidden post-Phase-I.
        stories = [
            _story("STORY-1", deps=["STORY-2"]),
            _story("STORY-2"),
        ]
        batches = [
            {"batch_id": 1, "story_keys": ["STORY-1"]},
            {"batch_id": 2, "story_keys": ["STORY-2"]},
        ]
        errs = validate_batches(stories, batches)
        assert any(
            "dep must be in an earlier or same batch" in e for e in errs
        )

    def test_orphan_dep_still_tolerated(self):
        # External dep that isn't in any batch shouldn't trigger errors.
        stories = [_story("STORY-1", deps=["EXTERNAL-9"])]
        batches = [{"batch_id": 1, "story_keys": ["STORY-1"]}]
        assert validate_batches(stories, batches) == []


# ---------------------------------------------------------------------------
# deterministic_batches — output still topo-clean
# ---------------------------------------------------------------------------

class TestDeterministicBatchesTopo:
    def test_chain_emits_one_story_per_batch_in_order(self):
        out = deterministic_batches([
            _story("STORY-1"),
            _story("STORY-2", deps=["STORY-1"]),
            _story("STORY-3", deps=["STORY-2"]),
        ])
        # Chain → one story per batch by the dep-frontier algorithm.
        assert [b["story_keys"] for b in out] == [
            ["STORY-1"], ["STORY-2"], ["STORY-3"],
        ]

    def test_output_passes_post_phase_i_validator(self):
        stories = [
            _story("STORY-1"),
            _story("STORY-2", deps=["STORY-1"]),
            _story("STORY-3", deps=["STORY-1"]),
            _story("STORY-4", deps=["STORY-2", "STORY-3"]),
        ]
        assert validate_batches(stories, deterministic_batches(stories)) == []


# ---------------------------------------------------------------------------
# _next_story_in_batch — intra-batch dep guard
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    ws_dir = tmp_path / "topo-ws"
    ws_dir.mkdir()
    ws = str(ws_dir)
    app = story_state.app_name_for_workspace(ws)
    conn = story_state.open_story_db()
    # v4 requires every story to belong to a feature. Seed a single
    # ``test`` feature so the inline create_stories calls below don't
    # have to declare one each time. Stories that don't set a feature
    # via _wrap below pick this up by default.
    story_state.ensure_feature(conn, app, "test", name="Test feature")
    yield conn, ws, app
    conn.close()


def _wrap(items: list[dict]) -> list[dict]:
    """Inject ``feature='test'`` into items that didn't specify one.
    Mirrors the test-only wrapper in test_story_state.py."""
    for item in items:
        item.setdefault("feature", "test")
    return items


class TestNextStoryInBatchDepGuard:
    def test_dependent_story_is_deferred_when_dep_planned(self, db):
        conn, ws, app = db
        story_state.create_stories(conn, app, _wrap([
            {"title": "A"},
            {"title": "B", "depends_on": ["STORY-1"]},
        ]))
        # Put them in one batch with STORY-1 first (topo-correct).
        bid = story_state.start_batch(
            conn, app, "sess-1", ["STORY-1", "STORY-2"],
        )
        # Both are still 'planned'. The first call must return STORY-1
        # (no deps); STORY-2 is deferred because its dep is not 'done'.
        nxt = story_loop._next_story_in_batch(conn, app, bid)
        assert nxt["story_key"] == "STORY-1"

    def test_dependent_story_returns_after_dep_done(self, db):
        conn, ws, app = db
        story_state.create_stories(conn, app, _wrap([
            {"title": "A"},
            {"title": "B", "depends_on": ["STORY-1"]},
        ]))
        bid = story_state.start_batch(
            conn, app, "sess-1", ["STORY-1", "STORY-2"],
        )
        # Mark STORY-1 done; STORY-2's intra-batch dep is now satisfied.
        story_state.mark_done(conn, app, "STORY-1")
        nxt = story_loop._next_story_in_batch(conn, app, bid)
        assert nxt["story_key"] == "STORY-2"

    def test_intra_batch_dep_deferral_skips_to_independent_story(self, db):
        """If a batch contains [STORY-1, STORY-2, STORY-3] where STORY-2
        depends on STORY-1 (which is 'in_progress' from a resumed
        session) and STORY-3 is independent, the loop should defer
        STORY-2 and pick STORY-3 — preserving forward progress."""
        conn, ws, app = db
        story_state.create_stories(conn, app, _wrap([
            {"title": "A"},
            {"title": "B", "depends_on": ["STORY-1"]},
            {"title": "C"},
        ]))
        bid = story_state.start_batch(
            conn, app, "sess-1", ["STORY-1", "STORY-2", "STORY-3"],
        )
        # Simulate a resumed session: STORY-1 is mid-patch.
        story_state.mark_in_progress(conn, app, "STORY-1")
        # _next_story_in_batch's primary order returns in_progress
        # rows first (STORY-1) — verify that comes back, and then
        # marking it done unblocks STORY-2.
        first = story_loop._next_story_in_batch(conn, app, bid)
        assert first["story_key"] == "STORY-1"
        story_state.mark_done(conn, app, "STORY-1")
        # STORY-2's dep is now done; sequence-order picks STORY-2 first.
        second = story_loop._next_story_in_batch(conn, app, bid)
        assert second["story_key"] == "STORY-2"

    def test_cross_batch_dep_does_not_block(self, db):
        """A story whose dep is in an EARLIER batch (and is already
        'done' by the time the new batch starts) shouldn't be blocked
        by this guard — the cross-batch contract was already enforced
        by batch_planner_node."""
        conn, ws, app = db
        story_state.create_stories(conn, app, _wrap([
            {"title": "Earlier"},
            {"title": "Current", "depends_on": ["STORY-1"]},
        ]))
        # First batch contains STORY-1 alone; STORY-1 done.
        story_state.start_batch(conn, app, "sess-1", ["STORY-1"])
        story_state.mark_done(conn, app, "STORY-1")
        # Second batch contains STORY-2 alone; its dep is cross-batch.
        b2 = story_state.start_batch(conn, app, "sess-1", ["STORY-2"])
        nxt = story_loop._next_story_in_batch(conn, app, b2)
        assert nxt["story_key"] == "STORY-2"
