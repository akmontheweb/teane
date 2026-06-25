"""Phase K — mid-batch resume via per-gate progress markers.

Verifies:

- ``_batch_gate_passed`` reads ``batch_gate_progress[str(batch_id)][gate]``
- ``_mark_batch_gate`` returns an immutably-updated dict with the new flag
  set, no-op when no batch is active
- ``route_after_story_loop`` (Phase K extension):
  * no batch_complete → ``patching_node`` (existing)
  * batch_complete + no flags → ``speculative_node`` (full chain)
  * batch_complete + compile_passed only → ``code_review_node``
  * batch_complete + compile+review passed → ``batch_commit_node``
- ``create_initial_state`` seeds ``batch_gate_progress`` as ``{}``
"""

from __future__ import annotations

import pytest

from harness import story_loop
from harness.graph import (
    _batch_gate_passed,
    _mark_batch_gate,
    create_initial_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestBatchGatePassed:
    def test_returns_false_when_no_batch_active(self):
        st = {
            "current_batch_id": 0,
            "batch_gate_progress": {"1": {"compile_passed": True}},
        }
        assert _batch_gate_passed(st, "compile_passed") is False

    def test_returns_false_when_progress_dict_missing(self):
        st = {"current_batch_id": 1}
        assert _batch_gate_passed(st, "compile_passed") is False

    def test_returns_true_when_flag_set_for_current_batch(self):
        st = {
            "current_batch_id": 3,
            "batch_gate_progress": {"3": {"compile_passed": True}},
        }
        assert _batch_gate_passed(st, "compile_passed") is True

    def test_returns_false_for_different_batch_id(self):
        # State carries progress for batch 2 but we're now in batch 5.
        st = {
            "current_batch_id": 5,
            "batch_gate_progress": {"2": {"compile_passed": True}},
        }
        assert _batch_gate_passed(st, "compile_passed") is False


class TestMarkBatchGate:
    def test_returns_existing_dict_when_no_batch_active(self):
        existing = {"5": {"compile_passed": True}}
        st = {"current_batch_id": 0, "batch_gate_progress": existing}
        out = _mark_batch_gate(st, "review_passed")
        assert out == existing
        assert out is not existing  # always returns a fresh copy

    def test_sets_flag_on_fresh_state(self):
        st = {"current_batch_id": 4, "batch_gate_progress": {}}
        out = _mark_batch_gate(st, "compile_passed")
        assert out == {"4": {"compile_passed": True}}

    def test_preserves_existing_gates_and_other_batches(self):
        st = {
            "current_batch_id": 7,
            "batch_gate_progress": {
                "1": {"compile_passed": True, "review_passed": True},
                "7": {"compile_passed": True},
            },
        }
        out = _mark_batch_gate(st, "review_passed")
        assert out == {
            "1": {"compile_passed": True, "review_passed": True},
            "7": {"compile_passed": True, "review_passed": True},
        }

    def test_does_not_mutate_input_state(self):
        st = {"current_batch_id": 2, "batch_gate_progress": {}}
        _mark_batch_gate(st, "compile_passed")
        # Original dict reference unchanged.
        assert st["batch_gate_progress"] == {}


# ---------------------------------------------------------------------------
# route_after_story_loop — resume short-circuits
# ---------------------------------------------------------------------------

class TestRouteAfterStoryLoopResume:
    def test_story_picked_routes_to_patching(self):
        st = {
            "node_state": {"batch_complete": False},
            "current_batch_id": 3,
        }
        assert story_loop.route_after_story_loop(st) == "patching_node"

    def test_batch_complete_no_progress_enters_full_chain(self):
        # First run, no crash — every gate runs.
        st = {
            "node_state": {"batch_complete": True},
            "current_batch_id": 1,
            "batch_gate_progress": {},
        }
        assert story_loop.route_after_story_loop(st) == "speculative_node"

    def test_batch_complete_compile_passed_skips_to_review(self):
        # Resume after crash: compile already passed, review didn't run.
        st = {
            "node_state": {"batch_complete": True},
            "current_batch_id": 1,
            "batch_gate_progress": {"1": {"compile_passed": True}},
        }
        assert story_loop.route_after_story_loop(st) == "code_review_node"

    def test_batch_complete_both_passed_skips_to_commit(self):
        # Crash inside batch_commit itself — both gates passed; the
        # resume needs to re-enter the seal step (idempotent).
        st = {
            "node_state": {"batch_complete": True},
            "current_batch_id": 1,
            "batch_gate_progress": {
                "1": {"compile_passed": True, "review_passed": True},
            },
        }
        assert story_loop.route_after_story_loop(st) == "batch_commit_node"

    def test_progress_for_different_batch_is_ignored(self):
        # The progress dict carries data for batch 1, but we're in
        # batch 2 — must NOT skip the verification chain.
        st = {
            "node_state": {"batch_complete": True},
            "current_batch_id": 2,
            "batch_gate_progress": {
                "1": {"compile_passed": True, "review_passed": True},
            },
        }
        assert story_loop.route_after_story_loop(st) == "speculative_node"


# ---------------------------------------------------------------------------
# create_initial_state seeds the empty dict
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_create_initial_state_seeds_batch_gate_progress_empty(self):
        state = create_initial_state(
            workspace_path="/tmp/fake",
            initial_prompt="x",
            build_command="true",
        )
        assert state.get("batch_gate_progress") == {}


# ---------------------------------------------------------------------------
# batch_commit_node clears progress on seal
# ---------------------------------------------------------------------------

from harness import story_state  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "gate-ws"
    ws.mkdir()
    return str(ws)


class TestBatchCommitClearsProgress:
    def test_seal_pops_batch_progress_entry(self, workspace):
        app = story_state.app_name_for_workspace(workspace)
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "test", name="Test feature")
            story_state.create_stories(
                conn, app, [{"title": "S1", "feature": "test"}],
            )
            bid = story_state.start_batch(conn, app, "sess-1", ["STORY-1"])
        finally:
            conn.close()

        state = {
            "workspace_path": workspace,
            "session_id": "sess-1",
            "current_batch_id": bid,
            "current_story_id": "",
            "story_scope_files": [],
            "story_modified_baseline": [],
            "batch_modified_files": ["a.py"],
            "batch_gate_progress": {
                str(bid): {"compile_passed": True, "review_passed": True},
                "999": {"compile_passed": True},  # other batch entry
            },
            "commit_on_story": False,
            "loop_counter": {},
        }
        out = story_loop.batch_commit_node(state)
        # Sealed batch's entry was popped; other batch's entry preserved.
        next_bgp = out["batch_gate_progress"]
        assert str(bid) not in next_bgp
        assert next_bgp.get("999") == {"compile_passed": True}
