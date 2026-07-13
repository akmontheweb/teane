"""Tests for the universal stale-mental-model defense in
``harness.graph._pre_patch_screen``.

Extends guard 2 with a session-wide "files modified since the LLM last
read them" set that catches every case of a stale mental model —
whether the mutation came from a prior patcher round, autofix, a
formatter, a cross-story write, or human-intervention edit. No file-
type enumeration involved; any modified file becomes stuck until the
LLM emits a fresh READ_FILE against it.

Root cause: finsearch build (session 5f65a887, 2026-07-09) spent ~12
minutes ping-ponging on ``pyproject.toml`` REPLACE_BLOCK misses before
hitting the HITL auto-resume 3/3 cap and terminating silently.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from harness import graph
from harness.patcher import OperationType, Placement


def _mk_block(*, file: str, operation: OperationType = OperationType.REPLACE_BLOCK,
              search: str = "old", replace: str = "new") -> SimpleNamespace:
    """Minimal PatchBlock-shaped stub — the screen only reads .operation,
    .file, .search."""
    return SimpleNamespace(
        operation=operation,
        file=file,
        search=search,
        replace=replace,
        content="",
        anchor="",
        placement=Placement.AFTER,
        count="unique",
        line=0,
        end_line=0,
        expected_file_hash="",
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestMarkFilesModified:
    def test_records_single_file(self):
        lc: dict = {}
        graph._mark_files_modified(lc, ["server/main.py"])
        assert graph._get_files_modified(lc) == {"server/main.py"}

    def test_idempotent_and_dedupes(self):
        lc: dict = {}
        graph._mark_files_modified(lc, ["a.py", "b.py"])
        graph._mark_files_modified(lc, ["a.py", "c.py"])
        assert graph._get_files_modified(lc) == {"a.py", "b.py", "c.py"}

    def test_persisted_as_sorted_list(self):
        """Deterministic JSON serialisation for checkpoint replay."""
        lc: dict = {}
        graph._mark_files_modified(lc, ["z.py", "a.py", "m.py"])
        assert lc[graph._FILES_MODIFIED_LOOP_KEY] == ["a.py", "m.py", "z.py"]

    def test_survives_list_round_trip(self):
        """Round-tripping through checkpoint (dict-of-lists) preserves
        the set semantics on the next read."""
        lc: dict = {graph._FILES_MODIFIED_LOOP_KEY: ["a.py", "b.py"]}
        assert graph._get_files_modified(lc) == {"a.py", "b.py"}
        graph._mark_files_modified(lc, ["c.py"])
        assert graph._get_files_modified(lc) == {"a.py", "b.py", "c.py"}

    def test_empty_input_is_noop(self):
        lc: dict = {"other_key": "preserved"}
        graph._mark_files_modified(lc, [])
        assert lc == {"other_key": "preserved"}

    def test_get_defaults_to_empty(self):
        assert graph._get_files_modified({}) == set()
        assert graph._get_files_modified({"unrelated": 5}) == set()


class TestFilesSeenPersistence:
    """Task 11 (2026-07-12): ``files_seen_by_llm`` must survive
    ``story_loop_node``'s wholesale ``node_state`` replacement so
    round N+1's ``_pre_patch_screen`` sees the fresh hashes round
    N's reactive inject wrote. ``loop_counter`` is the persistent
    home; ``node_state`` is a backwards-compat mirror only.
    """

    def test_read_prefers_loop_counter_over_node_state(self):
        state = {
            "loop_counter": {graph._FILES_SEEN_LOOP_KEY: {"a.py": "hash-a"}},
            "node_state": {"files_seen_by_llm": {"b.py": "hash-b-stale"}},
        }
        assert graph._read_files_seen(state) == {"a.py": "hash-a"}

    def test_read_falls_back_to_node_state(self):
        # Resume checkpoints written before the loop_counter migration
        # only had node_state; the fallback keeps them working.
        state = {"node_state": {"files_seen_by_llm": {"b.py": "hash-b"}}}
        assert graph._read_files_seen(state) == {"b.py": "hash-b"}

    def test_read_falls_back_when_loop_counter_empty(self):
        state = {
            "loop_counter": {graph._FILES_SEEN_LOOP_KEY: {}},
            "node_state": {"files_seen_by_llm": {"b.py": "hash-b"}},
        }
        assert graph._read_files_seen(state) == {"b.py": "hash-b"}

    def test_read_returns_fresh_copy(self):
        # Callers mutate the returned dict freely (add fresh hashes as
        # they READ_FILE); those mutations must not leak into state.
        src = {"a.py": "hash-a"}
        state = {"loop_counter": {graph._FILES_SEEN_LOOP_KEY: src}}
        r = graph._read_files_seen(state)
        r["c.py"] = "hash-c"
        assert "c.py" not in src

    def test_read_ignores_non_string_entries(self):
        state = {
            "loop_counter": {
                graph._FILES_SEEN_LOOP_KEY: {
                    "a.py": "hash-a", "b.py": 42, 3: "hash-c", "d.py": None,
                },
            },
        }
        assert graph._read_files_seen(state) == {"a.py": "hash-a"}

    def test_stash_writes_into_loop_counter(self):
        lc: dict = {}
        graph._stash_files_seen(lc, {"a.py": "hash-a"})
        assert lc[graph._FILES_SEEN_LOOP_KEY] == {"a.py": "hash-a"}

    def test_stash_overwrites_prior_entry(self):
        # The map is round-N cumulative; the last-writer-wins semantics
        # match how ``tool_files_seen`` is threaded through a node.
        lc: dict = {graph._FILES_SEEN_LOOP_KEY: {"a.py": "hash-a"}}
        graph._stash_files_seen(lc, {"a.py": "hash-a-v2", "b.py": "hash-b"})
        assert lc[graph._FILES_SEEN_LOOP_KEY] == {
            "a.py": "hash-a-v2", "b.py": "hash-b",
        }

    def test_stash_isolates_from_source_mutation(self):
        # After stashing, the caller's local dict may keep growing;
        # loop_counter's stored copy must not follow those mutations.
        lc: dict = {}
        src = {"a.py": "hash-a"}
        graph._stash_files_seen(lc, src)
        src["b.py"] = "hash-b-added-later"
        assert lc[graph._FILES_SEEN_LOOP_KEY] == {"a.py": "hash-a"}

    def test_survives_node_state_replacement_via_read(self):
        # End-to-end shape check: patching_node stashes into
        # loop_counter, an intermediate node returns a fresh node_state
        # without files_seen_by_llm (LangGraph replaces it wholesale),
        # next patching call's _read_files_seen still finds the hashes.
        lc: dict = {}
        graph._stash_files_seen(lc, {"a.py": "hash-a"})
        # Simulate story_loop_node's node_state replacement.
        state = {"loop_counter": lc, "node_state": {"current_node": "story_loop"}}
        assert graph._read_files_seen(state) == {"a.py": "hash-a"}


class TestPartialProgressDemote:
    """The partial-progress gate demotes rounds where a stray
    CREATE_FILE succeeds while the story's intended surgical edits
    were rejected as stuck-reread. Without demotion, story_loop's
    ``_patch_success > 0`` advance test would mark the story done
    and the intended edits stay on the floor. Root cause: finsearch
    STORY-004 ran patches=10 succeed=1 fail=9 (all 9 fails were
    stuck-reread) and silently advanced.
    """

    def test_no_real_success_never_demoted(self):
        # Real zero-patch is already caught by the existing tripwire;
        # the gate must not double-count it.
        assert graph._partial_progress_demote(0, 5) is False
        assert graph._partial_progress_demote(0, 0) is False

    def test_no_stuck_reread_never_demoted(self):
        # A clean round with real successes and no screen rejections
        # is exactly what advance is for — leave it alone.
        assert graph._partial_progress_demote(3, 0) is False
        assert graph._partial_progress_demote(100, 0) is False

    def test_finsearch_story_004_pattern_demoted(self):
        # 1 landed / 9 stuck-reread: the canonical silent-gap case.
        assert graph._partial_progress_demote(1, 9) is True

    def test_ratio_at_threshold_not_demoted(self):
        # 3 landed / 3 stuck-reread → ratio exactly 0.5 → NOT
        # demoted (strict less-than). Rationale: half-real progress
        # is real enough to advance; retry cost would exceed benefit.
        assert graph._partial_progress_demote(3, 3) is False

    def test_ratio_just_below_threshold_demoted(self):
        # 2 landed / 3 stuck-reread → ratio 0.4 < 0.5 → demoted.
        assert graph._partial_progress_demote(2, 3) is True

    def test_ratio_just_above_threshold_not_demoted(self):
        # 3 landed / 2 stuck-reread → ratio 0.6 > 0.5 → advance.
        assert graph._partial_progress_demote(3, 2) is False

    def test_custom_ratio_threshold(self):
        # A caller wanting to be stricter (e.g. 0.8) sees the same
        # 3-vs-2 case demote instead of advance.
        assert graph._partial_progress_demote(3, 2, min_ratio=0.8) is True
        # And more permissive (0.1) keeps the finsearch case as advance.
        assert graph._partial_progress_demote(1, 9, min_ratio=0.1) is False


class TestUniversalModifiedGuard:
    """Any file in ``files_modified_this_session`` is treated as stuck
    regardless of type — no manifest-specific enumeration involved."""

    @pytest.mark.parametrize("path", [
        "pyproject.toml",
        "server/main.py",
        "package.json",
        "client/src/components/Foo.tsx",
        "pom.xml",
        "tailwind.config.js",
        "deep/nested/module/util.ts",
    ])
    def test_modified_file_with_stale_view_rejected(self, tmp_path, path):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# module A\n" * 5 + "def hello():\n    return 42\n" * 3,
            encoding="utf-8",
        )

        loop_counter: dict = {}
        graph._mark_files_modified(loop_counter, [path])

        blocks = [_mk_block(file=path)]
        files_seen = {path: _sha256("old-view")}   # LLM's view is stale
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter=loop_counter,
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert kept == []
        assert len(rejections) == 1
        assert "screen:stuck-reread" in rejections[0].error
        assert path in rejections[0].error

    def test_modified_file_with_fresh_view_allowed(self, tmp_path):
        (tmp_path / "server").mkdir()
        body = "# module A\n" * 5 + "def hello():\n    return 42\n" * 3
        (tmp_path / "server" / "main.py").write_text(body, encoding="utf-8")

        loop_counter: dict = {}
        graph._mark_files_modified(loop_counter, ["server/main.py"])

        blocks = [_mk_block(file="server/main.py")]
        # LLM re-read after the modification — seen hash matches disk.
        files_seen = {"server/main.py": _sha256(body)}
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter=loop_counter,
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert len(kept) == 1
        assert rejections == []

    def test_unmodified_file_first_attempt_not_stuck(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "main.py").write_text(
            "# module A\n" * 5 + "def hello():\n    return 42\n" * 3,
            encoding="utf-8",
        )
        blocks = [_mk_block(file="server/main.py")]
        # No modifications tracked, LLM hasn't seen the file — guard 2
        # stays quiet, block goes through.
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter={},
            files_seen_by_llm={},
            workspace_path=str(tmp_path),
        )
        assert len(kept) == 1
        assert rejections == []


class TestExternallyMutatedFallback:
    """Belt-and-suspenders — even when a write site hasn't been threaded
    through ``_mark_files_modified``, on-disk divergence from
    files_seen_by_llm still triggers guard 2."""

    def test_hash_divergence_rejected(self, tmp_path):
        (tmp_path / "server").mkdir()
        target = tmp_path / "server" / "config.py"
        target.write_text(
            "# config\nDATABASE_URL = 'postgres://localhost/x'\n" * 3,
            encoding="utf-8",
        )

        blocks = [_mk_block(file="server/config.py")]
        # LLM saw old content, disk has new — no explicit tracking,
        # but sha256 divergence still catches it.
        files_seen = {"server/config.py": _sha256("stale-view-content\n" * 3)}
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter={},
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert kept == []
        assert len(rejections) == 1
        assert "screen:stuck-reread" in rejections[0].error

    def test_empty_seen_hash_ignored(self, tmp_path):
        """A files_seen entry with an empty hash (never actually read)
        skips the divergence check — doesn't accidentally block edits."""
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "main.py").write_text(
            "# module A\n" * 5 + "def hello():\n    return 42\n" * 3,
            encoding="utf-8",
        )
        blocks = [_mk_block(file="server/main.py")]
        files_seen = {"server/main.py": ""}   # sentinel: never seen
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter={},
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert len(kept) == 1
        assert rejections == []


class TestStuckRereadAutoInjectRecovery:
    """When guard 2 rejects a batch of surgical edits with
    ``[screen:stuck-reread]``, ``patching_node`` / ``repair_node``
    auto-inject the current on-disk content into ``messages`` and
    refresh ``files_seen_by_llm`` so the LLM's next round is accepted
    without a round-trip on a READ_FILE the model was too eager to
    skip. Root cause: finsearch build session finsearch-agile-
    1783819612 hit the zero_patch_loop:2 tripwire twice on STORY-004
    because guard 2 rejected edits against files STORY-003 had just
    created, and the LLM ignored the "READ_FILE first" directive.
    """

    def test_refreshing_seen_hash_via_resolve_clears_guard_2(self, tmp_path):
        # Prove the primitive the node relies on: after
        # ``_resolve_read_blocks(record_hashes_into=files_seen)``, a
        # second screen call against the same block passes because
        # files_seen[path] now matches the on-disk sha256.
        (tmp_path / "server").mkdir()
        body = "def compute():\n    return 1 + 2\n"
        (tmp_path / "server" / "svc.py").write_text(body, encoding="utf-8")

        loop_counter: dict = {}
        graph._mark_files_modified(loop_counter, ["server/svc.py"])

        blocks = [_mk_block(file="server/svc.py")]
        files_seen: dict[str, str] = {}   # simulates cross-story reset

        # First screen call — guard 2 rejects.
        kept, rejections = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter=loop_counter,
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert kept == []
        assert len(rejections) == 1
        assert "screen:stuck-reread" in rejections[0].error

        # Auto-inject step (what patching_node / repair_node do inline
        # after harvesting stuck-reread rejections).
        injected = graph._resolve_read_blocks(
            [("server/svc.py", None)],
            workspace_path=str(tmp_path),
            record_hashes_into=files_seen,
        )
        assert "def compute" in injected
        assert files_seen.get("server/svc.py") == _sha256(body)

        # Second screen call — guard 2 now passes because seen hash
        # matches disk. This is what the LLM's next dispatch sees.
        kept2, rejections2 = graph._pre_patch_screen(
            blocks=blocks,
            loop_counter=loop_counter,
            files_seen_by_llm=files_seen,
            workspace_path=str(tmp_path),
        )
        assert len(kept2) == 1
        assert rejections2 == []


class TestPatchingNodePersistsFilesSeenByLlm:
    """Bug B (2026-07-10): patching_node's return did not carry
    ``files_seen_by_llm`` back onto ``node_state`` — repair_node's
    parity line at graph.py:15125 was missing here. The next
    patching call would then read an empty dict, seed a fresh
    ``tool_files_seen``, and _pre_patch_screen's guard 2 rejected
    every edit against a previously-modified file. Session
    44c5e194 batch 85 (STORY-033–037) lost every patch to this
    across 5 stories.
    """

    def test_source_contains_files_seen_by_llm_in_return_delta(self):
        """Static parity check: patching_node's return node_state
        MUST include ``files_seen_by_llm`` so the next call's
        _pre_patch_screen sees prior hashes. Failure mode is silent
        pre-flight rejection of every edit — not the sort of bug
        pytest-driven integration coverage catches easily."""
        import inspect
        from harness.graph import patching_node
        src = inspect.getsource(patching_node)
        # There are two shapes of value that would be correct — the
        # tool-loop-returned local ``tool_files_seen`` (populated by
        # in-loop READ_FILE tool calls AND by preflight) or the
        # pattern used in repair_node. Either match — we just want
        # to catch a regression where the line is dropped again.
        assert "\"files_seen_by_llm\": tool_files_seen" in src or \
               "'files_seen_by_llm': tool_files_seen" in src, (
            "patching_node's return must persist files_seen_by_llm "
            "back onto node_state — otherwise the next patching call "
            "loses every hash the LLM has been shown and guard 2 in "
            "_pre_patch_screen rejects every edit against files "
            "modified this session."
        )
