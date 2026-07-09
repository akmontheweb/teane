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
