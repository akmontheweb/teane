"""Regression tests for the post-cleanse READ_FILE re-injection.

Ciod session 523e86a7 observed 2-3 wasted round trips per HITL resume
because the LLM re-emitted ``<<<READ_FILE>>>`` for files it had
already been shown in a prior round. Root cause:
``_trim_mid_loop_messages`` and ``apply_memory_cleanse`` strip the
READ_FILE resolution messages between iterations. The LLM's context
loses the file content and it re-requests the same path.

The fix is a bounded re-injection: on every repair-node preflight,
walk ``files_seen_by_llm`` (which survives cleanses in ``node_state``)
and append paths that aren't already in ``files_for_preflight`` so
their current on-disk content lands in the prompt directly. The LLM
sees the file without asking, no extra dispatch. Capped at
``_REREAD_INJECTION_CAP`` (5) so the prompt can't unbounded-inflate
on sessions that touched dozens of files.

These tests exercise the selection logic in isolation because standing
up a full ``repair_node`` fixture is heavy; the caller-side integration
is verified end-to-end by the existing ``repair_node`` regression tests.
"""

from __future__ import annotations

import os


class TestRereadCandidateSelection:
    """The selection logic that decides which ``files_seen_by_llm``
    paths to re-inject on the next preflight pass. Mirrors the loop
    inline in ``repair_node`` — keep the two implementations aligned."""

    def _select(self, files_seen, existing_preflight, workspace, cap):
        """Replica of the inline selection from ``repair_node``. Kept
        as a separate helper here so future refactors can extract the
        production loop into a callable and share this test suite."""
        existing = set(existing_preflight)
        candidates = []
        for path in files_seen:
            if path in existing:
                continue
            if not os.path.isfile(os.path.join(workspace, path)):
                continue
            candidates.append(path)
            if len(candidates) >= cap:
                break
        return candidates

    def test_selects_seen_files_not_in_existing_preflight(self, tmp_path):
        (tmp_path / "a.py").write_text("# a\n")
        (tmp_path / "b.py").write_text("# b\n")
        (tmp_path / "c.py").write_text("# c\n")
        picked = self._select(
            files_seen={"a.py": "h1", "b.py": "h2", "c.py": "h3"},
            existing_preflight=["a.py"],  # already in preflight
            workspace=str(tmp_path),
            cap=5,
        )
        # a.py filtered out (already there); b.py, c.py picked up.
        assert set(picked) == {"b.py", "c.py"}

    def test_respects_cap(self, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text(f"# {i}\n")
        picked = self._select(
            files_seen={f"f{i}.py": "h" for i in range(10)},
            existing_preflight=[],
            workspace=str(tmp_path),
            cap=5,
        )
        assert len(picked) == 5

    def test_skips_missing_files(self, tmp_path):
        # A file recorded in files_seen_by_llm but deleted from disk
        # (patched-away by a REWRITE_FILE, or user cleanup) must not
        # land as a "(file not found)" entry in the prompt.
        (tmp_path / "kept.py").write_text("# kept\n")
        picked = self._select(
            files_seen={"kept.py": "h", "deleted.py": "h"},
            existing_preflight=[],
            workspace=str(tmp_path),
            cap=5,
        )
        assert picked == ["kept.py"]

    def test_empty_seen_dict_is_noop(self, tmp_path):
        picked = self._select(
            files_seen={},
            existing_preflight=[],
            workspace=str(tmp_path),
            cap=5,
        )
        assert picked == []

    def test_all_seen_already_in_preflight_returns_empty(self, tmp_path):
        (tmp_path / "a.py").write_text("# a\n")
        picked = self._select(
            files_seen={"a.py": "h"},
            existing_preflight=["a.py", "b.py"],
            workspace=str(tmp_path),
            cap=5,
        )
        assert picked == []


class TestRereadInjectionCap:
    def test_module_cap_is_set(self):
        from harness.graph import _REREAD_INJECTION_CAP
        # Cap chosen empirically: 5 files at 300 lines / 6 KB each
        # adds roughly 30 KB to the prompt worst-case, well inside
        # the 32 K token per-role budget. Larger caps risk starving
        # the diagnostic + patch-failure sections that follow.
        assert 1 <= _REREAD_INJECTION_CAP <= 20
