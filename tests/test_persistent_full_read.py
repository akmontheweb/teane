"""Verify _collect_workspace_file_content lifts the preflight truncation
caps for files marked as persistent-blocker (fix for the "LLM cannot
compose a REPLACE_BLOCK spanning line 172 because the prompt only shows
lines 1-133" pattern observed in build v5)."""

import os
import tempfile

from harness.graph import _collect_workspace_file_content


def _write(root: str, rel: str, content: str) -> None:
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _fat_line(prefix: str, i: int) -> str:
    # ~90 chars per line so the 300-line / 6000-char default caps
    # truncate before we hit line 200.
    return f"# {prefix} tag={i:04d} " + "." * 75


def test_default_caps_truncate_large_files():
    with tempfile.TemporaryDirectory() as td:
        # 500 fat lines — well past both the 300-line and 6000-char caps.
        big = "\n".join(_fat_line("edgar", i) for i in range(500))
        _write(td, "backend/services/edgar.py", big)

        pairs = _collect_workspace_file_content(
            td, ["backend/services/edgar.py"],
        )
        assert len(pairs) == 1
        _, rendered = pairs[0]
        assert "truncated" in rendered
        # Line 400 is past both caps — must be truncated out.
        assert "tag=0400" not in rendered
        # Line 20 is well within the head; must be present.
        assert "tag=0020" in rendered


def test_full_read_paths_lifts_caps_for_specified_files():
    with tempfile.TemporaryDirectory() as td:
        big = "\n".join(_fat_line("edgar", i) for i in range(500))
        _write(td, "backend/services/edgar.py", big)

        pairs = _collect_workspace_file_content(
            td, ["backend/services/edgar.py"],
            full_read_paths={"backend/services/edgar.py"},
        )
        assert len(pairs) == 1
        _, rendered = pairs[0]
        # No truncation marker for this file — the whole thing came through.
        assert "truncated" not in rendered
        # The line that was previously invisible is now visible.
        assert "tag=0400" in rendered
        # And so is the tail.
        assert "tag=0499" in rendered


def test_full_read_paths_only_affects_matching_files():
    with tempfile.TemporaryDirectory() as td:
        big_a = "\n".join(_fat_line("a", i) for i in range(500))
        big_b = "\n".join(_fat_line("b", i) for i in range(500))
        _write(td, "a.py", big_a)
        _write(td, "b.py", big_b)

        pairs = _collect_workspace_file_content(
            td, ["a.py", "b.py"],
            full_read_paths={"a.py"},
        )
        by_path = dict(pairs)
        # a.py: no truncation, full 500 lines visible.
        assert "truncated" not in by_path["a.py"]
        assert "tag=0499" in by_path["a.py"]
        # b.py: truncated as before, line 400 unreachable.
        assert "truncated" in by_path["b.py"]
        assert "tag=0400" not in by_path["b.py"]


def test_full_read_paths_none_matches_default_behavior():
    # Backward-compatibility: omitting full_read_paths (or passing None)
    # must give the exact same output as before the parameter existed.
    with tempfile.TemporaryDirectory() as td:
        big = "\n".join(f"# line {i:03d}" for i in range(200))
        _write(td, "m.py", big)

        pairs_omitted = _collect_workspace_file_content(td, ["m.py"])
        pairs_none = _collect_workspace_file_content(
            td, ["m.py"], full_read_paths=None,
        )
        pairs_empty = _collect_workspace_file_content(
            td, ["m.py"], full_read_paths=set(),
        )
        assert pairs_omitted == pairs_none == pairs_empty


def test_missing_persistent_file_is_silently_skipped():
    with tempfile.TemporaryDirectory() as td:
        pairs = _collect_workspace_file_content(
            td, ["does_not_exist.py"],
            full_read_paths={"does_not_exist.py"},
        )
        assert pairs == []
