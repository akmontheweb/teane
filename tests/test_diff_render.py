"""Phase 5 regression: server-side unified-diff renderer.

Kept in a dedicated file because ``harness/diff_render.py`` is pure
and can be exercised without any dashboard fixtures.
"""

from __future__ import annotations

from harness.diff_render import (
    looks_binary,
    render_patch_list,
    render_unified_diff,
)


def test_render_unified_diff_marks_add_del_ctx_lines():
    body = render_unified_diff(
        "line 1\nline 2\nline 3\n",
        "line 1\nline 2 changed\nline 3\n",
        file_path="src/x.py",
    )
    assert "diff__file" in body
    assert "src/x.py" in body
    assert "diff-add" in body
    assert "diff-del" in body
    # Context should render at least the unchanged neighbour lines.
    assert "diff-ctx" in body


def test_render_unified_diff_identical_files_states_it():
    body = render_unified_diff("same\n", "same\n", file_path="a.py")
    assert "identical" in body.lower()


def test_render_unified_diff_binary_short_circuits():
    body = render_unified_diff("\x00\x00binary", "different", file_path="a.bin")
    assert "Binary file" in body
    assert "diff-add" not in body
    assert "diff-del" not in body


def test_render_unified_diff_truncates_giant_diffs():
    a = "\n".join(f"line-{i}" for i in range(2000))
    b = "\n".join(f"other-{i}" for i in range(2000))
    body = render_unified_diff(a, b, file_path="huge.txt", max_lines=100)
    assert "truncated" in body


def test_render_unified_diff_escapes_html():
    body = render_unified_diff("<script>", "<b>", file_path="x.html")
    assert "&lt;script&gt;" in body
    assert "<script>" not in body


def test_render_patch_list_multi_file():
    payload = [
        {"path": "a.py", "operation": "create_file",
         "before": "", "after": "print(1)\n", "is_binary": False},
        {"path": "b.py", "operation": "replace_block",
         "before": "old\n", "after": "new\n", "is_binary": False},
    ]
    body = render_patch_list(payload)
    assert "a.py" in body
    assert "b.py" in body
    # Two file headers implies two stacked diff blocks.
    assert body.count("diff__file") == 2


def test_render_patch_list_binary_entries_render_summary_only():
    payload = [
        {"path": "logo.png", "operation": "create_file",
         "is_binary": True, "size_after": 2048},
    ]
    body = render_patch_list(payload)
    assert "logo.png" in body
    assert "Binary file" in body
    # No before/after byte payload leaks into the HTML.
    assert "diff-add" not in body


def test_render_patch_list_empty_state():
    assert "No patches" in render_patch_list([])


def test_looks_binary_detects_null_bytes():
    assert looks_binary(b"hello\x00world")
    assert looks_binary("hello\x00world")
    assert not looks_binary(b"hello world\n")
    assert not looks_binary("hello world\n")
