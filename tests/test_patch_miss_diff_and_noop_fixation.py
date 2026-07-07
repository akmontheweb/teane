"""Regression tests for two 2026-07-07 patcher signal upgrades.

Both target LLM behaviour observed in session cec4d124 HITL#1: the repair
LLM kept emitting REPLACE_BLOCK searches that didn't match, and (in the
same batch) 20+ byte-identical REWRITE_FILEs across ~15 files. The
existing rejection surfaces told the LLM "search didn't match" and
"you emitted what's already there" but neither carried enough delta or
persistence signal to break the fixation.

Fix #1 — ``_render_search_miss_diff`` prepends an ``ndiff``-style delta
between the LLM's search block and the closest region in the file, so
the LLM sees WHY its search missed (whitespace? renamed identifier?
moved brace?) instead of having to re-derive the delta from the raw
line-numbered window.

Fix #3 — ``rewrite_file_no_ops_per_file`` tracks consecutive no-op
REWRITE_FILEs per file. ``_format_rewrite_file_noop_directive`` emits a
"you're in a no-op fixation loop" block at ≥ 2, calling out the
persistence signal that the patcher's per-emission error can't.
"""

from __future__ import annotations

from harness.graph import _format_rewrite_file_noop_directive
from harness.patcher import _render_search_miss_diff


# ---------------------------------------------------------------------------
# Fix #1 — closest-match diff on REPLACE_BLOCK miss
# ---------------------------------------------------------------------------


class TestRenderSearchMissDiff:
    """The diff should be present + informative for typical near-misses
    and empty (fall back to the raw window) when the search has no close
    match — the raw window is more useful than a misleading diff."""

    def test_diff_surfaces_renamed_parameter(self):
        # The exact failure shape the fix targets: LLM searches for the
        # old signature after the file already renamed the parameter.
        file_content = (
            "def get_health_score(company_id):\n"
            "    data = fetch_company_data(company_id)\n"
            "    return calculate_score(data)\n"
        )
        search = (
            "def get_health_score(company: Company):\n"
            "    data = fetch_company_data(company.id)\n"
            "    return calculate_score(data)"
        )
        diff = _render_search_miss_diff(file_content, search)
        assert diff, "expected a diff for a high-similarity near-miss"
        # Delta lines carry the ``- `` / ``+ `` markers ndiff produces.
        # We check both directions of the delta appear — the LLM needs
        # to see both what it typed AND what the file actually says.
        assert "- def get_health_score(company: Company):" in diff
        assert "+ def get_health_score(company_id):" in diff
        assert "- " in diff and "+ " in diff
        # Header must locate the region so the LLM can jump to it.
        assert "similarity" in diff.lower()

    def test_diff_returns_empty_for_no_close_match(self):
        # When nothing in the file even remotely resembles the search
        # block, the fuzzy anchor would be arbitrary noise — better to
        # return empty and let the caller show the raw window alone.
        file_content = (
            "import os\n"
            "import sys\n"
            "from typing import Any\n"
        )
        search = "SELECT * FROM users WHERE id = 42;"
        assert _render_search_miss_diff(file_content, search) == ""

    def test_diff_skips_when_search_is_huge(self):
        # An 80-line search block being diff'd against an 80-line region
        # would produce more noise than signal. Cap kicks in at
        # ``max_search_lines`` (default 30) and returns empty.
        file_content = "\n".join(f"line {i}" for i in range(200))
        search = "\n".join(f"line {i}" for i in range(80))
        assert _render_search_miss_diff(file_content, search) == ""

    def test_diff_handles_empty_inputs(self):
        assert _render_search_miss_diff("", "some search") == ""
        assert _render_search_miss_diff("some file\n", "") == ""

    def test_diff_shows_whitespace_delta(self):
        # A common failure mode: the LLM's search uses 4-space indent
        # but the file uses tabs (or vice-versa). ``ndiff`` surfaces the
        # exact character-level delta so the LLM can copy the correct
        # whitespace next round.
        file_content = "def foo():\n\treturn 42\n"
        search = "def foo():\n    return 42"
        diff = _render_search_miss_diff(file_content, search)
        assert diff, "expected a diff even for a whitespace-only delta"
        # Both the tab-line and the 4-space-line should appear in the
        # diff so the LLM can see which side has which whitespace.
        assert "    return 42" in diff
        assert "\treturn 42" in diff

    def test_diff_header_names_the_region_by_line_numbers(self):
        # The header must tell the LLM where in the file to look. That's
        # what distinguishes the diff view from a naive text-vs-text
        # comparison — it grounds the LLM in the actual file coordinates.
        file_content = "\n".join([
            "# preamble",
            "# more preamble",
            "def target():",
            "    return 1",
            "# footer",
        ])
        search = "def target():\n    return 2"
        diff = _render_search_miss_diff(file_content, search)
        assert diff
        # Region starts at line 3 (1-indexed, "def target():").
        assert "lines 3-" in diff


# ---------------------------------------------------------------------------
# Fix #3 — REWRITE_FILE no-op fixation directive
# ---------------------------------------------------------------------------


class TestRewriteFileNoopDirective:

    def test_fires_at_two_consecutive_noops(self):
        out = _format_rewrite_file_noop_directive({"backend/models/x.py": 2})
        assert out, "directive must fire at threshold (≥ 2)"
        assert "REWRITE_FILE no-op fixation trap" in out
        assert "backend/models/x.py" in out
        # Must direct the LLM away from the same-content fixation and
        # toward looking elsewhere (callers, tests, fixtures).
        assert "READ_FILE" in out
        assert "caller" in out.lower() or "call chain" in out.lower()

    def test_silent_below_threshold(self):
        # One no-op is normal (LLM misjudged once). The escalation is
        # scoped to persistence, so 1 → silent, 2 → fires.
        assert _format_rewrite_file_noop_directive({"foo.py": 1}) == ""
        assert _format_rewrite_file_noop_directive({}) == ""

    def test_fires_only_for_files_at_or_above_threshold(self):
        # Mixed input: one file at 1 (silent), one at 3 (fires). The
        # directive must include the ≥ 2 file only.
        out = _format_rewrite_file_noop_directive({
            "a.py": 1,
            "b.py": 3,
        })
        assert "b.py" in out
        assert "a.py" not in out

    def test_includes_consecutive_count(self):
        # The count is the whole point of the directive — the LLM
        # needs to see the persistence signal. Format is
        # ``- ``file`` (N)`` per stuck file.
        out = _format_rewrite_file_noop_directive({"foo.py": 4})
        assert "foo.py" in out
        assert "(4)" in out
