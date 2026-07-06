"""Unit tests for the four HITL-prevention harness improvements.

Each Fix is exercised end-to-end against a scratch workspace:

  Fix 1 — patcher REPLACE_BLOCK rejection surfaces a same-basename
          sibling when the LLM's search text lives in the wrong file.

  Fix 2 — HITL escalation summary is corrected when the LLM claims a
          symbol is missing but it's actually present in the workspace.

  Fix 3 — repair diagnostic includes production-code pointers when the
          failing frame is in a test file with no deeper user frame.

  Fix 4 — repair loop injects a fixation-breaker system message when
          the same file has been edited across ≥3 recent rounds and
          the loop has stalled.
"""

from __future__ import annotations

import textwrap

import pytest

from harness.patcher import (
    _find_search_in_other_files,
    _format_sibling_hits,
    _pick_distinctive_line,
)
from harness.graph import (
    _detect_fixation_files,
    _extract_hitl_claimed_symbols,
    _extract_identifiers_near_line,
    _format_fixation_breaker_message,
    _format_production_symbols_under_test,
    _grep_production_definitions,
    _grep_symbol_in_workspace,
    _is_test_file,
    _verify_hitl_summary_claims,
)


# ---------------------------------------------------------------------------
# Fix 1 — patcher cross-file grep on REPLACE_BLOCK rejection
# ---------------------------------------------------------------------------


class TestFix1CrossFileGrep:
    def test_distinctive_line_prefers_longest_non_boilerplate(self):
        search = "}\ndef reconcile_annual_quarterly(records):\n    pass\n"
        # ``}`` and ``pass`` are trivial; the def line wins.
        needle = _pick_distinctive_line(search)
        assert needle is not None
        assert "reconcile_annual_quarterly" in needle

    def test_distinctive_line_returns_none_for_only_trivial_lines(self):
        assert _pick_distinctive_line("}\n{\n") is None

    def test_finds_search_in_sibling_file(self, tmp_path):
        # Two same-basename siblings, only one contains the search text.
        target = tmp_path / "tests" / "test_edgar.py"
        target.parent.mkdir(parents=True)
        target.write_text("# empty test file with no matching content\n")

        sibling = tmp_path / "tests" / "unit" / "backend" / "test_edgar.py"
        sibling.parent.mkdir(parents=True)
        sibling.write_text(textwrap.dedent("""
            async def test_prefix_match(self):
                results = await client.search("GO")
                tickers = {r.ticker for r in results}
                assert "GOOGL" in tickers
        """))

        search = "results = await client.search(\"GO\")\n"
        hits = _find_search_in_other_files(
            str(tmp_path), "tests/test_edgar.py", search,
        )
        assert hits
        assert any("test_edgar.py" in p for p, _ in hits)
        # Should hit the sibling, NOT the target.
        assert all("tests/unit/backend" in p for p, _ in hits)

    def test_format_sibling_hits_ranks_same_basename_first(self):
        hits = [
            ("other/random.py", 12),
            ("tests/unit/backend/test_edgar.py", 157),
        ]
        tail = _format_sibling_hits(hits, "tests/test_edgar.py")
        assert tail  # non-empty
        # Same-basename hit appears before the unrelated file.
        i_test = tail.index("tests/unit/backend/test_edgar.py")
        i_other = tail.index("other/random.py")
        assert i_test < i_other
        # Explicit sibling-file callout is present.
        assert "matches the basename" in tail

    def test_format_sibling_hits_empty_when_no_hits(self):
        assert _format_sibling_hits([], "tests/test_edgar.py") == ""

    def test_skips_git_and_node_modules(self, tmp_path):
        # Content lives inside skipped dirs — must not surface as a hit.
        for d in (".git", "node_modules", "__pycache__"):
            hidden = tmp_path / d / "test_edgar.py"
            hidden.parent.mkdir(parents=True)
            hidden.write_text("distinctive_marker_line_for_scan\n")
        target = tmp_path / "src" / "test_edgar.py"
        target.parent.mkdir(parents=True)
        target.write_text("# empty\n")
        hits = _find_search_in_other_files(
            str(tmp_path), "src/test_edgar.py",
            "distinctive_marker_line_for_scan\n",
        )
        assert hits == []


# ---------------------------------------------------------------------------
# Fix 2 — HITL summary claim grounding
# ---------------------------------------------------------------------------


class TestFix2SummaryGrounding:
    def test_extract_missing_function_claim(self):
        summary = (
            "The root cause is the missing function `reconcile_annual_quarterly` "
            "in backend/services/normalizer.py."
        )
        symbols = _extract_hitl_claimed_symbols(summary)
        assert "reconcile_annual_quarterly" in symbols

    def test_extract_is_not_defined_claim(self):
        summary = "`EdgarClient` is not defined anywhere in the workspace."
        assert "EdgarClient" in _extract_hitl_claimed_symbols(summary)

    def test_extract_add_missing_helper_claim(self):
        summary = (
            "Manually add the missing helper `reconcile_annual_quarterly` "
            "to normalizer.py."
        )
        assert "reconcile_annual_quarterly" in _extract_hitl_claimed_symbols(summary)

    def test_extract_ignores_stopword_identifiers(self):
        summary = "The missing `file` handle is what caused the crash."
        # `file` is a stopword — should not become a grep candidate.
        assert _extract_hitl_claimed_symbols(summary) == []

    def test_grep_finds_definition(self, tmp_path):
        (tmp_path / "backend").mkdir()
        (tmp_path / "backend" / "normalizer.py").write_text(
            "def reconcile_annual_quarterly(records):\n    return records\n"
        )
        hits = _grep_symbol_in_workspace(str(tmp_path), "reconcile_annual_quarterly")
        assert hits
        assert any("normalizer.py" in p for p, _ in hits)

    def test_verify_appends_correction_when_symbol_present(self, tmp_path):
        (tmp_path / "backend").mkdir()
        (tmp_path / "backend" / "normalizer.py").write_text(
            "def reconcile_annual_quarterly(records):\n    return records\n"
        )
        summary = (
            "The repair loop stopped because the missing function "
            "`reconcile_annual_quarterly` in normalizer.py could not "
            "be added by the LLM."
        )
        correction = _verify_hitl_summary_claims(summary, str(tmp_path))
        assert "HARNESS CORRECTION" in correction
        assert "reconcile_annual_quarterly" in correction
        assert "normalizer.py" in correction

    def test_verify_empty_when_claim_holds(self, tmp_path):
        # Symbol genuinely missing — no correction to append.
        summary = "The missing function `truly_missing_symbol_xyz` blocks the loop."
        assert _verify_hitl_summary_claims(summary, str(tmp_path)) == ""

    def test_verify_empty_on_empty_input(self, tmp_path):
        assert _verify_hitl_summary_claims("", str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# Fix 3 — production-symbols-under-test hint
# ---------------------------------------------------------------------------


class TestFix3ProductionPointers:
    def test_is_test_file_detects_common_patterns(self):
        assert _is_test_file("tests/unit/backend/test_edgar.py")
        assert _is_test_file("src/edgar_test.py")
        assert _is_test_file("app/foo.test.ts")
        assert _is_test_file("src/__tests__/foo.spec.ts")
        assert not _is_test_file("backend/services/edgar.py")
        assert not _is_test_file("")

    def test_extract_identifiers_drops_pytest_scaffolding(self):
        src = textwrap.dedent("""
            async def test_prefix_match(self):
                results = await client.search("GO")
                tickers = {r.ticker for r in results}
                assert "GOOGL" in tickers
                assert "GOOG" in tickers
        """)
        idents = _extract_identifiers_near_line(src, line=4, radius=3)
        # Production-code identifier surfaces.
        assert "client" in idents or "search" in idents
        # pytest scaffolding is filtered.
        assert "assert" not in idents

    def test_grep_production_definitions_excludes_tests(self, tmp_path):
        (tmp_path / "backend" / "services").mkdir(parents=True)
        (tmp_path / "backend" / "services" / "edgar.py").write_text(
            "class EdgarClient:\n"
            "    async def search(self, query):\n"
            "        return []\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_edgar.py").write_text(
            "class EdgarClient:\n    pass\n"  # decoy in a test file
        )
        hits = _grep_production_definitions(str(tmp_path), "EdgarClient")
        assert hits
        assert all("backend/services" in p for p, _, _ in hits)

    def test_format_production_symbols_under_test_end_to_end(self, tmp_path):
        # Realistic mini-workspace mirroring the cec4d124 shape.
        (tmp_path / "backend" / "services").mkdir(parents=True)
        (tmp_path / "backend" / "services" / "edgar.py").write_text(
            "class EdgarClient:\n"
            "    async def search(self, query):\n"
            "        return []\n"
        )
        test_dir = tmp_path / "tests" / "unit" / "backend"
        test_dir.mkdir(parents=True)
        test_path = test_dir / "test_edgar.py"
        test_path.write_text(textwrap.dedent("""
            async def test_prefix_match(self):
                client = EdgarClient()
                results = await client.search("GO")
                tickers = {r.ticker for r in results}
                assert "GOOGL" in tickers
                assert "GOOG" in tickers
        """).lstrip())

        errors = [{
            "file": "tests/unit/backend/test_edgar.py",
            "line": 6,  # `assert "GOOG" in tickers`
            "column": 0,
            "severity": "error",
            "error_code": "AssertionError",
            "message": "assert 'GOOG' in {'GOOGL'}",
            "semantic_context": "",
        }]
        rendered = _format_production_symbols_under_test(errors, str(tmp_path))
        assert "Code under test" in rendered
        assert "EdgarClient" in rendered
        assert "backend/services/edgar.py" in rendered
        assert "Default to patching the production code" in rendered

    def test_format_returns_empty_when_no_test_failures(self, tmp_path):
        errors = [{"file": "backend/services/edgar.py", "line": 5,
                   "error_code": "SyntaxError", "message": "..."}]
        assert _format_production_symbols_under_test(errors, str(tmp_path)) == ""

    def test_format_returns_empty_on_no_workspace(self):
        errors = [{"file": "tests/test_x.py", "line": 3, "error_code": "E"}]
        assert _format_production_symbols_under_test(errors, "") == ""


# ---------------------------------------------------------------------------
# Fix 4 — fixation-breaker
# ---------------------------------------------------------------------------


class TestFix4FixationBreaker:
    def _entry(self, rnd: int, success: bool = True, no_op: bool = False):
        return [rnd, "replace_block", success, no_op, ""]

    def test_detects_file_edited_in_three_recent_rounds(self):
        loop_counter = {
            "file_modification_history": {
                "tests/test_edgar.py": [
                    self._entry(5), self._entry(6), self._entry(7),
                ],
            },
        }
        fixated = _detect_fixation_files(loop_counter, threshold=3, window=5)
        assert fixated == [("tests/test_edgar.py", 3)]

    def test_ignores_files_below_threshold(self):
        loop_counter = {
            "file_modification_history": {
                "a.py": [self._entry(4), self._entry(5)],   # only 2 rounds
                "b.py": [self._entry(3), self._entry(4), self._entry(5)],  # 3 → fires
            },
        }
        fixated = _detect_fixation_files(loop_counter, threshold=3, window=5)
        assert ("b.py", 3) in fixated
        assert not any(p == "a.py" for p, _ in fixated)

    def test_ignores_failed_and_no_op_rounds(self):
        loop_counter = {
            "file_modification_history": {
                "flaky.py": [
                    self._entry(3, success=False),        # failure — skip
                    self._entry(4, success=True, no_op=True),  # no-op — skip
                    self._entry(5, success=True),
                ],
            },
        }
        # Only 1 real edit — below threshold.
        assert _detect_fixation_files(loop_counter, threshold=3) == []

    def test_deduplicates_multiple_edits_within_same_round(self):
        # Two entries in the same round for the same file — still counts once.
        loop_counter = {
            "file_modification_history": {
                "a.py": [
                    self._entry(5), self._entry(5),      # both round 5
                    self._entry(6), self._entry(7),
                ],
            },
        }
        fixated = _detect_fixation_files(loop_counter, threshold=3, window=5)
        assert fixated == [("a.py", 3)]

    def test_handles_missing_history(self):
        assert _detect_fixation_files({}, threshold=3) == []
        assert _detect_fixation_files({"file_modification_history": None}) == []

    def test_message_names_the_fixated_file(self):
        msg = _format_fixation_breaker_message([("tests/test_edgar.py", 3)])
        assert "Fixation-breaker" in msg
        assert "tests/test_edgar.py" in msg
        assert "3 recent rounds" in msg
        assert "READ_FILE" in msg  # steers toward reading, not blind editing

    def test_message_lists_multiple_fixated_files(self):
        msg = _format_fixation_breaker_message([
            ("a.py", 4), ("b.py", 3),
        ])
        assert "a.py" in msg and "b.py" in msg

    def test_message_empty_for_empty_list(self):
        assert _format_fixation_breaker_message([]) == ""


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
