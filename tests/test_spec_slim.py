"""Tests for the planner-only-section stripper.

The RSD injected into the system prompt used to carry planner-only
fields (business drivers, success metrics, priority, wave) that the
code-emission LLM does not use. Finsearch session 156032347 shipped
a 243 KB system prompt; that overhead compounds across every LLM
call in the session. :func:`_slim_spec_for_prompt` cuts it once at
load time so the immutable system prompt is smaller while every
code-grounding field (assumptions, story titles, scope, ACs) is
preserved.
"""

from __future__ import annotations

from harness.cli import _slim_spec_for_prompt


class TestPlannerOnlyStripped:
    """Fields the planner uses but the code-writer doesn't."""

    def test_business_driver_line_removed(self) -> None:
        spec = (
            "## Epic: E-1 — Search\n\n"
            "**Vision statement:** Enable users to search.\n"
            "**Business driver:** Reduce time-to-info by 90%.\n"
            "**Scope:** Ticker input.\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert "Business driver" not in out
        assert "Reduce time-to-info" not in out
        # neighbouring fields preserved
        assert "Vision statement" in out
        assert "Scope" in out

    def test_success_metrics_block_removed(self) -> None:
        spec = (
            "**Success metrics:**\n"
            "- Search < 3s for 95% of queries.\n"
            "- Autocomplete < 200ms.\n"
            "**Priority:** Must Have\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert "Success metrics" not in out
        assert "Search < 3s" not in out
        assert "Autocomplete" not in out
        assert "Priority" not in out

    def test_estimated_size_removed(self) -> None:
        spec = "**Estimated size:** M\n**Dependencies:** External API\n"
        out = _slim_spec_for_prompt(spec)
        assert "Estimated size" not in out
        assert "Dependencies" in out  # kept — code-relevant

    def test_wave_and_iteration_removed(self) -> None:
        spec = "**Wave:** 1\n**Iteration:** 2\n**Vision statement:** X\n"
        out = _slim_spec_for_prompt(spec)
        assert "Wave" not in out
        assert "Iteration" not in out
        assert "Vision statement" in out


class TestCodeRelevantPreserved:
    """Every field the LLM actually needs to write code must survive."""

    def test_ac_lines_preserved(self) -> None:
        spec = (
            "### STORY-1.AC-1\n"
            "Given a valid ticker, return the CIK within 200ms.\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert "STORY-1.AC-1" in out
        assert "Given a valid ticker" in out

    def test_scope_and_out_of_scope_preserved(self) -> None:
        spec = (
            "**Scope:** Ticker search with autocomplete.\n"
            "**Out of scope:** Multi-company search.\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert "Scope" in out
        assert "Out of scope" in out
        assert "Ticker search" in out
        assert "Multi-company" in out

    def test_dependencies_preserved(self) -> None:
        spec = "**Dependencies:** External – SEC EDGAR APIs.\n"
        out = _slim_spec_for_prompt(spec)
        assert "Dependencies" in out
        assert "EDGAR" in out

    def test_assumptions_block_preserved(self) -> None:
        spec = (
            "## Assumptions\n"
            "- SEC EDGAR APIs remain publicly available.\n"
            "- XBRL data is available for all listed companies.\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert "Assumptions" in out
        assert "SEC EDGAR APIs" in out


class TestEdgeCases:
    def test_empty_input(self) -> None:
        assert _slim_spec_for_prompt("") == ""

    def test_no_planner_fields_noop(self) -> None:
        spec = (
            "## Epic: E-1\n"
            "**Vision statement:** Do the thing.\n"
            "**Scope:** X.\n"
        )
        out = _slim_spec_for_prompt(spec)
        # Same shape — no fields matched.
        assert "Vision statement" in out
        assert "Scope" in out

    def test_realistic_agile_snippet_size_reduction(self) -> None:
        # A representative RSD fragment mixing kept + stripped fields.
        # We assert only that the stripped fields disappear and that
        # some real reduction happens; the exact byte count is not the
        # contract.
        spec = (
            "## Epic: EPIC-001 — Company Search\n\n"
            "**Vision statement:** Enable any user to find a company.\n"
            "**Business driver:** Eliminate manual CIK lookups.\n"
            "**Scope:** Ticker / name search with autocomplete.\n"
            "**Out of scope:** Multi-company search.\n"
            "**Success metrics:**\n"
            "- Search-to-confirmation < 3s for 95% of queries.\n"
            "- Autocomplete < 200ms.\n"
            "- Filing list within 1s.\n"
            "**Priority:** Must Have\n"
            "**Estimated size:** M\n"
            "**Dependencies:** External – SEC EDGAR APIs.\n"
        )
        out = _slim_spec_for_prompt(spec)
        assert len(out) < len(spec)
        for stripped in (
            "Business driver", "Success metrics", "Priority",
            "Estimated size",
        ):
            assert stripped not in out
        for kept in (
            "Vision statement", "Scope", "Out of scope", "Dependencies",
        ):
            assert kept in out
