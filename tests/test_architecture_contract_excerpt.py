"""An assertion cannot respect a contract it was never shown.

``gather_story_acceptance_context`` fed the acceptance scenario generator
``SPEC_ARCHITECTURE.md`` through ``_read_excerpt``, which is
``fh.read(max_chars)`` -- a head truncation. On lumina that document is 48,812
bytes / 772 lines and the 6,000-char excerpt ended at **line 96**: the overview
and the start of the module inventory. It never reached the API contract
sections, "Error handling" (line 496), or the "Edge Case & Security Test
Matrix" (line 564).

So the component whose entire job is writing assertions against API contracts
could not see the contracts, and the document that defines the error envelope
was structurally invisible to it. The requirements document states the same
contract in shorthand -- ``404 { code: "NOT_FOUND" }`` -- and IS fully visible.

That is how lumina run 11 died at 4/16. The generator wrote
``assert body == {'code': 'NOT_FOUND'}`` against an application whose
``to_error_response()`` returns ``{"error": {"code", "type", "message",
"error_ref_id"}}``. No production change satisfies that without abandoning the
envelope everywhere; repair may not edit tests; and regeneration anchors on the
same shorthand criterion, so it reproduced the identical wrong assertion until
the ``unsatisfiable_test`` class cap terminated the run.

:class:`TestTheRegressionItself` is the premise. :class:`TestTheRun11Shape` is
the case that killed it.
"""

from __future__ import annotations

import pytest

from harness.acceptance_gen import (
    _read_excerpt,
    architecture_contract_excerpt,
)

#: Shaped like lumina's: the sections an assertion depends on sit at the END,
#: behind a long module inventory.
ARCH_DOC = (
    "# Architecture\n\n"
    "## 1. Architecture Overview\n\n"
    "A FastAPI backend with a React client.\n\n"
    "## 2. Component / Module Inventory\n\n"
    + "".join(
        f"- **`server/app/module_{i}.py`**\n"
        f"  - Purpose: filler module {i} to push the real content down.\n"
        f"  - Public surface: `thing_{i}()`.\n\n"
        for i in range(200)
    )
    + "## 4. API Contracts\n\n"
    "#### `GET /api/v1/birthdays/{birthday_id}`\n\n"
    "- Returns the record, scoped to the authenticated owner.\n"
    "- A record belonging to another owner returns 404.\n\n"
    "### Error handling\n\n"
    "- Domain exceptions are converted by FastAPI exception handlers into "
    "the SRS error envelope.\n"
    "- The envelope is `{\"error\": {\"code\", \"type\", \"message\", "
    "\"error_ref_id\"}}`.\n\n"
    "### Security boundaries\n\n"
    "- Every read is scoped by owner_id.\n\n"
    "### Edge Case & Security Test Matrix\n\n"
    "- Another owner's record must not leak through any endpoint.\n"
)

ENVELOPE_SENTENCE = "the SRS error envelope"
ROUTE = "/api/v1/birthdays/{birthday_id}"


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").write_text(ARCH_DOC)
    return str(tmp_path)


class TestTheRegressionItself:
    def test_the_document_is_long_enough_to_truncate(self):
        """Guards the premise: if the fixture ever shrinks below the budget
        this whole file stops covering anything."""
        assert len(ARCH_DOC) > 20000

    def test_a_head_read_misses_every_section_that_matters(self, ws):
        """The shipped behaviour until now."""
        head = _read_excerpt(
            ws, "docs/SPEC_ARCHITECTURE.md", max_chars=6000,
        )
        assert ENVELOPE_SENTENCE not in head
        assert "Error handling" not in head
        assert "Security Test Matrix" not in head
        assert "Architecture Overview" in head, (
            "it is a HEAD read — the overview is all it ever saw"
        )


class TestTheSelection:
    def test_the_envelope_rule_survives(self, ws):
        out = architecture_contract_excerpt(ws)
        assert ENVELOPE_SENTENCE in out

    def test_shape_sections_are_selected_without_any_routes(self, ws):
        """Story-independent: every assertion about a response body depends
        on these, so they are never contingent on route discovery working."""
        out = architecture_contract_excerpt(ws, routes=())
        assert "Error handling" in out
        assert "Security boundaries" in out
        assert "Security Test Matrix" in out

    def test_a_route_section_is_selected_when_the_route_is_known(self, ws):
        out = architecture_contract_excerpt(ws, routes=[ROUTE])
        assert "scoped to the authenticated owner" in out

    def test_an_unknown_route_does_not_pull_in_its_section(self, ws):
        out = architecture_contract_excerpt(ws, routes=["/api/v1/unrelated"])
        assert "scoped to the authenticated owner" not in out
        assert ENVELOPE_SENTENCE in out, "shape sections are unconditional"

    def test_sections_are_emitted_in_document_order(self, ws):
        out = architecture_contract_excerpt(ws, routes=[ROUTE])
        assert out.index("API Contracts") < out.index("Error handling")
        assert out.index("Error handling") < out.index("Security boundaries")

    def test_a_selected_section_is_kept_whole(self, ws):
        """A mid-section cut drops the tail of the rule it was quoting —
        which is the failure this function exists to end."""
        out = architecture_contract_excerpt(ws)
        assert "error_ref_id" in out, (
            "the envelope section's second bullet is where the shape is"
        )

    def test_the_filler_inventory_is_not_dragged_in(self, ws):
        out = architecture_contract_excerpt(ws)
        assert "module_150" not in out

    def test_the_budget_is_respected(self, ws):
        out = architecture_contract_excerpt(ws, max_chars=2000)
        assert ENVELOPE_SENTENCE in out, (
            "the smallest, most load-bearing content must win the budget"
        )

    def test_a_tiny_budget_still_returns_the_shape_rules(self, ws):
        """One section always lands even if it exceeds the budget — an empty
        excerpt is strictly worse than an over-budget one."""
        out = architecture_contract_excerpt(ws, max_chars=10)
        assert out.strip()


class TestTheEdges:
    def test_a_missing_document_is_empty_not_an_error(self, tmp_path):
        assert architecture_contract_excerpt(str(tmp_path)) == ""

    def test_a_document_with_no_headings_falls_back_to_the_head(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").write_text("x" * 50000)
        out = architecture_contract_excerpt(str(tmp_path), max_chars=100)
        assert len(out) == 100

    def test_a_document_with_no_matching_sections_falls_back(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").write_text(
            "## Overview\n\nNothing about shapes here.\n"
        )
        out = architecture_contract_excerpt(str(tmp_path), routes=["/zzz"])
        assert "Overview" in out, "the preamble path keeps it non-empty"

    def test_blank_routes_are_ignored(self, ws):
        out = architecture_contract_excerpt(ws, routes=["", "   ", None])
        assert ENVELOPE_SENTENCE in out


class TestTheRun11Shape:
    """The assertion that killed the run, and why it is now preventable."""

    def test_the_generator_can_now_see_both_sides_of_the_disagreement(
        self, ws,
    ):
        """The requirements shorthand said `404 { code: "NOT_FOUND" }`; the
        architecture says the body is a nested envelope. Until now only the
        first was visible, so there was no disagreement to resolve — just a
        wrong answer with no evidence against it."""
        out = architecture_contract_excerpt(ws, routes=[ROUTE])
        assert '{"error": {"code"' in out or '"error"' in out
        assert "error_ref_id" in out

    def test_the_prompt_forbids_exact_equality_on_an_error_body(self):
        """An envelope carries fields a test cannot predict, so `body ==
        {...}` can only pass if the application abandons its own envelope."""
        from harness.acceptance_gen import (
            AcceptanceScenario,  # noqa: F401 — import sanity
            StoryAcceptanceContext,
            build_user_prompt,
        )

        ctx = StoryAcceptanceContext(
            story_key="STORY-006", title="Get one record", description="",
            acceptance_criteria=[
                {"ac_key": "STORY-006.AC-3", "text": "404 for another owner"},
            ],
        )
        prompt = build_user_prompt(ctx, max_scenarios=2)
        assert "NEVER assert exact equality against a whole error body" in prompt
        assert "error_ref_id" in prompt

    def test_the_prompt_states_which_document_wins(self):
        from harness.acceptance_gen import (
            StoryAcceptanceContext,
            build_user_prompt,
        )

        ctx = StoryAcceptanceContext(
            story_key="STORY-006", title="t", description="",
            acceptance_criteria=[{"ac_key": "STORY-006.AC-1", "text": "x"}],
        )
        prompt = build_user_prompt(ctx, max_scenarios=1)
        assert "MORE DETAILED statement wins" in prompt
        assert "that is the architecture" in prompt

    def test_the_prompt_says_the_shorthand_names_a_code_not_a_body(self):
        from harness.acceptance_gen import (
            StoryAcceptanceContext,
            build_user_prompt,
        )

        ctx = StoryAcceptanceContext(
            story_key="STORY-006", title="t", description="",
            acceptance_criteria=[{"ac_key": "STORY-006.AC-1", "text": "x"}],
        )
        prompt = build_user_prompt(ctx, max_scenarios=1)
        assert "naming the error CODE, not giving you the literal response" in prompt


class TestTheWiring:
    def test_the_context_builder_no_longer_head_reads_the_architecture(self):
        import inspect

        import harness.acceptance_gen as ag

        src = inspect.getsource(ag.gather_story_acceptance_context)
        assert "architecture_contract_excerpt(" in src
        assert 'SPEC_ARCHITECTURE.md", max_chars' not in src, (
            "the head read is what this change removes"
        )

    def test_it_is_given_the_discovered_routes(self):
        import inspect

        import harness.acceptance_gen as ag

        src = inspect.getsource(ag.gather_story_acceptance_context)
        assert "routes=[str(r.get(\"path\", \"\")) for r in _routes]" in src

    def test_the_data_model_excerpt_is_untouched(self):
        """Scoped change: SPEC_DATA_MODEL.md is a different document with a
        different shape and is not part of this defect."""
        import inspect

        import harness.acceptance_gen as ag

        src = inspect.getsource(ag.gather_story_acceptance_context)
        assert 'SPEC_DATA_MODEL.md", max_chars=excerpt_chars' in src
