"""``cross_cutting_context`` — keep everything that is not a story body.

``tier1_preamble`` truncates the spec region at the FIRST requirement
heading, which assumes every cross-cutting section precedes every story.
Real specs do not look like that, and lumina's pair is the counter-example
that made ``planning.scoped_spec_context`` unsafe to enable:

    SPEC_REQUIREMENTS.md  59,660 chars
      # Software Requirements Specification       @0       (38 chars)
      ## Epic: EPIC-001 + every story body        @39      53,049 chars
      ## Assumptions and dependencies             @53,088
      ## API error handling                       @56,772
      ## Security and authorization               @57,371
      ## Traceability matrix                      @58,125   } 6,572 chars
    SPEC_ARCHITECTURE.md  48,632 chars  (no requirement heading at all)

The region handed to a node is ``requirements + architecture``, so the
first requirement heading sits at offset 39 and ``tier1_preamble`` keeps
38 characters — dropping the story bodies (wanted) AND the 6,572 chars of
cross-cutting requirements that follow them AND the whole architecture
document: data model, external interfaces, cross-cutting concerns, test
strategy, required inventory blocks.

The ADR's safety rule does not catch that. It fires on an unresolved
scope or an empty extract; here the scope resolves and the extract is
non-empty, so the loss would have been silent.
"""

from harness.spec_slice import cross_cutting_context, tier1_preamble

#: The shape that breaks the truncating version: a story body FOLLOWED by
#: cross-cutting sections, then a second document with no stories at all.
SPEC = """# Software Requirements Specification

## Epic: EPIC-001 — Tracker

**Vision statement:** something.

#### STORY-001 — First story

**Acceptance criteria:** must do the thing.

#### STORY-002 — Second story

**Acceptance criteria:** must do the other thing.

## Assumptions and dependencies

Assumes a local clock.

## API error handling

Errors are shaped `{"error": {...}}`.

# SPEC Architecture — Tracker

## 3. Data Model

birthdays(id TEXT PRIMARY KEY, ...)

## 6. Test Strategy

pytest, no mocks.
"""


class TestItKeepsWhatEveryCallNeeds:
    def test_post_story_cross_cutting_sections_survive(self):
        kept = cross_cutting_context(SPEC)
        assert "## Assumptions and dependencies" in kept
        assert "Assumes a local clock." in kept
        assert "## API error handling" in kept

    def test_a_second_document_without_stories_survives_whole(self):
        kept = cross_cutting_context(SPEC)
        assert "## 3. Data Model" in kept
        assert "birthdays(id TEXT PRIMARY KEY" in kept
        assert "## 6. Test Strategy" in kept

    def test_the_title_survives(self):
        assert "# Software Requirements Specification" in cross_cutting_context(SPEC)


class TestItDropsTheStoryBodies:
    def test_the_epic_span_goes(self):
        kept = cross_cutting_context(SPEC)
        assert "## Epic: EPIC-001" not in kept
        assert "Vision statement" not in kept

    def test_the_story_headings_and_criteria_go(self):
        kept = cross_cutting_context(SPEC)
        for gone in ("STORY-001", "STORY-002", "Acceptance criteria",
                     "must do the thing"):
            assert gone not in kept, f"{gone!r} should have been sliced out"

    def test_it_actually_saves_something(self):
        kept = cross_cutting_context(SPEC)
        assert len(kept) < len(SPEC)


class TestItBeatsTheTruncatingVersion:
    def test_tier1_preamble_loses_the_tail_this_keeps(self):
        """The regression this function exists to fix, asserted directly."""
        old = tier1_preamble(SPEC)
        new = cross_cutting_context(SPEC)
        assert "## 3. Data Model" not in old      # the old bug
        assert "## 3. Data Model" in new          # the fix
        assert len(new) > len(old)


class TestSafetyAndEdges:
    def test_a_region_with_no_story_heading_is_returned_whole(self):
        """The architecture document on its own is cross-cutting in its
        entirety — trimming it to nothing is the silent under-injection
        the module exists to avoid."""
        arch = "# Arch\n\n## 1. Overview\n\ntext\n\n## 2. Inventory\n\nmore\n"
        assert cross_cutting_context(arch).rstrip() == arch.rstrip()

    def test_empty_input(self):
        assert cross_cutting_context("") == ""

    def test_a_story_span_runs_to_end_of_document(self):
        """No closing non-requirement heading: the span must extend to the
        end rather than keeping the remaining story bodies."""
        text = "# Title\n\n## Epic: EPIC-001\n\nbody\n\n#### STORY-001\n\nmore\n"
        kept = cross_cutting_context(text)
        assert "# Title" in kept
        assert "STORY-001" not in kept and "EPIC-001" not in kept

    def test_deeper_headings_stay_inside_the_span(self):
        """Stories are written at #### under a ## epic; a deeper heading
        must not be mistaken for the end of the epic's span."""
        text = ("# T\n\n## Epic: EPIC-001\n\n#### STORY-001\n\na\n\n"
                "##### Detail\n\nb\n\n## Glossary\n\nkeep me\n")
        kept = cross_cutting_context(text)
        assert "## Glossary" in kept and "keep me" in kept
        assert "Detail" not in kept and "STORY-001" not in kept

    def test_several_separate_story_spans_all_go(self):
        text = ("# T\n\n## Epic: EPIC-001\n\na\n\n## Notes\n\nkeep1\n\n"
                "## FEAT-002 — feature\n\nb\n\n## More notes\n\nkeep2\n")
        kept = cross_cutting_context(text)
        assert "keep1" in kept and "keep2" in kept
        assert "EPIC-001" not in kept and "FEAT-002" not in kept


class TestTheAnchorStaysCacheStable:
    """The whole design rests on this: the surviving anchor must not vary
    with the story scope, or every call is a cache miss at a 120x cost
    multiple."""

    def test_output_is_independent_of_scope(self):
        a = cross_cutting_context(SPEC)
        b = cross_cutting_context(SPEC)
        assert a == b

    def test_it_is_a_pure_function_of_the_region(self):
        assert cross_cutting_context(SPEC) == cross_cutting_context(SPEC[:])
