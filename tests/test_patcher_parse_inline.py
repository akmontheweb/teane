"""Regression tests for the patcher parse loop's inline-``content:``
tolerance and the parse-miss diagnostic.

Root cause covered
------------------
Finsearch session 156032347 batch 111 died from a silent parser drop:

* The LLM's block regex required ``content:\\s*\\n<body>`` — a newline
  immediately after ``content:``.
* Under output-token pressure (a 32k cap forced continuation cycles),
  the LLM inlined the body: ``content:<body>``, saving one newline per
  block. 600/610 blocks in one round had this shape.
* The regex matched none of them, ``parse_patch_blocks`` returned ``[]``,
  ``[patcher] No patch blocks to apply.`` fired with no diagnostic, and
  the LLM's next round got zero corrective signal. Five stories were
  carried as defects before the auto-resume cap terminated the run.

Two-layer fix, exercised here:

1. ``_BODY_SEP = r'[ \\t]*\\n?'`` in ``harness/patcher.py`` accepts both
   ``content:\\n<body>`` (canonical) and ``content:<body>`` (inline).
2. ``harness.patcher.summarize_parse_miss`` returns an LLM-facing hint
   when marker openers were present but zero blocks parsed — wired
   into ``patching_node`` and ``repair_node`` so the next round's
   system message names the shape mismatch instead of staying silent.
"""

from __future__ import annotations

from harness.patcher import (
    OperationType,
    parse_patch_blocks,
    summarize_parse_miss,
)


class TestInlineContentTolerance:
    """Both ``content:\\n<body>`` and ``content:<body>`` MUST parse."""

    def test_rewrite_file_canonical_newline_shape_still_parses(self):
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: prompts/summary_v1.txt\n"
            "content:\n"
            "You are a financial writer.\n"
            "<<<END_REWRITE_FILE>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.REWRITE_FILE
        assert blocks[0].file == "prompts/summary_v1.txt"
        assert "You are a financial writer" in blocks[0].content

    def test_rewrite_file_inline_content_parses(self):
        """The finsearch-signature shape — body on the same line as
        ``content:``, no newline between."""
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: prompts/summary_v1.txt\n"
            "content:You are a financial writer.\n"
            "<<<END_REWRITE_FILE>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].file == "prompts/summary_v1.txt"
        assert blocks[0].content.startswith("You are a financial writer")

    def test_create_file_inline_content_parses(self):
        text = (
            "<<<CREATE_FILE>>>\n"
            "file: config/health_score_benchmarks.json\n"
            'content:{"global": {"revenue_growth": {"values_pct": []}}}\n'
            "<<<END_CREATE_FILE>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.CREATE_FILE
        assert blocks[0].content.startswith('{"global"')

    def test_inline_content_with_triple_quote_parses(self):
        """LLM often opens a docstring on the same line as ``content:``
        then closes several lines later. This is the exact shape from
        finsearch STORY-014's rejected output."""
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: server/app/schemas.py\n"
            'content:"""Pydantic schemas package."""\n'
            "\n"
            "from pydantic import BaseModel\n"
            "<<<END_REWRITE_FILE>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert '"""Pydantic schemas package."""' in blocks[0].content
        assert "from pydantic import BaseModel" in blocks[0].content

    def test_inline_content_preserves_leading_indentation_on_next_lines(self):
        """The relaxation MUST NOT eat leading whitespace from the
        body itself. ``\\s*`` would greedily consume indentation of
        the first non-inline line, breaking python/js files.
        ``[ \\t]*\\n?`` bounds the tolerance to the label line only."""
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: app.py\n"
            "content:def outer():\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner\n"
            "<<<END_REWRITE_FILE>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        # The four-space indentation on ``    def inner():`` MUST be
        # preserved verbatim in the captured body.
        assert "    def inner():" in blocks[0].content
        assert "        return 1" in blocks[0].content

    def test_replace_block_inline_search_and_replace_parse(self):
        text = (
            "<<<REPLACE_BLOCK>>>\n"
            "file: foo.py\n"
            "search:old_value = 1\n"
            "replace:new_value = 2\n"
            "<<<END_REPLACE_BLOCK>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.REPLACE_BLOCK
        assert blocks[0].search == "old_value = 1"
        assert blocks[0].replace == "new_value = 2"

    def test_delete_block_inline_search_parses(self):
        text = (
            "<<<DELETE_BLOCK>>>\n"
            "file: foo.py\n"
            "search:# TODO: remove\n"
            "<<<END_DELETE_BLOCK>>>"
        )
        blocks = parse_patch_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.DELETE_BLOCK
        assert blocks[0].search == "# TODO: remove"

    def test_finsearch_signature_batch_of_mixed_blocks_all_parse(self):
        """Reconstruct the exact shape the finsearch LLM produced:
        many inline-body ``REWRITE_FILE`` blocks in a row, mixed with
        canonical-form ones. The pre-fix parser dropped all 600 inline
        blocks silently. This test seeds five of each and asserts all
        ten parse."""
        parts = []
        for i in range(5):
            parts.append(
                f"<<<REWRITE_FILE>>>\n"
                f"file: prompts/p{i}.txt\n"
                f"content:Prompt body {i}.\n"
                f"<<<END_REWRITE_FILE>>>\n"
            )
        for i in range(5):
            parts.append(
                f"<<<REWRITE_FILE>>>\n"
                f"file: prompts/q{i}.txt\n"
                f"content:\n"
                f"Prompt body {i}.\n"
                f"<<<END_REWRITE_FILE>>>\n"
            )
        blocks = parse_patch_blocks("\n".join(parts))
        assert len(blocks) == 10, (
            f"Expected all 10 blocks to parse (5 inline + 5 newline). "
            f"Pre-fix, the 5 inline blocks were dropped. Got: "
            f"{[b.file for b in blocks]}"
        )


class TestParseMissDiagnostic:
    """``summarize_parse_miss`` MUST stay silent when the LLM
    legitimately emitted no patches, and MUST name the shape mismatch
    when it did but nothing parsed. This is the surface that feeds the
    LLM's next-round system message."""

    def test_empty_when_no_markers_present(self):
        """LLM emitted prose only — nothing to diagnose. The caller
        should log the usual 'no patches' line."""
        assert summarize_parse_miss(
            "I think we should change the schema. Do you agree?"
        ) == ""

    def test_names_shape_when_inline_body_present(self):
        """The finsearch signature: opener + closer paired, zero
        blocks parsed by whatever caller invoked this. Diag must call
        out the inline-body cause and give a concrete sample."""
        # Note: this test invokes ``summarize_parse_miss`` in
        # isolation with a text that WOULD parse post-fix, because the
        # diagnostic runs at the patching/repair site based on the
        # LIVE parse result, not the diag's own re-parse. We simulate
        # the "inline body but caller saw 0 blocks" contract.
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: prompts/foo.txt\n"
            "content:You are a financial writer.\n"
            "<<<END_REWRITE_FILE>>>"
        )
        diag = summarize_parse_miss(text)
        assert diag != ""
        assert "REWRITE_FILE=1/1" in diag
        assert "same line" in diag.lower() or "line after" in diag.lower()

    def test_names_truncation_when_opener_has_no_closer(self):
        """LLM hit output-token cap mid-block. Tail opener has no
        matching END marker — diag calls that out so the LLM knows
        to shrink the batch, not to reformat."""
        text = (
            "<<<REWRITE_FILE>>>\n"
            "file: a.py\n"
            "content:\n"
            "print('a')\n"
            "<<<END_REWRITE_FILE>>>\n"
            "<<<REWRITE_FILE>>>\n"
            "file: b.py\n"
            "content:\n"
            "print('cut off here"  # no END marker
        )
        diag = summarize_parse_miss(text)
        assert diag != ""
        assert "truncat" in diag.lower() or "no matching closer" in diag.lower()

    def test_reports_marker_counts_for_multiple_kinds(self):
        text = (
            "<<<CREATE_FILE>>>\n"
            "file: a\ncontent:x\n<<<END_CREATE_FILE>>>\n"
            "<<<REPLACE_BLOCK>>>\n"
            "file: b\nsearch:x\nreplace:y\n<<<END_REPLACE_BLOCK>>>\n"
        )
        diag = summarize_parse_miss(text)
        assert diag != ""
        assert "CREATE_FILE=1/1" in diag
        assert "REPLACE_BLOCK=1/1" in diag
