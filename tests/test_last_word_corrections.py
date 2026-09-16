"""A correction the model never reads last is a correction it does not apply.

lumina-run7-20260915-1345. The REWRITE_FILE fixation trap fired correctly,
named the right file, carried the right repeat count, and gave the right
diagnosis -- "the bug is NOT in this file's current state... the failing
tests are actually caught by a DIFFERENT file" -- which was exactly true:
the defect was in ``test_body_size_limit.py``, which posts raw bytes to an
endpoint that parses JSON.

It landed 92.8% of the way through a 239,342-character prompt, followed by
~9,800 chars of diagnostics, a 6,326-char bare list of 83 workspace
filenames, and a format reminder closing with "Generate your fix patches
NOW." Immediately before all of that sat the model's own previous answer.

It re-emitted the same byte-identical content five rounds running (repair
calls 0072/0075/0078/0081/0084, every response md5 8cea442d8d68). The
correction was present on every one of those calls. It simply never had the
last word.
"""

from __future__ import annotations

from harness.graph import _format_last_word_corrections

FILE = "server/app/middleware/body_size_limit.py"


class TestSilentWhenNothingIsBlocking:
    """This must not become unconditional prompt bloat."""

    def test_empty_loop_counter(self) -> None:
        assert _format_last_word_corrections({}) == ""

    def test_non_dict_is_tolerated(self) -> None:
        assert _format_last_word_corrections(None) == ""  # type: ignore[arg-type]

    def test_single_noop_is_not_yet_blocking(self) -> None:
        """One no-op is ambiguous — the trap itself only fires at two."""
        out = _format_last_word_corrections(
            {"rewrite_file_no_ops_per_file": {FILE: 1}}
        )
        assert out == ""

    def test_malformed_counts_do_not_raise(self) -> None:
        out = _format_last_word_corrections(
            {"rewrite_file_no_ops_per_file": {FILE: "four", 7: None}}
        )
        assert out == ""


class TestFiresOnRealFixation:
    def test_repeat_noop_names_the_file(self) -> None:
        out = _format_last_word_corrections(
            {"rewrite_file_no_ops_per_file": {FILE: 4}}
        )
        assert FILE in out
        assert "byte-identical" in out

    def test_offers_all_three_sanctioned_exits(self) -> None:
        """A prohibition with no way out is what produced the loop."""
        out = _format_last_word_corrections(
            {"rewrite_file_no_ops_per_file": {FILE: 4}}
        )
        assert "DIFFERENT file" in out
        assert "READ_FILE" in out
        assert "UNSATISFIABLE_TEST" in out

    def test_layer5_assertion_alone_is_enough(self) -> None:
        """Layer 5 can flag a file before the per-file counter crosses 2."""
        out = _format_last_word_corrections({"asserted_correct_files": [FILE]})
        assert FILE in out

    def test_both_sources_merge_without_duplicating(self) -> None:
        out = _format_last_word_corrections({
            "rewrite_file_no_ops_per_file": {FILE: 3},
            "asserted_correct_files": [FILE],
        })
        assert out.count(f"`{FILE}`") == 1

    def test_multiple_files_are_all_named(self) -> None:
        other = "server/app/services/birthday_service.py"
        out = _format_last_word_corrections(
            {"rewrite_file_no_ops_per_file": {FILE: 4, other: 2}}
        )
        assert FILE in out and other in out
