"""Post-patcher LLM feedback composition — shared between patching_node
and repair_node.

Root case for this module (finsearch 156032347): patching_node's next-
round feedback was 38 chars ("[System]: Failed to apply 8 patch(es).") —
no file names, no reasons, no directive. The LLM had no way to know a
"file missing" failure meant "switch to CREATE_FILE", so it re-emitted
the same broken block every round until the zero-patch tripwire fired.
Repair_node had the same thin status_msg alongside its richer error-
summary path. Both are now routed through
:func:`harness.patch_feedback.compose_patch_feedback`; the tests below
lock in per-classification directives, prefix parameterisation, and the
allowlist / parse-miss carve-outs.
"""

from __future__ import annotations

from harness.patch_feedback import (
    _DEFAULT_DIRECTIVE,
    _DIRECTIVE_BY_TAG,
    _PATCH_ERROR_WIDER_CONTEXT_MARKER,
    _store_patch_failure_error,
    compose_patch_feedback,
)
from harness.patcher import OperationType, PatchResult


def _fail(file: str, op: OperationType, err: str) -> PatchResult:
    return PatchResult(success=False, file=file, operation=op, error=err)


def _ok(file: str, op: OperationType, *, no_op: bool = False) -> PatchResult:
    return PatchResult(
        success=True, file=file, operation=op, error="", no_op=no_op,
    )


# -----------------------------------------------------------------------
# Directive coverage — every classifier tag has a matching directive
# -----------------------------------------------------------------------

class TestDirectiveCoverage:
    """Every tag that :func:`harness.patcher._classify_patch_failure`
    produces must map to a non-empty directive; otherwise the LLM would
    fall back to the generic default and lose the specific corrective
    hint (which was the whole point of this refactor)."""

    def test_all_tags_present(self) -> None:
        expected = {
            "file missing",
            "search miss",
            "ambiguous match",
            "rejected: file already exists",
            "path denied",
            "allowlist denied",
            "no blocks parsed",
        }
        assert expected.issubset(_DIRECTIVE_BY_TAG.keys())

    def test_no_empty_directives(self) -> None:
        for tag, body in _DIRECTIVE_BY_TAG.items():
            assert body.strip(), f"directive for {tag!r} is empty"

    def test_default_directive_non_empty(self) -> None:
        assert _DEFAULT_DIRECTIVE.strip()


# -----------------------------------------------------------------------
# Per-classification feedback in status_msg
# -----------------------------------------------------------------------

class TestFileMissingFeedback:
    """The finsearch signature failure. LLM emitted INSERT_AT_BLOCK
    against a package's `__init__.py` that hadn't been CREATE_FILE'd
    yet. Feedback must name the file AND tell the LLM to CREATE_FILE."""

    def test_zero_success_all_missing(self) -> None:
        err = (
            "File not found: server/app/services/ai_analysis/__init__.py. "
            "Use CREATE_FILE for new files."
        )
        results = [
            _fail(
                "server/app/services/ai_analysis/__init__.py",
                OperationType.INSERT_AT_BLOCK,
                err,
            ),
        ]
        msg, failures, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        assert "CREATE_FILE" in msg
        assert "server/app/services/ai_analysis/__init__.py" in msg
        assert "file missing" in msg
        assert "insert_at_block" in msg
        assert len(failures) == 1

    def test_partial_success_still_includes_directive(self) -> None:
        err = (
            "File not found: foo/__init__.py. Use CREATE_FILE for new "
            "files."
        )
        results = [
            _ok("foo/bar.py", OperationType.CREATE_FILE),
            _fail("foo/__init__.py", OperationType.INSERT_AT_BLOCK, err),
        ]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=1, fail_count=1, no_op_count=0,
        )
        assert "Applied 1/2 patches successfully" in msg
        assert "Failed on: foo/__init__.py" in msg
        assert "CREATE_FILE" in msg
        assert "file missing" in msg


class TestSearchMissFeedback:
    def test_directive_names_read_file(self) -> None:
        err = "Search block not found in foo.py."
        results = [_fail("foo.py", OperationType.REPLACE_BLOCK, err)]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        assert "READ_FILE" in msg
        assert "search miss" in msg


class TestAmbiguousMatchFeedback:
    def test_directive_asks_for_more_context(self) -> None:
        err = "Search block matched 3 regions in foo.py — tolerant search."
        results = [_fail("foo.py", OperationType.REPLACE_BLOCK, err)]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        assert "surrounding context" in msg
        assert "ambiguous match" in msg


class TestFileAlreadyExistsFeedback:
    def test_directive_asks_for_replace_block(self) -> None:
        err = "CREATE_FILE rejected: foo.py already exists on disk."
        results = [_fail("foo.py", OperationType.CREATE_FILE, err)]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        assert "REPLACE_BLOCK" in msg
        assert "rejected: file already exists" in msg


# -----------------------------------------------------------------------
# Prefix parameterisation — patching_node vs repair_node
# -----------------------------------------------------------------------

class TestPrefixParameterisation:
    """The two callers pass different prefixes so the LLM can tell whether
    it's a fresh patching pass or a repair retry. Empty message body
    otherwise identical — the shared directives should render both ways.
    """

    def test_patching_prefix(self) -> None:
        results = [_ok("foo.py", OperationType.CREATE_FILE)]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=1, fail_count=0, no_op_count=0,
        )
        assert msg.startswith("[System]: Applied 1/1 patches successfully.")

    def test_repair_prefix(self) -> None:
        results = [_ok("foo.py", OperationType.CREATE_FILE)]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]: Repair attempt 3:",
            success_count=1, fail_count=0, no_op_count=0,
        )
        assert msg.startswith(
            "[System]: Repair attempt 3: Applied 1/1 patches successfully."
        )


# -----------------------------------------------------------------------
# Parse-miss (cc9ab6a path) still fires when fail_count == 0
# -----------------------------------------------------------------------

class TestParseMissDiagnostic:
    """cc9ab6a introduced the parse-miss diagnostic for the case where
    the LLM emitted marker openers but zero blocks parsed. That's a
    distinct failure mode from "patcher rejected N blocks" — the shared
    helper must still surface the diagnostic verbatim when it fires."""

    def test_parse_miss_only_fires_when_zero_fail_count(self) -> None:
        results: list[PatchResult] = []
        diag = "Body field starts on the same line as its label."
        msg, _, _ = compose_patch_feedback(
            results, [], diag, prefix="[System]:",
            success_count=0, fail_count=0, no_op_count=0,
        )
        assert "parser could not extract" in msg
        assert "Body field starts" in msg

    def test_parse_miss_ignored_when_real_failures(self) -> None:
        # If patches actually failed with real errors, the parse-miss
        # diagnostic is not relevant (LLM did emit real blocks; they
        # just rejected). Directives take precedence.
        err = (
            "File not found: foo.py. Use CREATE_FILE for new files."
        )
        results = [_fail("foo.py", OperationType.INSERT_AT_BLOCK, err)]
        msg, _, _ = compose_patch_feedback(
            results, [], "irrelevant parse diag", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        assert "parser could not extract" not in msg
        assert "CREATE_FILE" in msg


# -----------------------------------------------------------------------
# Allowlist carve-out — separate bucket, separate section
# -----------------------------------------------------------------------

class TestAllowlistRejections:
    def test_allowlist_rejections_get_own_section(self) -> None:
        err = "Path 'evil.py' not in skill allowlist."
        results = [_fail("evil.py", OperationType.CREATE_FILE, err)]
        msg, failures, rejects = compose_patch_feedback(
            results, ["src/", "tests/"], "", prefix="[System]:",
            success_count=0, fail_count=1, no_op_count=0,
        )
        # Allowlist failures do NOT go into patch_failures — they get
        # their own bucket so the LLM doesn't see them classified as
        # "generic" errors.
        assert failures == []
        assert len(rejects) == 1
        assert rejects[0]["file"] == "evil.py"
        # They still show up in status_msg — but in the Allowlist
        # section, not the per-file rejection details block.
        assert "Allowlist" in msg
        assert "evil.py" in msg
        assert "src/" in msg

    def test_mixed_allowlist_and_generic_failures(self) -> None:
        err_missing = (
            "File not found: foo.py. Use CREATE_FILE for new files."
        )
        err_allowlist = "Path 'evil.py' not in skill allowlist."
        results = [
            _fail("foo.py", OperationType.INSERT_AT_BLOCK, err_missing),
            _fail("evil.py", OperationType.CREATE_FILE, err_allowlist),
        ]
        msg, failures, rejects = compose_patch_feedback(
            results, ["src/"], "", prefix="[System]:",
            success_count=0, fail_count=2, no_op_count=0,
        )
        assert len(failures) == 1
        assert failures[0]["file"] == "foo.py"
        assert len(rejects) == 1
        assert rejects[0]["file"] == "evil.py"
        assert "CREATE_FILE" in msg  # directive for foo.py
        assert "Allowlist" in msg    # section for evil.py


# -----------------------------------------------------------------------
# Bounds & idempotency
# -----------------------------------------------------------------------

class TestBoundsAndStorage:
    def test_patch_failures_capped_at_five_by_default(self) -> None:
        err = "File not found: X. Use CREATE_FILE for new files."
        results = [
            _fail(
                f"f{i}.py", OperationType.INSERT_AT_BLOCK, err,
            )
            for i in range(10)
        ]
        _, failures, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=10, no_op_count=0,
        )
        assert len(failures) == 5

    def test_max_failures_stored_is_honored(self) -> None:
        err = "File not found: X. Use CREATE_FILE for new files."
        results = [
            _fail(f"f{i}.py", OperationType.INSERT_AT_BLOCK, err)
            for i in range(5)
        ]
        _, failures, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=0, fail_count=5, no_op_count=0,
            max_failures_stored=2,
        )
        assert len(failures) == 2

    def test_no_op_count_surfaced_in_message(self) -> None:
        # Idempotency no-ops must be visible to the LLM so it doesn't
        # count them as real progress on the next round — this is what
        # the "consecutive_zero_patch_rounds" tripwire depends on.
        results = [
            _ok("foo.py", OperationType.CREATE_FILE, no_op=True),
            _ok("bar.py", OperationType.CREATE_FILE, no_op=False),
        ]
        msg, _, _ = compose_patch_feedback(
            results, [], "", prefix="[System]:",
            success_count=2, fail_count=0, no_op_count=1,
        )
        assert "1 were idempotency no-ops" in msg


# -----------------------------------------------------------------------
# _store_patch_failure_error — moved here from graph.py in this refactor
# -----------------------------------------------------------------------

class TestStorePatchFailureError:
    def test_none_and_empty(self) -> None:
        assert _store_patch_failure_error(None) == ""
        assert _store_patch_failure_error("") == ""

    def test_regular_error_capped_at_3000(self) -> None:
        long_err = "boom: " + ("X" * 5000)
        stored = _store_patch_failure_error(long_err)
        assert len(stored) == 3000
        assert stored.startswith("boom:")

    def test_wider_context_survives_uncapped(self) -> None:
        body = "X" * 2500
        err = (
            "Search block not found in foo.py. "
            f"{_PATCH_ERROR_WIDER_CONTEXT_MARKER}\n{body}"
        )
        stored = _store_patch_failure_error(err)
        assert stored == err
