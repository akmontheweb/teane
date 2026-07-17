"""Repair-loop test-tamper guard (reward-hacking defense).

The patching node has always stripped test-targeting patch blocks
(``_filter_test_patch_blocks`` / ``_filter_test_blocks_from_patch_response``)
so phase-1 only writes production code. The *repair* node — which runs on a
RED build, the exact moment an LLM is tempted to delete or weaken a failing
test to turn it green — historically had no such filter: it parsed blocks and
handed them straight to ``apply_patch_blocks``.

``harness.graph._reject_test_patch_blocks`` closes that hole. It splits parsed
blocks into (production, test-rejections); test edits become
``PatchResult(success=False)`` objects carrying the ``[test-protected]``
sentinel, which ``harness.patcher._classify_patch_failure`` maps to the
``"test file protected"`` tag and ``harness.patch_feedback`` renders as a
"fix the production code, not the test" directive.

These tests lock in: the split, the pass-through (non-tamper runs must be
byte-identical to before), the classifier tag, the directive, and the
end-to-end status message the LLM sees.
"""

from __future__ import annotations

from harness.graph import _reject_test_patch_blocks
from harness.patch_feedback import _DIRECTIVE_BY_TAG, compose_patch_feedback
from harness.patcher import OperationType, PatchResult, _classify_patch_failure


class _Block:
    """Minimal PatchBlock stand-in: the guard only reads .file/.operation."""

    def __init__(self, file: str, operation: OperationType) -> None:
        self.file = file
        self.operation = operation


# -----------------------------------------------------------------------
# The split: production kept, test edits refused
# -----------------------------------------------------------------------

class TestRejectTestPatchBlocks:
    def test_production_blocks_survive_test_blocks_refused(self) -> None:
        blocks = [
            _Block("src/app/service.py", OperationType.REPLACE_BLOCK),
            _Block("tests/test_service.py", OperationType.DELETE_BLOCK),
            _Block("tests/test_api.py", OperationType.REPLACE_BLOCK),
            _Block("conftest.py", OperationType.REPLACE_BLOCK),
        ]
        kept, rejections = _reject_test_patch_blocks(blocks)
        assert [b.file for b in kept] == ["src/app/service.py"]
        assert {r.file for r in rejections} == {
            "tests/test_service.py",
            "tests/test_api.py",
            "conftest.py",
        }
        assert all(isinstance(r, PatchResult) and not r.success
                   for r in rejections)
        assert all("[test-protected]" in (r.error or "") for r in rejections)

    def test_delete_block_on_test_is_refused(self) -> None:
        # The most dangerous reward-hack: deleting a failing test outright.
        kept, rejections = _reject_test_patch_blocks(
            [_Block("tests/test_core.py", OperationType.DELETE_BLOCK)]
        )
        assert kept == []
        assert len(rejections) == 1
        assert rejections[0].operation == OperationType.DELETE_BLOCK

    def test_pass_through_when_no_test_blocks(self) -> None:
        # Safety: non-tamper runs must be a pure pass-through so the repair
        # path stays byte-identical to pre-guard behavior.
        prod = [
            _Block("src/a.py", OperationType.REPLACE_BLOCK),
            _Block("lib/b.py", OperationType.CREATE_FILE),
        ]
        kept, rejections = _reject_test_patch_blocks(prod)
        assert kept == prod
        assert rejections == []

    def test_empty_input(self) -> None:
        assert _reject_test_patch_blocks([]) == ([], [])


# -----------------------------------------------------------------------
# Parse-error carve-out (lumina 019f7054)
# -----------------------------------------------------------------------

class TestParseErrorCarveOut:
    """A test file whose CURRENT diagnostic is a parse error may be
    repaired: a file that doesn't parse can't run any assertion, so
    there's nothing to weaken — and pytest collection failure on one
    test-infra file blocks the whole suite. Lumina 019f7054: the
    harness's own @tests autofix wrote a `//` comment into
    tests/__init__.py; the repair LLM emitted the correct one-line fix
    twice and this guard rejected it both times → zero-patch HITL."""

    def test_syntax_broken_test_file_is_editable(self) -> None:
        from harness.graph import _syntax_broken_test_files
        diags = [{
            "file": "tests/__init__.py", "line": 1, "severity": "error",
            "error_code": "SyntaxError",
            "message": "SyntaxError: invalid syntax",
        }]
        allow = _syntax_broken_test_files(diags, "/ws")
        kept, rejections = _reject_test_patch_blocks(
            [_Block("tests/__init__.py", OperationType.REPLACE_BLOCK)],
            allow_parse_broken=allow, workspace_path="/ws",
        )
        assert [b.file for b in kept] == ["tests/__init__.py"]
        assert rejections == []

    def test_absolute_diag_path_matches_relative_block(self) -> None:
        # The lumina diagnostic carried an ABSOLUTE path (anchored via
        # pytest's assertion-rewrite File line); patch blocks may use
        # either form. Both must normalize to the same key.
        from harness.graph import _syntax_broken_test_files
        diags = [{
            "file": "/ws/tests/__init__.py", "line": 1,
            "severity": "error", "error_code": "SyntaxError",
            "message": "SyntaxError: invalid syntax",
        }]
        allow = _syntax_broken_test_files(diags, "/ws")
        for block_path in ("tests/__init__.py", "/ws/tests/__init__.py"):
            kept, rejections = _reject_test_patch_blocks(
                [_Block(block_path, OperationType.REPLACE_BLOCK)],
                allow_parse_broken=allow, workspace_path="/ws",
            )
            assert kept and not rejections, block_path

    def test_carve_out_is_per_file_not_blanket(self) -> None:
        # Only the parse-broken file is editable — sibling test files
        # stay protected in the same round.
        from harness.graph import _syntax_broken_test_files
        diags = [{
            "file": "tests/__init__.py", "line": 1, "severity": "error",
            "error_code": "SyntaxError",
            "message": "SyntaxError: invalid syntax",
        }]
        allow = _syntax_broken_test_files(diags, "/ws")
        kept, rejections = _reject_test_patch_blocks(
            [
                _Block("tests/__init__.py", OperationType.REPLACE_BLOCK),
                _Block("tests/test_api.py", OperationType.REPLACE_BLOCK),
            ],
            allow_parse_broken=allow, workspace_path="/ws",
        )
        assert [b.file for b in kept] == ["tests/__init__.py"]
        assert {r.file for r in rejections} == {"tests/test_api.py"}

    def test_assertion_failures_do_not_open_the_carve_out(self) -> None:
        # A plain failing assertion in a test file is exactly the
        # reward-hacking case — no carve-out.
        from harness.graph import _syntax_broken_test_files
        diags = [{
            "file": "tests/test_api.py", "line": 40, "severity": "error",
            "error_code": "AssertionError",
            "message": "AssertionError: assert 404 == 200",
        }]
        assert _syntax_broken_test_files(diags, "/ws") == frozenset()

    def test_parse_error_in_production_file_opens_nothing(self) -> None:
        from harness.graph import _syntax_broken_test_files
        diags = [{
            "file": "server/app/main.py", "line": 3, "severity": "error",
            "error_code": "SyntaxError",
            "message": "SyntaxError: invalid syntax",
        }]
        assert _syntax_broken_test_files(diags, "/ws") == frozenset()


# -----------------------------------------------------------------------
# Classifier + directive wiring
# -----------------------------------------------------------------------

class TestTestProtectedDirective:
    def test_classifier_maps_sentinel_to_tag(self) -> None:
        _, rejections = _reject_test_patch_blocks(
            [_Block("tests/test_x.py", OperationType.REPLACE_BLOCK)]
        )
        assert _classify_patch_failure(rejections[0].error) == \
            "test file protected"

    def test_directive_exists_and_names_the_antipattern(self) -> None:
        directive = _DIRECTIVE_BY_TAG.get("test file protected", "")
        assert directive.strip()
        assert "reward-hacking" in directive
        # It must steer toward fixing production code.
        assert "production" in directive.lower()


# -----------------------------------------------------------------------
# End-to-end: the status message the LLM actually receives
# -----------------------------------------------------------------------

class TestFeedbackSurfacesGuard:
    def test_status_message_carries_directive_and_counts_as_failure(
        self,
    ) -> None:
        _, rejections = _reject_test_patch_blocks(
            [
                _Block("tests/test_service.py", OperationType.DELETE_BLOCK),
                _Block("src/app/service.py", OperationType.REPLACE_BLOCK),
            ]
        )
        # Only the test block is a rejection; simulate the repair merge where
        # rejections are prepended before counts are computed.
        status, failures, _allow = compose_patch_feedback(
            list(rejections),
            allowed_paths=["src/"],
            parse_miss_diag="",
            prefix="[System]: Repair attempt 2:",
            success_count=0,
            fail_count=len(rejections),
            no_op_count=0,
        )
        assert "Repair attempt 2" in status
        assert "test_service.py" in status
        assert "test file protected" in status
        assert "reward-hacking" in status
        # The refused edit is a genuine failure, not a silent no-op.
        assert any(f["file"] == "tests/test_service.py" for f in failures)
