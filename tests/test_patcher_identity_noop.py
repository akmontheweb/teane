"""REPLACE_BLOCK identity patches must be rejected as no-ops.

lumina session 01a079dc: across repair rounds 5-8 the LLM emitted four
REPLACE_BLOCK patches whose ``replace`` was byte-identical to their
``search`` (``server/app/main.py`` twice, ``server/app/middleware/audit.py``
twice, plus ``server/app/utils/dates.py`` and ``server/requirements.txt``
earlier in the run). Every match tier found the region, wrote the same bytes
back, and returned ``success=True, no_op=False, lines_changed=0`` — so the
round rolled up as "[patcher] Applied 3/3 patches" while the failing-test
count sat at 3 and the loop's zero-progress detector never fired. The run
escalated to HITL on the low-signal verdict cap 3 rounds later.

The guard mirrors the pre-existing REWRITE_FILE no-op contract: report the
block as an actionable FAILURE (so it reaches the LLM's patch-failure
surface next round) with ``no_op=True`` (so progress accounting can tell an
identity patch apart from a search miss).
"""

import os
import tempfile

import pytest

from harness.patcher import (
    HybridPatcher,
    OperationType,
    PatchBlock,
    _is_identity_replacement,
)

SRC = "def f():\n    return 1\n"


class TestIsIdentityReplacement:
    def test_exact_equality(self):
        assert _is_identity_replacement("a = 1\n", "a = 1\n") is True

    def test_crlf_drift_is_identity(self):
        assert _is_identity_replacement("a = 1\r\nb = 2\r\n", "a = 1\nb = 2\n") is True

    def test_trailing_newline_only_drift_is_identity(self):
        assert _is_identity_replacement("a = 1\n", "a = 1") is True

    def test_real_edit_is_not_identity(self):
        assert _is_identity_replacement("a = 1\n", "a = 2\n") is False

    def test_trailing_whitespace_edit_is_not_identity(self):
        # A deliberate lint fix. Narrow by design: we must not reject a
        # patch the LLM needs to land.
        assert _is_identity_replacement("a = 1   \n", "a = 1\n") is False

    def test_indent_edit_is_not_identity(self):
        # Python/YAML/Makefile indent repairs are real edits.
        assert _is_identity_replacement("  a = 1\n", "    a = 1\n") is False

    def test_added_line_is_not_identity(self):
        assert _is_identity_replacement("a = 1\n", "a = 1\nb = 2\n") is False


class TestReplaceBlockIdentityRejected:
    @pytest.mark.asyncio
    async def test_identity_patch_reported_as_failed_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(SRC)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search="    return 1\n",
                replace="    return 1\n",
            )])
            assert results[0].success is False
            assert results[0].no_op is True
            assert results[0].lines_changed == 0

    @pytest.mark.asyncio
    async def test_identity_patch_error_is_actionable(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(SRC)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search="    return 1\n",
                replace="    return 1\n",
            )])
            err = results[0].error or ""
            # Names the file, says nothing changed, and tells the LLM what
            # to do instead — the same shape as the REWRITE_FILE no-op.
            assert "m.py" in err
            assert "changes nothing" in err
            assert "READ_FILE" in err

    @pytest.mark.asyncio
    async def test_identity_patch_leaves_file_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.py")
            with open(path, "w") as f:
                f.write(SRC)
            patcher = HybridPatcher(td)
            await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search=SRC,
                replace=SRC,
            )])
            with open(path) as f:
                assert f.read() == SRC

    @pytest.mark.asyncio
    async def test_real_edit_still_lands(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.py")
            with open(path, "w") as f:
                f.write(SRC)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search="    return 1\n",
                replace="    return 2\n",
            )])
            assert results[0].success is True
            assert results[0].no_op is False
            with open(path) as f:
                assert "return 2" in f.read()

    @pytest.mark.asyncio
    async def test_identity_block_does_not_sink_sibling_patches(self):
        """A no-op block must not roll back the good patches beside it —
        the lumina rounds mixed one identity block with one real one."""
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "a.py"), "w") as f:
                f.write(SRC)
            with open(os.path.join(td, "b.py"), "w") as f:
                f.write(SRC)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([
                PatchBlock(
                    operation=OperationType.REPLACE_BLOCK,
                    file="a.py",
                    search="    return 1\n",
                    replace="    return 1\n",
                ),
                PatchBlock(
                    operation=OperationType.REPLACE_BLOCK,
                    file="b.py",
                    search="    return 1\n",
                    replace="    return 42\n",
                ),
            ])
            by_file = {r.file: r for r in results}
            assert by_file["a.py"].success is False
            assert by_file["a.py"].no_op is True
            assert by_file["b.py"].success is True
            with open(os.path.join(td, "b.py")) as f:
                assert "return 42" in f.read()

    @pytest.mark.asyncio
    async def test_empty_search_on_missing_file_still_creates(self):
        """The empty-search CREATE_FILE escape hatch predates this guard and
        must keep working: search == replace == "" would otherwise read as
        an identity patch."""
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="new.py",
                search="",
                replace="x = 1\n",
            )])
            assert results[0].success is True
            assert os.path.isfile(os.path.join(td, "new.py"))
