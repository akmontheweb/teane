"""Verify HybridPatcher rolls back patches that turn a clean Python or JSON
file into a syntactically broken one, without regressing partial fixes on
files that were already broken pre-patch (see harness/patcher.py
``_validate_syntax`` docstring)."""

import os
import tempfile

import pytest

from harness.patcher import (
    HybridPatcher,
    OperationType,
    PatchBlock,
    _validate_syntax,
)


def test_validate_syntax_python_ok():
    assert _validate_syntax("m.py", "def f():\n    return 1\n") is None


def test_validate_syntax_python_broken():
    err = _validate_syntax("m.py", "def f(\n")
    assert err is not None and "SyntaxError" in err


def test_validate_syntax_json_ok():
    assert _validate_syntax("cfg.json", '{"a": 1}') is None


def test_validate_syntax_json_broken():
    err = _validate_syntax("cfg.json", "{invalid")
    assert err is not None and "JSONDecodeError" in err


def test_validate_syntax_skips_non_validated_extensions():
    # Markdown, YAML, TS etc. get no validation — return None regardless.
    assert _validate_syntax("README.md", "```python\ndef f(\n```") is None
    assert _validate_syntax("app.ts", "function() {") is None


class TestPostPatchValidation:
    @pytest.mark.asyncio
    async def test_create_file_valid_python_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            result = await patcher.apply_patch(PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="ok.py", content="def f():\n    return 1\n",
            ))
            assert result.success is True
            assert os.path.isfile(os.path.join(td, "ok.py"))

    @pytest.mark.asyncio
    async def test_create_file_broken_python_rolls_back(self):
        # A CREATE_FILE that lands but leaves the workspace with an
        # unparseable file used to be a common repair-loop poison
        # (session 674bfdbd). Verify the guard catches it via apply_all
        # (where the wrapping validation lives) and removes the file.
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="broken.py",
                content="def f(\n    return 1\n",
            )])
            assert len(results) == 1
            assert results[0].success is False
            assert "Post-patch validation" in (results[0].error or "")
            # File was rolled back (never existed → deleted).
            assert not os.path.exists(os.path.join(td, "broken.py"))

    @pytest.mark.asyncio
    async def test_replace_block_that_breaks_clean_file_rolls_back(self):
        with tempfile.TemporaryDirectory() as td:
            src = "def f():\n    return 1\n"
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(src)
            patcher = HybridPatcher(td)
            # Patch corrupts by unbalancing the parens.
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search="def f():\n    return 1\n",
                replace="def f(\n    return 1\n",
            )])
            assert results[0].success is False
            # Rollback wrote the pre-patch content back.
            with open(os.path.join(td, "m.py")) as f:
                assert f.read() == src

    @pytest.mark.asyncio
    async def test_replace_block_on_already_broken_file_lets_partial_fix_through(self):
        # If the file was ALREADY unparseable pre-patch, don't regress
        # partial fixes — the LLM may be iteratively repairing.
        with tempfile.TemporaryDirectory() as td:
            broken = "def f(\n    return 1\n"
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(broken)
            patcher = HybridPatcher(td)
            # Post-patch is STILL broken (unbalanced brace) but pre-patch
            # was also broken — must not roll back.
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search="def f(\n    return 1\n",
                replace="def f(\n    return 1  # partial fix\n",
            )])
            assert results[0].success is True
            with open(os.path.join(td, "m.py")) as f:
                assert "partial fix" in f.read()

    @pytest.mark.asyncio
    async def test_non_validated_extension_unaffected(self):
        # A markdown file with content that a Python parser would reject
        # must apply cleanly — validation is python/json only.
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="notes.md",
                content="```python\ndef f(\n```\n",
            )])
            assert results[0].success is True
            assert os.path.isfile(os.path.join(td, "notes.md"))

    @pytest.mark.asyncio
    async def test_rewrite_file_overwrites_existing(self):
        # REWRITE_FILE is the escape hatch: unlike CREATE_FILE it does
        # clobber an existing file. Sanity-check the happy path.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write("def old():\n    return 1\n")
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REWRITE_FILE,
                file="m.py",
                content="def new():\n    return 42\n",
            )])
            assert results[0].success is True
            with open(os.path.join(td, "m.py")) as f:
                text = f.read()
            assert "def new" in text and "def old" not in text

    @pytest.mark.asyncio
    async def test_rewrite_file_noop_signals_actionable_failure(self):
        # Session b9369w5uu (ciod) had the LLM emit REWRITE_FILE with
        # content byte-identical to disk twice in a row while the judge
        # kept flagging a missing symbol. Silently marking the no-op as
        # success hid the "you're stuck" signal from the LLM. It must
        # surface as an actionable failure with a hint about READ_FILE.
        with tempfile.TemporaryDirectory() as td:
            existing = "def f():\n    return 1\n"
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(existing)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REWRITE_FILE,
                file="m.py",
                content="def f():\n    return 1",  # trailing \n added by patcher
            )])
            assert len(results) == 1
            r = results[0]
            assert r.success is False, (
                "REWRITE_FILE no-op must be reported as failure so the "
                "LLM sees it in the next round's patch-failure surface"
            )
            assert r.no_op is True
            assert "no-op" in (r.error or "").lower()
            assert "read_file" in (r.error or "").lower()
            # File must be unchanged on disk.
            with open(os.path.join(td, "m.py")) as f:
                assert f.read() == existing

    @pytest.mark.asyncio
    async def test_rewrite_file_rolls_back_on_broken_syntax(self):
        # Post-patch validation applies to REWRITE_FILE the same as
        # every other op — a rewrite that produces unparseable Python
        # must roll the file back to its pre-patch content.
        with tempfile.TemporaryDirectory() as td:
            good = "def ok():\n    return 1\n"
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(good)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REWRITE_FILE,
                file="m.py",
                content="def broken(\n    return 1\n",
            )])
            assert results[0].success is False
            with open(os.path.join(td, "m.py")) as f:
                assert f.read() == good

    @pytest.mark.asyncio
    async def test_rewrite_file_creates_when_missing(self):
        # REWRITE_FILE on a path that doesn't exist yet should still
        # succeed — same as CREATE_FILE in that case. Keeps the LLM's
        # options open when the escape hatch is unlocked after a file
        # has already been deleted mid-round.
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REWRITE_FILE,
                file="new.py",
                content="def f():\n    return 1\n",
            )])
            assert results[0].success is True
            assert os.path.isfile(os.path.join(td, "new.py"))

    @pytest.mark.asyncio
    async def test_create_file_overwrites_empty_existing(self):
        # Regression for the DELETE_BLOCK + CREATE_FILE trap: after the
        # LLM has failed REPLACE_BLOCK twice on a file, the harness
        # directs it to use DELETE_BLOCK + CREATE_FILE instead. If the
        # first step lands (file becomes empty) and the CREATE_FILE step
        # then rejects on "file already exists with different content",
        # the file stays empty and the loop cycles. An empty file has
        # no author-preserving content, so CREATE_FILE should overwrite
        # it. Session b61f48a7 spent 3+ HITL cycles on this exact loop
        # for ``backend/api/search.py``.
        with tempfile.TemporaryDirectory() as td:
            # Case: file exists with a single newline (typical LLM
            # DELETE_BLOCK residue).
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write("\n")
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="m.py",
                content="def f():\n    return 1\n",
            )])
            assert results[0].success is True
            with open(os.path.join(td, "m.py")) as f:
                assert "def f" in f.read()

    @pytest.mark.asyncio
    async def test_create_file_still_rejects_non_empty_existing(self):
        # The empty-file overwrite must NOT weaken the guard for real
        # content — a file with author code still gets the classic
        # "file already exists with different content" rejection.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write("existing = 1\n")
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="m.py",
                content="new = 2\n",
            )])
            assert results[0].success is False
            assert "already exists" in (results[0].error or "").lower()
            with open(os.path.join(td, "m.py")) as f:
                assert "existing = 1" in f.read()

    @pytest.mark.asyncio
    async def test_apply_all_continues_past_validation_rollback(self):
        # A validation-rolled-back block must not stop subsequent blocks
        # from applying — same invariant as
        # ``test_apply_all_continues_past_middle_failure`` in test_harness.py.
        with tempfile.TemporaryDirectory() as td:
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([
                PatchBlock(operation=OperationType.CREATE_FILE,
                           file="a.py", content="x = 1\n"),
                PatchBlock(operation=OperationType.CREATE_FILE,
                           file="bad.py", content="def f(\n"),
                PatchBlock(operation=OperationType.CREATE_FILE,
                           file="c.py", content="z = 3\n"),
            ])
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is True
            assert os.path.isfile(os.path.join(td, "a.py"))
            assert not os.path.exists(os.path.join(td, "bad.py"))
            assert os.path.isfile(os.path.join(td, "c.py"))
