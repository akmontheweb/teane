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


# Finsearch session 44c5e194 root cause E4: LLM repair corrupted
# requirements.txt to ``lxml==6.1.0.`` (trailing dot). Post-patch
# validator now catches invalid PEP 508 requirement lines and rolls
# back the patch instead of shipping the corruption to pip.

def test_validate_requirements_txt_ok():
    content = (
        "# core deps\n"
        "fastapi>=0.100.0\n"
        "sqlalchemy[asyncio]==2.0.30\n"
        "lxml==6.1.0\n"
        "pytest; python_version >= '3.10'\n"
        "-r requirements-base.txt\n"
    )
    assert _validate_syntax("requirements.txt", content) is None


def test_validate_requirements_txt_rejects_trailing_dot_version():
    # The exact corruption from the finsearch session — LLM patched
    # requirements.txt and left an extra dot after the version.
    content = "lxml==6.1.0.\n"
    err = _validate_syntax("requirements.txt", content)
    assert err is not None
    assert "Invalid requirement" in err
    assert "line 1" in err
    assert "lxml==6.1.0." in err


def test_validate_requirements_dev_and_test_also_checked():
    # The validator covers requirements-dev.txt and requirements-test.txt
    # under the same rule, since they use identical PEP 508 syntax.
    err = _validate_syntax("requirements-dev.txt", "pytest==\n")
    assert err is not None and "Invalid requirement" in err
    err = _validate_syntax("requirements-test.txt", "coverage==\n")
    assert err is not None and "Invalid requirement" in err


def test_validate_requirements_ignores_comments_and_flags():
    # Comments, blank lines, and pip flags (-r, -e, -c, --extra-index-url)
    # must not be parsed as requirements.
    content = (
        "# leading comment\n"
        "\n"
        "-r requirements-base.txt\n"
        "-e .\n"
        "--extra-index-url https://example.com/simple\n"
        "-c constraints.txt\n"
        "requests>=2.0\n"
    )
    assert _validate_syntax("requirements.txt", content) is None


def test_validate_requirements_txt_only_by_basename():
    # A file that happens to end in .txt but isn't a pip manifest is
    # not validated (the check keys on basename == requirements*.txt).
    content = "not a requirement at all\n"
    assert _validate_syntax("notes.txt", content) is None


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
    async def test_replace_block_on_json_rejects_multiline_edits(self):
        # Class fix (finsearch 156032347): multi-line REPLACE_BLOCK on
        # structural files (JSON/YAML/TOML) is fragile — one misplaced
        # brace or trailing comma breaks the whole file. Finsearch
        # shipped 2 broken JSON patches in one run (tsconfig.test.json,
        # health_score_benchmarks.json), each rolled back correctly by
        # the post-patch validator but burning a repair round. Reject at
        # the patcher entry point with REWRITE_FILE guidance so the LLM
        # skips the doomed round entirely.
        with tempfile.TemporaryDirectory() as td:
            initial = (
                "{\n"
                '  "a": 1,\n'
                '  "b": 2,\n'
                '  "c": 3,\n'
                '  "d": 4\n'
                "}\n"
            )
            with open(os.path.join(td, "cfg.json"), "w") as f:
                f.write(initial)
            patcher = HybridPatcher(td)
            # Search + replace both span 5 lines — well above the 4-line
            # threshold for structural files.
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="cfg.json",
                search='  "a": 1,\n  "b": 2,\n  "c": 3,\n  "d": 4\n',
                replace='  "a": 10,\n  "b": 20,\n  "c": 30,\n  "d": 40\n',
            )])
            assert results[0].success is False
            err = (results[0].error or "").lower()
            assert "structural" in err or "rewrite_file" in err
            # Original file untouched — reject happens before the write.
            with open(os.path.join(td, "cfg.json")) as f:
                assert f.read() == initial

    @pytest.mark.asyncio
    async def test_replace_block_on_json_allows_small_edits(self):
        # Single-line REPLACE_BLOCK on JSON stays on the fast path.
        # Only multi-line edits (>= 4 lines) are gated.
        with tempfile.TemporaryDirectory() as td:
            initial = '{\n  "a": 1,\n  "b": 2\n}\n'
            with open(os.path.join(td, "cfg.json"), "w") as f:
                f.write(initial)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="cfg.json",
                search='  "a": 1,',
                replace='  "a": 42,',
            )])
            assert results[0].success is True
            with open(os.path.join(td, "cfg.json")) as f:
                assert '"a": 42' in f.read()

    @pytest.mark.asyncio
    async def test_replace_block_on_python_ignores_structural_gate(self):
        # The structural-file guard MUST NOT catch .py files — Python
        # REPLACE_BLOCK on a multi-line block is a normal codegen shape.
        with tempfile.TemporaryDirectory() as td:
            initial = (
                "def foo():\n"
                "    x = 1\n"
                "    y = 2\n"
                "    z = 3\n"
                "    return x + y + z\n"
            )
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(initial)
            patcher = HybridPatcher(td)
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.REPLACE_BLOCK,
                file="m.py",
                search=(
                    "    x = 1\n    y = 2\n    z = 3\n"
                    "    return x + y + z\n"
                ),
                replace=(
                    "    x = 10\n    y = 20\n    z = 30\n"
                    "    return x * y * z\n"
                ),
            )])
            assert results[0].success is True

    @pytest.mark.asyncio
    async def test_create_file_promotion_rolls_back_broken_syntax(self):
        # Class fix (finsearch 156032347): CREATE_FILE now auto-promotes
        # to REWRITE_FILE when the LLM's new content is highly similar
        # to what's on disk (typical repair-round re-emit shape). The
        # safety net that makes this move safe is post-patch parse
        # validation — if the promoted rewrite produces broken syntax,
        # ``_validate_and_maybe_rollback`` restores the pre-patch
        # content just like it would for any REWRITE_FILE. This test
        # locks that invariant so a future patcher change can't
        # silently drop the rollback and let a broken auto-promotion
        # land on disk.
        with tempfile.TemporaryDirectory() as td:
            clean = "def foo():\n    return 1\n"
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write(clean)
            patcher = HybridPatcher(td)
            # Second create is highly similar (would promote) but
            # syntactically broken (missing closing paren).
            broken_promotion = "def foo(:\n    return 2\n"
            results = await patcher.apply_all([PatchBlock(
                operation=OperationType.CREATE_FILE,
                file="m.py",
                content=broken_promotion,
            )])
            assert results[0].success is False
            # File must be restored to the pre-patch content.
            with open(os.path.join(td, "m.py")) as f:
                assert f.read() == clean

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
