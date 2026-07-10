"""Tests for the missing-``__init__.py`` autofix.

Session 44c5e194's fourth resume hit ``reflection_distraction_loop:3``
with ``ModuleNotFoundError: No module named 'server.middleware'`` —
``server/middleware/rate_limit.py`` had been created by STORY-NFR-015
but no ``server/middleware/__init__.py``. Five repair turns bounced
around patching import statements instead of noticing the missing
marker. This autofix converts the failure class into a zero-token
deterministic fix.
"""

from __future__ import annotations

import pytest

from harness.autofix import _try_missing_init_py
from harness.patcher import OperationType


def _diag(message: str, error_code: str = "PROD_IMPORT_SMOKE:ModuleNotFoundError") -> dict:
    return {
        "file": "server/main.py",
        "line": 0,
        "column": 0,
        "severity": "error",
        "error_code": error_code,
        "message": message,
        "semantic_context": "",
    }


class TestMissingInitPy:
    def test_creates_init_for_directory_missing_marker(self, tmp_path):
        # server/middleware/ exists on disk with a source file, but no
        # __init__.py. Autofix should propose creating the marker.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "server" / "middleware").mkdir()
        (tmp_path / "server" / "middleware" / "rate_limit.py").write_text(
            "def limit(): pass\n", encoding="utf-8",
        )
        block = _try_missing_init_py(
            _diag(
                "Production module `server.main` failed to import "
                "(ModuleNotFoundError): No module named 'server.middleware'"
            ),
            str(tmp_path),
        )
        assert block is not None
        assert block.operation == OperationType.CREATE_FILE
        assert block.file == "server/middleware/__init__.py"
        assert block.content == ""

    def test_noop_when_init_already_exists(self, tmp_path):
        # Already fixed — must not propose a redundant create that
        # would fail the patcher's file-exists guard.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "server" / "middleware").mkdir()
        (tmp_path / "server" / "middleware" / "__init__.py").write_text(
            "", encoding="utf-8",
        )
        block = _try_missing_init_py(
            _diag("No module named 'server.middleware'"),
            str(tmp_path),
        )
        # The directory has its marker — no fix here. Some deeper
        # reason for the ModuleNotFoundError is real; leave it to the
        # LLM.
        assert block is None

    def test_noop_when_directory_does_not_exist(self, tmp_path):
        # ``No module named 'server.middleware'`` when server/ exists
        # but middleware/ does not. That's not an __init__.py miss —
        # it's a real missing-code situation the LLM should handle.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("", encoding="utf-8")
        block = _try_missing_init_py(
            _diag("No module named 'server.middleware'"),
            str(tmp_path),
        )
        assert block is None

    def test_noop_for_bare_top_level_name(self, tmp_path):
        # A bare ``No module named 'foo'`` is either a missing third-
        # party dep (handled by _try_deps_not_installed) or a missing
        # top-level package. Neither is fixed by writing an
        # ``__init__.py`` at the workspace root.
        block = _try_missing_init_py(
            _diag("No module named 'foo'"),
            str(tmp_path),
        )
        assert block is None

    def test_only_fires_on_prod_import_smoke_module_not_found(self, tmp_path):
        # Wrong error_code (e.g. a syntax-level failure) must not
        # cannibalize this handler.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "middleware").mkdir()
        assert _try_missing_init_py(
            _diag(
                "No module named 'server.middleware'",
                error_code="TEST_FAILURE:missing_verifies_marker",
            ),
            str(tmp_path),
        ) is None
        assert _try_missing_init_py(
            _diag(
                "No module named 'server.middleware'",
                error_code="PROD_IMPORT_SMOKE:SyntaxError",
            ),
            str(tmp_path),
        ) is None

    def test_deepest_miss_is_the_one_created(self, tmp_path):
        # Two levels of nesting BOTH exist as directories, but the
        # deepest one is missing its __init__.py. Autofix creates ONLY
        # the deepest. If a shallower level were also missing an
        # __init__.py, the next smoke check re-fires this handler on
        # the next round — no LLM turn spent either way.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "server" / "middleware").mkdir()
        (tmp_path / "server" / "middleware" / "__init__.py").write_text(
            "", encoding="utf-8",
        )
        (tmp_path / "server" / "middleware" / "auth").mkdir()
        (tmp_path / "server" / "middleware" / "auth" / "jwt.py").write_text(
            "def verify(): pass\n", encoding="utf-8",
        )
        block = _try_missing_init_py(
            _diag(
                "Production module `server.main` failed to import "
                "(ModuleNotFoundError): No module named 'server.middleware.auth'"
            ),
            str(tmp_path),
        )
        assert block is not None
        assert block.file == "server/middleware/auth/__init__.py"

    def test_message_variants_parsed(self, tmp_path):
        # The classifier can format the message in a few shapes; each
        # must be recognised by the regex.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "middleware").mkdir()
        for msg in (
            "No module named 'server.middleware'",
            'No module named "server.middleware"',
            "Production module `server.main` failed to import (ModuleNotFoundError): No module named 'server.middleware'",
        ):
            block = _try_missing_init_py(_diag(msg), str(tmp_path))
            assert block is not None, f"failed to parse: {msg!r}"
            assert block.file == "server/middleware/__init__.py"


class TestApplyAutofixesRoutesToInitPy:
    """End-to-end: apply_autofixes must route the diagnostic through
    the new dispatcher and land the CREATE_FILE on disk."""

    @pytest.mark.asyncio
    async def test_end_to_end_creates_marker_and_removes_from_unhandled(
        self, tmp_path,
    ):
        from harness.autofix import apply_autofixes
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "server" / "middleware").mkdir()
        (tmp_path / "server" / "middleware" / "rate_limit.py").write_text(
            "def limit(): pass\n", encoding="utf-8",
        )
        diag = _diag(
            "Production module `server.main` failed to import "
            "(ModuleNotFoundError): No module named 'server.middleware'"
        )
        unhandled, applied = await apply_autofixes([diag], str(tmp_path))
        # Diagnostic was resolved
        assert unhandled == []
        assert len(applied) == 1
        assert applied[0].fix_kind == "init_py"
        assert applied[0].file == "server/middleware/__init__.py"
        # File exists on disk
        assert (tmp_path / "server" / "middleware" / "__init__.py").is_file()
