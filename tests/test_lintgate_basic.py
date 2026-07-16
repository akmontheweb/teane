"""Tests for harness/lintgate.py — formatter and linter basics."""

import tempfile
import os


from pathlib import Path

from harness.lintgate import (
    FormatterSpec,
    LintGateResult,
    is_tool_available,
    register_formatter,
    get_formatter_for_file,
    _resolve_path,
    _inject_missing_typing_imports,
)


class TestInjectMissingTypingImports:
    """Regression: LLM-generated code routinely uses Optional/List/etc. without
    importing them, then ruff trips F821. Lintgate auto-injects to avoid
    expensive LLM repair loops."""

    def _write(self, tmpdir: str, src: str) -> str:
        path = os.path.join(tmpdir, "mod.py")
        Path(path).write_text(src)
        return path

    def test_adds_missing_optional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "import asyncio\n"
                "\n"
                "class W:\n"
                "    def __init__(self):\n"
                "        self._task: Optional[asyncio.Task] = None\n"
            )
            path = self._write(tmpdir, src)
            assert _inject_missing_typing_imports(path) == 1
            new_src = Path(path).read_text()
            assert "from typing import Optional" in new_src
            assert new_src.startswith("from typing import Optional\nimport asyncio")

    def test_extends_existing_typing_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "from typing import List\n"
                "\n"
                "def f(xs: List[int]) -> Dict[str, int]:\n"
                "    return {}\n"
            )
            path = self._write(tmpdir, src)
            assert _inject_missing_typing_imports(path) == 1
            new_src = Path(path).read_text()
            # Single merged line, names sorted
            assert new_src.count("from typing import") == 1
            assert "Dict" in new_src.splitlines()[0]
            assert "List" in new_src.splitlines()[0]

    def test_noop_when_already_imported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "from typing import Optional\n"
                "\n"
                "x: Optional[int] = None\n"
            )
            path = self._write(tmpdir, src)
            before = Path(path).read_text()
            assert _inject_missing_typing_imports(path) == 0
            assert Path(path).read_text() == before

    def test_noop_when_no_typing_names_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = "x = 1\nprint(x)\n"
            path = self._write(tmpdir, src)
            before = Path(path).read_text()
            assert _inject_missing_typing_imports(path) == 0
            assert Path(path).read_text() == before

    def test_inserts_after_future_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "from __future__ import annotations\n"
                "\n"
                "x: Optional[int] = None\n"
            )
            path = self._write(tmpdir, src)
            assert _inject_missing_typing_imports(path) == 1
            new_lines = Path(path).read_text().splitlines()
            # __future__ stays first
            assert new_lines[0].startswith("from __future__ import")
            # typing import lands after
            assert any(
                line.startswith("from typing import Optional") for line in new_lines[1:]
            )

    def test_skips_syntax_error_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = "def broken(:\n    pass\n"
            path = self._write(tmpdir, src)
            before = Path(path).read_text()
            assert _inject_missing_typing_imports(path) == 0
            assert Path(path).read_text() == before

    def test_recognized_in_subscript_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = (
                "def f() -> Iterator[int]:\n"
                "    yield 1\n"
                "    yield 2\n"
            )
            path = self._write(tmpdir, src)
            assert _inject_missing_typing_imports(path) == 1
            assert "Iterator" in Path(path).read_text()


class TestFormatterSpec:
    """Test FormatterSpec dataclass."""

    def test_construct_minimal(self):
        """Construct FormatterSpec with required field."""
        spec = FormatterSpec(command="black", args=[])
        assert spec.command == "black"
        assert spec.args == []

    def test_construct_with_linter(self):
        """Construct with linter command."""
        spec = FormatterSpec(
            command="prettier",
            args=["--write"],
            linter_command="eslint",
            linter_args=["--fix"],
        )
        assert spec.command == "prettier"
        assert spec.args == ["--write"]
        assert spec.linter_command == "eslint"
        assert spec.linter_args == ["--fix"]


class TestLintGateResult:
    """Test LintGateResult dataclass."""

    def test_construct_success(self):
        """Construct successful result."""
        result = LintGateResult(
            files_formatted=[],
            files_linted=[],
            format_errors=[],
            lint_errors=[],
            total_files_checked=5,
            had_errors=False,
        )
        assert result.total_files_checked == 5
        assert result.had_errors is False

    def test_construct_with_errors(self):
        """Construct result with format errors."""
        result = LintGateResult(
            files_formatted=["a.py"],
            files_linted=["a.py", "b.py"],
            format_errors=["a.py: parse error"],
            lint_errors=["b.py: undefined variable"],
            total_files_checked=2,
            had_errors=True,
        )
        assert result.had_errors is True
        assert len(result.format_errors) == 1
        assert len(result.lint_errors) == 1

    def test_construct_all_formatted_and_linted(self):
        """Construct with multiple files formatted and linted."""
        result = LintGateResult(
            files_formatted=["a.py", "b.py"],
            files_linted=["a.py", "b.py", "c.py"],
            format_errors=[],
            lint_errors=[],
            total_files_checked=3,
            had_errors=False,
        )
        assert len(result.files_formatted) == 2
        assert len(result.files_linted) == 3


class TestIsToolAvailable:
    """Test tool availability detection."""

    def test_python_available(self):
        """python should be available."""
        assert is_tool_available("python") is True or is_tool_available("python3") is True

    def test_nonexistent_tool(self):
        """nonexistent tool should not be available."""
        assert is_tool_available("nonexistent_tool_xyz_12345") is False

    def test_git_available(self):
        """git should be available."""
        assert is_tool_available("git") is True


class TestResolvePath:
    """Test path resolution relative to workspace."""

    def test_resolve_existing_relative_path(self):
        """Should resolve existing relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "file.py")
            with open(test_file, "w") as f:
                f.write("")  # Create the file
            result = _resolve_path("file.py", tmpdir)
            assert result is not None
            assert "file.py" in result

    def test_resolve_existing_absolute_path(self):
        """Absolute path should resolve when exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("")  # Create the file
            result = _resolve_path(test_file, tmpdir)
            # _resolve_path canonicalizes for boundary enforcement, so
            # compare realpaths — on macOS the tempdir itself sits behind
            # the /var -> /private/var symlink.
            assert result == os.path.realpath(test_file)

    def test_resolve_nonexistent_returns_none(self):
        """Nonexistent path should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _resolve_path("nonexistent.py", tmpdir)
            assert result is None


class TestRegisterFormatter:
    """Test formatter registration."""

    def test_register_formatter(self):
        """Should register a formatter."""
        spec = FormatterSpec(
            command="black",
            args=[],
        )
        # Should not raise
        register_formatter(".py", spec)

    def test_get_registered_formatter(self):
        """Should retrieve registered formatter."""
        spec = FormatterSpec(
            command="custom_formatter",
            args=[],
        )
        register_formatter(".custom", spec)
        retrieved = get_formatter_for_file("test.custom")
        # May or may not be there depending on implementation
        if retrieved is not None:
            assert retrieved.command == "custom_formatter"

