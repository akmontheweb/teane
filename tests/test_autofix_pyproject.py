"""Verify _try_missing_dep_pyproject writes into pyproject.toml's
``[project].dependencies`` list without disturbing surrounding TOML.
Regression baseline for the FinancialResearch sessions that spent 3-6
LLM rounds per missing dep (``pdfplumber``, ``pdf2image``, ``aiofiles``,
``beautifulsoup4`` etc.) because autofix bailed out when pyproject.toml
was present."""

import os
import tempfile

from harness.autofix import (
    _find_pyproject_dependencies_span,
    _try_missing_dep_pyproject,
)
from harness.patcher import OperationType


_BASE_PYPROJECT = """\
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myapp"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "fastapi>=0.110.0",
    "pydantic>=2.5.0",
]

[tool.setuptools.packages.find]
where = ["backend"]
"""


def _write(tmp_path, name, content):
    full = os.path.join(tmp_path, name)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


class TestFindPyprojectDependenciesSpan:
    def test_finds_standard_deps_list(self):
        span = _find_pyproject_dependencies_span(_BASE_PYPROJECT)
        assert span is not None
        start, end, inner = span
        assert _BASE_PYPROJECT[start] == "["
        assert _BASE_PYPROJECT[end] == "]"
        assert "fastapi" in inner and "pydantic" in inner

    def test_returns_none_when_no_deps_list(self):
        assert _find_pyproject_dependencies_span(
            "[build-system]\nrequires = []\n"
        ) is None

    def test_handles_empty_deps_list(self):
        text = "[project]\nname = 'x'\ndependencies = [\n]\n"
        span = _find_pyproject_dependencies_span(text)
        assert span is not None
        _, _, inner = span
        assert inner.strip() == ""

    def test_ignores_brackets_inside_strings(self):
        # A dep like ``"pkg[extra]>=1.0"`` contains an inner ``[`` that
        # would confuse a naive bracket walker. Verify we skip it.
        text = (
            "[project]\ndependencies = [\n"
            '    "pkg[extra]>=1.0",\n'
            '    "other",\n'
            "]\n"
        )
        span = _find_pyproject_dependencies_span(text)
        assert span is not None
        _, _, inner = span
        assert "pkg[extra]" in inner
        assert "other" in inner


class TestTryMissingDepPyproject:
    def test_appends_new_dep_before_closing_bracket(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", _BASE_PYPROJECT)
            block = _try_missing_dep_pyproject("aiofiles", td)
            assert block is not None
            assert block.operation is OperationType.REPLACE_BLOCK
            assert block.file == "pyproject.toml"
            # The replace contains the new entry followed by the
            # original closing-bracket line.
            assert '"aiofiles"' in block.replace
            assert block.replace.endswith("]")

    def test_idempotent_when_dep_already_present(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", _BASE_PYPROJECT)
            assert _try_missing_dep_pyproject("fastapi", td) is None

    def test_idempotent_across_version_pin_shapes(self):
        # Dep already present with a version pin — still counts.
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", _BASE_PYPROJECT)
            assert _try_missing_dep_pyproject("pydantic", td) is None

    def test_idempotent_across_extras(self):
        text = (
            "[project]\ndependencies = [\n"
            '    "pkg[extra]>=1.0",\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", text)
            assert _try_missing_dep_pyproject("pkg", td) is None

    def test_case_insensitive_dedup(self):
        text = (
            "[project]\ndependencies = [\n"
            '    "PyYAML",\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", text)
            assert _try_missing_dep_pyproject("pyyaml", td) is None

    def test_preserves_operator_indent_style(self):
        text = (
            "[project]\ndependencies = [\n"
            '  "existing",\n'   # 2-space indent
            "]\n"
        )
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", text)
            block = _try_missing_dep_pyproject("newpkg", td)
            assert block is not None
            # The replace line should retain the 2-space indent style
            # the operator was using.
            assert '  "newpkg",' in block.replace

    def test_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            assert _try_missing_dep_pyproject("aiofiles", td) is None

    def test_returns_none_when_no_deps_list(self):
        with tempfile.TemporaryDirectory() as td:
            _write(
                td,
                "pyproject.toml",
                "[build-system]\nrequires = ['setuptools']\n",
            )
            assert _try_missing_dep_pyproject("aiofiles", td) is None

    def test_search_block_is_uniquely_matchable(self):
        # The REPLACE_BLOCK search block must actually match the on-disk
        # content — otherwise the patch is DOA at the patcher.
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", _BASE_PYPROJECT)
            block = _try_missing_dep_pyproject("aiofiles", td)
            assert block is not None
            assert block.search in _BASE_PYPROJECT

    def test_handles_empty_deps_list(self):
        text = "[project]\nname = 'x'\ndependencies = [\n]\n"
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pyproject.toml", text)
            block = _try_missing_dep_pyproject("aiofiles", td)
            assert block is not None
            assert '"aiofiles"' in block.replace
