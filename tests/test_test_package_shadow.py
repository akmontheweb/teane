"""Regression tests for the 2026-07-09 pytest package-shadow fix.

User-reported symptom (session fr-agile-20260709): pytest hit
``ModuleNotFoundError: No module named 'backend.core'`` on every test
collection despite ``pytest.ini`` having ``pythonpath = .``, a root
``conftest.py`` that inserted the workspace root, and a per-test
``sys.path.insert(0, root)`` at the top of the failing test file.
The reflection judge kept diagnosing the shape as "PYTHONPATH unset
/ wrong CWD" and steered the repair LLM toward pytest.ini / conftest
edits — none of which touched the real bug. HITL cap trip after 10
rounds; $0.23 burned; no forward progress.

Root cause: ``tests/unit/backend/__init__.py`` present while
``tests/unit/__init__.py`` missing. Pytest's rootdir walk stops at
the first init-less ancestor (``tests/unit/``), treats ``backend`` as
a top-level package name, prepends ``tests/unit/`` to sys.path, and
subsequent ``import backend.core`` resolves to ``tests/unit/backend/``
(the shadow) instead of the workspace ``backend/`` (the real source).

Fix: ``_ensure_test_package_init_chain`` in ``compiler_node`` walks
``tests/`` before every build and creates missing intermediate
``__init__.py`` files so pytest's walk anchors at the workspace root.
"""

from __future__ import annotations

import subprocess
import sys

from harness.graph import (
    _ensure_test_package_init_chain,
    _test_dir_shadows_source_package,
)


class TestEnsureTestPackageInitChain:
    def test_creates_missing_intermediate_init(self, tmp_path):
        # Layout: tests/__init__.py + tests/unit/backend/__init__.py
        # but MISSING tests/unit/__init__.py — the fr-agile-20260709 shape.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "__init__.py").write_text("")
        (tmp_path / "tests" / "unit" / "backend").mkdir(parents=True)
        (tmp_path / "tests" / "unit" / "backend" / "__init__.py").write_text("")

        created = _ensure_test_package_init_chain(str(tmp_path))
        assert created == ["tests/unit/__init__.py"]
        assert (tmp_path / "tests" / "unit" / "__init__.py").is_file()

    def test_creates_tests_root_init_when_leaf_packages_exist(self, tmp_path):
        # Layout: tests/unit/backend/__init__.py exists but NEITHER
        # tests/__init__.py NOR tests/unit/__init__.py — both need
        # creating so pytest's rootdir climbs all the way up.
        (tmp_path / "tests" / "unit" / "backend").mkdir(parents=True)
        (tmp_path / "tests" / "unit" / "backend" / "__init__.py").write_text("")

        created = _ensure_test_package_init_chain(str(tmp_path))
        assert sorted(created) == [
            "tests/__init__.py",
            "tests/unit/__init__.py",
        ]

    def test_idempotent_when_chain_complete(self, tmp_path):
        (tmp_path / "tests" / "unit" / "backend").mkdir(parents=True)
        (tmp_path / "tests" / "__init__.py").write_text("")
        (tmp_path / "tests" / "unit" / "__init__.py").write_text("")
        (tmp_path / "tests" / "unit" / "backend" / "__init__.py").write_text("")

        assert _ensure_test_package_init_chain(str(tmp_path)) == []

    def test_no_op_when_no_tests_dir(self, tmp_path):
        assert _ensure_test_package_init_chain(str(tmp_path)) == []

    def test_no_op_when_tests_dir_has_no_init_packages(self, tmp_path):
        # Bare `tests/foo.py` (no __init__.py anywhere) → nothing to do.
        # Pytest uses rootdir=workspace_path from ``testpaths = tests`` in
        # this layout, and there's no shadow risk to preempt.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text("def test(): pass\n")
        assert _ensure_test_package_init_chain(str(tmp_path)) == []

    def test_end_to_end_breaks_the_shadow(self, tmp_path):
        # Full reproduction: build a workspace that fails on pytest
        # collection with the shadow error, apply the fix, confirm
        # pytest now passes.
        (tmp_path / "backend" / "core").mkdir(parents=True)
        (tmp_path / "backend" / "__init__.py").write_text("")
        (tmp_path / "backend" / "core" / "__init__.py").write_text("")
        (tmp_path / "backend" / "core" / "config.py").write_text("VALUE = 42\n")
        (tmp_path / "tests" / "unit" / "backend").mkdir(parents=True)
        (tmp_path / "tests" / "__init__.py").write_text("")
        (tmp_path / "tests" / "unit" / "backend" / "__init__.py").write_text("")
        (tmp_path / "tests" / "unit" / "backend" / "test_config.py").write_text(
            "from backend.core.config import VALUE\n"
            "def test_v(): assert VALUE == 42\n"
        )
        (tmp_path / "pytest.ini").write_text(
            "[pytest]\ntestpaths = tests\npythonpath = .\n"
        )

        # Before the fix: pytest collection fails with ModuleNotFoundError.
        pre = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/unit/backend/test_config.py", "-x", "--tb=line"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert pre.returncode != 0
        assert "ModuleNotFoundError" in (pre.stdout + pre.stderr)

        # Apply the harness fix.
        _ensure_test_package_init_chain(str(tmp_path))

        # After the fix: pytest passes cleanly.
        post = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/unit/backend/test_config.py", "-x", "--tb=line"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert post.returncode == 0, (
            f"pytest still failed after fix. stdout=\n{post.stdout}\n"
            f"stderr=\n{post.stderr}"
        )


class TestTestDirShadowsSourcePackage:
    def test_detects_shadow_directory(self, tmp_path):
        (tmp_path / "backend").mkdir()
        (tmp_path / "tests" / "unit" / "backend").mkdir(parents=True)
        assert _test_dir_shadows_source_package(str(tmp_path), "backend") == (
            "tests/unit/backend"
        )

    def test_returns_none_when_no_shadow(self, tmp_path):
        (tmp_path / "backend").mkdir()
        (tmp_path / "tests" / "unit").mkdir(parents=True)
        assert _test_dir_shadows_source_package(str(tmp_path), "backend") is None

    def test_returns_none_when_no_tests_dir(self, tmp_path):
        (tmp_path / "backend").mkdir()
        assert _test_dir_shadows_source_package(str(tmp_path), "backend") is None

    def test_returns_none_on_empty_inputs(self, tmp_path):
        assert _test_dir_shadows_source_package(str(tmp_path), "") is None
        assert _test_dir_shadows_source_package("", "backend") is None
