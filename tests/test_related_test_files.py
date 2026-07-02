"""Verify _related_test_files finds ``test_<stem>.py`` under ``tests/`` for
the persistent-blocker banner's "fix may belong in the test file" hint."""

import os
import tempfile

from harness.graph import _related_test_files


def _touch(root: str, rel: str) -> None:
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").close()


def test_finds_top_level_tests_dir():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "tests/test_edgar.py")
        matches = _related_test_files("backend/services/edgar.py", td)
        assert matches == [os.path.join("tests", "test_edgar.py")]


def test_finds_nested_tests_dir_under_source_tree():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "backend/tests/test_edgar.py")
        matches = _related_test_files("backend/services/edgar.py", td)
        assert os.path.join("backend", "tests", "test_edgar.py") in matches


def test_finds_mirror_layout():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "tests/backend/services/test_edgar.py")
        matches = _related_test_files("backend/services/edgar.py", td)
        assert os.path.join("tests", "backend", "services", "test_edgar.py") in matches


def test_ignores_bare_test_file_at_root():
    # A bare ``test_edgar.py`` NOT under a ``tests/`` or ``test/`` dir is
    # almost never a real test module. Excluding it keeps the hint sharp.
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "test_edgar.py")
        matches = _related_test_files("backend/services/edgar.py", td)
        assert matches == []


def test_skips_dot_and_cache_dirs():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "node_modules/tests/test_edgar.py")
        _touch(td, ".venv/tests/test_edgar.py")
        _touch(td, "__pycache__/tests/test_edgar.py")
        _touch(td, "tests/test_edgar.py")
        matches = _related_test_files("backend/services/edgar.py", td)
        assert matches == [os.path.join("tests", "test_edgar.py")]


def test_no_match_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        _touch(td, "tests/test_something_else.py")
        assert _related_test_files("backend/services/edgar.py", td) == []


def test_never_suggests_editing_the_source_itself():
    # If someone named a source file ``test_edgar.py`` under
    # ``backend/tests/`` and it IS the stuck file, we must not suggest
    # editing itself.
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/tests/test_edgar.py")
        assert _related_test_files(
            "backend/tests/test_edgar.py", td,
        ) == []


def test_handles_empty_inputs():
    with tempfile.TemporaryDirectory() as td:
        assert _related_test_files("", td) == []
        assert _related_test_files("edgar.py", "") == []


def test_bounds_results():
    with tempfile.TemporaryDirectory() as td:
        _touch(td, "backend/services/edgar.py")
        for i in range(10):
            _touch(td, f"module{i}/tests/test_edgar.py")
        matches = _related_test_files(
            "backend/services/edgar.py", td, max_matches=3,
        )
        assert len(matches) == 3
