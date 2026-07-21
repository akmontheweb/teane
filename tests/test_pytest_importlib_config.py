"""Deterministic pytest.ini writer (``harness.test_generation``).

Two same-named test files in different packages, or a full-stack app with a
flat ``tests/`` tree plus a nested ``server/tests/`` tree, collide on
collection under pytest's default ``prepend`` import mode
(ImportPathMismatchError / silently-dropped tier — lumina 019f82af). The
harness writes a ``pytest.ini`` selecting ``--import-mode=importlib`` so the
trees coexist, plus ``pythonpath = .`` so first-party imports still resolve
(importlib mode does not prepend rootdir the way prepend mode does).

These tests lock in: the pythonpath line, self-gating on Python-test
PRESENCE (not primary stack), and the no-op / warn behavior when a config
already exists.
"""

from __future__ import annotations

import os

from harness.test_generation import (
    _PYTEST_IMPORTLIB_INI,
    _ensure_pytest_importlib_config,
    _workspace_has_python_tests,
)


def _touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def test_template_has_importlib_and_pythonpath():
    assert "--import-mode=importlib" in _PYTEST_IMPORTLIB_INI
    assert "pythonpath = ." in _PYTEST_IMPORTLIB_INI


def test_writes_config_for_python_tests(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "server", "tests", "test_x.py"), "def test_x(): pass\n")
    written = _ensure_pytest_importlib_config(ws)
    assert written == "pytest.ini"
    content = open(os.path.join(ws, "pytest.ini"), encoding="utf-8").read()
    assert "--import-mode=importlib" in content
    assert "pythonpath = ." in content


def test_self_gates_on_python_test_presence_not_primary_stack(tmp_path):
    # A workspace whose *primary* stack is the JS frontend but which HAS a
    # Python test tree must still get the config — the writer keys on
    # Python-test presence, so no `primary` argument is even needed.
    ws = str(tmp_path)
    _touch(os.path.join(ws, "client", "src", "App.tsx"), "export default 1;\n")
    _touch(os.path.join(ws, "server", "tests", "conftest.py"), "# fixtures\n")
    assert _workspace_has_python_tests(ws)
    assert _ensure_pytest_importlib_config(ws) == "pytest.ini"


def test_noop_without_python_tests(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "src", "app.py"), "x = 1\n")  # source, no tests
    assert not _workspace_has_python_tests(ws)
    assert _ensure_pytest_importlib_config(ws) is None
    assert not os.path.exists(os.path.join(ws, "pytest.ini"))


def test_noop_when_pytest_ini_already_exists(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    _touch(os.path.join(ws, "pytest.ini"), "[pytest]\naddopts = --import-mode=importlib\n")
    assert _ensure_pytest_importlib_config(ws) is None


def test_no_overwrite_of_existing_pyproject_pytest_section(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    _touch(
        os.path.join(ws, "pyproject.toml"),
        "[tool.pytest.ini_options]\naddopts = '--import-mode=importlib'\n",
    )
    assert _ensure_pytest_importlib_config(ws) is None
    assert not os.path.exists(os.path.join(ws, "pytest.ini"))
