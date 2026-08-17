"""Deterministic pytest.ini writer (``harness.test_generation``).

Two same-named test files in different packages, or a full-stack app with a
flat ``tests/`` tree plus a nested ``server/tests/`` tree, collide on
collection under pytest's default ``prepend`` import mode
(ImportPathMismatchError / silently-dropped tier — lumina 019f82af). The
harness writes a ``pytest.ini`` selecting ``--import-mode=importlib`` so the
trees coexist, plus ``pythonpath = .`` so first-party imports still resolve
(importlib mode does not prepend rootdir the way prepend mode does).

These tests lock in: the pythonpath line, self-gating on Python-test
PRESENCE (not primary stack), the no-op when a config already selects
importlib mode, and the in-place MERGE of ``--import-mode=importlib`` into an
existing pytest config that lacks it (preserving ``asyncio_mode`` etc.).
"""

from __future__ import annotations

import os

from harness.test_generation import (
    _PYTEST_IMPORTLIB_INI,
    _ensure_pytest_importlib_config,
    _merge_importlib_into_pytest_config,
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
    # Already selects importlib → no-op, and never writes a competing pytest.ini.
    assert _ensure_pytest_importlib_config(ws) is None
    assert not os.path.exists(os.path.join(ws, "pytest.ini"))


# --- in-place merge into an existing config that LACKS importlib -----------

def test_merges_into_pyproject_preserving_asyncio_mode(tmp_path):
    # The lumina 019f82af shape: [tool.pytest.ini_options] present with
    # asyncio_mode but no addopts, so importlib was never selected. The node
    # must merge the flag in place (not write a shadowing pytest.ini that would
    # drop asyncio_mode) and report pyproject.toml as the modified file.
    ws = str(tmp_path)
    _touch(os.path.join(ws, "server", "tests", "test_x.py"), "def test_x(): pass\n")
    _touch(
        os.path.join(ws, "pyproject.toml"),
        '[tool.pytest.ini_options]\nasyncio_mode = "auto"\n',
    )
    assert _ensure_pytest_importlib_config(ws) == "pyproject.toml"
    content = open(os.path.join(ws, "pyproject.toml"), encoding="utf-8").read()
    assert "--import-mode=importlib" in content
    assert 'pythonpath = ["."]' in content
    assert 'asyncio_mode = "auto"' in content  # preserved
    # No competing pytest.ini (would shadow the pyproject table wholesale).
    assert not os.path.exists(os.path.join(ws, "pytest.ini"))


def test_merges_by_appending_to_existing_addopts(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    _touch(
        os.path.join(ws, "pyproject.toml"),
        '[tool.pytest.ini_options]\naddopts = "-q -ra"\nasyncio_mode = "auto"\n',
    )
    assert _ensure_pytest_importlib_config(ws) == "pyproject.toml"
    content = open(os.path.join(ws, "pyproject.toml"), encoding="utf-8").read()
    assert 'addopts = "-q -ra --import-mode=importlib"' in content


def test_merges_into_pytest_ini_ini_syntax(tmp_path):
    ws = str(tmp_path)
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    _touch(os.path.join(ws, "pytest.ini"), "[pytest]\naddopts = -q\n")
    assert _ensure_pytest_importlib_config(ws) == "pytest.ini"
    content = open(os.path.join(ws, "pytest.ini"), encoding="utf-8").read()
    assert "addopts = -q --import-mode=importlib" in content
    assert "pythonpath = ." in content


def test_merges_into_setup_cfg_and_tox_ini(tmp_path):
    for fname, header in (("setup.cfg", "[tool:pytest]"), ("tox.ini", "[pytest]")):
        ws = str(tmp_path / fname.replace(".", "_"))
        os.makedirs(ws, exist_ok=True)
        _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
        _touch(os.path.join(ws, fname), f"{header}\nasyncio_mode = auto\n")
        assert _ensure_pytest_importlib_config(ws) == fname
        content = open(os.path.join(ws, fname), encoding="utf-8").read()
        assert "--import-mode=importlib" in content
        assert "asyncio_mode = auto" in content


def test_unsafe_toml_array_addopts_falls_back_to_warn(tmp_path):
    # A TOML array addopts can't be text-merged safely → return None (warn),
    # leave the file untouched, and never write a competing pytest.ini.
    ws = str(tmp_path)
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    original = '[tool.pytest.ini_options]\naddopts = ["-q", "-ra"]\n'
    _touch(os.path.join(ws, "pyproject.toml"), original)
    assert _ensure_pytest_importlib_config(ws) is None
    assert open(os.path.join(ws, "pyproject.toml"), encoding="utf-8").read() == original
    assert not os.path.exists(os.path.join(ws, "pytest.ini"))


def test_merge_preserves_sibling_sections(tmp_path):
    ws = str(tmp_path)
    p = os.path.join(ws, "pyproject.toml")
    _touch(os.path.join(ws, "tests", "test_x.py"), "def test_x(): pass\n")
    content = (
        '[tool.pytest.ini_options]\nasyncio_mode = "auto"\n\n'
        "[tool.mypy]\nstrict = true\n"
    )
    _touch(p, content)
    assert _merge_importlib_into_pytest_config(p, "pyproject.toml", content) is True
    out = open(p, encoding="utf-8").read()
    assert "[tool.mypy]" in out and "strict = true" in out
    assert "--import-mode=importlib" in out
