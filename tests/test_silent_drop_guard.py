"""Silent-drop guard for the build gate (``harness.sandbox``).

A pytest run can exit 0 while quietly collecting only a subset of the test
files that exist on disk — the classic cause is two ``tests`` packages
colliding on the same dotted module name so pytest drops one tier
(lumina 019f82af: 41 of 78 tests ran, the gate went green, and the entire
``server/tests/`` tier never executed). ``extract_diagnostics`` only lifts
FAILED/ERROR rows and never checks collection completeness, so this
false-green shipped unnoticed.

``_detect_dropped_test_files`` closes that hole: it compares the test files
present on disk against the file paths pytest actually referenced in its
(verbose) output. These tests lock in the detection and its guardrails
against false positives.
"""

from __future__ import annotations

import os

from harness.sandbox import _detect_dropped_test_files


def _write(path: str, body: str = "def test_ok():\n    assert True\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def test_flags_file_present_on_disk_but_never_collected(tmp_path):
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    _write(os.path.join(ws, "server", "tests", "test_b.py"))
    # pytest (verbose) only referenced the flat tests/ tree — server/tests/
    # was silently dropped by a package-name collision.
    output = (
        "tests/test_a.py::test_ok PASSED\n"
        "===== 1 passed in 0.01s =====\n"
    )
    dropped = _detect_dropped_test_files(output, "python3 -m pytest -vv", ws)
    assert dropped == [os.path.join("server", "tests", "test_b.py")]


def test_no_drop_when_every_file_collected(tmp_path):
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    _write(os.path.join(ws, "server", "tests", "test_b.py"))
    output = (
        "tests/test_a.py::test_ok PASSED\n"
        "server/tests/test_b.py::test_ok PASSED\n"
        "===== 2 passed in 0.01s =====\n"
    )
    assert _detect_dropped_test_files(output, "python3 -m pytest -vv", ws) == []


def test_skipped_module_still_counts_as_collected(tmp_path):
    # A pytest.importorskip module reports its node id as SKIPPED — the path
    # still appears in output, so it must NOT be flagged as dropped.
    ws = str(tmp_path)
    _write(
        os.path.join(ws, "tests", "test_prop.py"),
        "import pytest\npytest.importorskip('hypothesis')\n"
        "def test_ok():\n    assert True\n",
    )
    output = "tests/test_prop.py::test_ok SKIPPED (could not import 'hypothesis')\n"
    assert _detect_dropped_test_files(output, "python3 -m pytest -vv", ws) == []


def test_ignores_helper_files_without_test_functions(tmp_path):
    # A test_*.py that contains no `def test` is a helper/scaffold — not a
    # dropped test, even if pytest never referenced it.
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    _write(
        os.path.join(ws, "tests", "test_helpers.py"),
        "SHARED = 1\n\ndef make_client():\n    return None\n",
    )
    output = "tests/test_a.py::test_ok PASSED\n===== 1 passed =====\n"
    assert _detect_dropped_test_files(output, "python3 -m pytest -vv", ws) == []


def test_bails_on_quiet_run_with_no_node_ids(tmp_path):
    # A green `-q` run prints only dots, no `path::test` node ids — per-file
    # completeness is unknowable, so the guard must not guess (no false drop).
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    _write(os.path.join(ws, "server", "tests", "test_b.py"))
    output = "..\n===== 2 passed in 0.01s =====\n"
    assert _detect_dropped_test_files(output, "python3 -m pytest -q", ws) == []


def test_noop_when_command_is_not_pytest(tmp_path):
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    assert _detect_dropped_test_files("build ok", "make build", ws) == []


def test_ignores_vendored_dirs(tmp_path):
    # Test files under .venv/node_modules are dependencies, not our suite.
    ws = str(tmp_path)
    _write(os.path.join(ws, "tests", "test_a.py"))
    _write(os.path.join(ws, ".venv", "lib", "pkg", "test_vendor.py"))
    output = "tests/test_a.py::test_ok PASSED\n===== 1 passed =====\n"
    assert _detect_dropped_test_files(output, "python3 -m pytest -vv", ws) == []
