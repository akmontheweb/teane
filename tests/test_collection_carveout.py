"""Collection-error carve-out for the repair-loop test-tamper guard.

The tamper guard refuses repair edits to test files (reward-hacking defense).
But when a pytest *collection* failure's only fix is in test infrastructure —
a broken ``conftest.py``, a missing/duplicate ``__init__.py``, a pytest config
— refusing those edits leaves the repair loop with zero legal moves. lumina
019f82af deadlocked exactly this way: two ``tests`` packages collided
(ImportPathMismatchError), the only fix was editing conftest/``__init__.py``,
the guard refused it every round, and the run ping-ponged through every HITL
auto-resume while the production code was already green.

``_syntax_broken_test_files`` now opens a carve-out for collection/import
errors — scoped to test-INFRASTRUCTURE files (conftest, test-tree
``__init__.py``, pytest config), never test-CASE files, so assertions still
can't be weakened.
"""

from __future__ import annotations

import os

from harness.graph import (
    _has_collection_error,
    _is_test_infra_file,
    _reject_test_patch_blocks,
    _syntax_broken_test_files,
)
from harness.patcher import OperationType


class _Block:
    def __init__(self, file: str, operation=OperationType.REPLACE_BLOCK) -> None:
        self.file = file
        self.operation = operation


def _mk_workspace(tmp_path) -> str:
    ws = str(tmp_path)
    for rel in (
        "tests/conftest.py",
        "tests/__init__.py",
        "tests/test_contacts.py",
        "server/tests/conftest.py",
        "server/tests/__init__.py",
        "server/app/__init__.py",  # source pkg marker — must NOT be opened
        "pytest.ini",
    ):
        p = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("# scaffold\n")
    return ws


def test_has_collection_error_matches_code_and_message():
    assert _has_collection_error([{"error_code": "ImportPathMismatchError"}])
    assert _has_collection_error([{"message": "ERROR collecting tests/foo.py"}])
    assert _has_collection_error([{"message": "import file mismatch"}])
    assert not _has_collection_error([{"error_code": "AssertionError"}])
    assert not _has_collection_error([])


def test_is_test_infra_file_scope():
    assert _is_test_infra_file("tests/conftest.py")
    assert _is_test_infra_file("server/tests/__init__.py")
    assert _is_test_infra_file("pytest.ini")
    # test-CASE files and source-package __init__.py are NOT infra
    assert not _is_test_infra_file("tests/test_contacts.py")
    assert not _is_test_infra_file("server/app/__init__.py")
    # a nested (subpackage) pyproject is not the root config
    assert not _is_test_infra_file("server/pyproject.toml")


def test_collection_error_opens_infra_files_only(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{"error_code": "ImportPathMismatchError", "file": "conftest.py"}]
    allowed = _syntax_broken_test_files(errors, ws)
    assert "tests/conftest.py" in allowed
    assert "tests/__init__.py" in allowed
    assert os.path.join("server", "tests", "conftest.py") in allowed
    # the nested second `tests` package marker — the actual lumina collider
    assert os.path.join("server", "tests", "__init__.py") in allowed
    assert "pytest.ini" in allowed
    # test-case file stays protected even during a collection error
    assert "tests/test_contacts.py" not in allowed
    # source-package __init__.py is never opened
    assert os.path.join("server", "app", "__init__.py") not in allowed


def test_no_carveout_without_collection_error(tmp_path):
    ws = _mk_workspace(tmp_path)
    # A plain assertion failure opens nothing (no parse error, no collection).
    allowed = _syntax_broken_test_files(
        [{"error_code": "AssertionError", "file": "tests/test_contacts.py"}], ws,
    )
    assert allowed == frozenset()


def test_reject_permits_conftest_edit_under_collection_error(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{"error_code": "ImportPathMismatchError", "file": "conftest.py"}]
    allowed = _syntax_broken_test_files(errors, ws)
    blocks = [
        _Block("server/app/routes.py"),      # production — always kept
        _Block("tests/conftest.py"),          # infra — kept under carve-out
        _Block("tests/test_contacts.py"),     # test case — still refused
    ]
    kept, rejections = _reject_test_patch_blocks(
        blocks, allow_parse_broken=allowed, workspace_path=ws,
    )
    kept_files = {b.file for b in kept}
    assert "server/app/routes.py" in kept_files
    assert "tests/conftest.py" in kept_files
    assert "tests/test_contacts.py" not in kept_files
    assert [r.file for r in rejections] == ["tests/test_contacts.py"]
